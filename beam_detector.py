"""
Rule-based beam-detail detector for vector PDF beam sheets (no training needed).

    python beam_detector.py drawing.pdf [page_no]      -> <name>_p<page>_detections.csv/.json/_overlay.png
    add --round-bends to keep rebar bends as drawn arcs (default: squared to sharp corners)
    --cover 25: link sizes with this cover (default: measured in the sheet's sections)
    --hook-d 12: hook allowance per hook, in bar diameters (default 12; a closed link has 2 hooks)

As a module (used by the Label Studio ML backend):
    from beam_detector import detect
    res = detect("drawing.pdf", 0)       # res["dets"], res["page_w"], res["page_h"] (PDF points, rotation removed)

Needs:  pip install pypdfium2 scipy matplotlib rapidocr_onnxruntime   (PDF access: pdf_backend.py)
"""
import csv
import sys
from collections import Counter
import sys
import json, math, os, re
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Patch, Rectangle
from matplotlib.path import Path as MplPath
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from pdf_backend import pymupdf      # pypdfium2 behind PyMuPDF's API (BEAM_PDF=pymupdf: PyMuPDF itself)

_OCR = None
def get_ocr():
    global _OCR
    if _OCR is None:
        from rapidocr_onnxruntime import RapidOCR
        _OCR = RapidOCR()
    return _OCR

PT_MM = 25.4 / 72

# a rebar callout at the start of a string, e.g. "T20-200 SIDE BAR" -> "T20-200", "SIDE BAR"
_BAR = r"(?:\d+x)?\d*[HRTY]\d{1,2}(?:-\d{2,3})?"
RE_LABEL_HEAD = re.compile(rf"^\s*(?P<lab>(?:\d{{1,3}}\s*[xX×]\s*)?{_BAR}(?:\s*\+\s*{_BAR})*(?:\s*\((?:T|B)\))?)(?:\s+(?P<sfx>.*))?$")
RE_NUM_TXT = re.compile(r"^\d{2,6}$")
LEAD_JUNK = "[|-_~–— "      # OCR: a leader line touching a label is read as one of these in front of it


# --- title block: drawing number and drawing title (sheet-level, no relations) -----------------------
RE_DWGNO_LAB = re.compile(r"(?i)\b(?:drawing|dwg)\.?\s*no\b|\bno\.?\s*lukisan\b|\bthis\s+sheet\s+no\b|\bsheet\s+no\b")
RE_TITLE_LAB = re.compile(r"(?i)\bdrawing\s+title\b|\btajuk\s+lukisan\b|^\s*lukisan\b|^\s*\(?drawing\)?\s*$|^\s*title\s*$")
RE_DWGNO = re.compile(r"^(?=.*\d)(?=.*[A-Za-z])[A-Za-z0-9]+(?:[_/\-.][A-Za-z0-9]+){2,}$")      # WEB786_L11_BM05, WH/RC/L7/14
RE_TITLE_WORDS = re.compile(r"(?i)\b(?:details?|plans?|layout|sections?|elevations?|schedules?|beams?|slabs?|columns?|walls?|"
                            r"footings?|piles?|pile\s*caps?|stairs?|reinforcement|framing|key\s*plan)\b")


def title_block(page, pdf_stem=""):
    """Drawing number and drawing title from the title block's text layer.
    With field labels as text ('No. Lukisan / Drawing No.', 'Drawing Title'): the bold/larger text
    under the label. Labels drawn as strokes (JNK): by pattern inside the title block - the first
    sheet number ('this sheet', above 'next sheet', or the one matching the file name), and the
    largest line with drawing-title words (BEAM DETAILS, PLAN ...)."""
    W, H = page.rect.width, page.rect.height
    lines = []
    for b in page.get_text("dict")["blocks"]:
        for l in b.get("lines", []):
            t = " ".join(s["text"].strip() for s in l["spans"] if s["text"].strip())
            if t:
                lines.append(dict(t=t, bb=[float(v) for v in l["bbox"]], sz=max(s["size"] for s in l["spans"]),
                                  bold=any("bold" in s["font"].lower() or s["flags"] & 16 for s in l["spans"])))
    in_tb = lambda L: L["bb"][0] >= 0.75 * W or L["bb"][1] >= 0.82 * H         # right strip or bottom strip
    is_lab = lambda L: bool(RE_DWGNO_LAB.search(L["t"]) or RE_TITLE_LAB.search(L["t"])) or L["t"].rstrip().endswith((":", "/"))
    out = []

    def under(lab, title):
        x0, y0, x1, y1 = lab["bb"]; h = max(y1 - y0, 1.0)
        c = [L for L in lines if L is not lab and in_tb(L) and y1 - 0.4 * h <= L["bb"][1] <= y1 + 12 * h
             and x0 - 3 * h <= L["bb"][0] <= x0 + 40 * h and not is_lab(L) and (L["bold"] or L["sz"] >= 1.15 * lab["sz"])]
        if not title:
            c = [L for L in c if RE_DWGNO.match(L["t"].replace(" ", ""))]
        if not c:
            return None
        first = min(c, key=lambda L: L["bb"][1])
        got = [first]
        if title:                                    # a title may run over two or three lines
            for L in sorted(c, key=lambda L: L["bb"][1]):
                prev = got[-1]
                if L is not first and abs(L["sz"] - first["sz"]) < 0.5 and 0 <= L["bb"][1] - prev["bb"][3] < 0.8 * (prev["bb"][3] - prev["bb"][1]) \
                        and abs(L["bb"][0] - first["bb"][0]) < 6 * h:
                    got.append(L)
        return got

    labs = [L for L in lines if in_tb(L) and "refer" not in L["t"].lower()]
    for cls, rx, title in (("DrawingNo", RE_DWGNO_LAB, False), ("DrawingTitle", RE_TITLE_LAB, True)):
        found = None
        for lab in sorted([L for L in labs if rx.search(L["t"])], key=lambda L: -L["bb"][1]):     # lowest label first
            found = under(lab, title)
            if found:
                how = "pdf-text"; break
        if not found:                                # field labels not in the text layer: by pattern
            tb = [L for L in lines if in_tb(L) and "refer" not in L["t"].lower()]
            if not title:
                stem = re.sub(r"[^a-z0-9]", "", pdf_stem.lower())
                c = [L for L in tb if RE_DWGNO.match(L["t"].replace(" ", "")) and (L["bb"][0] >= 0.75 * W and L["bb"][1] >= 0.5 * H or L["bb"][1] >= 0.82 * H)]
                if c:
                    found = [min(c, key=lambda L: (not (stem and re.sub(r"[^a-z0-9]", "", L["t"].lower()) in stem), -round(L["sz"]), L["bb"][1]))]
            else:
                med = float(np.median([L["sz"] for L in tb])) if tb else 0
                c = [L for L in tb if RE_TITLE_WORDS.search(L["t"]) and len(L["t"].split()) >= 2 and L["sz"] >= 1.3 * med
                     and L["t"].upper() == L["t"]]
                if c:
                    found = [max(c, key=lambda L: (round(L["sz"]), -L["bb"][1]))]
            how = "title block (by pattern)"
        if found:
            bb = [min(L["bb"][0] for L in found), min(L["bb"][1] for L in found), max(L["bb"][2] for L in found), max(L["bb"][3] for L in found)]
            out.append(dict(cls=cls, text=" ".join(L["t"] for L in found), bbox=bb, source=how,
                            conf=1.0 if how == "pdf-text" else 0.8))
    return out


def ocr_tiles(page, rect=None, dpi=300, tile_px=1600, overlap=60.0, rotated=True, log=print):
    """OCR small annotation text (labels, dimension numbers) drawn as strokes.
    The page is cut into overlapping tiles at `dpi` (small CAD text is lost on a whole-page image).
    rotated=True: a second pass on tiles turned 90 deg reads vertical (bottom-to-top) dimension text.
    Returns [dict(txt, conf, bb=[x0,y0,x1,y1] in pt, dir=[dx,dy])]."""
    rect = rect or page.rect
    z = dpi / 72
    T = tile_px / z
    out = []
    xs = np.arange(rect.x0, rect.x1, T - overlap)
    ys = np.arange(rect.y0, rect.y1, T - overlap)
    for y in ys:
        for x in xs:
            r = pymupdf.Rect(x, y, min(x + T, rect.x1), min(y + T, rect.y1))
            pix = page.get_pixmap(clip=r, dpi=dpi, colorspace=pymupdf.csGRAY)
            img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width)
            if (img < 128).mean() < 0.002:                 # empty tile
                continue
            H = img.shape[0]
            passes = [(img, False)] + ([(np.ascontiguousarray(np.rot90(img, -1)), True)] if rotated else [])
            for im, rot in passes:
                res, _ = get_ocr()(np.stack([im] * 3, -1))
                for quad, txt, conf in res or []:
                    q = np.array(quad, float)
                    if rot:                                 # rotated clockwise: (x', y') = (H-1-y, x)
                        q = np.column_stack([q[:, 1], H - 1 - q[:, 0]])
                    q = q / z + [r.x0, r.y0]
                    bb = [*q.min(0), *q.max(0)]
                    w, h = bb[2] - bb[0], bb[3] - bb[1]
                    if rot and not (h > 1.5 * w and (RE_NUM_TXT.match(txt) or RE_LABEL_HEAD.match(txt))):
                        continue                            # rotated pass: keep vertical numbers/labels only
                    out.append(dict(txt=txt.strip(), conf=float(conf), bb=bb, dir=[0.0, -1.0] if rot else [1.0, 0.0]))
    # tiles overlap: keep one copy (the longer text, then the higher confidence)
    out.sort(key=lambda o: (-len(o["txt"]), -o["conf"]))
    keep = []
    for o in out:
        a = o["bb"]
        dup = False
        for k in keep:
            b = k["bb"]
            ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
            if ix > 0.5 * min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])):
                dup = True
                break
        if not dup:
            keep.append(o)
    log(f"OCR tiles @ {dpi} dpi: {len(keep)} text items")
    return keep


def link_set(size, cover, pattern=None, pattern_size=None, pattern_name=""):
    """The links of one set in a beam of `size` (width, depth, other depth or None) with `cover` (sides,
    top and bottom) in mm. The outer link is (width - 2 x cover) x (depth - 2 x cover), at each depth.
    `pattern`: the links drawn in a section [(kind, w, h)], outer first. It adds the inner links: width as
    drawn (scaled when this beam is wider or narrower than the section's beam, `pattern_size`), depth as the
    outer link's less the same difference as drawn. -> ([(kind, a, [b per depth])], note)"""
    r5 = lambda v: int(5 * round(v / 5))
    cx, cy = cover
    hs = [r5(h - 2 * cy) for h in size[1:] if h]
    a_out = r5(size[0] - 2 * cx)
    if not pattern:
        return [("outer", a_out, hs)], ""
    _, ow, oh = pattern[0]
    same_w = pattern_size[0] == size[0] if pattern_size else abs(a_out - ow) <= 10
    out = [("outer", a_out, hs) if kind == "outer" else
           ("inner", w if same_w else r5(w * a_out / ow), [b - (oh - h) for b in hs]) for kind, w, h in pattern]
    note = "" if same_w or len(out) < 2 else \
        f"inner link width scaled from {pattern_name} ({pattern_size[0] if pattern_size else '?'} wide beam), check"
    return out, note


def link_cut(a, b, dia, hook_d=12, hooks=2):
    """Cut length of a closed link a x b (mm): 2 x (a + b) + hooks x hook_d x bar size."""
    return int(round(2 * (a + b) + hooks * hook_d * dia))


def detect(pdf_path, page_no=0, verbose=True, square_bends=True, force_ocr=False, cover=None, hook_d=12):
    """Detect beam names/sizes, rebar labels, dimensions and their lines on one PDF page.
    square_bends=True: each rebar bend (arc) is replaced by a sharp corner where the straight legs meet.
    force_ocr=True: ignore labels/numbers in the text layer and read them with OCR (garbled text layers).
    cover: link sizes with this cover in mm (None: measured in the sheet's sections, concrete face to link).
    hook_d: hook allowance per hook in bar diameters; link cut length = 2(a + b) + 2 x hook_d x bar size."""
    log = print if verbose else (lambda *a, **k: None)
    PDF_PATH = Path(pdf_path)
    PAGE_NO = int(page_no)
    ENDPOINT_TOL, TOUCH_TOL, MARK_MAX_DIAG = 0.05, 0.15, 1.5


    doc = pymupdf.open(PDF_PATH)
    page = doc[PAGE_NO]
    if page.rotation:
        log(f"page /Rotate {page.rotation} removed (in memory) so coordinates match the rendered image")
        page.remove_rotation()
    PAGE_W, PAGE_H = page.rect.width, page.rect.height
    log(f"{PDF_PATH.name} p{PAGE_NO}: {PAGE_W:.0f} x {PAGE_H:.0f} pt = {PAGE_W/72*25.4:.0f} x {PAGE_H/72*25.4:.0f} mm")

    def _word(chars, span, d):
        bb = np.array([c["bbox"] for c in chars])
        return dict(kind="text", text="".join(c["c"] for c in chars),
                    bbox=[bb[:, 0].min(), bb[:, 1].min(), bb[:, 2].max(), bb[:, 3].max()],
                    size=span["size"], dir=d.tolist(), font=span["font"], color=span["color"])

    def extract_words(page, gap_factor=0.6):
        # Split on spaces AND on wide gaps: CAD exports often place every character separately.
        words = []
        for block in page.get_text("rawdict")["blocks"]:
            for line in block.get("lines", []):
                d = np.array(line["dir"], float)
                for span in line["spans"]:
                    cur, prev_hi = [], None
                    for ch in span["chars"]:
                        x0, y0, x1, y1 = ch["bbox"]
                        proj = [d @ (x0, y0), d @ (x1, y1), d @ (x0, y1), d @ (x1, y0)]
                        lo, hi = min(proj), max(proj)
                        wide_gap = prev_hi is not None and lo - prev_hi > gap_factor * span["size"]
                        if ch["c"].isspace() or wide_gap:
                            if cur:
                                words.append(_word(cur, span, d))
                            cur, prev_hi = [], None
                            if ch["c"].isspace():
                                continue
                        cur.append(ch)
                        prev_hi = hi
                    if cur:
                        words.append(_word(cur, span, d))
        return words

    words = extract_words(page)
    # CAD exports often stack identical text objects; keep one
    _seen, _uniq = set(), []
    for w in words:
        key = (w["text"], *(round(v) for v in w["bbox"]))
        if key not in _seen:
            _seen.add(key); _uniq.append(w)
    words = _uniq

    # Annotation text exported as strokes (SHX fonts): the text layer then holds only the title
    # block. Read labels and dimension numbers with OCR and treat them as text nodes.
    if force_ocr:
        words = [w for w in words if not (RE_LABEL_HEAD.match(w["text"]) or RE_NUM_TXT.match(w["text"]))]
    n_lab = sum(bool(RE_LABEL_HEAD.match(w["text"])) for w in words)
    OCR_TEXT = n_lab < 5
    ocr_words = []
    W_bb = np.array([w["bbox"] for w in words]) if words else np.zeros((0, 4))
    if OCR_TEXT:
        log(f"only {n_lab} rebar labels in the text layer -> labels are drawn as strokes, reading them with OCR")
        for it in ocr_tiles(page, log=log):
            txt, (x0, y0, x1, y1), vert = it["txt"], it["bb"], it["dir"][1] != 0
            if len(W_bb):                                   # already in the text layer (e.g. real-text dims)
                ix = (np.minimum(x1, W_bb[:, 2]) - np.maximum(x0, W_bb[:, 0])).clip(0) * \
                     (np.minimum(y1, W_bb[:, 3]) - np.maximum(y0, W_bb[:, 1])).clip(0)
                if ix.max() > 0.3 * (x1 - x0) * (y1 - y0):
                    continue
            n_lead = len(txt) - len(txt.lstrip(LEAD_JUNK))      # leader end read as '-', '_' ... in front
            if n_lead and RE_LABEL_HEAD.match(txt[n_lead:]):
                k = n_lead / len(txt)
                if vert: y1 -= (y1 - y0) * k
                else: x0 += (x1 - x0) * k
                txt = txt[n_lead:]
            m = RE_LABEL_HEAD.match(txt)
            if m:
                lab = re.sub(r"\s+", "", m["lab"]).replace("(T)", " (T)").replace("(B)", " (B)")
                frac = min(1.0, (len(m["lab"].strip()) + 0.3) / max(len(txt), 1))   # box share of the label part
                bb = [x0, y1 - (y1 - y0) * frac, x1, y1] if vert else [x0, y0, x0 + (x1 - x0) * frac, y1]
                sfx = m["sfx"] or ""
            elif RE_NUM_TXT.match(txt):
                lab, bb, sfx = txt, [x0, y0, x1, y1], ""
            else:
                continue                                    # titles etc.: the title pass below reads them
            h = (bb[2] - bb[0]) if vert else (bb[3] - bb[1])
            ocr_words.append(dict(kind="text", text=lab, bbox=bb, size=h / 1.1, dir=it["dir"], font="ocr",
                                  color=0, conf=it["conf"], suffix=sfx, source="ocr-text"))
        words += ocr_words
    sizes = [w["size"] for w in words if RE_LABEL_HEAD.match(w["text"]) or RE_NUM_TXT.match(w["text"])] \
        or [w["size"] for w in words]                       # annotation text sets the scale of tolerances
    UNIT = float(np.median(sizes)) if sizes else 7.0
    log(f"{len(words)} words ({len(ocr_words)} from OCR), UNIT (median text size) = {UNIT:.2f} pt")
    OCR_BB = np.array([w["bbox"] for w in ocr_words]) if ocr_words else np.zeros((0, 4))


    def bezier(p0, p1, p2, p3, n=8):
        t = np.linspace(0, 1, n + 1)[:, None]
        p0, p1, p2, p3 = (np.array(p, float) for p in (p0, p1, p2, p3))
        return (1 - t)**3 * p0 + 3 * (1 - t)**2 * t * p1 + 3 * (1 - t) * t**2 * p2 + t**3 * p3

    def path_segments(path):
        # One get_drawings() path -> straight segments (curves flattened), shape (n, 2, 2)
        segs, flags, curved = [], [], False          # flags: True = piece of a flattened curve
        for it in path["items"]:
            op = it[0]
            if op == "l":
                segs.append([tuple(it[1]), tuple(it[2])]); flags.append(False)
            elif op == "c":
                pts = bezier(*it[1:5]); curved = True
                segs += [[tuple(pts[i]), tuple(pts[i + 1])] for i in range(len(pts) - 1)]
                flags += [True] * (len(pts) - 1)
            elif op in ("re", "qu"):
                q = it[1].quad if op == "re" else it[1]
                c = [tuple(q.ul), tuple(q.ur), tuple(q.lr), tuple(q.ll)]
                segs += [[c[i], c[(i + 1) % 4]] for i in range(4)]; flags += [False] * 4
        if path.get("closePath") and len(segs) > 1 and math.dist(segs[-1][1], segs[0][0]) > 1e-6:
            segs.append([segs[-1][1], segs[0][0]]); flags.append(False)
        return np.array(segs, float).reshape(-1, 2, 2), np.array(flags, bool), curved

    def split_components(segs, tol):
        # A PDF path can hold many unrelated lines; split it into connected polylines.
        n = len(segs)
        if n == 1:
            return [np.arange(1)]
        pairs = cKDTree(segs.reshape(-1, 2)).query_pairs(tol, output_type="ndarray")
        g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0] // 2, pairs[:, 1] // 2)), shape=(n, n))
        k, lab = connected_components(g, directed=False)
        return [np.where(lab == i)[0] for i in range(k)]

    def geom_props(segs):
        v = segs[:, 1] - segs[:, 0]
        seglen = np.hypot(v[:, 0], v[:, 1])
        ang2 = 2 * np.arctan2(v[:, 1], v[:, 0])               # doubled angle = direction-free orientation
        theta = 0.5 * np.arctan2((seglen * np.sin(ang2)).sum(), (seglen * np.cos(ang2)).sum())
        pts = segs.reshape(-1, 2)
        proj = pts @ np.array([np.cos(theta), np.sin(theta)])
        _, cnt = np.unique(np.round(pts, 1), axis=0, return_counts=True)
        return dict(length=float(seglen.sum()), span=float(proj.max() - proj.min()), theta=float(theta),
                    bbox=[*pts.min(0), *pts.max(0)], n_vertices=len(cnt),
                    closed=bool(len(segs) >= 3 and np.all(cnt == 2)))

    def rgb(c):
        if not c: return (0.0, 0.0, 0.0)
        if len(c) == 1: return (c[0],) * 3
        if len(c) == 4: return tuple((1 - c[i]) * (1 - c[3]) for i in range(3))   # CMYK
        return tuple(c)

    nodes = [dict(w) for w in words]          # text nodes first
    for path in page.get_drawings():
        segs, seg_curve, curved = path_segments(path)
        if len(segs) == 0:
            continue
        filled = path.get("fill") is not None
        dashed = bool(re.search(r"\[\s*\d", path.get("dashes") or ""))
        for idx in split_components(segs, ENDPOINT_TOL * UNIT):
            s = segs[idx]
            g = geom_props(s)
            if len(OCR_BB) and (path.get("width") or 0) <= 0.3:     # stroke glyph of OCR'd text: not linework
                pad = 0.15 * UNIT
                inside = (g["bbox"][0] >= OCR_BB[:, 0] - pad) & (g["bbox"][1] >= OCR_BB[:, 1] - pad) & \
                         (g["bbox"][2] <= OCR_BB[:, 2] + pad) & (g["bbox"][3] <= OCR_BB[:, 3] + pad)
                if inside.any():
                    continue
            diag = math.hypot(g["bbox"][2] - g["bbox"][0], g["bbox"][3] - g["bbox"][1])
            kind = "mark" if diag <= MARK_MAX_DIAG * UNIT and (filled or g["closed"]) else "line"
            nodes.append(dict(kind=kind, segs=s, seg_curve=seg_curve[idx], curved=curved, filled=filled, dashed=dashed,
                              width=path.get("width") or 0.0, color=rgb(path.get("color") or path.get("fill")),
                              layer=path.get("layer") or "", **g))

    for n in nodes:
        n["bbox"] = [float(v) for v in n["bbox"]]
        n["c"] = ((n["bbox"][0] + n["bbox"][2]) / 2, (n["bbox"][1] + n["bbox"][3]) / 2)

    kinds = np.array([n["kind"] for n in nodes])
    text_ids = np.where(kinds == "text")[0]
    geo_ids = np.where(kinds != "text")[0]
    log({k: int((kinds == k).sum()) for k in ("text", "line", "mark")})
    layers = sorted({n["layer"] for n in nodes if n.get("layer")})
    log("PDF layers:", layers[:20] if layers else "none")

    STEP = TOUCH_TOL * UNIT / 2
    S_xy, S_own = [], []
    for i in geo_ids:
        for a, b in nodes[i]["segs"]:
            m = max(1, math.ceil(math.dist(a, b) / STEP))
            S_xy.append(a + np.linspace(0, 1, m + 1)[:, None] * (b - a))
            S_own.append(np.full(m + 1, i))
    S_xy, S_own = np.vstack(S_xy), np.concatenate(S_own)      # S_own is sorted by node id
    S_tree = cKDTree(S_xy)

    def node_pts(i):
        lo, hi = np.searchsorted(S_own, [i, i + 1])
        return S_xy[lo:hi]

    log(f"{len(S_xy):,} sample points, step {STEP:.2f} pt")

    # =====================================================================================
    # Detection rules (everything above is the same extraction code as the notebook)
    # =====================================================================================

    line_ids = [i for i in geo_ids if nodes[i]["kind"] == "line"]
    W = np.array([round(nodes[i]["width"], 2) for i in line_ids])

    # --- line-style roles, learned from this sheet ------------------------------------
    # rebar = thickest stroke that is used often; dimension style = the width found under dim numbers (below)
    wc = Counter(W.tolist())
    REBAR_W = max(w for w, n in wc.items() if n >= 10)
    log("stroke widths:", dict(sorted(wc.items())), "-> rebar width", REBAR_W)

    # segment-level arrays (dimension lines are taken per straight segment, not per polyline)
    SEG, SEG_own = [], []
    for i in line_ids:
        for s in nodes[i]["segs"]:
            SEG.append(s); SEG_own.append(i)
    SEG, SEG_own = np.array(SEG), np.array(SEG_own)
    SEG_w = np.array([round(nodes[i]["width"], 2) for i in SEG_own])
    SEG_v = SEG[:, 1] - SEG[:, 0]
    SEG_len = np.hypot(*SEG_v.T)
    SEG_u = SEG_v / np.maximum(SEG_len, 1e-9)[:, None]

    def segs_under(c, d, max_perp, min_len=0.0, max_angle=0.1, widths=None):
        """Segments roughly parallel to direction d, whose extent contains point c, within max_perp of it."""
        rel = c - SEG[:, 0]
        t = (rel * SEG_u).sum(1)
        perp = np.abs(rel[:, 0] * SEG_u[:, 1] - rel[:, 1] * SEG_u[:, 0])
        par = np.abs(SEG_u[:, 0] * d[1] - SEG_u[:, 1] * d[0]) < max_angle
        ok = par & (t >= -1) & (t <= SEG_len + 1) & (perp <= max_perp) & (SEG_len >= min_len)
        if widths is not None:
            ok &= np.isin(SEG_w, widths)
        idx = np.where(ok)[0]
        return idx[np.argsort(perp[idx])], perp

    # --- 1. text-layer words ----------------------------------------------------------
    # glue "(T)" onto the label in front of it: "H10-300" "(T)" -> "H10-300 (T)"
    merged = []
    for i in text_ids:
        n = nodes[i]
        if n["text"] in ("(T)", "(B)") and merged:
            p = nodes[merged[-1]]
            if abs(p["bbox"][3] - n["bbox"][3]) < 1 and 0 <= n["bbox"][0] - p["bbox"][2] < UNIT:
                p["text"] += " " + n["text"]
                p["bbox"] = [min(p["bbox"][0], n["bbox"][0]), min(p["bbox"][1], n["bbox"][1]),
                             max(p["bbox"][2], n["bbox"][2]), max(p["bbox"][3], n["bbox"][3])]
                p["c"] = ((p["bbox"][0] + p["bbox"][2]) / 2, (p["bbox"][1] + p["bbox"][3]) / 2)
                continue
        merged.append(i)
    text_ids = np.array(merged)

    VALID_DIA = {6, 8, 10, 12, 13, 16, 20, 25, 28, 32, 40}
    PART = re.compile(r"^(?:(?P<sets>\d+)x)?(?P<n>\d+)?(?P<g>[HRTY])(?P<d>\d{1,2})(?:-(?P<s>\d{2,3}))?$")
    def parse_label(s):
        s = re.sub(r"\s*([xX×+])\s*(?=\d*[HRTY]?\d)", lambda m: "+" if m[1] == "+" else "x", s.strip())
        core = re.sub(r"\s*\((T|B)\)$", "", s.strip())
        parts = []
        for p in core.split("+"):
            m = PART.match(p)
            if not m:
                return None
            part = dict(sets=int(m["sets"] or 1), count=int(m["n"] or 1), grade=m["g"], dia=int(m["d"]),
                        spacing=int(m["s"]) if m["s"] else None)
            if part["dia"] not in VALID_DIA or not 1 <= part["count"] <= 40 or \
                    (part["spacing"] is not None and not 50 <= part["spacing"] <= 600):
                return None          # e.g. OCR noise like 'H1-01', 'H02'
            parts.append(part)
        kind = "link" if all(p["spacing"] for p in parts) else "bar" if not any(p["spacing"] for p in parts) else None
        if not kind:
            return None
        lab = dict(parts=parts, kind=kind, face=re.search(r"\((T|B)\)$", s.strip()).group(1) if "(" in s else None)
        if kind == "link" and parts[0]["sets"] > 1:
            # '3 x T12-100 CLOSED LINKS' = 3 links in each set (nested closed links, as the section
            # shows), sets at 100 c/c along the link zone
            lab["links_per_set"] = parts[0]["sets"]
        return lab

    RE_NUM = re.compile(r"^\d{2,6}$")
    dets = []                      # every detected entity
    def add_det(cls, text, bbox, source, **kw):
        dets.append(dict(id=len(dets), cls=cls, text=text, bbox=[float(v) for v in bbox], source=source, **kw))
        return dets[-1]

    def suffix_of(i, max_words=3):
        """Words after a label on the same line ('SIDE BAR', 'CLOSED LINKS + LINKS')."""
        n = nodes[i]
        if n.get("suffix") is not None:
            return n["suffix"]
        d = np.array(n["dir"], float); nrm = np.array([-d[1], d[0]])
        out, cur = [], n
        for _ in range(max_words):
            c = np.array(cur["c"]); hi = max(d @ (cur["bbox"][0], cur["bbox"][1]), d @ (cur["bbox"][2], cur["bbox"][3]))
            best = None
            for j in text_ids:
                o = nodes[j]
                if o is cur or o is n: continue
                oc = np.array(o["c"])
                lo = min(d @ (o["bbox"][0], o["bbox"][1]), d @ (o["bbox"][2], o["bbox"][3]))
                if abs((oc - c) @ nrm) < 0.5 * cur["size"] and -0.2 * UNIT <= lo - hi <= 1.5 * UNIT:
                    if best is None or lo < best[0]: best = (lo, o)
            if not best: break
            out.append(best[1]["text"]); cur = best[1]
        return " ".join(out)

    def prefix_of(i):
        """'3 x' written as separate words in front of a label -> those words (nearest first)."""
        n = nodes[i]
        d = np.array(n["dir"], float); nrm = np.array([-d[1], d[0]])
        def before(cur):
            lo = min(d @ (cur["bbox"][0], cur["bbox"][1]), d @ (cur["bbox"][2], cur["bbox"][3]))
            best = None
            for j in text_ids:
                o = nodes[j]
                if o is cur: continue
                hi = max(d @ (o["bbox"][0], o["bbox"][1]), d @ (o["bbox"][2], o["bbox"][3]))
                if abs((np.array(o["c"]) - np.array(cur["c"])) @ nrm) < 0.5 * cur["size"] and -0.2 * UNIT <= lo - hi <= 1.2 * UNIT:
                    if best is None or lo - hi < best[0]: best = (lo - hi, int(j))
            return best[1] if best else None
        j1 = before(n)
        if j1 is None: return []
        t1 = nodes[j1]["text"].strip()
        if re.match(r"^\d{1,3}\s*[xX×]$", t1): return [j1]
        if t1 in ("x", "X", "×"):
            j0 = before(nodes[j1])
            if j0 is not None and re.match(r"^\d{1,3}$", nodes[j0]["text"].strip()): return [j1, j0]
        return []

    def length_word(i):
        """the '(3300)' word right after a label on the same line"""
        n = nodes[i]; best = None
        for j in text_ids:
            o = nodes[j]
            if o is n or not re.match(r"^\(\d{3,5}\)", o["text"].strip()): continue
            if abs(o["c"][1] - n["c"][1]) < 0.5 * n["size"] and -0.2 * UNIT <= o["bbox"][0] - n["bbox"][2] <= 1.5 * UNIT:
                if best is None or o["bbox"][0] < nodes[best]["bbox"][0]: best = int(j)
        return best

    consumed, prefix = set(), {}
    for i in text_ids:
        if parse_label(nodes[i]["text"]) and not re.match(r"^\d+x", nodes[i]["text"]):
            pre = prefix_of(i)
            if pre:
                prefix[int(i)] = pre; consumed.update(pre)

    for i in text_ids:
        if int(i) in consumed:
            continue
        n = nodes[i]
        src = n.get("source", "pdf-text")
        extra = dict(conf=n["conf"]) if "conf" in n else {}
        if int(i) in prefix:                                 # '3 x T12-100' -> '3xT12-100'
            pw = [nodes[j] for j in prefix[int(i)]]
            num = re.match(r"^\d+", pw[-1]["text"].strip())[0]
            n = dict(n, text=f"{num}x{n['text']}",
                     bbox=[min(o["bbox"][0] for o in pw + [n]), min(o["bbox"][1] for o in pw + [n]),
                           max(o["bbox"][2] for o in pw + [n]), max(o["bbox"][3] for o in pw + [n])])
        lab = parse_label(n["text"])
        blen = None
        if not lab and (mL := re.match(r"^(.+?)\((\d{3,5})\)$", n["text"].strip())):    # TSSC: '1T25(4700)' bar + length
            lab = parse_label(mL[1]); blen = int(mL[2]) if lab else None
        if lab:
            sfx = suffix_of(i).upper()
            if blen is None and (mS := re.match(r"^\((\d{3,5})\)", sfx)):             # '1T25 (3300)': length as the next word
                blen = int(mS[1]); nx = length_word(i)
                if nx is not None:
                    o = nodes[nx]; consumed.add(int(nx))
                    n = dict(n, text=f"{n['text']}({blen})", bbox=[min(n["bbox"][0], o["bbox"][0]), min(n["bbox"][1], o["bbox"][1]),
                                                                  max(n["bbox"][2], o["bbox"][2]), max(n["bbox"][3], o["bbox"][3])])
                sfx = sfx[len(mS[0]):].strip()
            if blen is not None: lab["length"] = blen
            if "SIDE" in sfx or re.match(r"^(S\.?B\.?|E\.?F\.?)(\s|$)", sfx):
                lab["kind"], lab["side"] = "bar", True            # 'T20-200 SIDE BAR': side face bars, not a link
            add_det("RebarSize", n["text"], n["bbox"], src, node=int(i), label=lab, size=n["size"], dir=n["dir"],
                    suffix=sfx, **extra)

        elif RE_NUM.match(n["text"]):
            add_det("Dim", n["text"], n["bbox"], src, node=int(i), value=float(n["text"]), size=n["size"], dir=n["dir"], **extra)

    # --- 2. outlined text via OCR (beam titles, section labels) ------------------------
    OCR_DPI = 150
    pix = page.get_pixmap(dpi=OCR_DPI)
    ocr_img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)[..., :3]
    ocr, _ = get_ocr()(ocr_img)
    k = 72 / OCR_DPI
    T_bb = np.array([nodes[i]["bbox"] for i in text_ids])
    NAME = r"\d{0,2}[A-Za-z]{0,3}B[A-Z]?\d{1,4}[A-Za-z]?"     # lower case too: Ramboll names beams 11aB1a, 1aB13, 1aBM4
    # (600x500)  600x500  600(WD)x1125/825(DP)  (width x depth; two depths = depth changes along the beam)
    SIZE = r"\(?\s*(?P<b>\d{2,4})\s*(?:\(\s*W[A-Z]*\s*\))?\s*[xX×]\s*(?P<h>\d{2,4}(?:\s*/\s*\d{2,4})?)"
    RE_TITLE = re.compile(rf"^\s*(?P<name>{NAME})\s*(?P<rest>{SIZE}.*)$")      # 7B224 (600x500), 7B144 (300x600) + (125x600)
    RE_NAME_ONLY = re.compile(rf"^\s*(?P<name>{NAME})\s*$")                     # title split over two lines
    RE_SIZE_ONLY = re.compile(rf"^\s*(?P<rest>{SIZE}.*)$")
    items = []
    for quad, txt, conf in ocr or []:
        q = np.array(quad) * k
        bb = [*q.min(0), *q.max(0)]
        # skip anything the text layer already has (OCR boxes are looser than text boxes)
        if len(T_bb):
            ov = (np.minimum(bb[2], T_bb[:, 2]) - np.maximum(bb[0], T_bb[:, 0])).clip(0) * \
                 (np.minimum(bb[3], T_bb[:, 3]) - np.maximum(bb[1], T_bb[:, 1])).clip(0)
            a_t = (T_bb[:, 2] - T_bb[:, 0]) * (T_bb[:, 3] - T_bb[:, 1])
            a_o = (bb[2] - bb[0]) * (bb[3] - bb[1])
            if (ov / np.maximum(np.minimum(a_t, a_o), 1e-6)).max(initial=0) > 0.5:
                continue
        items.append(dict(bb=bb, txt=txt.strip(), conf=float(conf), used=False, src="ocr"))
    # titles written as real text (name and size are separate words): same matching as OCR text
    RE_SECNO = re.compile(r"^\s*(\w{1,6})\s*-\s*\1\s*$")                      # SECTION 14-14, 15A-15A, A-A: the same id twice
    RE_SECID1 = re.compile(r"^\s*[A-Za-z]{0,3}\d{1,3}[A-Za-z]?\s*$")             # Ramboll's one id: SECTION 3, SECTION 14b
    sec_words = [nodes[i]["bbox"] for i in text_ids if nodes[i]["text"].strip().upper() == "SECTION"]

    def after_section(bb):                              # the word right after a SECTION, on the same line
        h = bb[3] - bb[1]
        return any(abs((sw[1] + sw[3]) / 2 - (bb[1] + bb[3]) / 2) < 0.5 * max(h, sw[3] - sw[1])
                   and -0.2 * UNIT <= bb[0] - sw[2] <= 2 * UNIT for sw in sec_words)
    for i in text_ids:
        t = nodes[i]["text"]
        if RE_TITLE.match(t) or RE_NAME_ONLY.match(t) or RE_SIZE_ONLY.match(t) or t.strip().upper() == "SECTION" \
                or RE_SECNO.match(t) or (RE_SECID1.match(t) and after_section(nodes[i]["bbox"])):
            items.append(dict(bb=list(nodes[i]["bbox"]), txt=t, conf=1.0, used=False, src="pdf-text"))

    for i in text_ids:                                  # TSSC: '1 - 1' written as three words under the section
        if nodes[i]["text"].strip() != "-": continue
        c = nodes[i]["c"]; h = nodes[i]["size"]
        L = [j for j in text_ids if nodes[j]["bbox"][2] <= nodes[i]["bbox"][0] + 0.1 and nodes[i]["bbox"][0] - nodes[j]["bbox"][2] < UNIT and abs(nodes[j]["c"][1] - c[1]) < 0.5 * h]
        R = [j for j in text_ids if nodes[j]["bbox"][0] >= nodes[i]["bbox"][2] - 0.1 and nodes[j]["bbox"][0] - nodes[i]["bbox"][2] < UNIT and abs(nodes[j]["c"][1] - c[1]) < 0.5 * h]
        for a_ in L:
            for b_ in R:
                ta, tb = nodes[a_]["text"].strip(), nodes[b_]["text"].strip()
                if ta == tb and re.match(r"^[A-Za-z0-9]{1,4}$", ta):
                    bb = [nodes[a_]["bbox"][0], min(nodes[a_]["bbox"][1], nodes[b_]["bbox"][1]), nodes[b_]["bbox"][2], max(nodes[a_]["bbox"][3], nodes[b_]["bbox"][3])]
                    items.append(dict(bb=bb, txt=f"SECTION {ta}-{tb}", conf=1.0, used=False, src="pdf-text"))

    def partner(it, pattern):
        """OCR item matching pattern to the right on the same line, or on the next line below."""
        x0, y0, x1, y1 = it["bb"]; h = y1 - y0
        best = None
        for o in items:
            if o is it or o["used"] or not pattern.match(o["txt"]): continue
            ox0, oy0, ox1, oy1 = o["bb"]
            same_line = abs((oy0 + oy1) / 2 - (y0 + y1) / 2) < 0.5 * h and -UNIT < ox0 - x1 < 2 * UNIT
            below = -0.2 * h < oy0 - y1 < 1.5 * h and min(x1, ox1) - max(x0, ox0) > -UNIT
            if same_line or below:
                dist = math.dist(((x0 + x1) / 2, y1), ((ox0 + ox1) / 2, oy0))
                if best is None or dist < best[0]: best = (dist, o)
        return best[1] if best else None

    def add_title(name, name_bb, rest, rest_bb, conf, src="ocr"):
        m = re.search(SIZE, rest)
        hs = [int(v) for v in re.findall(r"\d+", m["h"])]
        a = add_det("BeamName", name, name_bb, src, conf=conf)
        b = add_det("BeamSize", rest.strip(), rest_bb, src, conf=conf, b=int(m["b"]), h=hs[0],
                    **({"h2": hs[1]} if len(hs) > 1 else {}))
        a["rel"] = b["id"]; b["rel"] = a["id"]

    n_ocr = 0
    for it in items:
        if it["used"]: continue
        txt, bb, conf = it["txt"], it["bb"], it["conf"]
        m = RE_TITLE.match(txt)
        if m:
            cut = bb[0] + (bb[2] - bb[0]) * (len(m["name"]) + 0.6) / len(txt)    # split box by character share (+ half the space)
            add_title(m["name"], [bb[0], bb[1], cut, bb[3]], m["rest"], [cut, bb[1], bb[2], bb[3]], conf, it["src"])
            it["used"] = True; n_ocr += 1
        elif RE_NAME_ONLY.match(txt) and (o := partner(it, RE_SIZE_ONLY)):
            add_title(txt, bb, o["txt"], o["bb"], min(conf, o["conf"]), it["src"])
            it["used"] = o["used"] = True; n_ocr += 1
    for it in items:
        if it["used"]: continue
        txt, bb, conf = it["txt"], it["bb"], it["conf"]
        if it["src"] != "ocr" and RE_SECNO.match(txt) and not txt.upper().startswith("SECTION"):
            txt = "SECTION " + re.sub(r"\s+", "", txt)        # TSSC writes the title as '1 - 1' under the section
        if txt.upper().startswith("SECTION"):
            if txt.strip().upper() == "SECTION" and (o := partner(it, re.compile(
                    rf"^\s*(?:{NAME}|(?P<sid>\w{{1,6}})\s*-\s*(?P=sid)|[A-Za-z]{{0,3}}\d{{1,3}}[A-Za-z]?\s*$)"))):
                txt = f"SECTION {o['txt']}"; o["used"] = True
                bb = [min(bb[0], o["bb"][0]), bb[1], max(bb[2], o["bb"][2]), o["bb"][3]]
            txt = re.sub(r"(?i)^SECTION\s*", "SECTION ", txt.strip())     # OCR may drop the space: SECTION1B135-1B135
            add_det("SectionTitle", txt, bb, it["src"], conf=conf)
        elif it["src"] != "ocr":
            continue                                    # text-layer words: titles only (labels came from step 1)
        elif (m := RE_LABEL_HEAD.match(txt.lstrip(LEAD_JUNK))) and parse_label(re.sub(r"\s+", "", m["lab"])):
            n_lead = len(txt) - len(txt.lstrip(LEAD_JUNK))  # the end of a leader line read as '-', '_', '[' ...
            if n_lead:
                bb = [bb[0] + (bb[2] - bb[0]) * n_lead / len(txt), bb[1], bb[2], bb[3]]
                txt = txt[n_lead:]
            t = re.sub(r"\s+", "", m["lab"])
            lab, sfx = parse_label(t), (m["sfx"] or "").upper()
            if m["sfx"]:
                bb = [bb[0], bb[1], bb[0] + (bb[2] - bb[0]) * (len(m["lab"].strip()) + 0.3) / len(txt), bb[3]]
            if "SIDE" in sfx:
                lab["kind"], lab["side"] = "bar", True
            add_det("RebarSize", t, bb, "ocr", label=lab, conf=conf, dir=[1.0, 0.0], size=UNIT, suffix=sfx)
    log(f"OCR: {len(ocr or [])} items; {n_ocr} beam titles (OCR + text layer), "
        f"{sum(d['cls'] == 'SectionTitle' for d in dets)} section titles")

    # --- 3. Dim -> DimLine ---------------------------------------------------------------
    # link-zone lengths: number printed right under a link label (e.g. H10-275 / 440)
    link_lbls = [d for d in dets if d["cls"] == "RebarSize" and d["label"]["kind"] == "link"]
    for d in dets:
        if d["cls"] != "Dim": continue
        for L in link_lbls:
            if 0 <= d["bbox"][1] - L["bbox"][3] < 0.6 * UNIT and abs((d["bbox"][0] + d["bbox"][2]) / 2 - (L["bbox"][0] + L["bbox"][2]) / 2) < 2 * UNIT:
                d["cls"], d["zone_of"] = "ZoneLen", L["id"]
                L["zone_len"] = d["value"]

    def extent_to_ext_lines(s, reach=1.2, overshoot=1.5):
        """Measured extent of a dimension = between its extension lines (gridlines), not the drawn ends.
        Arrow style: the dim line stops at the arrow tips, short of the extension lines -> look outward.
        Tick style: the dim line runs past the extension lines (overshoot) -> look inward too.
        A 45-deg tick centred on the axis marks the exact point and wins; otherwise the
        perpendicular crossing closest to the drawn end."""
        a, b = SEG[s]
        u = SEG_u[s]
        nrm = np.array([-u[1], u[0]])
        L = float(np.hypot(*(b - a)))
        cosang = np.abs((SEG_u * u).sum(1))
        perp_ok = np.where(cosang < 0.1)[0]
        tick_ok = np.where((cosang > 0.42) & (cosang < 0.91) & (SEG_len <= 2.5 * UNIT))[0]   # 25-65 deg
        ends = []
        for p, out in ((a, -u), (b, u)):
            lo = -min(overshoot * UNIT, 0.45 * L)          # how far inside the drawn end to look
            hi = reach * UNIT
            ticks = []
            for j in tick_ok:
                m = SEG[j].mean(0)
                if abs((m - p) @ nrm) <= 0.6 and lo <= (m - p) @ out <= hi:
                    ticks.append(float((m - p) @ out))
            if ticks:
                ends.append(p + out * min(ticks, key=abs)); continue
            best = None
            for j in perp_ok:
                q0, q1 = SEG[j]
                # where the perpendicular segment meets the dim axis, and is it actually crossing it?
                t = float(((q0 - p) * out).sum())
                side0, side1 = (q0 - p) @ nrm, (q1 - p) @ nrm
                if lo <= t <= hi and min(side0, side1) <= 1.0 and max(side0, side1) >= -1.0 and j != s:
                    if best is None or abs(t) < abs(best):
                        best = t
            ends.append(p + out * (best if best is not None else 0.0))
        return ends[0], ends[1]

    # dimension line: the parallel segment just "under" the number (per text direction)
    cand_w = Counter()
    for d in dets:
        if d["cls"] != "Dim": continue
        idx, perp = segs_under(np.array([(d["bbox"][0] + d["bbox"][2]) / 2, (d["bbox"][1] + d["bbox"][3]) / 2]),
                               np.array(d["dir"]), 1.5 * UNIT, min_len=1.5 * UNIT)
        if len(idx): cand_w[SEG_w[idx[0]]] += 1
    DIM_W = cand_w.most_common(1)[0][0]
    log("dimension-line width:", DIM_W, dict(cand_w))
    for d in dets:
        if d["cls"] != "Dim": continue
        c = np.array([(d["bbox"][0] + d["bbox"][2]) / 2, (d["bbox"][1] + d["bbox"][3]) / 2])
        idx, perp = segs_under(c, np.array(d["dir"]), 1.5 * UNIT, min_len=1.0 * UNIT, widths=[DIM_W])
        if len(idx):
            s = int(idx[0])
            a, b = extent_to_ext_lines(s)
            d["dimline"] = [a.tolist(), b.tolist()]; d["dimline_len"] = float(np.hypot(*(b - a))); d["seg"] = s
            d["mm_per_pt"] = d["value"] / d["dimline_len"]

    # numbers in the dimension text size that found no dimension line stay as unmatched Dim;
    # numbers in other sizes (title block, notes) are dropped
    dim_size = Counter(round(d["size"], 1) for d in dets if d["cls"] == "Dim" and d.get("dimline")).most_common(1)[0][0]
    dets[:] = [d for d in dets if not (d["cls"] == "Dim" and not d.get("dimline") and abs(d["size"] - dim_size) > 0.2)]
    remap = {d["id"]: k for k, d in enumerate(dets)}
    for d in dets:
        d["id"] = remap[d["id"]]
        for key in ("rel", "zone_of"):
            if key in d: d[key] = remap[d[key]]

    # --- 4. views (elevation / section groups) and their scale ----------------------------
    VIEW_GAP = 1.0
    k_sub = max(1, round(VIEW_GAP * UNIT / 2 / STEP))
    pv = cKDTree(S_xy[::k_sub]).query_pairs(VIEW_GAP * UNIT, output_type="ndarray")
    own = S_own[::k_sub]
    g = coo_matrix((np.ones(len(pv)), (own[pv[:, 0]], own[pv[:, 1]])), shape=(len(nodes), len(nodes)))
    view = connected_components(g, directed=False)[1]
    def view_of_point(c):
        return int(view[S_own[S_tree.query(c)[1]]])

    PT_MM = 25.4 / 72
    view_scale = defaultdict(list)
    for d in dets:
        if d.get("mm_per_pt"):
            d["view"] = int(view[SEG_own[d["seg"]]])
            if d["dimline_len"] >= 4 * UNIT:          # short dims carry more drafting error
                view_scale[d["view"]].append(d["mm_per_pt"])
    VIEW_SCALE = {v: float(np.median(s)) for v, s in view_scale.items()}
    all_s = np.array([d["mm_per_pt"] for d in dets if d.get("mm_per_pt")])
    SHEET_SCALE = float(np.median(all_s))
    for d in dets:
        if d.get("mm_per_pt"):
            d["scale_ok"] = bool(abs(d["mm_per_pt"] / VIEW_SCALE.get(d["view"], SHEET_SCALE) - 1) < 0.03)
    log(f"sheet scale ~1:{SHEET_SCALE / PT_MM:.1f}; per view:",
          sorted({round(s / PT_MM) for s in VIEW_SCALE.values()}))

    # --- 5. RebarSize -> RebarLine ---------------------------------------------------------
    rebar_nodes = set(i for i in line_ids if round(nodes[i]["width"], 2) >= REBAR_W - 0.01)
    adj = defaultdict(set)
    Q, Q_own = [], []
    for i in geo_ids:
        v = np.unique(np.round(nodes[i]["segs"].reshape(-1, 2), 2), axis=0)
        Q.append(v); Q_own.append(np.full(len(v), i))
        if nodes[i]["kind"] == "mark" or nodes[i]["length"] < 2 * UNIT:
            Q.append(np.array([nodes[i]["c"]])); Q_own.append(np.array([i]))
    Q, Q_own = np.vstack(Q), np.concatenate(Q_own)
    for qi, hit in enumerate(S_tree.query_ball_point(Q, TOUCH_TOL * UNIT + STEP / 2)):
        for j in np.unique(S_own[hit]):
            if j != Q_own[qi]:
                adj[int(Q_own[qi])].add(int(j)); adj[int(j)].add(int(Q_own[qi]))

    # some offices draw rebar, dimension and leader lines with one pen: width can't tell them apart
    UNIFORM = DIM_W >= REBAR_W - 0.01
    if UNIFORM:
        log(f"rebar, dimension and leader lines share width {REBAR_W} -> bar = line at the arrow tip")

    def leader_ok(j):          # pieces a leader can be made of: arrowheads, short thin lines
        n = nodes[j]
        if UNIFORM:
            return n["kind"] == "mark" or n["length"] <= 30 * UNIT
        return j not in rebar_nodes and (n["kind"] == "mark" or n["length"] <= 30 * UNIT)

    def arrow_tip(j):
        """Tip of an arrowhead = vertex opposite its shortest side (centre for dots)."""
        n = nodes[j]
        v = np.unique(np.round(n["segs"].reshape(-1, 2), 2), axis=0)
        if n["curved"] or len(v) != 3:
            return np.array(n["c"])
        side = [math.dist(v[(k + 1) % 3], v[(k + 2) % 3]) for k in range(3)]
        return v[int(np.argmin(side))]

    def follow_leader(bbox, max_hops=5):
        x0, y0, x1, y1 = bbox
        pad = 0.6 * UNIT
        hit = S_tree.query_ball_point(((x0 + x1) / 2, (y0 + y1) / 2), math.hypot(x1 - x0, y1 - y0) / 2 + pad)
        P, o = S_xy[hit], S_own[hit]
        inside = (P[:, 0] >= x0 - pad) & (P[:, 0] <= x1 + pad) & (P[:, 1] >= y0 - pad) & (P[:, 1] <= y1 + pad)
        start = {int(j) for j in np.unique(o[inside]) if leader_ok(j)}
        seen, frontier = set(start), set(start)
        lab_c = ((x0 + x1) / 2, (y0 + y1) / 2)
        TOUCH = max(0.5, 0.06 * UNIT)
        plain_end = None                     # fallback: leader line ending on a bar without an arrowhead
        for hop in range(max_hops):
            best = None
            for f in frontier:
                # lines at an arrowhead: the bar touches its tip, its own leader only its base, so the
                # nearest-to-tip wins (one-pen sheets: a short bar may also have been walked as 'leader')
                bars = [j for j in adj[f] if j in rebar_nodes and nodes[j]["kind"] == "line"]
                if not bars: continue
                if nodes[f]["kind"] == "mark":
                    tip = arrow_tip(f)
                    vtx = np.unique(np.round(nodes[f]["segs"].reshape(-1, 2), 2), axis=0)
                    axis = tip - vtx.mean(0)
                    axis = axis / np.hypot(*axis) if not nodes[f]["curved"] and np.hypot(*axis) > 1e-6 else None
                    for j in bars:
                        sg = nodes[j]["segs"]                   # nearest segment of j to the tip
                        ab = sg[:, 1] - sg[:, 0]
                        t = np.clip(((tip - sg[:, 0]) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-12), 0, 1)
                        dd = np.hypot(*(sg[:, 0] + t[:, None] * ab - tip).T)
                        k = int(np.argmin(dd)); dist = float(dd[k])
                        if axis is not None and np.hypot(*ab[k]) > 1e-6 and \
                                abs(ab[k] @ axis) / np.hypot(*ab[k]) > 0.9:
                            continue                            # runs along the arrow: its own leader
                        # touching arrows (within 0.5 pt) rank by closeness to the label: a neighbouring
                        # label's leader can be reached in the same step, and its arrow touches a bar too
                        key = (dist > TOUCH, math.dist(tip, lab_c) if dist <= TOUCH else dist)
                        if best is None or key < best[0]:
                            best = (key, j, tip)
                elif plain_end is None and hop > 0:
                    long_bars = [j for j in bars if not leader_ok(j)]
                    if long_bars:
                        plain_end = (long_bars[0], np.array(nodes[f]["c"]))
            if best:
                return {best[1]}, best[2], seen
            frontier = {j for f in frontier for j in adj[f] if j not in seen and leader_ok(j)}
            seen |= frontier
            if not frontier: break
        if plain_end:
            return {plain_end[0]}, plain_end[1], seen
        return set(), None, seen

    def clip_zone(s, c):
        """Link zone = part of the zone line between the nearest crossings (link lines, ticks,
        arrow tips) either side of the label; ends extended to extension lines if none."""
        u = SEG_u[s]; nrm = np.array([-u[1], u[0]])
        # join collinear pieces of the same zone line (drawn broken between arrows)
        base = SEG[s][0]
        col = np.where((np.abs((SEG_u * u).sum(1)) > 0.999) &
                       (np.abs((SEG[:, 0] - base) @ nrm) < 0.5) & (np.abs((SEG[:, 1] - base) @ nrm) < 0.5))[0]
        T = np.concatenate([(SEG[col, 0] - base) @ u, (SEG[col, 1] - base) @ u])
        a0, b0 = base + u * T.min(), base + u * T.max()
        L = float(T.max() - T.min()); tc = float((c - a0) @ u)
        ts, arrows = [0.0, L], []
        # segments crossing the zone line
        for j in np.where(np.abs((SEG_u * u).sum(1)) < 0.95)[0]:
            q0, q1 = SEG[j]
            s0, s1 = (q0 - a0) @ nrm, (q1 - a0) @ nrm
            if s0 * s1 > 0 or s0 == s1: continue
            p = q0 + (q1 - q0) * (s0 / (s0 - s1))
            t = float((p - a0) @ u)
            if 0.5 < t < L - 0.5: ts.append(t)
        # arrowheads / ticks touching it
        for j in set(S_own[S_tree.query_ball_point(((a0 + b0) / 2), L / 2 + UNIT)]):
            if nodes[j]["kind"] == "mark":
                tip = arrow_tip(j)
                if abs((tip - a0) @ nrm) < 0.6 * UNIT:
                    t = float((tip - a0) @ u)
                    if 0.5 < t < L - 0.5: arrows.append(t)
        # drafting convention: zone ends are arrow tips; other crossings only where no arrow exists
        lo = max([t for t in arrows if t <= tc], default=None)
        hi = min([t for t in arrows if t >= tc], default=None)
        if lo is None: lo = max([t for t in ts if t <= tc], default=0.0)
        if hi is None: hi = min([t for t in ts if t >= tc], default=L)
        return a0 + u * lo, a0 + u * hi

    for d in dets:
        if d["cls"] != "RebarSize": continue
        c = np.array([(d["bbox"][0] + d["bbox"][2]) / 2, (d["bbox"][1] + d["bbox"][3]) / 2])
        if d["label"]["kind"] == "link":
            # link zone: the line the label sits on (arrowed extent of that spacing)
            idx, _ = segs_under(c, np.array([1.0, 0.0]), 1.2 * UNIT, min_len=1.5 * UNIT, max_angle=0.35)
            if len(idx):
                s = int(idx[0])
                a, b = clip_zone(s, c)
                d["line"] = [[a.tolist(), b.tolist()]]; d["line_len"] = float(np.hypot(*(b - a))); d["line_view"] = int(view[SEG_own[s]])
        else:
            found, contact, path = follow_leader(d["bbox"])
            d["leader"] = sorted(path)
            if found:
                d["rebar_nodes"] = sorted(found); d["contact"] = contact.tolist()
                d["line"] = [nodes[j]["segs"].tolist() for j in found]
                d["line_len"] = float(max(nodes[j]["length"] for j in found))   # one bar line per arrow
                d["line_view"] = int(view[min(found)])

    # --- 5a. stacked labels on one leader ('6T32' over '2T20'): one label per bar layer -----------
    # Both labels reach the same arrow. The label farthest from the arrow tip names the outer layer
    # (the line the arrow touches); the next one names the next bar line inward along the arrow.
    REBAR_SEGS = np.where(SEG_w >= REBAR_W - 0.01)[0]
    groups = defaultdict(list)
    for d in dets:
        if d["cls"] == "RebarSize" and d.get("rebar_nodes") and d.get("contact") and d["label"]["kind"] == "bar":
            groups[tuple(np.round(d["contact"], 0))].append(d)
    n_layer = 0
    for grp in groups.values():
        if len(grp) < 2:
            continue
        tip = np.asarray(grp[0]["contact"], float)
        grp.sort(key=lambda d: -math.dist(tip, ((d["bbox"][0] + d["bbox"][2]) / 2, (d["bbox"][1] + d["bbox"][3]) / 2)))
        j0 = grp[0]["rebar_nodes"][0]
        sg = nodes[j0]["segs"]
        k = int(np.argmin([np.hypot(*(tip - (a + np.clip((tip - a) @ (b - a) / max((b - a) @ (b - a), 1e-12), 0, 1) * (b - a)))) for a, b in sg]))
        u = sg[k][1] - sg[k][0]; u = u / max(np.hypot(*u), 1e-9)
        nrm = np.array([-u[1], u[0]])
        lab_c = np.array(((grp[0]["bbox"][0] + grp[0]["bbox"][2]) / 2, (grp[0]["bbox"][1] + grp[0]["bbox"][3]) / 2))
        if nrm @ (tip - lab_c) < 0:
            nrm = -nrm                                    # inward = away from the labels
        hits = {j0: 0.0}                                  # parallel bar lines along the arrow:
        for j in REBAR_SEGS:                              # from behind the head to past the tip
            if SEG_own[j] == j0 or abs(SEG_u[j] @ u) < 0.99 or SEG_len[j] < 2 * UNIT:
                continue
            a = SEG[j][0]
            t = float((a - tip) @ nrm)                    # signed distance inward from the tip
            along = float((tip - a) @ SEG_u[j])
            if -1.2 * UNIT <= t <= 2 * UNIT and abs(t) > 0.3 and -0.5 <= along <= SEG_len[j] + 0.5:
                o = int(SEG_own[j])
                if o not in hits or abs(t) < abs(hits[o]):
                    hits[o] = t
        layers = sorted(hits.items(), key=lambda kv: kv[1])   # outer layer first
        for m, d in enumerate(grp):
            if m < len(layers):
                o, t = layers[m]
                if o != d["rebar_nodes"][0] or m:
                    d["rebar_nodes"], d["contact"], d["layer"] = [o], (tip + nrm * t).tolist(), m + 1
                    d.setdefault("why", []).append(
                        f"stacked labels on one leader ({', '.join(g['text'] for g in grp)}): layer {m + 1} of "
                        f"{len(layers)} bar lines along the arrow, {t:+.1f} pt from the tip")
                    n_layer += 1
            else:
                d.setdefault("why", []).append(f"stacked under '{grp[0]['text']}' but no layer {m + 1} line found: same bar")
    if n_layer:
        log(f"stacked labels: {n_layer} assigned to inner bar layers")

    # --- 5b. whole bar: follow the arrowed line through its bends to the end legs -------------
    # CAD often exports a bend (+ its leg) as a separate path from the straight run, so the arrow
    # only touches the run. Walk end-to-end through rebar-width paths that join with a bend or a
    # corner, then (square_bends) replace each bend arc by the corner where the straight legs meet.
    CURVE_SEG = max(1.0, 0.25 * UNIT)          # pieces shorter than this are a flattened arc
    JOIN_TOL = max(0.6, 0.15 * UNIT)            # gap allowed where two paths of one bar meet
    LEG_MIN = 1.0 * UNIT                        # a straight piece at least this long is a leg
    _poly = {}

    def chain_segs(segs, flags, tol=0.05):
        """Unordered segments -> ordered polylines [(points, curve flag per segment)]."""
        keep = np.hypot(*(segs[:, 1] - segs[:, 0]).T) > 1e-6
        seen_k = set()                              # drop repeats (closed 2-point paths, CAD duplicates)
        for k in np.where(keep)[0]:
            key = tuple(sorted([tuple(np.round(segs[k][0], 2)), tuple(np.round(segs[k][1], 2))]))
            if key in seen_k:
                keep[k] = False
            seen_k.add(key)
        segs, flags = segs[keep], flags[keep]
        n = len(segs)
        if n == 0:
            return []
        P = segs.reshape(-1, 2)
        tree = cKDTree(P)
        at = [[(k // 2, k % 2) for k in tree.query_ball_point(P[e], tol)] for e in range(2 * n)]
        used, runs = np.zeros(n, bool), []
        for e in sorted(range(2 * n), key=lambda e: len(at[e]) != 1):     # free ends first
            s, end = e // 2, e % 2
            if used[s]:
                continue
            pts, fl, used[s] = [segs[s][end], segs[s][1 - end]], [flags[s]], True
            last = 2 * s + (1 - end)
            while True:
                nxt = [(s2, e2) for s2, e2 in at[last] if not used[s2]]
                if len(nxt) != 1:            # free end, or a branch (T): stop
                    break
                s2, e2 = nxt[0]
                pts.append(segs[s2][1 - e2]); fl.append(flags[s2]); used[s2] = True
                last = 2 * s2 + (1 - e2)
            runs.append((np.array(pts), np.array(fl, bool)))
        return runs

    def node_poly(j):
        """Rebar node -> its longest ordered polyline (points, is-arc flag per segment)."""
        if j not in _poly:
            n = nodes[j]
            L = np.hypot(*(n["segs"][:, 1] - n["segs"][:, 0]).T)
            runs = chain_segs(n["segs"], n["seg_curve"] | (L < CURVE_SEG))
            _poly[j] = max(runs, key=lambda r: np.hypot(*np.diff(r[0], axis=0).T).sum()) if runs else None
        return _poly[j]                       # None: zero-length path

    def unit(v):
        return v / max(np.hypot(*v), 1e-9)

    def end_dir(P, F, at_end):
        """Outward direction of the last straight piece at one end of a polyline (None: all arc)."""
        idx = range(len(F) - 1, -1, -1) if at_end else range(len(F))
        for k in idx:
            if not F[k]:
                v = P[k + 1] - P[k]
                return unit(v if at_end else -v)
        return None

    # pieces = unbranched runs of the rebar-width paths. A path that branches (a lap stub drawn
    # from the middle of a bar line, two bars drawn as one path) gives several pieces.
    PIECES, pieces_of = [], defaultdict(list)
    for j in sorted(rebar_nodes):
        n = nodes[j]
        L = np.hypot(*(n["segs"][:, 1] - n["segs"][:, 0]).T)
        for P_, F_ in chain_segs(n["segs"], n["seg_curve"] | (L < CURVE_SEG)):
            pieces_of[j].append(len(PIECES)); PIECES.append((j, P_, F_))
    rb_ends = np.array([p for (_, P_, _) in PIECES for p in (P_[0], P_[-1])]) if PIECES else np.zeros((0, 2))
    rb_ends_own = np.repeat(np.arange(len(PIECES)), 2)
    rb_tree = cKDTree(rb_ends) if len(rb_ends) else None

    def poly_dist(P_, c):
        """Distance from point c to polyline P_, and the arc-length position of the nearest point."""
        a, b = P_[:-1], P_[1:]
        ab = b - a
        t = np.clip(((c - a) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-12), 0, 1)
        dd = np.hypot(*(a + t[:, None] * ab - c).T)
        k = int(np.argmin(dd))
        cum = np.concatenate([[0.0], np.cumsum(np.hypot(*ab.T))])
        return float(dd[k]), float(cum[k] + t[k] * np.hypot(*ab[k]))

    def piece_at(j, c):
        """The piece of path j under point c (the arrow tip); its longest piece if c is unknown."""
        cand = pieces_of.get(j, [])
        if not cand:
            return None
        if c is None:
            return max(cand, key=lambda q: np.hypot(*np.diff(PIECES[q][1], axis=0).T).sum())
        return min(cand, key=lambda q: poly_dist(PIECES[q][1], np.asarray(c, float))[0])

    def describe(j):
        n = nodes[j]
        return f"path {j} ({n['kind']}, width {n['width']:.2f}, {n['length']:.1f} pt)"

    def describe_piece(q):
        j, P_, _ = PIECES[q]
        part = f" part {pieces_of[j].index(q) + 1}/{len(pieces_of[j])}" if len(pieces_of[j]) > 1 else ""
        return f"path {j}{part} (width {nodes[j]['width']:.2f}, {np.hypot(*np.diff(P_, axis=0).T).sum():.1f} pt)"

    def near_end(end, used, why, others=(), k=5):
        """Diagnostics: linework near a bar end that was not joined, and why."""
        rows = []
        for j in set(S_own[S_tree.query_ball_point(end, 3 * UNIT)].tolist()) - set(used):
            n = nodes[j]
            if n["kind"] != "line":
                continue
            ends = [n["segs"][0][0], n["segs"][-1][1]] if node_poly(j) is None else [node_poly(j)[0][0], node_poly(j)[0][-1]]
            de = min(math.dist(e, end) for e in ends)
            dl = float(np.hypot(*(node_pts(j) - end).T).min())
            rows.append((de, dl, j))
        rows.sort()
        if not rows:
            why.append("      nothing else drawn within 3 text heights")
        for de, dl, j in rows[:k]:
            flags = []
            if j not in rebar_nodes: flags.append(f"not rebar width ({REBAR_W:.2f})")
            if j in others: flags.append("another label's bar")
            if de > JOIN_TOL: flags.append(f"its end is {de:.2f} pt away (join tolerance {JOIN_TOL:.2f})" if de < 3 * UNIT
                                            else f"passes {dl:.2f} pt away, no end here")
            why.append(f"      {describe(j)}: " + ("; ".join(flags) or "joinable"))

    def grow_bar(t, others, why=None):
        """Start from piece t; at each end add bend/leg pieces (not other labels' bars)."""
        why = why if why is not None else []
        P, F = PIECES[t][1], PIECES[t][2]
        P, F, used = P.copy(), F.copy(), {t}
        why.append(f"arrow touches {describe_piece(t)}, ends {np.round(P[0], 1).tolist()} and {np.round(P[-1], 1).tolist()}")
        for at_end in ((True, False) if rb_tree is not None else ()):
            legs = 0
            for _ in range(4):
                end = P[-1] if at_end else P[0]
                d_out = end_dir(P, F, at_end)                   # direction we arrive in
                cands, rejected = [], []
                for j in {int(rb_ends_own[k]) for k in rb_tree.query_ball_point(end, JOIN_TOL)}:
                    if j in used:
                        continue
                    if j in others:
                        rejected.append(f"{describe_piece(j)}: another label's bar"); continue
                    Q, G = PIECES[j][1], PIECES[j][2]
                    if math.dist(Q[-1], end) < math.dist(Q[0], end):
                        Q, G = Q[::-1], G[::-1]                 # orient: Q[0] at the join
                    if math.dist(Q[0], end) > JOIN_TOL:
                        continue
                    st = np.where(~G)[0]                        # Q's first straight, pointing away from the join
                    d_in = unit(Q[st[0] + 1] - Q[st[0]]) if len(st) else None
                    turn = 0.0 if d_out is None or d_in is None else math.degrees(
                        math.acos(np.clip(np.dot(d_out, d_in), -1, 1)))
                    # continues the bar: starts with a bend, or a sharp corner that doesn't
                    # double back (a straight butt joint or a fold-back is a different bar)
                    if len(Q) == 2 and not G[0] and math.dist(Q[0], Q[1]) <= 1.5 * UNIT and \
                            (25 <= turn <= 65 or 115 <= turn <= 155):
                        rejected.append(f"{describe_piece(j)}: slanted end tick (lap mark)"); continue
                    if G[0] or (not UNIFORM and 20 <= turn <= 150):     # one-pen sheets: bends only
                        cands.append((j, Q, G, math.dist(Q[0], end)))
                    else:
                        rejected.append(f"{describe_piece(j)}: " + ("corner without a bend (one-pen sheet)" if UNIFORM and 20 <= turn <= 150
                                        else f"turns {turn:.0f} deg (straight butt joint or fold-back)"))
                tag = f"   end {np.round(end, 1).tolist()}: "
                if len(cands) > 1:          # a bend drawn as tiny pieces: the next piece meets exactly,
                    dmin = min(c[3] for c in cands)               # the one after it only within tolerance
                    cands = [c for c in cands if c[3] <= dmin + 0.3]
                if len(cands) != 1:
                    if len(cands) > 1:
                        why.append(tag + "stop, branch: " + ", ".join(describe_piece(c[0]) for c in cands))
                    else:
                        why.append(tag + "stop, nothing joinable" + (" (" + "; ".join(rejected) + ")" if rejected else ""))
                        near_end(end, {PIECES[q][0] for q in used}, why, {PIECES[q][0] for q in others})
                    break                                       # free end, or a real branch
                j, Q, G, _ = cands[0]
                seg_len = np.hypot(*np.diff(Q, axis=0).T)
                has_leg = bool(((~G) & (seg_len >= LEG_MIN)).any())
                if legs and seg_len.sum() > 0.5 * legs:
                    why.append(tag + f"stop: {describe_piece(j)} is longer than a hook after the leg")
                    break                                       # after a leg only a short hook/return
                why.append(tag + f"joined {describe_piece(j)}" + (" (starts with a bend)" if G[0] else " (corner)"))
                if at_end:
                    P, F = np.vstack([P, Q[1:]]), np.concatenate([F, G])
                else:
                    P, F = np.vstack([Q[::-1][:-1], P]), np.concatenate([G[::-1], F])
                used.add(j)
                if has_leg:
                    if legs:
                        break
                    legs = float(seg_len[~G].max())
        return P, F, used

    def arc_tangent(A):
        """Tangent at A[0] of a flattened arc A, pointing into it: perpendicular to the radius of a
        circle fitted through the arc points (the first chord alone is off by half its turn)."""
        chord = unit(A[1] - A[0])
        if len(A) < 3:
            return chord
        M = np.column_stack([A[:, 0], A[:, 1], np.ones(len(A))])
        (D, E, Fc), *_ = np.linalg.lstsq(M, -(A ** 2).sum(1), rcond=None)
        ctr = np.array([-D / 2, -E / 2])
        r = math.sqrt(max(ctr @ ctr - Fc, 0.0))
        if r < 1e-6 or np.abs(np.hypot(*(A - ctr).T) - r).max() > 0.1 * r + 0.05:
            return chord                                         # not a circular arc
        t = unit(np.array([-(A[0] - ctr)[1], (A[0] - ctr)[0]]))
        return t if t @ chord > 0 else -t

    def line_x(a, da, b, db):
        """Intersection of line a + s*da and line b + t*db (None if parallel)."""
        den = da[0] * db[1] - da[1] * db[0]
        if abs(den) < 1e-6:
            return None
        s = ((b[0] - a[0]) * db[1] - (b[1] - a[1]) * db[0]) / den
        return a + s * da

    def square_corners(P, F):
        """Replace each arc run by the corner where its straight neighbours (or its end tangents) meet."""
        pieces, k = [], 0                       # (is_arc, first point index, last point index)
        while k < len(F):
            m = k
            while m + 1 < len(F) and F[m + 1] == F[k]:
                m += 1
            pieces.append((bool(F[k]), k, m + 1))
            k = m + 1
        n_sq, repl = 0, {}
        for q, (arc, i0, i1) in enumerate(pieces):
            if not arc:
                continue
            a, b = P[i0], P[i1]
            prev = pieces[q - 1] if q > 0 else None
            nxt = pieces[q + 1] if q + 1 < len(pieces) else None
            d_a = unit(P[i0] - P[i0 - 1]) if prev else arc_tangent(P[i0:i1 + 1])        # heading into the arc
            d_b = unit(P[i1 + 1] - P[i1]) if nxt else -arc_tangent(P[i0:i1 + 1][::-1])  # heading out of it
            if not prev and not nxt:
                continue                                        # a lone arc: keep as drawn
            arc_len = float(np.hypot(*np.diff(P[i0:i1 + 1], axis=0).T).sum())
            turn = math.degrees(math.acos(np.clip(np.dot(d_a, d_b), -1, 1)))
            if turn < 10:                                       # no net bend (tiny pieces, S-crank): chord
                repl[q] = [a, b]
                continue
            if turn <= 150:
                c = line_x(a, d_a, b, d_b)
                if c is None or max(math.dist(c, a), math.dist(c, b)) > 3 * arc_len + 1:
                    continue
                repl[q] = [c]          # at a bar end the leg simply stops at the corner (no arc stub)
            else:                                               # 180-deg hook -> squared U
                t_max = float(((P[i0:i1 + 1] - a) @ d_a).max())
                corner = [a + d_a * t_max, b + d_a * (t_max - float((b - a) @ d_a))]
                repl[q] = ([] if prev else [a]) + corner + ([] if nxt else [b])
            n_sq += 1
        out = []
        for q, (arc, i0, i1) in enumerate(pieces):
            if arc:
                out.extend(repl[q] if q in repl else list(P[i0:i1 + 1]))
            else:                                               # straight piece keeps its own corners
                if not (q > 0 and q - 1 in repl): out.append(P[i0])
                out.extend(P[i0 + 1:i1])
                if not (q + 1 < len(pieces) and q + 1 in repl): out.append(P[i1])
        Q = [np.asarray(out[0], float)]
        for p in out[1:]:                                       # drop repeats and collinear middles
            p = np.asarray(p, float)
            if math.dist(p, Q[-1]) < 1e-3:
                continue
            v1, v2 = unit(Q[-1] - Q[-2]) if len(Q) >= 2 else None, unit(p - Q[-1])
            if len(Q) >= 2 and abs(v1[0] * v2[1] - v1[1] * v2[0]) < 1e-3 \
                    and np.dot(Q[-1] - Q[-2], p - Q[-1]) > 0:
                Q[-1] = p
                continue
            Q.append(p)
        return np.array(Q), n_sq

    # --- laps ------------------------------------------------------------------------------------
    # Lapped bars end in short slanted stubs that hang off the bar line and point AWAY from the bar
    # they end (outward along the line). Walking out from the arrow along the drawn line, a stub
    # pointing back towards us is where the next bar starts (skip it); the first stub pointing away
    # is our end. CAD often draws the shared length once, so the line can run on through the lap
    # (cut the bar at its end) or stop at the other bar's stub (run on to our end). The two
    # RebarLines then overlap over the lap. Crossed ticks with no direction: far tick of a pair.
    REBAR_SEG = SEG_w >= REBAR_W - 0.01
    TICK_TOL = max(0.6, 0.12 * UNIT)

    def ticks_on_line(a, u, t_lo, t_hi):
        """Slanted ticks touching the line a + t*u, t in [t_lo, t_hi] -> [(t, sign)]; sign +1/-1 = the
        stub points to +u/-u along the line, 0 = crossing tick (no direction)."""
        nrm = np.array([-u[1], u[0]])
        cos = np.abs(SEG_u @ u)
        TICK_MIN = max(CURVE_SEG, 0.35 * UNIT)          # shorter slanted pieces are flattened arcs
        out = []
        for j in np.where((cos > 0.42) & (cos < 0.91) & (SEG_len <= 1.5 * UNIT) & (SEG_len >= TICK_MIN))[0]:
            q0, q1 = SEG[j]
            s0, s1 = float((q0 - a) @ nrm), float((q1 - a) @ nrm)
            if abs(s0) <= TICK_TOL < abs(s1):           # q0 on the line, q1 hangs off it
                p_on, sign = q0, int(np.sign((q1 - q0) @ u))
            elif abs(s1) <= TICK_TOL < abs(s0):
                p_on, sign = q1, int(np.sign((q0 - q1) @ u))
            elif s0 * s1 < 0:                            # crosses the line
                p_on, sign = q0 + (q1 - q0) * (s0 / (s0 - s1)), 0
            else:
                continue
            t = float((p_on - a) @ u)
            if t_lo <= t <= t_hi:
                out.append((t, sign))
        out.sort()
        merged = []
        for t, sg in out:                                # one tick may be several segments
            if merged and t - merged[-1][0] <= 0.5 * UNIT:
                if merged[-1][1] != sg:
                    merged[-1] = (merged[-1][0], 0)      # two stubs at one point (an X): no direction
                continue
            merged.append((t, sg))
        return merged

    def walk_to_end(ticks, s0, direction, R, min_off=0.0, limit=float("inf")):
        """Walk from s0 along `direction` (+1/-1) over [(s, sign)] ticks: the first stub pointing the
        way we walk is our end. Returns (end s, lap length in pt or None, how), or (None, None, '')."""
        seq = sorted(((s, sg) for s, sg in ticks if min_off < (s - s0) * direction <= limit),
                     key=lambda x: (x[0] - s0) * direction)
        skipped, crossing = [], []
        for s, sg in seq:
            if sg == 0:
                crossing.append(s); continue
            if sg * direction > 0:                       # points away: the end of our bar
                lap = abs(s - skipped[-1]) if skipped and abs(s - skipped[-1]) <= R else None
                return s, lap, "stub", (skipped[-1] if lap else None)
            skipped.append(s)                            # points back: the next bar starts here
        if len(crossing) >= 2 and abs(crossing[1] - crossing[0]) <= R:
            return crossing[1], abs(crossing[1] - crossing[0]), "tick pair", crossing[0]
        return None, None, "", None

    def stub_tip(p, u):
        """Free end of the slanted stub whose foot is at p on the bar line (None if no stub there)."""
        best = None
        for j in np.where((np.abs(SEG_u @ u) > 0.42) & (np.abs(SEG_u @ u) < 0.91) & (SEG_len <= 1.5 * UNIT))[0]:
            q0, q1 = SEG[j]
            for on, off in ((q0, q1), (q1, q0)):
                dd = math.dist(on, p)
                if dd <= max(1.0, 0.15 * UNIT) and (best is None or dd < best[0]):
                    best = (dd, np.asarray(off, float))
        return best[1] if best else None

    def add_stub(P, F, d, at_end):
        """Finish the bar line with its own end stub (the bar's drawn end mark), as drawn."""
        if len(P) < 2:
            return P, F
        foot, prev = (P[-1], P[-2]) if at_end else (P[0], P[1])
        tip = stub_tip(foot, unit(foot - prev))
        if tip is None:
            return P, F
        d["stub_len_pt"] = d.get("stub_len_pt", 0.0) + math.dist(foot, tip)
        if at_end:
            return np.vstack([P, tip]), np.concatenate([F, [False]])
        return np.vstack([tip, P]), np.concatenate([[False], F])

    def stub_side(p, u):
        """Unit normal pointing to the side of the line (through p, direction u) a stub at p hangs on."""
        nrm = np.array([-u[1], u[0]])
        best = None
        for j in np.where((np.abs(SEG_u @ u) > 0.42) & (np.abs(SEG_u @ u) < 0.91) & (SEG_len <= 1.5 * UNIT))[0]:
            q0, q1 = SEG[j]
            for on, off in ((q0, q1), (q1, q0)):
                dd = math.dist(on, p)
                if dd <= max(1.0, 0.15 * UNIT) and (best is None or dd < best[0]):
                    best = (dd, float(np.sign((off - on) @ nrm)))
        return nrm * (best[1] if best and best[1] else 1.0)

    def lap_geom(a, b, mm, u):
        """Lap between the two stubs: exact ends, plus a copy set just off the bar line on the stubs'
        side so it shows (and can be clicked) in Label Studio instead of hiding under the bar lines."""
        a, b = np.asarray(a, float), np.asarray(b, float)
        off = stub_side(b, u) * 0.25 * UNIT
        return dict(a=a.tolist(), b=b.tolist(), mm=mm, show=[(a + off).tolist(), (b + off).tolist()])

    def drawn_between(a, u, t0, t1):
        """Is the line a + t*u drawn at rebar width all the way from t0 to t1?"""
        nrm = np.array([-u[1], u[0]])
        ok = REBAR_SEG & (np.abs(SEG_u @ u) > 0.999) & (np.abs((SEG[:, 0] - a) @ nrm) < 0.6) & \
             (np.abs((SEG[:, 1] - a) @ nrm) < 0.6)
        iv = sorted(tuple(sorted(((SEG[j, 0] - a) @ u, (SEG[j, 1] - a) @ u))) for j in np.where(ok)[0])
        reach = t0
        for lo, hi in iv:
            if lo <= reach + JOIN_TOL and hi > reach:
                reach = hi
        return reach >= t1 - JOIN_TOL

    def lap_reach(d):
        lab = d.get("label") or {}
        mm_pt = VIEW_SCALE.get(d.get("line_view"), SHEET_SCALE)
        return min(80 * lab["parts"][0]["dia"], 3000) / mm_pt, mm_pt   # longest lap looked for (pt)

    def slice_poly(P, F, s_lo, s_hi):
        """Part of polyline P between arc-length positions s_lo and s_hi."""
        L = np.hypot(*np.diff(P, axis=0).T)
        cum = np.concatenate([[0.0], np.cumsum(L)])
        def at(sv):
            k = int(np.clip(np.searchsorted(cum, sv) - 1, 0, len(L) - 1))
            return k, P[k] + (P[k + 1] - P[k]) * ((sv - cum[k]) / max(L[k], 1e-9))
        k0, p0 = at(s_lo); k1, p1 = at(s_hi)
        Q = np.vstack([p0, P[k0 + 1:k1 + 1], p1])
        G = F[k0:k1 + 1].copy()
        keep = np.r_[True, np.hypot(*np.diff(Q, axis=0).T) > 1e-6]      # drop zero-length steps
        idx = np.where(keep)[0]
        return Q[idx], np.array([G[min(i - 1, len(G) - 1)] for i in idx[1:]], bool)

    def cut_laps(P, F, d):
        """The drawn line runs on through a lap: end the bar at its own stub on each side of the arrow."""
        lab = d.get("label") or {}
        d["_lap_ends"] = set()                  # ends settled at a stub: no run-on there
        if lab.get("kind") != "bar" or lab.get("side") or not d.get("contact") or len(P) < 2:
            return P, F, []
        R, mm_pt = lap_reach(d)
        L = np.hypot(*np.diff(P, axis=0).T)
        cum = np.concatenate([[0.0], np.cumsum(L)])
        total = float(cum[-1])
        _, s_c = poly_dist(P, np.asarray(d["contact"], float))
        tks = []                                # (arc-length position, sign along increasing s)
        for k in range(len(L)):
            if F[k] or L[k] < 0.5:
                continue
            tks += [(cum[k] + t, sg) for t, sg in ticks_on_line(P[k], unit(P[k + 1] - P[k]), -0.3, L[k] + 0.3)]
        def at(sv):
            k = int(np.clip(np.searchsorted(cum, sv) - 1, 0, len(L) - 1))
            return (P[k] + (P[k + 1] - P[k]) * ((sv - cum[k]) / max(L[k], 1e-9))).tolist()
        lo, hi, laps, notes, stub_ends = 0.0, total, [], [], []
        for direction in (+1, -1):
            s_end, lap, how, s_lap = walk_to_end(tks, s_c, direction, R)
            if s_end is None:
                continue
            d["_lap_ends"].add(direction > 0)
            if how == "stub":
                stub_ends.append(direction > 0)
            if lap:
                laps.append(round(lap * mm_pt))
                pa, pb = np.array(at(s_lap)), np.array(at(s_end))
                d.setdefault("lap_geoms", []).append(lap_geom(pa, pb, laps[-1], unit(pb - pa)))
            if direction > 0 and s_end < total - 0.5 * UNIT:
                hi = s_end; notes.append(f"{total - s_end:.1f} pt past its {how}")
            if direction < 0 and s_end > 0.5 * UNIT:
                lo = s_end; notes.append(f"{s_end:.1f} pt before its {how}")
        if notes:
            d.setdefault("why", []).append("   lap: the drawn line runs through the lap; bar cut "
                                           + " and ".join(notes) + (", lap " + ", ".join(f"{v} mm" for v in laps) if laps else ""))
            P, F = slice_poly(P, F, lo, hi)
        elif laps:
            d.setdefault("why", []).append("   lap: bar ends at its own stub, lap " + ", ".join(f"{v} mm" for v in laps))
        for at_end in stub_ends:
            P, F = add_stub(P, F, d, at_end)
        return P, F, laps

    def extend_laps(P, F, d):
        """The drawn line of this bar stops short of its stub (overlap drawn once, by the other
        bar): run on along the drawn line to our own stub."""
        lab = d.get("label") or {}
        if lab.get("kind") != "bar" or lab.get("side"):
            return P, F, []
        R, mm_pt = lap_reach(d)
        laps = []
        for at_end in (True, False):
            if at_end in d.get("_lap_ends", ()):
                continue                                    # this end already sits at its stub
            Q, G = (P, F) if at_end else (P[::-1], F[::-1])
            L = np.hypot(*np.diff(Q, axis=0).T)
            main = [k for k in range(len(L)) if not G[k] and L[k] >= LEG_MIN]
            if not main:
                continue
            k = main[-1]
            if L[k + 1:].sum() > 0:                          # ends in a hook/stub/leg: its own end
                continue
            a, u = Q[k], unit(Q[k + 1] - Q[k])
            t_end = float((Q[k + 1] - a) @ u)
            tks = ticks_on_line(a, u, t_end - 0.5 * UNIT, t_end + R)
            target, lap, how, t_lap = walk_to_end(tks, t_end, +1, R, min_off=-0.5 * UNIT, limit=R)
            if target is None or target <= t_end + 0.5 * UNIT or not drawn_between(a, u, t_end, target):
                continue
            Q = Q.copy(); Q[-1] = a + u * target
            if how == "stub":
                Q, G = add_stub(Q, G, d, True)
            if lap:
                laps.append(round(lap * mm_pt))
                d.setdefault("lap_geoms", []).append(lap_geom(a + u * t_lap, a + u * target, laps[-1], u))
            d.setdefault("why", []).append(f"   lap: line stops short; run on {(target - t_end):.1f} pt along the drawn line to its {how}"
                                           + (f", lap {laps[-1]} mm" if lap else ""))
            P, F = (Q, G) if at_end else (Q[::-1], G[::-1])
        return P, F, laps

    start = {}
    for d in dets:
        if d["cls"] == "RebarSize" and d.get("rebar_nodes"):
            q = piece_at(d["rebar_nodes"][0], d.get("contact"))
            if q is not None:
                start[d["id"]] = q
    targets = set(start.values())
    n_ext = n_sq_tot = n_lap = 0
    for d in dets:
        t = start.get(d["id"])
        if t is None:
            continue
        d["why"] = d.get("why", [])
        P, F, used = grow_bar(t, targets - {t}, d["why"])
        if P is None:
            continue
        n_ext += len(used) > 1
        P, F, laps_c = cut_laps(P, F, d)
        P, F, laps_e = extend_laps(P, F, d)
        laps = laps_c + laps_e
        d.pop("_lap_ends", None)
        if laps:
            n_lap += 1
            d["laps_mm"] = laps
        if square_bends:
            P, nsq = square_corners(P, F)
            n_sq_tot += nsq
            d["bends_squared"] = nsq
        d["rebar_nodes"] = sorted({PIECES[q][0] for q in used})
        d["line"] = [np.stack([P[:-1], P[1:]], 1).tolist()]
        # run + bends + legs; the end stubs are drawn in the line but are bar-end marks, not length
        d["line_len"] = float(np.hypot(*np.diff(P, axis=0).T).sum()) - d.get("stub_len_pt", 0.0)
    log(f"bars followed through bends: {n_ext} extended to their legs, {n_lap} with a lap"
        + (f", {n_sq_tot} bends squared to corners" if square_bends else ""))

    # lengths in mm through the view's own scale
    for d in dets:
        if d.get("line_len"):
            sc = VIEW_SCALE.get(d["line_view"], SHEET_SCALE)
            d["scale_src"] = "view" if d["line_view"] in VIEW_SCALE else "sheet"
            d["length_mm"] = round(d["line_len"] * sc)

    # --- 6. which beam does each label belong to? -------------------------------------------
    # Titles sit under their drawing: a rebar line belongs to the nearest title below it that
    # overlaps it horizontally. Sections take their SECTION title. Labels without a line take
    # the beam of the nearest label that has one.
    titles = [d for d in dets if d["cls"] == "BeamName"]
    sec_titles = [d for d in dets if d["cls"] == "SectionTitle"]
    def title_box(t):
        """Whole title: the beam name plus its size text ('11B14a' + '600(WD)x1125/825(DP)')."""
        b = list(t["bbox"])
        if t["cls"] == "BeamName" and "rel" in t:
            o = dets[t["rel"]]["bbox"]
            b = [min(b[0], o[0]), min(b[1], o[1]), max(b[2], o[2]), max(b[3], o[3])]
        return b

    def beam_below(bb, pool, px=None):
        """Nearest title below bb that overlaps it horizontally. Titles side by side at the same level (less
        than half a text height apart: the beams of one row, each title under its own span): with px (where
        the label sits) the title of the span px is in, i.e. the last title starting before px (a title starts
        near its span's left end); without px, or left of them all, the left one."""
        x0, y0, x1, y1 = bb
        best, cands = None, []
        for t in pool:
            tx0, ty0, tx1, _ = title_box(t)
            if ty0 < y1 - UNIT or tx1 < x0 - 6 * UNIT or tx0 > x1 + 6 * UNIT:
                continue
            cands.append((ty0 - y1, t))
        if cands:
            g = min(c[0] for c in cands)
            row = [c for c in cands if c[0] <= g + 0.5 * UNIT]
            before = [c for c in row if px is not None and title_box(c[1])[0] <= px + 3 * UNIT]
            best = max(before, key=lambda c: title_box(c[1])[0]) if before else \
                min(row, key=lambda c: title_box(c[1])[0])
        if best is None:     # title off to one side: nearest title below the drawing
            cand = [(math.hypot((title_box(t)[0] + title_box(t)[2]) / 2 - (x0 + x1) / 2, t["bbox"][1] - y1), t)
                    for t in pool if t["bbox"][1] >= y1 - UNIT]
            if not cand: return "?"
            best = min(cand, key=lambda z: z[0])
        t = best[1]
        return t["text"] if t["cls"] == "BeamName" else re.sub(r"^SECTION\s*", "Sec ", t["text"])

    def densify(pts, step):
        out = [pts[:1]]
        for a, b in zip(pts[:-1], pts[1:]):
            m = max(1, math.ceil(math.dist(a, b) / step))
            out.append(a + np.linspace(0, 1, m + 1)[1:, None] * (b - a))
        return np.vstack(out)

    ctr = lambda d: np.array([(d["bbox"][0] + d["bbox"][2]) / 2, (d["bbox"][1] + d["bbox"][3]) / 2])
    # 1) longitudinal bars (long lines): title below
    bars = [d for d in dets if d["cls"] == "RebarSize" and d.get("rebar_nodes")]
    for d in bars:
        P = np.vstack([np.array(s_).reshape(-1, 2) for s_ in d["line"]])
        d["beam"] = beam_below([*P.min(0), *P.max(0)], titles, px=(d["bbox"][0] + d["bbox"][2]) / 2)
    bar_pts = [(d["beam"], np.vstack([np.array(s_).reshape(-1, 2) for s_ in d["line"]])) for d in bars]
    bar_trees = []
    for d in bars:
        S = np.vstack([np.array(s_).reshape(-1, 2, 2) for s_ in d["line"]])
        bar_trees.append((d["beam"], cKDTree(np.vstack([densify(sg, STEP * 4) for sg in S]))))
    # 2) section labels: the section title below, when it is nearer than any bar line;
    # 3) everything else: beam of the nearest bar line
    def section_below(bb):
        """Section title under a label/dimension (horizontally overlapping), and its distance."""
        x0, y0, x1, y1 = bb
        best = None
        for t in sec_titles:
            tx0, ty0, tx1, _ = t["bbox"]
            if ty0 < y1 - UNIT or tx1 < x0 - 10 * UNIT or tx0 > x1 + 10 * UNIT:
                continue
            gap = ty0 - y1
            if best is None or gap < best[0]:
                best = (gap, t)
        return best
    for d in dets:
        if d["cls"] not in ("RebarSize", "Dim", "ZoneLen") or d.get("beam"):
            continue
        c = ctr(d) if not d.get("line") else np.array(d["line"][0]).reshape(-1, 2).mean(0)
        d_bar = min((bt[1].query(c)[0] for bt in bar_trees), default=np.inf)
        sb = section_below(d["bbox"]) if sec_titles else None
        if sb and sb[0] < 40 * UNIT and (d["source"] == "ocr" or sb[0] < d_bar):
            d["beam"] = re.sub(r"^SECTION\s*", "Sec ", sb[1]["text"])
        elif bar_trees:
            d["beam"] = min(bar_trees, key=lambda bt: bt[1].query(c)[0])[0]
    for d in dets:
        if d["cls"] == "ZoneLen":
            d["beam"] = dets[d["zone_of"]].get("beam", "?")

    # --- 6b. continued drawings: the part drawn twice where a beam continues in the next row ----------
    # A long beam is split over two rows ('CONTINUE BELOW'). The end of the upper row and the start of the
    # lower row show the same part of the beam (around the shared grid line): same labels, same
    # dimensions, at the same place relative to that grid line. Each such element is counted once:
    # where it is drawn complete if the other copy is cut by the drawing edge, otherwise in the row whose
    # span it lies in (left of the shared grid = upper row, right = lower row). The other copy keeps its
    # box and line (it is drawn) and gets repeat_of = the counted copy, and no beam membership.
    RE_GRID = re.compile(r"^(?=.*[A-Z0-9])[A-Z]{0,2}\d{0,3}[A-Z]?'?$")
    toks = [nodes[i] for i in text_ids if RE_GRID.match(nodes[i]["text"])]
    bubbles = []
    for n in nodes:
        if n["kind"] != "line" or not np.any(n.get("seg_curve", [])):
            continue
        x0, y0, x1, y1 = n["bbox"]; w, h = x1 - x0, y1 - y0
        if not (1.2 * UNIT <= w <= 6 * UNIT and abs(w - h) < 0.08 * w + 0.3):
            continue
        S_ = n["segs"].reshape(-1, 2); c = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
        rr = np.hypot(*(S_ - c).T)
        if rr.std() > 0.06 * w or not n.get("closed", False) and math.dist(S_[0], S_[-1]) > 0.1 * w:
            continue                                         # not a circle
        ins = [t for t in toks if math.dist(t["c"], c) < 0.3 * w]
        if len(ins) == 1 and not any(np.linalg.norm(b["c"] - c) < 0.5 * w for b in bubbles):
            bubbles.append(dict(label=ins[0]["text"], c=c, r=w / 2, k=len(bubbles)))
    rows = []
    for b in sorted(bubbles, key=lambda b: b["c"][1]):
        if rows and abs(rows[-1][-1]["c"][1] - b["c"][1]) < UNIT:
            rows[-1].append(b)
        else:
            rows.append([b])
    rows = [sorted(r, key=lambda b: b["c"][0]) for r in rows]
    for ri, r in enumerate(rows):
        for b in r:
            b["row"] = ri
    base_of = lambda t: (m.group(1) if (m := re.match(r"^(.*\d)([a-z]{1,2})$", t)) else None)
    MATCH_CLS = ("RebarSize", "Dim", "ZoneLen")

    def mid_x(d):                       # where an element is along the beam: its line, else its text
        if d.get("line"):
            P = np.vstack([np.array(s_).reshape(-1, 2) for s_ in d["line"]]); return float((P[:, 0].min() + P[:, 0].max()) / 2)
        if d.get("dimline") is not None:
            q = np.array(d["dimline"], float).reshape(-1, 2); return float(q[:, 0].mean())
        return float(ctr(d)[0])

    def clip_len(line, lo, hi):         # length (pt) of a bar line between x = lo and x = hi
        tot = 0.0
        for x0, y0, x1, y1 in np.vstack([np.array(s_, float).reshape(-1, 4) for s_ in line]):
            L = math.hypot(x1 - x0, y1 - y0)
            if abs(x1 - x0) < 1e-6:
                tot += L if lo <= x0 <= hi else 0.0
                continue
            t0, t1 = sorted(((lo - x0) / (x1 - x0), (hi - x0) / (x1 - x0)))
            tot += L * max(0.0, min(1.0, t1) - max(0.0, t0))
        return tot

    def geo_x(d):                       # x-extent of an element's geometry (text, line, dimension line)
        xs = [d["bbox"][0], d["bbox"][2]]
        if d.get("line"):
            P = np.vstack([np.array(s_).reshape(-1, 2) for s_ in d["line"]]); xs += [P[:, 0].min(), P[:, 0].max()]
        if d.get("dimline") is not None:
            xs += list(np.array(d["dimline"], float).reshape(-1, 2)[:, 0])
        return min(xs), max(xs)

    claimed, CONT = set(), []
    for i, Ru in enumerate(rows):
        for j in range(i + 1, len(rows)):
            Rl = rows[j]
            if j in claimed:
                continue
            yu, yl = Ru[0]["c"][1], Rl[0]["c"][1]
            if yl - yu < 8 * UNIT:
                continue
            found = None
            for bu in Ru:
                for bl in Rl:
                    if bu["label"] != bl["label"]:
                        continue
                    o = bu["c"] - bl["c"]
                    if o[0] < 10 * UNIT:                     # same x: parallel beams on the same grids, not a continuation
                        continue
                    ok, shared = True, []
                    for b2 in Rl:                            # the grids must agree wherever the rows overlap
                        q = b2["c"] + o
                        hit = [b3 for b3 in Ru if np.linalg.norm(b3["c"] - q) < 1.5 * UNIT]
                        if hit and hit[0]["label"] != b2["label"]:
                            ok = False
                        elif hit:
                            shared.append(hit[0])
                    if not ok or Ru[-1]["k"] not in {b3["k"] for b3 in shared}:   # the upper row must end inside the overlap
                        continue
                    yb = (yu - 2 * UNIT, yl - 2 * UNIT, yl + (yl - yu))
                    A = [d for d in dets if d["cls"] in MATCH_CLS and yb[0] < ctr(d)[1] < yb[1]]
                    B = [d for d in dets if d["cls"] in MATCH_CLS and yb[1] < ctr(d)[1] < yb[2]]
                    tu = [t for t in titles if yb[0] < t["bbox"][1] < yb[1]]
                    tl = [t for t in titles if yb[1] < t["bbox"][1] < yb[2]]
                    lastu = min(tu, key=lambda t: abs(ctr(t)[0] - bu["c"][0]) + (0 if ctr(t)[0] < bu["c"][0] else 1e4), default=None)
                    firstl = min(tl, key=lambda t: abs(ctr(t)[0] - bl["c"][0]) + (0 if ctr(t)[0] > bl["c"][0] else 1e4), default=None)
                    if lastu and firstl and base_of(lastu["text"]) and base_of(firstl["text"]) \
                            and base_of(lastu["text"]) != base_of(firstl["text"]):
                        continue                             # 11B14c -> 11B15d: a different beam line
                    cand = sorted((float(np.linalg.norm(ctr(a) - ctr(b) - o)), a["id"], b["id"]) for a in A for b in B
                                  if a["cls"] == b["cls"] and a["text"] == b["text"] and np.linalg.norm(ctr(a) - ctr(b) - o) < UNIT)
                    used, pairs = set(), []
                    for _, ia, ib in cand:
                        if ia not in used and ib not in used:
                            used |= {ia, ib}; pairs.append((dets[ia], dets[ib]))
                    if len(pairs) >= 2 and (found is None or len(pairs) > len(found[3])):
                        found = (bu, bl, o, pairs, shared, A, B, tu, tl)
            if found:
                claimed.add(j)
                CONT.append(found)
                break

    CONT_INFO = []
    for bu, bl, o, pairs, shared, A, B, tu, tl in CONT:
        ox = float(o[0])
        grid_x = sorted({round(float(b["c"][0]), 1) for b in rows[bu["row"]]} |
                        {round(float(b["c"][0]) + ox, 1) for b in rows[bl["row"]]})
        ttl = [(float(ctr(t)[0]), "upper", t) for t in tu] + [(float(ctr(t)[0]) + ox, "lower", t) for t in tl]
        split = min(float(b["c"][0]) for b in shared)        # first shared grid line (upper coordinates)

        def owner(x):
            """Row and title of the span (between grid lines) that x lies in, upper-row coordinates."""
            lo = max([g for g in grid_x if g <= x], default=-1e9); hi = min([g for g in grid_x if g > x], default=1e9)
            inside = [z for z in ttl if lo <= z[0] <= hi]
            sides = {z[1] for z in inside}
            if len(sides) == 1:
                z = min(inside, key=lambda z: abs(z[0] - x)); return z[1], z[2]["text"]
            side = "upper" if x < split else "lower"
            pool = [z for z in ttl if z[1] == side]
            z = min(pool, key=lambda z: abs(z[0] - x), default=None)
            return side, (z[2]["text"] if z else None)

        gl = bu["label"]
        yu_, yl_ = float(bu["c"][1]), float(bl["c"][1])
        CONT_INFO.append(dict(grid=gl, ox=ox, oy=float(o[1]), split_upper=split, split_lower=split - ox,
                              upper_y=[yu_ - 2 * UNIT, yl_ - 2 * UNIT], lower_y=[yl_ - 2 * UNIT, yl_ + (yl_ - yu_)],
                              upper_end=max(float(b["c"][0]) for b in rows[bu["row"]]) + 5 * UNIT,
                              lower_start=min(float(b["c"][0]) for b in rows[bl["row"]]) - 5 * UNIT))
        n_rep = 0
        for a, b in pairs:                                   # a: upper copy, b: lower copy
            la, lb = a.get("line_len") or 0.0, b.get("line_len") or 0.0
            if (la > 0) != (lb > 0):
                keep, why = (a, "the other copy has no bar line") if la > 0 else (b, "the other copy has no bar line")
            elif la and abs(la - lb) > max(0.02 * max(la, lb), UNIT):
                keep, why = (a, "drawn complete here") if la > lb else (b, "drawn complete here")
            else:
                side, _ = owner(mid_x(a))
                keep, why = (a, "in its span") if side == "upper" else (b, "in its span")
            rep = b if keep is a else a
            # the shared grid line this pair sits at, in each copy's own coordinates: each copy's bar line is
            # drawn only on its own side of it (upper row: left of it, lower row: right); past it, it is the
            # other row's bar drawn again (YS, 17 Oct)
            a["grid_x"], b["grid_x"] = split, split - ox     # the grid where the upper row hands over (owner())
            a["grid_side"], b["grid_side"] = "left", "right"
            xk = mid_x(keep) + (ox if keep is b else 0.0)
            _, beam = owner(xk)
            beam = beam or keep.get("beam")
            keep["beam"] = rep["beam"] = beam
            rep["repeat_of"] = keep["id"]
            keep.setdefault("repeats", []).append(rep["id"])
            # takeoff length: each copy is the real bar only on its own side of the grid (upper: A, lower: D);
            # the bar = A + D, counted once on the kept label (number of bars from it)
            if a["cls"] == "RebarSize" and a.get("line") and b.get("line") and a.get("length_mm") and b.get("length_mm"):
                pa, pb = clip_len(a["line"], -1e9, split), clip_len(b["line"], split - ox, 1e9)
                sa, sb = a["length_mm"] / max(a["line_len"], 1e-6), b["length_mm"] / max(b["line_len"], 1e-6)
                stub = (a.get("stub_len_pt", 0.0) * sa + b.get("stub_len_pt", 0.0) * sb) / 2
                tot = (b["length_mm"] if pa < 1e-6 else a["length_mm"] if pb < 1e-6    # one copy all in the overlap
                       else pa * sa + pb * sb - stub)
                keep["length_drawn_mm"] = keep["length_mm"]
                keep["length_mm"] = round(tot)
                keep["length_note"] = (f"continued at grid {gl}: {round(pa * sa)} mm up to the grid (upper row) + "
                                       f"{round(pb * sb)} mm from it (lower row) = {round(tot)} mm")
            where = "in the row below" if keep is b else "in the row above"
            rep["repeat_note"] = f"repeat (drawn twice where the beam continues, grid {gl}): counted {where} - not counted"
            keep["repeat_note"] = f"also drawn in the row {'above' if keep is b else 'below'} (grid {gl}): counted here ({why})"
            n_rep += 1
        # elements drawn only once in the overlap, but on the other row's side of the grid: that row's beam
        paired = {d["id"] for ab in pairs for d in ab}
        ua = [geo_x(d) for d in A if d.get("line") or d.get("dimline") is not None]
        lb_ = [geo_x(d) for d in B if d.get("line") or d.get("dimline") is not None]
        ov0 = min((g[0] for g in lb_), default=1e9) + ox
        ov1 = max((g[1] for g in ua), default=-1e9)
        moved, Bids = 0, {d["id"] for d in B}
        for d in A + B:
            if d["id"] in paired or d["cls"] == "ZoneLen":
                continue
            inB = d["id"] in Bids
            x = mid_x(d) + (ox if inB else 0.0)
            if not ov0 <= x <= ov1:
                continue
            side, beam = owner(x)
            if beam and side == ("upper" if inB else "lower") and d.get("beam") != beam:
                d["repeat_note"] = f"drawn only in this row, in the span of {beam} (other side of grid {gl})"
                d["beam"] = beam; moved += 1
        for d in dets:
            if d["cls"] == "ZoneLen" and d.get("zone_of") is not None:
                d["beam"] = dets[d["zone_of"]].get("beam", d.get("beam"))
        log(f"continued drawing at grid {gl} ({(tu and max(tu, key=lambda t: ctr(t)[0])['text']) or '?'} -> "
            f"{(tl and min(tl, key=lambda t: ctr(t)[0])['text']) or '?'}): {n_rep} elements drawn twice, "
            f"{n_rep} repeats marked (not counted){f', {moved} reassigned by span' if moved else ''}")
    if not CONT:
        log(f"continued drawings: none ({len(bubbles)} grid bubbles in {len(rows)} rows)")

    # --- 6c. sections: cut marks in the elevations, link shapes from the section drawings ------------
    # 'SECTION 15A-15A' is drawn for the cut marked '15A' (top and bottom of the beam) in an elevation.
    # The section shows the links of one set: each closed link is a rectangle whose sides are bar lines
    # (one line, or two lines = the bar's faces drawn to scale), with main bars (dots) in its corners.
    SEC_RE = re.compile(r"SECTION\s*([A-Z0-9]{1,8})\s*-\s*([A-Z0-9]{1,8})", re.I)
    SEC_RE1 = re.compile(r"SECTION\s*([A-Z]{0,3}\d[A-Z0-9]{0,6})\s*$", re.I)    # Ramboll's one id: SECTION 13A, SECTION 14b
    secs = []
    for T in [d for d in dets if d["cls"] == "SectionTitle"]:
        m, m1 = SEC_RE.search(T["text"]), SEC_RE1.search(T["text"].strip())
        if m:
            secs.append(dict(T=T, ids={m.group(1).upper(), m.group(2).upper()}, marks=[], links=[]))
        elif m1:
            secs.append(dict(T=T, ids={m1.group(1).upper()}, marks=[], links=[]))
    in_bb = lambda b, B, pad=0.0: b[0] >= B[0] - pad and b[1] >= B[1] - pad and b[2] <= B[2] + pad and b[3] <= B[3] + pad
    title_bbs = [d["bbox"] for d in dets if d["cls"] in ("BeamName", "BeamSize", "SectionTitle")]
    beam_names = {d["text"].upper(): d["text"] for d in titles}
    ARROWS = [n for n in nodes if n["kind"] == "mark" and n.get("filled")]
    BUBBLES = [n["bbox"] for n in nodes if n["kind"] == "line" and n.get("curved") and n.get("closed")
               and n["bbox"][2] - n["bbox"][0] < 4 * UNIT and n["bbox"][3] - n["bbox"][1] < 4 * UNIT]

    def in_bubble(b):                                   # text in a small circle: a grid line's or the section's bubble
        return any(in_bb(b, B, 0.2 * UNIT) for B in BUBBLES)

    def arrow_beside(b):                                # a cut's arrowhead within 1.5 text heights of the text
        return any(max(b[0] - a["bbox"][2], a["bbox"][0] - b[2], 0) < 1.5 * UNIT and
                   max(b[1] - a["bbox"][3], a["bbox"][1] - b[3], 0) < 1.5 * UNIT for a in ARROWS)
    for sc in secs:                                     # 1) cut marks: the section id written on its own
        named = [beam_names[i_] for i_ in sc["ids"] if i_ in beam_names]
        if named:                                       # 'SECTION 7B229-7B229': the section of that beam, no cut marks
            sc["T"]["beam"] = sc["direct"] = named[0]
            continue
        # The id at a cut: in pairs at the same x above and below the beam, or beside the cut's arrowhead.
        # Not the same text anywhere else (grid bubbles, the section's own bubble, notes, dimensions).
        def in_scale_note(b):                           # the '1' of 'SCALE 1 : 25' (YS, 15 Oct: not a cut mark)
            for j in text_ids:
                t, bb = nodes[j]["text"].strip().upper(), nodes[j]["bbox"]
                if abs((bb[1] + bb[3]) / 2 - (b[1] + b[3]) / 2) > 0.5 * UNIT: continue
                if t.startswith(":") and 0 <= bb[0] - b[2] < 1.5 * UNIT: return True
                if t.startswith("SCALE") and 0 <= b[0] - bb[2] < 1.5 * UNIT: return True
            return False
        cand = [i for i in text_ids if nodes[i]["text"].strip().upper() in sc["ids"]
                and not any(in_bb(nodes[i]["bbox"], bb, 1.0) for bb in title_bbs) and not in_bubble(nodes[i]["bbox"])
                and not in_scale_note(nodes[i]["bbox"])]
        for i in cand:
            n = nodes[i]
            paired = any(j != i and abs(nodes[j]["c"][0] - n["c"][0]) < 1.0 * UNIT
                         and 1.5 * UNIT < abs(nodes[j]["c"][1] - n["c"][1]) < 40 * UNIT for j in cand)
            if not (paired or arrow_beside(n["bbox"])):
                continue
            mk = add_det("SectionMark", n["text"].strip(), n["bbox"], n.get("source", "pdf-text"), conf=1.0,
                         section=sc["T"]["id"])
            mk["beam"] = beam_below(n["bbox"], titles)
            sc["marks"].append(mk)

    MKS = [n for n in nodes if n["kind"] == "mark" and n.get("filled")]
    MK_C = np.array([n["c"] for n in MKS]) if MKS else np.zeros((0, 2))
    SEG_dash = np.array([bool(nodes[i].get("dashed")) for i in SEG_own])
    SEG_h = np.abs(SEG_u[:, 1]) < 0.035                 # within 2 degrees of horizontal
    SEG_vv = np.abs(SEG_u[:, 0]) < 0.035

    def bars_of(idx, vertical):
        """Parallel segments -> bars: one line, or the two faces of a bar drawn to scale (same pen, same extent)."""
        it = []
        for k in idx:
            a, b = SEG[k]
            if vertical:
                it.append(((a[0] + b[0]) / 2, min(a[1], b[1]), max(a[1], b[1]), SEG_w[k]))
            else:
                it.append(((a[1] + b[1]) / 2, min(a[0], b[0]), max(a[0], b[0]), SEG_w[k]))
        it.sort()
        out, used = [], [False] * len(it)
        for i_, a in enumerate(it):
            if used[i_]:
                continue
            grp, used[i_] = [a], True
            for j_ in range(i_ + 1, len(it)):
                o = it[j_]
                if o[0] - a[0] > 0.45 * UNIT:
                    break
                if not used[j_] and o[3] == a[3] and abs(o[1] - a[1]) < 0.3 * UNIT and abs(o[2] - a[2]) < 0.3 * UNIT:
                    grp.append(o); used[j_] = True
                    break                               # a bar has two faces at most
            out.append((float(np.mean([g[0] for g in grp])), min(g[1] for g in grp), max(g[2] for g in grp), a[3]))
        return out

    def dots_in(x0, x1, y0, y1):                        # distinct bar dots in a box
        if not len(MK_C):
            return 0
        P = MK_C[(MK_C[:, 0] > x0) & (MK_C[:, 0] < x1) & (MK_C[:, 1] > y0) & (MK_C[:, 1] < y1)]
        seen = []
        for q in P:
            if all(math.dist(q, o) > 0.15 * UNIT for o in seen):
                seen.append(q)
        return len(seen)

    def covered(bars, lo, hi):                          # share of [lo, hi] covered by collinear pieces
        iv = sorted((max(b_[1], lo), min(b_[2], hi)) for b_ in bars if b_[2] > lo and b_[1] < hi)
        tot, cur = 0.0, lo
        for a_, b_ in iv:
            if b_ > cur:
                tot += b_ - max(a_, cur); cur = b_
        return tot / max(hi - lo, 1e-9)

    def link_rects(region):
        """Closed links: a top and a bottom side of the same extent, left and right sides covering the height
        between them (one line or several pieces), one pen, main bars (dots) inside at the top and bottom."""
        x0r, y0r, x1r, y1r = region
        inreg = (SEG[:, :, 0].min(1) >= x0r) & (SEG[:, :, 0].max(1) <= x1r) & (SEG[:, :, 1].min(1) >= y0r) & (SEG[:, :, 1].max(1) <= y1r)
        base = inreg & ~SEG_dash & (SEG_len >= 0.4 * UNIT)
        V, Hb = bars_of(np.where(base & SEG_vv)[0], True), bars_of(np.where(base & SEG_h)[0], False)
        rr = 0.6 * UNIT                                 # room for the bends at the corners
        out = []
        for i1, h1 in enumerate(Hb):
            for h2 in Hb:
                if h2[0] - h1[0] < 0.8 * UNIT or h2[3] != h1[3] or abs(h2[1] - h1[1]) > 0.5 * UNIT or abs(h2[2] - h1[2]) > 0.5 * UNIT:
                    continue
                if h1[2] - h1[1] < 0.4 * UNIT:
                    continue
                ya, yb = h1[0] + rr, h2[0] - rr
                def side(x_ref, left):
                    xs = sorted({round(v[0], 2) for v in V if v[3] == h1[3] and
                                 (x_ref - rr - 0.3 * UNIT <= v[0] <= x_ref + 0.05 * UNIT if left else x_ref - 0.05 * UNIT <= v[0] <= x_ref + rr + 0.3 * UNIT)},
                                key=lambda x: abs(x - x_ref))
                    for x in xs:
                        pcs = [v for v in V if v[3] == h1[3] and abs(v[0] - x) < 0.05 * UNIT]
                        ends_top = any(h1[0] - 0.3 * UNIT <= v[1] <= h1[0] + rr + 0.3 * UNIT for v in pcs)     # the side turns into
                        ends_bot = any(h2[0] - rr - 0.3 * UNIT <= v[2] <= h2[0] + 0.3 * UNIT for v in pcs)     # the top and bottom
                        if ends_top and ends_bot and covered(pcs, ya, yb) >= 0.85:
                            return x
                    return None
                xl = side(min(h1[1], h2[1]), True)
                xr = side(max(h1[2], h2[2]), False) if xl is not None else None
                if xr is None or xr - xl < 0.8 * UNIT:
                    continue
                r = dict(x0=xl, x1=xr, y0=h1[0], y1=h2[0], pen=h1[3])
                band = max(UNIT, 0.2 * (r["y1"] - r["y0"]))      # main bars in the corners: 2+ at the top and the bottom
                if dots_in(r["x0"], r["x1"], r["y0"], r["y0"] + band) >= 2 and dots_in(r["x0"], r["x1"], r["y1"] - band, r["y1"]) >= 2:
                    if not any(abs(o["x0"] - r["x0"]) < 0.1 * UNIT and abs(o["x1"] - r["x1"]) < 0.1 * UNIT and abs(o["y0"] - r["y0"]) < 0.1 * UNIT
                               and abs(o["y1"] - r["y1"]) < 0.1 * UNIT for o in out):
                        out.append(r)
        return out

    def face_gaps(r, mmpt):
        """The cover, measured: from each side of the outer link to the nearest concrete face, i.e. a solid line
        parallel to that side, 10-150 mm outside it, running along at least half of it. [left, right, top, bottom]
        in mm (None where no face is found)."""
        out_ = []
        for side in "lrtb":
            vert = side in "lr"
            k_ = np.where(~SEG_dash & (SEG_vv if vert else SEG_h))[0]
            P = SEG[k_]
            pos = P[:, :, 0].mean(1) if vert else P[:, :, 1].mean(1)
            lo_, hi_ = (P[:, :, 1].min(1), P[:, :, 1].max(1)) if vert else (P[:, :, 0].min(1), P[:, :, 0].max(1))
            ref = (r["y0"], r["y1"]) if vert else (r["x0"], r["x1"])
            edge = r[{"l": "x0", "r": "x1", "t": "y0", "b": "y1"}[side]]
            dist = edge - pos if side in "lt" else pos - edge
            ok = (dist >= 10 / mmpt) & (dist <= 150 / mmpt) & (hi_ > ref[0]) & (lo_ < ref[1])
            cand, found, i_ = sorted(zip(dist[ok], lo_[ok], hi_[ok])), None, 0
            while i_ < len(cand) and found is None:     # nearest first; collinear pieces count together
                grp = [cand[i_]]
                while i_ + 1 < len(cand) and cand[i_ + 1][0] - grp[0][0] < 0.3:
                    i_ += 1; grp.append(cand[i_])
                runs = []                               # pieces joined where they touch: a dashed line is no face
                for _, a_, b_ in sorted(grp, key=lambda g_: g_[1]):
                    if runs and a_ <= runs[-1][1] + 0.5:
                        runs[-1][1] = max(runs[-1][1], b_)
                    else:
                        runs.append([a_, b_])
                longest = max(min(b_, ref[1]) - max(a_, ref[0]) for a_, b_ in runs)
                if covered([(0.0, a_, b_) for _, a_, b_ in grp], *ref) >= 0.5 and longest >= 0.3 * (ref[1] - ref[0]):
                    found = float(np.mean([g_[0] for g_ in grp])) * mmpt
                i_ += 1
            out_.append(found)
        return out_

    for sc in secs:                                     # 2) the links drawn in each section
        T = sc["T"]; tb = T["bbox"]; tcx = (tb[0] + tb[2]) / 2
        R_ = link_rects((tcx - 30 * UNIT, tb[1] - 45 * UNIT, tcx + 30 * UNIT, tb[1] + 0.5 * UNIT))
        cont = lambda o, r: o["x0"] <= r["x0"] + 0.1 and o["x1"] >= r["x1"] - 0.1 and o["y0"] <= r["y0"] + 0.1 and o["y1"] >= r["y1"] - 0.1
        def outline(a):                                 # encloses another rectangle at an even small offset (the cover)
            for b_ in R_:
                if b_ is a or not cont(a, b_):
                    continue
                off = [b_["x0"] - a["x0"], a["x1"] - b_["x1"], b_["y0"] - a["y0"], a["y1"] - b_["y1"]]
                if min(off) >= 0.2 * UNIT and max(off) <= 2 * UNIT and max(off) - min(off) <= 0.6 * UNIT:
                    return True
            return False
        R_ = [r for r in R_ if not outline(r)]
        inner_pens = {r["pen"] for r in R_ if any(o is not r and cont(o, r) for o in R_)}
        if len({r["pen"] for r in R_}) > 1 and len(inner_pens) == 1:
            R_ = [r for r in R_ if r["pen"] in inner_pens]   # a closed concrete outline in another pen is not a link
        outers = [r for r in R_ if not any(o is not r and cont(o, r) for o in R_)]
        if not outers:
            continue
        outer = min(outers, key=lambda r: abs((r["x0"] + r["x1"]) / 2 - tcx) + 0.3 * abs(r["y1"] - tb[1]))
        links = sorted([r for r in R_ if cont(outer, r)], key=lambda r: (r is not outer, r["x0"]))
        words_below = sorted([nodes[i] for i in text_ids if tb[3] - 0.2 * UNIT <= nodes[i]["c"][1] <= tb[3] + 2.5 * UNIT
                              and tb[0] - 2 * UNIT <= nodes[i]["c"][0] <= tb[2] + 2 * UNIT], key=lambda n: n["c"][0])
        ms = re.search(r"1\s*:\s*(\d{1,3})\b", " ".join(n["text"] for n in words_below))
        if ms:
            mmpt, sc["scale"] = int(ms.group(1)) * PT_MM, f"1:{ms.group(1)}"
        else:
            mmpt = VIEW_SCALE.get(view_of_point((outer["x0"], outer["y0"])), SHEET_SCALE); sc["scale"] = f"1:{mmpt / PT_MM:.0f}"
        for r in links:
            r["kind"] = "outer" if r is outer else "inner"
            r["w_mm"], r["h_mm"] = int(5 * round((r["x1"] - r["x0"]) * mmpt / 5)), int(5 * round((r["y1"] - r["y0"]) * mmpt / 5))
        sc["links"] = links
        sc["faces"] = face_gaps(outer, mmpt)
        T["link_loops"] = [[[r["x0"], r["y0"]], [r["x1"], r["y0"]], [r["x1"], r["y1"]], [r["x0"], r["y1"]], [r["x0"], r["y0"]]] for r in links]
        T["link_dims"] = [(r["kind"], r["w_mm"], r["h_mm"]) for r in links]
        T["scale"] = sc["scale"]

    def zone_x(d):                                      # x-extent of a link label's zone line, else of its text
        if d.get("line"):
            P = np.vstack([np.array(s_).reshape(-1, 2) for s_ in d["line"]]); return float(P[:, 0].min()), float(P[:, 0].max())
        return d["bbox"][0], d["bbox"][2]
    link_labels = [d for d in dets if d["cls"] == "RebarSize" and (d.get("label") or {}).get("kind") == "link"
                   and not (d.get("label") or {}).get("face")
                   and d.get("repeat_of") is None and not str(d.get("beam", "")).startswith("Sec")
                   and (d.get("line") or "LINK" in str(d.get("suffix") or "").upper())]
    for sc in secs:                                     # 3) the link label of the cut beam -> that section
        if sc.get("direct"):                            # named after the beam: all its link labels
            for d in link_labels:
                if d.get("beam") == sc["direct"]:
                    d.setdefault("sections", []).append(sc["T"]["id"])
            continue
        by_beam = defaultdict(list)
        for mk in sc["marks"]:
            by_beam[mk["beam"]].append(mk)
        for beam, mks in by_beam.items():
            cx = float(np.mean([(m_["bbox"][0] + m_["bbox"][2]) / 2 for m_ in mks]))
            cands = [d for d in link_labels if d.get("beam") == beam]
            if not cands:
                continue
            gap = lambda d: (0.0 if zone_x(d)[0] <= cx <= zone_x(d)[1] else min(abs(cx - zone_x(d)[0]), abs(cx - zone_x(d)[1])))
            L = min(cands, key=gap)
            L.setdefault("sections", []).append(sc["T"]["id"])
            for mk in mks:
                mk["link_label"] = L["id"]
    for sc in secs:
        dims = ", ".join(f"{k} {w}x{h}" for k, w, h in sc["T"].get("link_dims", []))
        log(f"{sc['T']['text']}: " + (f"section of {sc['direct']}" if sc.get("direct") else
            f"cut marks {len(sc['marks'])} in {sorted({m_['beam'] for m_ in sc['marks']}) or '-'}") + ", "
            f"{len(sc['links'])} closed links ({dims or 'none found'}, {sc.get('scale', '?')})"
            + (", concrete face to link (l/r/t/b): " + "/".join("-" if g_ is None else f"{g_:.0f}" for g_ in sc["faces"]) + " mm" if sc.get("faces") else ""))

    # --- 7. report ----------------------------------------------------------------------
    cnt = Counter(d["cls"] for d in dets)
    rb = [d for d in dets if d["cls"] == "RebarSize"]
    dm = [d for d in dets if d["cls"] == "Dim"]
    log("detected:", dict(cnt))
    log(f"RebarSize with a RebarLine: {sum('line' in d for d in rb)}/{len(rb)}  "
          f"(bars {sum('line' in d for d in rb if d['label']['kind']=='bar')}/{sum(d['label']['kind']=='bar' for d in rb)}, "
          f"links {sum('line' in d for d in rb if d['label']['kind']=='link')}/{sum(d['label']['kind']=='link' for d in rb)})")
    log(f"Dim with a DimLine: {sum('dimline' in d for d in dm)}/{len(dm)}, "
          f"consistent with its view scale: {sum(d.get('scale_ok', False) for d in dm)}")

    for d in dets:
        if d.get("zone_len") and d.get("length_mm"):
            ok = abs(d["length_mm"] - d["zone_len"]) <= 0.01 * d["zone_len"] + 5
            d["check"] = "zone ok" if ok else f"drawn {d['length_mm']:.0f} vs printed {d['zone_len']:.0f} - use printed"
        elif d["cls"] == "Dim":
            d["check"] = "no dim line found" if not d.get("dimline") else ("on scale" if d.get("scale_ok") else
                         f"off scale (1:{d['mm_per_pt'] / PT_MM:.1f})")
        elif d["cls"] == "RebarSize" and not d.get("line"):
            d["check"] = "no line (section dot / side bars)"
    sec_by_id = {sc["T"]["id"]: sc for sc in secs}
    RE_SZ = re.compile(r"(\d{2,4})\s*(?:\(\s*W\w*\s*\))?\s*[xX×]\s*(\d{2,4})(?:\s*/\s*(\d{2,4}))?")
    def beam_size(name):
        """(width, depth, other depth or None) from the title's size text: '600(WD)x1125/825(DP)', '(600x500)'."""
        t = next((d for d in titles if d["text"] == name and "rel" in d), None)
        m = RE_SZ.search(dets[t["rel"]]["text"]) if t else None
        return (int(m[1]), int(m[2]), int(m[3]) if m[3] else None) if m else None

    def is_link_label(d):
        lab = d.get("label") or {}
        return (d["cls"] == "RebarSize" and lab.get("kind") == "link" and not lab.get("face") and d.get("repeat_of") is None
                and not str(d.get("beam", "")).startswith("Sec") and bool(d.get("line") or "LINK" in str(d.get("suffix") or "").upper()))

    def per_set(d):     # links in one set, from the label: '3 x T12-100' 3, 'H10-300+2H10-300' 3, 'T12-150 CLOSED LINKS + LINKS' 2
        lab = d["label"]
        if lab.get("links_per_set"):
            return lab["links_per_set"]
        return sum(p_["count"] for p_ in lab["parts"]) + (1 if "+" in str(d.get("suffix") or "") else 0)

    sig = lambda d: (per_set(d), len(d["label"]["parts"]), "+" in str(d.get("suffix") or ""))
    norm_t = lambda d: (re.sub(r"\s+", "", d["text"]).upper(), re.sub(r"[\s+]+", " ", str(d.get("suffix") or "")).strip().upper())
    line_of = lambda n: re.sub(r"[a-z]{1,2}$", "", str(n or ""))          # 11B15c -> 11B15 (the beam line)
    hs_txt = lambda hs: "..".join(str(h_) for h_ in hs)

    r5 = lambda v: int(5 * round(v / 5))
    for sc in secs:                   # sections with links: the beam they show, its depth at the cut, the cover
        sc["beam"] = sc.get("direct") or next((m_["beam"] for m_ in sc["marks"]), None)
        sc["size"] = sz = beam_size(sc["beam"]) if sc["beam"] else None
        if not sc["links"]:
            continue
        o_ = sc["links"][0]
        fl, fr, ft, fb = sc.get("faces") or [None] * 4
        mx, my = [g_ for g_ in (fl, fr) if g_], [g_ for g_ in (ft, fb) if g_]
        deps = [h_ for h_ in (sz[1:] if sz else []) if h_]
        cx = float(np.mean(mx)) if mx else ((sz[0] - o_["w_mm"]) / 2 if sz else None)
        drawn_h = o_["h_mm"] + (sum(my) if len(my) == 2 else 2 * (my[0] if my else (cx or 0)))   # concrete depth at the cut
        dep = min(deps, key=lambda h_: abs(h_ - drawn_h)) if deps else None
        cy = float(np.mean(my)) if my else ((dep - o_["h_mm"]) / 2 if dep else None)
        if cx is not None and cy is not None and 0 < cx < 120 and 0 < cy < 120:
            sc["cover"], sc["depth"] = (cx, cy), dep
            sc["cover_src"] = f"measured in {sc['T']['text']}" if mx and my else f"from {sc['T']['text']} and the beam size"
        if sz and len(mx) == 2 and len(my) == 2:        # the concrete as drawn vs the beam's title
            dw, dh = o_["w_mm"] + sum(mx), o_["h_mm"] + sum(my)
            if abs(dw - sz[0]) > max(25, 0.05 * sz[0]) or not any(abs(dh - h_) <= max(25, 0.05 * h_) for h_ in deps):
                sc["size_note"] = (f"drawn {r5(dw)}x{r5(dh)}, but {sc['beam']} is {sz[0]}x{'/'.join(map(str, deps))}: "
                                   f"check the section's beam or scale")
    covers = [sc for sc in secs if sc.get("cover")]
    if cover:                         # set by the user: the same cover everywhere
        sheet_cov, sheet_src = (float(cover), float(cover)), "as set"
    elif covers:                      # measured in this sheet's sections
        sheet_cov = (float(np.median([sc["cover"][0] for sc in covers])), float(np.median([sc["cover"][1] for sc in covers])))
        sheet_src = "measured in " + (covers[0]["T"]["text"] if len(covers) == 1 else f"{len(covers)} sections on this sheet")
    else:
        sheet_cov, sheet_src = (25.0, 25.0), "assumed: no section on this sheet to measure it, check"

    def cov_of(sc):                   # the cover for a beam sized after section sc (None: the sheet's)
        if cover or not sc or not sc.get("cover"):
            return sheet_cov, sheet_src
        return sc["cover"], sc["cover_src"]

    def cov_txt(cv, src):
        a_, b_ = int(round(cv[0])), int(round(cv[1]))
        return (f"cover {a_}" if a_ == b_ else f"cover {a_} sides / {b_} top and bottom") + f", {src}"

    HOOKS = 2                                           # a closed link has a hook at each end
    cut_len = lambda a, b, dia: link_cut(a, b, dia, hook_d, HOOKS)

    def set_for(sz, cv, sc):          # the links of one set in a beam of size sz, cover cv, pattern of section sc
        if not sc:
            return link_set(sz, cv)
        return link_set(sz, cv, [(r["kind"], r["w_mm"], r["h_mm"]) for r in sc["links"]], sc.get("size"), sc["T"]["text"])

    drawn = lambda sc: [(r["kind"], r["w_mm"], [r["h_mm"]]) for r in sc["links"]]

    def dias(d, n):                   # bar size of each link of a set: 'H10-300+2H10-300' -> 10, 10, 10
        lab = d["label"]
        ex = [p_["dia"] for p_ in lab["parts"] for _ in range(1 if lab.get("links_per_set") else p_["count"])]
        return [ex[min(i, len(ex) - 1)] for i in range(n)]

    def shape_txt(sh):
        o_ = [x for x in sh if x[0] == "outer"]; inn = [x for x in sh if x[0] == "inner"]
        t = f"outer {o_[0][1]}x{hs_txt(o_[0][2])}" if o_ else ""
        return t + (f" + {len(inn)} inner " + ", ".join(sorted({f'{w}x{hs_txt(hs)}' for _, w, hs in inn})) if inn else "")

    def cut_txt(sh, cuts, per, unknown):
        groups = []                   # identical links once: '2 x inner 2178'
        for (kind, _, _), c in zip(sh, cuts):
            g_ = next((g_ for g_ in groups if g_[0] == kind and g_[1] == c), None)
            if g_:
                g_[2] += 1
            else:
                groups.append([kind, c, 1])
        if len(sh) == 1 and not unknown:
            return f"{hs_txt(cuts[0])} mm"
        t = " + ".join(f"{'' if n_ == 1 else f'{n_} x '}{kind} {hs_txt(c)}" for kind, c, n_ in groups)
        return t + (f" + {unknown} unknown" if unknown else f" = {hs_txt(per)} mm per set")

    labels = [d for d in dets if is_link_label(d)]
    own = {d["id"]: next((sec_by_id[t] for t in d.get("sections", []) if sec_by_id[t]["links"]), None) for d in labels}
    for d in labels:                  # link labels: links per set, their sizes and cut lengths, after the other checks
        lab = d["label"]; sp = lab["parts"][0]["spacing"]
        k_lab, sc, szd = per_set(d), own[d["id"]], beam_size(d.get("beam"))
        extra, unknown, sh, cv, src = [], 0, None, None, None
        if d.get("sections") and not sc:
            extra.append("links not recognised in its section")
        if sc:                                          # its own section (cut marks, or named after the beam)
            k = lab.get("links_per_set") or len(sc["links"])
            head = f"{k} link{'s' if k > 1 else ''} per set @ {sp} c/c ({sc['T']['text']})"
            cv, src = cov_of(sc)
            if szd:
                sh, nt = set_for(szd, cv, sc); extra += [nt] if nt else []
            else:
                sh, cv = drawn(sc), None; extra.append("beam size not read: links as drawn")
            if lab.get("links_per_set") and lab["links_per_set"] != len(sc["links"]):
                extra.append(f"label says {lab['links_per_set']} per set, section shows {len(sc['links'])}: check")
        else:
            pool = [P for P in labels if own[P["id"]]]
            exact = [P for P in pool if norm_t(P) == norm_t(d)]
            same = exact or [P for P in pool if sig(P) == sig(d)]
            if same and k_lab > 1:                      # same label / pattern as a beam with a section: the precedent
                P = min(same, key=lambda P: (line_of(P.get("beam")) != line_of(d.get("beam")),
                                             abs(szd[1] - own[P["id"]]["depth"]) if szd and own[P["id"]].get("depth") else 1e6,
                                             abs(ctr(P)[1] - ctr(d)[1])))
                scP = own[P["id"]]
                k = lab.get("links_per_set") or len(scP["links"])
                head = (f"{k} link{'s' if k > 1 else ''} per set @ {sp} c/c, pattern of {scP['T']['text']} "
                        f"({'same label' if exact else 'same pattern'} in {P.get('beam')})")
                cv, src = cov_of(scP)
                if szd:
                    sh, nt = set_for(szd, cv, scP); extra += [nt] if nt else []
                else:
                    sh, cv = drawn(scP), None; extra.append("beam size not read: links as drawn")
                d["precedent"] = scP["T"]["id"]
            elif k_lab == 1:                            # generic simple beam: one outer link
                k, head = 1, f"1 outer link per set @ {sp} c/c (generic beam, no section)"
                cv, src = cov_of(None)
                if szd:
                    sh = set_for(szd, cv, None)[0]
                else:
                    extra.append("beam size not read")
            else:                                       # several links per set and nothing showing their shapes
                k, unknown = k_lab, k_lab - 1
                head = f"{k} links per set @ {sp} c/c, no section or precedent on this sheet"
                cv, src = cov_of(None)
                if szd:
                    sh = set_for(szd, cv, None)[0]
                extra.append(f"the other {unknown} link{'s' if unknown > 1 else ''}: shape unknown, check")
        note = head
        cuts = per = None
        if sh:
            st = shape_txt(sh)
            note += ": " + (st[6:] if k == 1 and not unknown else st) + " mm" + (f" ({cov_txt(cv, src)})" if cv else "")
            cuts = [[cut_len(a, b, dia) for b in bs] for (_, a, bs), dia in zip(sh, dias(d, len(sh)))]
            per = None if unknown else [sum(c[min(i, len(c) - 1)] for c in cuts) for i in range(max(len(c) for c in cuts))]
        if extra:
            note += "; " + "; ".join(extra)
        if sh:
            note += f"; cut length ({HOOKS} hooks x {hook_d:g}d): " + cut_txt(sh, cuts, per, unknown)
        d["link_shapes"] = (sh or []) + [("?", None, None)] * (unknown or (0 if sh else k))
        d["links_per_set"] = k
        d["link_cut_mm"], d["set_cut_mm"] = cuts, per
        if cv:
            d["cover_mm"], d["cover_src"] = [round(cv[0], 1), round(cv[1], 1)], src
        zone = d.get("zone_len") or d.get("length_mm")
        if zone:
            n_sets = int(zone // sp) + 1
            d["n_sets"], d["n_links"] = n_sets, n_sets * k
            note += f"; zone {zone:.0f} mm = {n_sets} sets = {n_sets * k} links"
            if per:
                d["total_link_m"] = [round(n_sets * v / 1000, 1) for v in per]
                note += f", {hs_txt(d['total_link_m'])} m"
        d["check"] = note + (f"; {d['check']}" if d.get("check") and "no line" not in d["check"] else "")

    # section labels (6T20 at the top bars, T20-200 SIDE BAR, 6T32 at the bottom) name the same bars as the
    # elevation's labels: the bars whose line crosses the cut (between the two cut marks), or for a section
    # named after its beam, that beam's labels. Same bar spec, matched top to bottom. Counted in the elevation.
    def bar_key(d):                   # '1H10-150' == 'H10-150'; 'T16-200' in a section == 'T16-200 SIDE BAR'
        lab = d.get("label") or {}
        return (tuple((p_["grade"], p_["dia"], p_["count"], p_["spacing"]) for p_ in lab.get("parts", [])),
                lab.get("links_per_set"))

    def dia_key(d):                   # bar size only: '2T10' ~ 'T10-250 SIDE BAR', '2T16-100 DOUBLE' ~ 'T16-100'
        lab = d.get("label") or {}
        return tuple(sorted({(p_["grade"], p_["dia"]) for p_ in lab.get("parts", [])}))

    def y_at(d, x, ylo, yhi):         # where d's bar line crosses the vertical x between ylo and yhi (None: it doesn't)
        hits = []
        for s_ in d.get("line") or []:
            P = np.array(s_, float).reshape(-1, 2)
            for a_, b_ in zip(P[:-1], P[1:]):
                if min(a_[0], b_[0]) - 0.3 * UNIT <= x <= max(a_[0], b_[0]) + 0.3 * UNIT:
                    t_ = 0.5 if abs(b_[0] - a_[0]) < 1e-6 else min(1.0, max(0.0, (x - a_[0]) / (b_[0] - a_[0])))
                    y_ = a_[1] + t_ * (b_[1] - a_[1])
                    if ylo - 0.5 * UNIT <= y_ <= yhi + 0.5 * UNIT:
                        hits.append(y_)
        return float(np.mean(hits)) if hits else None

    elev_labels = [d for d in dets if d["cls"] == "RebarSize" and d.get("repeat_of") is None
                   and not str(d.get("beam", "")).startswith("Sec")]
    for sc in secs:
        T = sc["T"]; sname = re.sub(r"^SECTION\s*", "Sec ", T["text"])
        labs = [d for d in dets if d["cls"] == "RebarSize" and d.get("beam") == sname]
        if not labs:
            continue
        o_ = sc["links"][0] if sc["links"] else None
        def pos_of(d):
            if "SIDE" in str(d.get("suffix") or "").upper():
                return "side"
            if not o_:
                return None
            cy_, h_ = ctr(d)[1], o_["y1"] - o_["y0"]
            return "top" if cy_ < o_["y0"] + 0.3 * h_ else "bottom" if cy_ > o_["y1"] - 0.3 * h_ else "side"
        mks = [m_ for m_ in sc["marks"] if m_.get("beam") == sc.get("beam")] or sc["marks"]
        if mks:                       # the cut: a vertical line between the cut marks
            cut_x = float(np.mean([(m_["bbox"][0] + m_["bbox"][2]) / 2 for m_ in mks]))
            ylo, yhi = min(m_["bbox"][1] for m_ in mks), max(m_["bbox"][3] for m_ in mks)
            def at_cut(e):            # links: Ramboll draws the link zone as a line below the beam
                if (e.get("label") or {}).get("kind") == "link":
                    return y_at(e, cut_x, ylo - 2 * UNIT, yhi + 10 * UNIT)
                return y_at(e, cut_x, ylo, yhi)
            cands = [(e, at_cut(e)) for e in elev_labels]
            cut_beams = {m_.get("beam") for m_ in mks} - {None}
            cands = [(e, y_) for e, y_ in cands if y_ is not None] + \
                    [(e, (ylo + yhi) / 2) for e, y_ in cands if y_ is None and not e.get("line")      # side bars drawn
                     and "SIDE" in str(e.get("suffix") or "").upper() and e.get("beam") in cut_beams]  # as dots: no line
            where = f"crossing the cut in the {sc.get('beam') or '?'} elevation"
        else:                         # a section named after its beam: that beam's labels
            cands = [(e, ctr(e)[1] if not e.get("line") else float(np.vstack([np.array(s_).reshape(-1, 2) for s_ in e["line"]])[:, 1].mean()))
                     for e in elev_labels if e.get("beam") == sc.get("beam")]
            where = f"in the {sc.get('beam') or '?'} elevation"
        used = set()
        by_key = defaultdict(list)
        for d in labs:
            by_key[bar_key(d)].append(d)
        is_lk = lambda d: (d.get("label") or {}).get("kind") == "link"
        def side_ok(d, e, y_):        # the elevation bar is on the section label's side of the beam
            pos = pos_of(d)
            e_side = "SIDE" in str(e.get("suffix") or "").upper()
            if is_lk(d) or pos is None or not mks:
                return True
            if pos == "side":         # side bars match side bars (or a bar at mid-depth), never a top / bottom bar
                return e_side or abs(y_ - (ylo + yhi) / 2) < 0.2 * (yhi - ylo)
            return not e_side and (y_ < (ylo + yhi) / 2) == (pos == "top")
        loose = []                    # section labels with no bar of the same spec at the cut: matched by bar size
        for key, group in by_key.items():
            group.sort(key=lambda d: ctr(d)[1])                               # top to bottom in the section
            cs = sorted([(e, y_) for e, y_ in cands if bar_key(e) == key], key=lambda c: c[1])
            if not cs:
                loose += group
                continue
            for rank, d in enumerate(group):
                pos, free = pos_of(d), [c for c in cs if c[0]["id"] not in used]
                if not free:
                    e = None
                elif len(cs) == len(group):
                    e = cs[rank][0] if cs[rank][0]["id"] not in used else free[0][0]
                elif pos == "top":
                    e = free[0][0]
                elif pos == "bottom":
                    e = free[-1][0]
                else:
                    e = free[len(free) // 2][0]
                is_link = (d.get("label") or {}).get("kind") == "link"
                d["sec_pos"] = None if is_link else pos
                bars = "links" if is_link else f"{pos} bars" if pos else "bars"
                if e is not None:
                    used.add(e["id"])
                    d["same_as"] = e["id"]
                    d["check"] = f"{bars} in {T['text']}: the same bars as {e['text']} in the {e.get('beam', '?')} elevation, counted there (not here)"
                    e.setdefault("sec_refs", []).append(T["text"])
                    if pos and not is_link:
                        e["bar_pos"] = pos
                    e["check"] = f"{bars}, as labelled in {T['text']}" + (f"; {e['check']}" if e.get("check") else "")
                elif re.search(r"E/?W|E/?F|SLAB", str(d.get("suffix") or ""), re.I):
                    d["check"] = f"slab bars shown in {T['text']} (refer to the slab details): not part of the beam"
                else:
                    d["check"] = (f"{bars} in {T['text']}: no {d['text']} {where}, so these {'links' if is_link else 'bars'} "
                                  f"are only labelled here: count them from the section (check)")
        # no bar of the same spec at the cut: the bar of the same size there, on the same side, is taken as the
        # same bars (the section and elevation word it differently: '2T10' / 'T10-250 SIDE BAR') and flagged;
        # a bar of another size there is a possible drawing error (flagged, not linked)
        loose.sort(key=lambda d: (not any(dia_key(e) == dia_key(d) and dia_key(d) and is_lk(e) == is_lk(d)
                                              for e, _ in cands), ctr(d)[1]))     # same-size matches first
        for d in loose:
            is_link = is_lk(d)
            pos = pos_of(d)
            d["sec_pos"] = None if is_link else pos
            bars = "links" if is_link else f"{pos} bars" if pos else "bars"
            near = [(e, y_) for e, y_ in cands if e["id"] not in used and is_lk(e) == is_link and side_ok(d, e, y_)]
            same_size = [(e, y_) for e, y_ in near if dia_key(e) == dia_key(d) and dia_key(d)]
            # another label of this section already took the bar at the cut: these are extra bars / inner links
            # labelled only in the section (not an error)
            taken = any(e["id"] in used and is_lk(e) == is_link and side_ok(d, e, y_) for e, y_ in cands)
            if same_size:
                e = (min if pos == "top" else max)(same_size, key=lambda c: c[1])[0] if pos in ("top", "bottom") else same_size[0][0]
                used.add(e["id"])
                d["same_as"] = e["id"]
                d["discrepancy"] = f"{T['text']} says {d['text']}, the elevation says {e['text']}"
                d["check"] = (f"DISCREPANCY (check): {bars} in {T['text']} are labelled {d['text']}, the bar at the cut in the "
                              f"{e.get('beam', '?')} elevation {e['text']}: taken as the same bars, counted there (not here)")
                e.setdefault("sec_refs", []).append(T["text"])
                e["check"] = (f"DISCREPANCY (check): labelled {d['text']} in {T['text']}"
                              + (f"; {e['check']}" if e.get("check") else ""))
            elif near and mks and not taken and is_link:   # spacing bars: often wall / slab steel in the section
                d["check"] = (f"{d['text']} in {T['text']} matches nothing at the cut in the {sc.get('beam') or '?'} elevation "
                              f"(there: {', '.join(sorted({e['text'] for e, _ in near}))}): wall / slab bars or links "
                              f"labelled only here? (check)")
            elif near and mks and not taken:   # (a section named after its beam has no cut to compare against)
                other = sorted({e["text"] for e, _ in near})
                d["discrepancy"] = f"{T['text']} says {d['text']}, the elevation has {', '.join(other)} at the cut"
                d["check"] = (f"POSSIBLE DRAWING ERROR (check): {bars} in {T['text']} are {d['text']}, but the "
                              f"{sc.get('beam') or '?'} elevation has {', '.join(other)} at the cut: not linked, "
                              f"count them from the section until confirmed")
            elif re.search(r"E/?W|E/?F|SLAB", str(d.get("suffix") or ""), re.I):
                d["check"] = f"slab bars shown in {T['text']} (refer to the slab details): not part of the beam"
            else:
                d["check"] = (f"{bars} in {T['text']}: no {d['text']} {where}, so these {'links' if is_link else 'bars'} "
                              f"are only labelled here: count them from the section (check)")
    for d in dets:
        if d.get("laps_mm"):
            note = "lap " + ", ".join(f"{v} mm" for v in d["laps_mm"])
            d["check"] = note + (f"; {d['check']}" if d.get("check") else "")
    for sc in secs:
        T = sc["T"]
        if sc["links"]:
            T["check"] = ("closed links as drawn: " + ", ".join(f"{k} {w}x{h}" for k, w, h in T["link_dims"]) + f" mm ({sc['scale']})"
                          + (f"; {cov_txt(sc['cover'], 'concrete face to link' if sc['cover_src'].startswith('measured') else 'from the beam size')}"
                             if sc.get("cover") else "") + (f"; {sc['size_note']}" if sc.get("size_note") else ""))
        for mk in sc["marks"]:
            mk["check"] = f"cut of {T['text']} in {mk.get('beam', '?')}"
    for d in dets:
        if d.get("repeat_note"):
            d["check"] = d["repeat_note"] + (f"; {d['check']}" if d.get("check") and d["check"] not in ("on scale", "zone ok") else "")
    bad = [d for d in dets if "use printed" in d.get("check", "")]
    log(f"link zones: {sum(d.get('check') == 'zone ok' for d in dets)} agree with printed length, {len(bad)} drawn off-scale")

    for t in title_block(page, PDF_PATH.stem):     # sheet-level: drawing number and title (no relations)
        dets.append(dict(id=len(dets), **t))
        log(f"{t['cls']}: {t['text']}  ({t['source']})")

    return dict(pdf=str(PDF_PATH), page=PAGE_NO, page_w=PAGE_W, page_h=PAGE_H, unit=UNIT,
                sheet_scale=SHEET_SCALE / PT_MM, dets=dets, continued=CONT_INFO)


def write_outputs(res, stem=None):
    """CSV + JSON + overlay PNG next to the PDF."""
    dets, PDF_PATH, PAGE_NO = res["dets"], Path(res["pdf"]), res["page"]
    PAGE_W, PAGE_H, UNIT = res["page_w"], res["page_h"], res["unit"]
    page = pymupdf.open(PDF_PATH)[PAGE_NO]
    if page.rotation:
        page.remove_rotation()
    titles = [d for d in dets if d["cls"] == "BeamName"]
    stem = stem or PDF_PATH.with_name(f"{PDF_PATH.stem}_p{PAGE_NO}")
    with open(f"{stem}_detections.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["beam", "class", "text", "kind", "sets", "count", "grade", "dia", "spacing", "face",
                    "line_len_pt", "length_mm", "bends_squared", "zone_len_printed", "repeat_of",
                    "links_per_set", "n_sets", "link_cut_mm", "cut_per_set_mm", "total_link_m", "check", "source", "x0", "y0", "x1", "y1"])
        for d in sorted(dets, key=lambda d: (d.get("beam", ""), d["cls"], d["bbox"][1], d["bbox"][0])):
            lab = d.get("label") or {}
            ps = lab.get("parts", [{}])
            w.writerow([d.get("beam", d["text"] if d["cls"] == "BeamName" else ""), d["cls"], d["text"], lab.get("kind", ""),
                        "+".join(str(p.get("sets", "")) for p in ps) if lab else "",
                        "+".join(str(p.get("count", "")) for p in ps) if lab else "",
                        ps[0].get("grade", ""), ps[0].get("dia", ""),
                        "+".join(str(p.get("spacing") or "") for p in ps) if lab else "",
                        lab.get("face") or "", round(d.get("line_len", d.get("dimline_len", 0)), 1) or "",
                        d.get("length_mm", ""), d.get("bends_squared", ""), d.get("zone_len", ""),
                        "" if d.get("repeat_of") is None else d["repeat_of"],
                        d.get("links_per_set", ""), d.get("n_sets", ""),
                        " + ".join("..".join(map(str, c)) for c in d["link_cut_mm"]) if d.get("link_cut_mm") else "",
                        "..".join(map(str, d["set_cut_mm"])) if d.get("set_cut_mm") else "",
                        "..".join(map(str, d["total_link_m"])) if d.get("total_link_m") else "",
                        d.get("check", ""), d["source"], *[round(v, 1) for v in d["bbox"]]])
    json.dump(dict(pdf=str(PDF_PATH), page=PAGE_NO, unit=UNIT, sheet_scale=res["sheet_scale"], detections=dets),
              open(f"{stem}_detections.json", "w"), default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))

    # --- 8. overlay -----------------------------------------------------------------------
    COL = {"BeamName": "#8e44ad", "BeamSize": "#16a085", "RebarSize": "#e74c3c", "Dim": "#2471a3",
           "ZoneLen": "#5dade2", "RebarLine": "#f39c12", "DimLine": "#00c3ff", "SectionTitle": "#7f8c8d",
           "DrawingNo": "#5d6d7e", "DrawingTitle": "#a04000"}
    def overlay(clip, path, figw=30):
        x0, y0, x1, y1 = clip
        g = page.get_pixmap(dpi=110, colorspace=pymupdf.csGRAY)
        img = np.frombuffer(g.samples, np.uint8).reshape(g.height, g.width, g.n)[..., 0]
        fig, ax = plt.subplots(figsize=(figw, figw * (y1 - y0) / (x1 - x0)))
        ax.imshow(img, cmap="gray", extent=(0, PAGE_W, PAGE_H, 0), alpha=0.3)
        for d in dets:
            a, b, c_, e = d["bbox"]
            rep = d.get("repeat_of") is not None
            ax.add_patch(Rectangle((a, b), c_ - a, e - b, fill=rep, fc="#bbbbbb", hatch="///" if rep else None,
                                   ec="#777777" if rep else COL.get(d["cls"], "#555555"), lw=1.6))
            cx, cy = (a + c_) / 2, (b + e) / 2
            if rep:                                         # drawn twice: link to the copy that is counted
                o = dets[d["repeat_of"]]["bbox"]
                ax.plot([cx, (o[0] + o[2]) / 2], [cy, (o[1] + o[3]) / 2], color="#777777", lw=1.2, ls="--")
            if d.get("line"):
                for s in d["line"]:
                    ax.add_collection(LineCollection(np.array(s).reshape(-1, 2, 2), colors="#999999" if rep else COL["RebarLine"],
                                                     linewidths=3, alpha=0.75))
                if d.get("contact"):
                    q = d["contact"]
                else:
                    P = np.vstack([np.array(s).reshape(-1, 2) for s in d["line"]])
                    q = P[np.argmin(np.hypot(*(P - (cx, cy)).T))]
                ax.plot([cx, q[0]], [cy, q[1]], color="green", lw=0.9)
            if d.get("dimline"):
                s = np.array(d["dimline"])
                ax.plot(s[:, 0], s[:, 1], color=COL["DimLine"] if d.get("scale_ok") else "magenta", lw=2.5, alpha=0.8)
            if d["cls"] == "RebarSize" and not d.get("line"):
                ax.add_patch(Rectangle((a - 2, b - 2), c_ - a + 4, e - b + 4, fill=False, ec="black", lw=2, ls=":"))
        for t in [t for t in titles if "rel" in t]:
            bs = dets[t["rel"]]
            ax.plot([t["bbox"][2], bs["bbox"][0]], [(t["bbox"][1] + t["bbox"][3]) / 2] * 2, color="green", lw=2)
        ax.set_xlim(x0, x1); ax.set_ylim(y1, y0); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.legend(handles=[Patch(color=c, label=k) for k, c in COL.items()] +
                  [Patch(color="magenta", label="DimLine (scale mismatch)"), Patch(color="green", label="relation"),
                   Patch(fill=False, ls=":", ec="black", label="RebarSize, no line found"),
                   Patch(fc="#bbbbbb", hatch="///", ec="#777777", label="repeat (drawn twice, not counted)")],
                  loc="upper left", bbox_to_anchor=(1, 1), fontsize=10)
        fig.savefig(path, dpi=110, bbox_inches="tight"); plt.close(fig)

    allbb = np.array([d["bbox"] for d in dets])
    overlay((allbb[:, 0].min() - 20, allbb[:, 1].min() - 20, allbb[:, 2].max() + 20, allbb[:, 3].max() + 20), f"{stem}_overlay.png")
    print("saved", f"{stem}_detections.csv / .json / _overlay.png")

    return stem


if __name__ == "__main__":
    argv = sys.argv[1:]
    opt = lambda k, f=str: f(argv[argv.index(k) + 1]) if k in argv else None
    why, cover_mm, hook_d = opt("--why"), opt("--cover", float), opt("--hook-d", float)
    args = [a for k, a in enumerate(argv) if not a.startswith("--") and not (k and argv[k - 1] in ("--why", "--cover", "--hook-d"))]
    res = detect(args[0] if args else "drawing.pdf", int(args[1]) if len(args) > 1 else 0,
                 square_bends="--round-bends" not in argv, force_ocr="--ocr" in argv,
                 cover=cover_mm, hook_d=12 if hook_d is None else hook_d)
    if why:                     # e.g. --why 6T20 : how each 6T20 label's bar line was traced
        hits = [d for d in res["dets"] if d["cls"] == "RebarSize" and d["text"].replace(" ", "") == why.replace(" ", "")]
        print(f"\n--why {why}: {len(hits)} label(s)")
        for d in hits:
            print(f"\n{d['text']} at {[round(v) for v in d['bbox']]} (beam {d.get('beam', '?')}), "
                  f"length {d.get('length_mm', '-')} mm, bends squared {d.get('bends_squared', 0)}")
            if not d.get("line"):
                lp = d.get("leader") or []
                print("   no bar line: the leader/arrow could not be followed to a rebar-width line"
                      + (f" (searched {len(lp)} pieces; a section label pointing at bar dots?)" if len(lp) > 12
                         else f" (leader pieces {lp})"))
            for line in d.get("why", []):
                print("  ", line)
    write_outputs(res)
