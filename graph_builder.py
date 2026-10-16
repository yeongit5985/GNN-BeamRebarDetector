"""
Graph builder: the steps of 01_pdf_labelstudio_to_graph.ipynb as functions. It is generated from the same
code as the notebook, so both build the same graph. Used by the backend's training (review page > Train,
Label Studio > Start Training) and by 02_train_gnn.ipynb.

    from graph_builder import build_graph, save_graph, export_to_graphs
    g = build_graph("drawing.pdf", 0, task)       # task: one task of a Label Studio JSON export (with an annotation)
    save_graph(g, "graphs")                        # graphs/drawing_p0.npz + .json, as the notebook saves them
    export_to_graphs("project-5.json", "D:/drawings", "graphs")   # every annotated task of an export

Needs: pip install pypdfium2 scipy matplotlib  (+ rapidocr_onnxruntime for outlined texts); PDF access via pdf_backend.py
"""
import builtins
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np
from matplotlib.path import Path as MplPath
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from pdf_backend import pymupdf      # pypdfium2 behind PyMuPDF's API (BEAM_PDF=pymupdf: PyMuPDF itself)

# ---- settings (the notebook's config cell) ----
# Your Label Studio label names -> classes used here (add any spelling you used)
LABEL_MAP = {
    "BeamID": "BeamName", "BeamName": "BeamName", "BeamDenotation": "BeamName",
    "BeamSize": "BeamSize",
    "RebarSize": "RebarSize",
    "Dimension": "Dim", "Dim": "Dim",
    "RebarLine": "RebarLine", "RebarCoordinate": "RebarLine",
    "DimensionLine": "DimLine", "DimLine": "DimLine",
    "SectionMark": "SectionMark", "SectionCut": "SectionMark", "CutMark": "SectionMark",
    "SectionTitle": "SectionTitle", "Section": "SectionTitle",
}
CLASSES      = ["Other", "BeamName", "BeamSize", "RebarSize", "Dim", "RebarLine", "DimLine", "SectionMark", "SectionTitle"]
LAP_LABELS   = {"Lap", "LapLine", "LapLength"}   # lap lines (stub to stub): kept as lap records, not as bar lines
# sheet-level boxes: kept as sheet attributes (drawing number / title), not as graph nodes
SHEET_LABELS = {"DrawingNo": "drawing_no", "DrawingNumber": "drawing_no", "DwgNo": "drawing_no", "DrawingRef": "drawing_no",
                "DrawingTitle": "drawing_title", "DwgTitle": "drawing_title"}
TEXT_CLASSES = {"BeamName", "BeamSize", "RebarSize", "Dim", "SectionMark", "SectionTitle"}
# a box over a CANCELLED stamp / revision cloud, linked to each beam it covers (Cancelled -> BeamID): those beams
# stay in the graph as training data, flagged 'cancelled' so the takeoff leaves them out. Not a graph node.
CANCEL_LABELS = {"Cancelled", "Canceled", "Superseded"}

# Tolerances in multiples of UNIT (= median text height on the page, set below),
# so the same numbers work for A1 and A3 plots.
ENDPOINT_TOL   = 0.05  # segment ends closer than this join into one polyline node
TOUCH_TOL      = 0.15  # a line end this close to another line = "touch" edge
NEAR_PAD       = 0.8   # text within this of a line = "near" edge
MARK_MAX_DIAG  = 1.5   # small closed/filled shapes (arrowheads, bar dots) become 'mark' nodes
LINE_MATCH_TOL = 0.5   # how far your hand-drawn LS vector may sit from the real PDF line
LINE_MATCH_MIN = 0.6   # share of a line node that must be covered by the LS vector
CAND_RADIUS    = 6.0   # text->line candidate pairs: lines within this radius ...
CAND_HOPS      = 4     # ... or reachable within this many touch/near hops
TEXT_KNN       = 6     # nearest texts linked to each text
BEAM_RADIUS    = 150   # beam membership: title-like text <-> rebar-like text pairs within this radius
VIEW_GAP       = 1.0   # line work closer than this = same view (elevation / section) for the scale check


def build_graph(pdf_path, page_no, task, img_clip=None, ocr_fallback=True, verbose=False):
    """One annotated Label Studio task + its PDF page -> dict(arrays=..., meta=..., stats=...).
    arrays/meta are what the notebook saves as .npz/.json; page_no is 0-based."""
    print = builtins.print if verbose else (lambda *a, **k: None)
    PDF_PATH, PAGE_NO, IMG_CLIP, OCR_FALLBACK = Path(pdf_path), int(page_no), img_clip, ocr_fallback
    doc = pymupdf.open(PDF_PATH)
    page = doc[PAGE_NO]
    if page.rotation:
        print(f"page /Rotate {page.rotation} removed (in memory) so coordinates match the rendered image")
        page.remove_rotation()
    PAGE_W, PAGE_H = page.rect.width, page.rect.height
    print(f"{PDF_PATH.name} p{PAGE_NO}: {PAGE_W:.0f} x {PAGE_H:.0f} pt = {PAGE_W/72*25.4:.0f} x {PAGE_H/72*25.4:.0f} mm")

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
    # CAD exports often print the same text twice on the same spot: keep one (as the ML backend does)
    _seen, _uniq = set(), []
    for w in words:
        key = (w["text"], *(round(v) for v in w["bbox"]))
        if key not in _seen:
            _seen.add(key); _uniq.append(w)
    words = _uniq
    UNIT = float(np.median([w["size"] for w in words])) if words else 7.0
    print(f"{len(words)} words, UNIT (median text size) = {UNIT:.2f} pt")
    if not words:
        print("WARNING: no text layer. Text was probably exported as SHX strokes -> needs the OCR fallback.")
    print([w["text"] for w in words[:25]])

    def bezier(p0, p1, p2, p3, n=8):
        t = np.linspace(0, 1, n + 1)[:, None]
        p0, p1, p2, p3 = (np.array(p, float) for p in (p0, p1, p2, p3))
        return (1 - t)**3 * p0 + 3 * (1 - t)**2 * t * p1 + 3 * (1 - t) * t**2 * p2 + t**3 * p3

    def path_segments(path):
        # One get_drawings() path -> straight segments (curves flattened), shape (n, 2, 2)
        segs, curved = [], False
        for it in path["items"]:
            op = it[0]
            if op == "l":
                segs.append([tuple(it[1]), tuple(it[2])])
            elif op == "c":
                pts = bezier(*it[1:5]); curved = True
                segs += [[tuple(pts[i]), tuple(pts[i + 1])] for i in range(len(pts) - 1)]
            elif op in ("re", "qu"):
                q = it[1].quad if op == "re" else it[1]
                c = [tuple(q.ul), tuple(q.ur), tuple(q.lr), tuple(q.ll)]
                segs += [[c[i], c[(i + 1) % 4]] for i in range(4)]
        if path.get("closePath") and len(segs) > 1 and math.dist(segs[-1][1], segs[0][0]) > 1e-6:
            segs.append([segs[-1][1], segs[0][0]])
        return np.array(segs, float).reshape(-1, 2, 2), curved

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
        segs, curved = path_segments(path)
        if len(segs) == 0:
            continue
        filled = path.get("fill") is not None
        dashed = bool(re.search(r"\[\s*\d", path.get("dashes") or ""))
        for idx in split_components(segs, ENDPOINT_TOL * UNIT):
            s = segs[idx]
            g = geom_props(s)
            diag = math.hypot(g["bbox"][2] - g["bbox"][0], g["bbox"][3] - g["bbox"][1])
            kind = "mark" if diag <= MARK_MAX_DIAG * UNIT and (filled or g["closed"]) else "line"
            nodes.append(dict(kind=kind, segs=s, curved=curved, filled=filled, dashed=dashed,
                              width=path.get("width") or 0.0, color=rgb(path.get("color") or path.get("fill")),
                              layer=path.get("layer") or "", **g))

    # Split lines where an arrowhead tip lands mid-line: link zones along one long line, bars hit by a
    # leader arrow. Each zone/piece becomes its own node (the information also exists at inference time).
    def arrow_tip(n):
        v = np.unique(np.round(n["segs"].reshape(-1, 2), 2), axis=0)
        if n["curved"] or len(v) != 3:
            return None
        side = [math.dist(v[(k + 1) % 3], v[(k + 2) % 3]) for k in range(3)]
        return v[int(np.argmin(side))]

    def cut_line(segs, cuts, tol):
        out = []
        for a, b in segs:
            ab = b - a; L2 = ab @ ab
            ts = sorted(t for c in cuts if L2 > 0 for t in [((c - a) @ ab) / L2]
                        if 1e-3 < t < 1 - 1e-3 and np.hypot(*(a + t * ab - c)) <= tol)
            pts = [a] + [a + t * ab for t in ts] + [b]
            out += [[pts[k], pts[k + 1]] for k in range(len(pts) - 1)]
        out = np.array(out)
        ends = out.reshape(-1, 2)
        pairs = cKDTree(ends).query_pairs(ENDPOINT_TOL * UNIT, output_type="ndarray")
        at_cut = cKDTree(np.array(cuts)).query(ends)[0] <= tol
        pairs = pairs[~(at_cut[pairs[:, 0]] & at_cut[pairs[:, 1]])]          # don't re-join at a cut
        g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0] // 2, pairs[:, 1] // 2)), shape=(len(out), len(out)))
        lab = connected_components(g, directed=False)[1]
        return [out[lab == k] for k in range(lab.max() + 1)]

    tips = [t for n in nodes if n["kind"] == "mark" for t in [arrow_tip(n)] if t is not None]
    if tips:
        tip_tree, tol = cKDTree(np.array(tips)), TOUCH_TOL * UNIT
        split_nodes, n_split = [], 0
        for n in nodes:
            if n["kind"] == "line" and n["length"] > 2 * UNIT:
                x0, y0, x1, y1 = n["bbox"]
                near = [tips[k] for k in tip_tree.query_ball_point(((x0 + x1) / 2, (y0 + y1) / 2),
                                                                   math.hypot(x1 - x0, y1 - y0) / 2 + tol)]
                ends = n["segs"].reshape(-1, 2)
                inner = [t for t in near if np.hypot(*(ends - t).T).min() > 0.5 * UNIT]   # not at the line's ends
                if inner:
                    pieces = cut_line(n["segs"], inner, tol)
                    if len(pieces) > 1:
                        n_split += 1
                        for pc in pieces:
                            split_nodes.append({**n, "segs": pc, **geom_props(pc)})
                        continue
            split_nodes.append(n)
        nodes = split_nodes
        print(f"split {n_split} lines at arrowhead tips")

    for n in nodes:
        n["bbox"] = [float(v) for v in n["bbox"]]
        n["c"] = ((n["bbox"][0] + n["bbox"][2]) / 2, (n["bbox"][1] + n["bbox"][3]) / 2)

    kinds = np.array([n["kind"] for n in nodes])
    text_ids = np.where(kinds == "text")[0]
    geo_ids = np.where(kinds != "text")[0]
    print({k: int((kinds == k).sum()) for k in ("text", "line", "mark")})
    layers = sorted({n["layer"] for n in nodes if n.get("layer")})
    print("PDF layers:", layers[:20] if layers else "none")

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

    print(f"{len(S_xy):,} sample points, step {STEP:.2f} pt")

    latest = sorted(task.get("annotations", []), key=lambda a: a.get("updated_at") or a.get("created_at") or "")
    page_type = str((task.get("meta") or {}).get("page_type", "")).lower()
    if page_type == "not a beam drawing" or (page_type != "beam drawing" and latest and latest[-1].get("was_cancelled")):
        raise ValueError(f"task {task.get('id')} is not a beam drawing (ticked in a review page, or skipped in Label Studio): "
                         "no graph for it - pick another page")
    anns = [a for a in task.get("annotations", []) if not a.get("was_cancelled")]
    if not anns:
        raise ValueError(f"task {task.get('id')} has no annotation: submit it in Label Studio, or save it from the review page")
    result = json.loads(json.dumps(anns[-1]["result"]))   # latest annotation of this task (a copy: fixed below)

    # Label Studio (1.2x) can save pre-annotated lines you didn't touch shrunk into the top-left corner (points
    # divided by the image size: x% / width, or x% * 100 / width; boxes untouched). Put them back with the factor
    # that lays them over the same part of the drawing as the boxes (the review page does the same).
    def unshrink_vectors(result):
        vec = [r for r in result if r.get("type") == "vectorlabels" and (r.get("value") or {}).get("vertices")
               and (r.get("original_width") or 0) > 200 and (r.get("original_height") or 0) > 200]
        small = [r for r in vec if max(p.get("x", 0) for p in r["value"]["vertices"]) <= 1e4 / r["original_width"] * 1.001
                 and max(p.get("y", 0) for p in r["value"]["vertices"]) <= 1e4 / r["original_height"] * 1.001]
        if not small:
            return 0
        W_, H_ = small[0]["original_width"], small[0]["original_height"]
        bx = [r["value"] for r in result if r.get("type") == "rectanglelabels" and "x" in (r.get("value") or {})]
        ref = (min(v["x"] for v in bx), min(v["y"] for v in bx), max(v["x"] + v["width"] for v in bx),
               max(v["y"] + v["height"] for v in bx)) if bx else (0.0, 0.0, 100.0, 100.0)
        pts = [p for r in small for p in r["value"]["vertices"]]
        def overlap(kx, ky):
            x0, x1 = min(p["x"] for p in pts) * kx, max(p["x"] for p in pts) * kx
            y0, y1 = min(p["y"] for p in pts) * ky, max(p["y"] for p in pts) * ky
            if x1 > 100.5 or y1 > 100.5:
                return -1.0
            iw, ih = max(0.0, min(x1, ref[2]) - max(x0, ref[0])), max(0.0, min(y1, ref[3]) - max(y0, ref[1]))
            union = (x1 - x0) * (y1 - y0) + (ref[2] - ref[0]) * (ref[3] - ref[1]) - iw * ih
            return iw * ih / union if union > 0 else 0.0
        score, kx, ky = max((overlap(kx, ky), kx, ky) for kx, ky in ((W_ / 100, H_ / 100), (W_, H_), ((W_ / 100) ** 2, (H_ / 100) ** 2)))
        if score < 0.05:
            return 0
        for r in small:
            for p in r["value"]["vertices"]:
                p["x"], p["y"] = p["x"] * kx, p["y"] * ky
        return len(small)

    # Regions written by another tool can lack the image size (original_width / original_height): give them the
    # size the other regions carry, or, if none has one, a size in the PDF page's proportions (percent coordinates
    # only need the ratio).
    sized = next((r for r in result if r.get("original_width") and r.get("original_height")), None)
    _cw, _ch = (IMG_CLIP[2] - IMG_CLIP[0], IMG_CLIP[3] - IMG_CLIP[1]) if IMG_CLIP else (PAGE_W, PAGE_H)
    W_fill, H_fill = (sized["original_width"], sized["original_height"]) if sized else (round(_cw * 4), round(_ch * 4))
    nosize = [r for r in result if r.get("type") in ("rectanglelabels", "vectorlabels", "polygonlabels") and not r.get("original_width")]
    for r in nosize:
        r["original_width"], r["original_height"] = W_fill, H_fill
    if nosize:
        print(f"{len(nosize)} regions had no image size: given {W_fill} x {H_fill}",
              "(as the other regions)" if sized else "(the PDF page's proportions)")

    n_unshrunk = unshrink_vectors(result)
    if n_unshrunk:
        print(f"{n_unshrunk} lines were saved by Label Studio shrunk into the top-left corner - put back in place")
    print(f"task {task.get('id')}: {task.get('data', {}).get('image', '')}  ({len(result)} result items)")

    # image pixels -> PDF points
    first = next(r for r in result if "original_width" in r)
    IMG_W, IMG_H = first["original_width"], first["original_height"]
    cx0, cy0, cx1, cy1 = IMG_CLIP or (0, 0, PAGE_W, PAGE_H)
    SX, SY = (cx1 - cx0) / IMG_W, (cy1 - cy0) / IMG_H
    print(f"image {IMG_W} x {IMG_H} px -> {SX:.4f} / {SY:.4f} pt per px (~{72 / SX:.0f} DPI)")
    if abs(SX / SY - 1) > 0.01:
        print("WARNING: x and y scales differ -> wrong page, rotated image, or IMG_CLIP needed")

    def px2pt(xy):
        xy = np.asarray(xy, float)
        return np.column_stack([cx0 + xy[:, 0] * SX, cy0 + xy[:, 1] * SY])

    # Vector vertices: pixels or percent? (pixels if any vertex is beyond 100)
    vec_xy = [(v["x"], v["y"]) for r in result if r.get("type") == "vectorlabels" for v in r["value"].get("vertices", [])]
    VECTOR_IN_PERCENT = bool(vec_xy) and max(max(p) for p in vec_xy) <= 100.5

    def ls_vertices(v, W, H):
        xy = np.array([[p["x"], p["y"]] for p in v["vertices"]], float)
        return xy / 100 * [W, H] if VECTOR_IN_PERCENT else xy
    print("vector vertices in", "percent" if VECTOR_IN_PERCENT else "pixels")

    regions, unknown, laps, sheet, cancels = {}, set(), [], {}, {}
    for r in result:
        t, v = r.get("type"), r.get("value", {})
        if t not in ("rectanglelabels", "vectorlabels", "polygonlabels"):
            continue
        raw = (v.get(t) or ["?"])[0]
        if raw in LAP_LABELS and t == "vectorlabels":
            W, H = r.get("original_width", IMG_W), r.get("original_height", IMG_H)
            xy = px2pt(ls_vertices(v, W, H))
            laps.append(dict(id=r["id"], pts=xy.round(2).tolist(), length_pt=round(float(np.hypot(*np.diff(xy, axis=0).T).sum()), 2),
                             note=(r.get("meta") or {}).get("text", [""])[0]))
            continue
        if raw in SHEET_LABELS and t == "rectanglelabels":
            W, H = r.get("original_width", IMG_W), r.get("original_height", IMG_H)
            b = px2pt(np.array([[v["x"], v["y"]], [v["x"] + v["width"], v["y"] + v["height"]]]) / 100 * [W, H])
            txt = page.get_textbox(pymupdf.Rect(*b[0], *b[1]))
            sheet[SHEET_LABELS[raw]] = " ".join(txt.split())
            continue
        if raw in CANCEL_LABELS and t == "rectanglelabels":
            W, H = r.get("original_width", IMG_W), r.get("original_height", IMG_H)
            b = px2pt(np.array([[v["x"], v["y"]], [v["x"] + v["width"], v["y"] + v["height"]]]) / 100 * [W, H])
            txt = page.get_textbox(pymupdf.Rect(*b[0], *b[1]))
            cancels[r["id"]] = dict(id=r["id"], bbox=b.ravel().round(2).tolist(), text=" ".join(txt.split()), beams=[])
            continue
        cls = LABEL_MAP.get(raw)
        if cls is None:
            unknown.add(raw); continue
        if r.get("image_rotation"):
            print("WARNING: region on a rotated image:", r["id"])
        W, H = r.get("original_width", IMG_W), r.get("original_height", IMG_H)
        if t == "rectanglelabels":
            x, y, w, h = v["x"] / 100 * W, v["y"] / 100 * H, v["width"] / 100 * W, v["height"] / 100 * H
            a = math.radians(v.get("rotation") or 0)       # LS rotates about the top-left corner
            R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
            xy = np.array([[0, 0], [w, 0], [w, h], [0, h]]) @ R.T + [x, y]
            closed = True
            if cls not in TEXT_CLASSES:                     # a box drawn around a line: use its long centre line
                mid = lambda p, q: (xy[p] + xy[q]) / 2
                xy = np.array([mid(0, 3), mid(1, 2)]) if w >= h else np.array([mid(0, 1), mid(3, 2)])
                closed = False
        elif t == "polygonlabels":
            xy, closed = np.array(v["points"], float) / 100 * [W, H], True
        else:
            xy = ls_vertices(v, W, H)
            closed = bool(v.get("closed"))
        regions[r["id"]] = dict(cls=cls, shape="box" if cls in TEXT_CLASSES else "line", pts=px2pt(xy), closed=closed)

    rels = [(r["from_id"], r["to_id"]) for r in result
            if r.get("type") == "relation" and r.get("from_id") in regions and r.get("to_id") in regions]

    # Cancelled (superseded) beams: Cancelled -> BeamID. Everything of such a beam (size, rebar labels and their
    # lines, cut marks, its sections) gets the flag; the walk never passes through another beam.
    for r in result:
        if r.get("type") == "relation":
            for c, b in ((r.get("from_id"), r.get("to_id")), (r.get("to_id"), r.get("from_id"))):
                if c in cancels and b in regions and regions[b]["cls"] == "BeamName" and b not in cancels[c]["beams"]:
                    cancels[c]["beams"].append(b)
    CANCEL_FOLLOW = {"BeamName": {"BeamSize", "RebarSize", "SectionMark", "SectionTitle"}, "SectionMark": {"SectionTitle"},
                     "SectionTitle": {"RebarSize"}, "RebarSize": {"RebarLine"}}
    rel_nb = defaultdict(set)
    for a, b in rels:
        rel_nb[a].add(b); rel_nb[b].add(a)
    cancelled_ents = {b for c in cancels.values() for b in c["beams"]}
    todo = list(cancelled_ents)
    while todo:
        a = todo.pop()
        for b in rel_nb[a]:
            if b not in cancelled_ents and regions[b]["cls"] in CANCEL_FOLLOW.get(regions[a]["cls"], ()):
                cancelled_ents.add(b); todo.append(b)

    if unknown:
        print("WARNING: labels not in LABEL_MAP (skipped):", unknown)
    if cancels:
        print(f"cancelled: {len(cancels)} Cancelled boxes -> {sum(len(c['beams']) for c in cancels.values())} beams "
              f"({len(cancelled_ents)} regions flagged: kept for training, left out of the takeoff)")
        for c in cancels.values():
            if not c["beams"]:
                print(f"WARNING: Cancelled box {c['id']} ({c['text']!r}) is not linked to a BeamID - link it to each beam it covers")
    if laps:
        print(f"laps: {len(laps)} lap lines kept as lap records (lengths in the saved .json)")
    print("sheet:", sheet or "no DrawingNo / DrawingTitle boxes")
    print("regions:", {c: n for c in CLASSES[1:] if (n := sum(g["cls"] == c for g in regions.values()))})
    pair_count = defaultdict(int)
    for a, b in rels:
        pair_count[" - ".join(sorted((regions[a]["cls"], regions[b]["cls"])))] += 1
    print("relations:", dict(pair_count))

    node_cls = np.zeros(len(nodes), np.int64)     # index into CLASSES, 0 = Other
    node_ent = [None] * len(nodes)                # LS region id the node belongs to
    T_c = np.array([nodes[i]["c"] for i in text_ids]).reshape(-1, 2)

    def poly_area(p):
        x, y = p[:, 0], p[:, 1]
        return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    unmatched_boxes = []
    for rid in sorted((k for k, g in regions.items() if g["shape"] == "box"),
                      key=lambda k: -poly_area(regions[k]["pts"])):        # big first, small boxes overwrite
        g = regions[rid]
        inside = MplPath(g["pts"]).contains_points(T_c) if len(T_c) else np.zeros(0, bool)
        hit = list(text_ids[inside])
        if not hit:                                    # fallback: word bbox mostly inside the box's extent
            bx0, by0 = g["pts"].min(0); bx1, by1 = g["pts"].max(0)
            for i in text_ids:
                x0, y0, x1, y1 = nodes[i]["bbox"]
                ov = max(0, min(x1, bx1) - max(x0, bx0)) * max(0, min(y1, by1) - max(y0, by0))
                if ov >= 0.5 * (x1 - x0) * (y1 - y0):
                    hit.append(i)
        if not hit:
            unmatched_boxes.append(rid); continue
        for i in hit:
            node_cls[i], node_ent[i] = CLASSES.index(g["cls"]), rid

    # Boxes with no PDF word under them = text drawn as outlines or SHX strokes (e.g. beam titles).
    # Read them with OCR and add them as text nodes so the graph still contains them.
    def ocr_crop(bb):
        global _ocr
        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError:
            return ""
        if "_ocr" not in globals():
            _ocr = RapidOCR()
        pix = page.get_pixmap(dpi=300, clip=pymupdf.Rect(*bb))
        img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)[..., :3]
        res, _ = _ocr(img)
        return " ".join(r[1] for r in res or []).strip()

    added = []
    if OCR_FALLBACK:
        for rid in list(unmatched_boxes):
            g = regions[rid]
            (x0, y0), (x1, y1) = g["pts"].min(0), g["pts"].max(0)
            vertical = (y1 - y0) > 1.5 * (x1 - x0)
            nodes.append(dict(kind="text", text=ocr_crop([x0, y0, x1, y1]) or "?",
                              bbox=[float(x0), float(y0), float(x1), float(y1)], size=float(min(x1 - x0, y1 - y0)),
                              dir=[0.0, -1.0] if vertical else [1.0, 0.0], font="ocr", color=0,
                              c=((x0 + x1) / 2, (y0 + y1) / 2)))
            node_cls = np.append(node_cls, CLASSES.index(g["cls"])); node_ent.append(rid)
            added.append(len(nodes) - 1); unmatched_boxes.remove(rid)
        if added:
            kinds = np.append(kinds, ["text"] * len(added))
            text_ids = np.append(text_ids, added)
            T_c = np.array([nodes[i]["c"] for i in text_ids])
            print(f"OCR fallback: {len(added)} outlined/SHX texts added as text nodes")

    n_box = sum(g["shape"] == "box" for g in regions.values())
    print(f"text boxes matched to words: {n_box - len(unmatched_boxes)}/{n_box}")
    for i in text_ids:
        if node_cls[i]:
            print(f"   {CLASSES[node_cls[i]]:<10} {nodes[i]['text']}")
    if unmatched_boxes:
        print(f"WARNING: {len(unmatched_boxes)} boxes have no PDF word under them "
              "(wrong page / IMG_CLIP, or install rapidocr_onnxruntime for the OCR fallback) ->",
              [regions[k]["cls"] for k in unmatched_boxes])

    def densify(pts, step):
        out = [pts[:1]]
        for a, b in zip(pts[:-1], pts[1:]):
            m = max(1, math.ceil(math.dist(a, b) / step))
            out.append(a + np.linspace(0, 1, m + 1)[1:, None] * (b - a))
        return np.vstack(out)

    is_line = kinds == "line"
    L_sel = is_line[S_own]
    L_tree, L_own = cKDTree(S_xy[L_sel]), S_own[L_sel]
    votes = defaultdict(lambda: defaultdict(float))     # node -> region -> voted length (pt)
    ann_dense, ann_owner = {}, {}
    for rid, g in regions.items():
        if g["shape"] != "line":
            continue
        pts = np.vstack([g["pts"], g["pts"][:1]]) if g["closed"] else g["pts"]
        dense = densify(pts, STEP)
        d, k = L_tree.query(dense, distance_upper_bound=LINE_MATCH_TOL * UNIT)
        ok = np.isfinite(d)
        owners = np.full(len(dense), -1)
        owners[ok] = L_own[k[ok]]
        ann_dense[rid], ann_owner[rid] = dense, owners
        for o in owners[ok]:
            votes[o][rid] += STEP

    ann_len = {rid: (len(d_) - 1) * STEP for rid, d_ in ann_dense.items()}
    partial_len = {}          # LS vector covering only part of a PDF line (e.g. one link zone of a long line)
    for i, vr in votes.items():
        rid, v = max(vr.items(), key=lambda kv: kv[1])
        full = v >= LINE_MATCH_MIN * nodes[i]["length"]
        part = v >= 0.8 * ann_len[rid]
        if full or part:
            node_cls[i], node_ent[i] = CLASSES.index(regions[rid]["cls"]), rid
            if part and not full:
                partial_len[rid] = v

    for rid in ann_dense:
        matched = [i for i in range(len(nodes)) if node_ent[i] == rid]
        cover = np.isin(ann_owner[rid], matched).mean()
        flag = "" if cover >= 0.5 else "   <-- WARNING: raise LINE_MATCH_TOL or check alignment"
        tag = " (part of a longer PDF line)" if rid in partial_len else ""
        print(f"{regions[rid]['cls']:<10} -> {len(matched)} node(s), covers {cover:.0%} of the LS vector{tag}{flag}")

    E = {}                                     # (src, dst) -> (type, gap in pt)
    def add_edge(i, j, t, gap):
        if i == j: return
        for a, b in ((i, j), (j, i)):
            if (a, b) not in E or E[(a, b)][1] > gap:
                E[(a, b)] = (t, gap)

    # touch: every vertex of a line/mark (+ centre of short pieces, e.g. a tick crossing a dim line)
    Q, Q_own = [], []
    for i in geo_ids:
        n = nodes[i]
        v = np.unique(np.round(n["segs"].reshape(-1, 2), 2), axis=0)
        Q.append(v); Q_own.append(np.full(len(v), i))
        if n["kind"] == "mark" or n["length"] < 2 * UNIT:
            Q.append(np.array([n["c"]])); Q_own.append(np.array([i]))
    Q, Q_own = np.vstack(Q), np.concatenate(Q_own)
    for qi, hit in enumerate(S_tree.query_ball_point(Q, TOUCH_TOL * UNIT + STEP / 2)):
        for j in np.unique(S_own[hit]):
            add_edge(int(Q_own[qi]), int(j), 0, 0.0)

    # near: text bbox expanded by NEAR_PAD
    pad = NEAR_PAD * UNIT
    for i in text_ids:
        x0, y0, x1, y1 = nodes[i]["bbox"]
        hit = S_tree.query_ball_point(nodes[i]["c"], math.hypot(x1 - x0, y1 - y0) / 2 + pad)
        if not hit: continue
        P, own = S_xy[hit], S_own[hit]
        gap = np.hypot(np.maximum(np.maximum(x0 - P[:, 0], P[:, 0] - x1), 0),
                       np.maximum(np.maximum(y0 - P[:, 1], P[:, 1] - y1), 0))
        for j in np.unique(own[gap <= pad]):
            add_edge(int(i), int(j), 1, float(gap[own == j].min()))

    # text_knn
    def bbox_gap(a, b):
        return math.hypot(max(0, a[0] - b[2], b[0] - a[2]), max(0, a[1] - b[3], b[1] - a[3]))

    text_nb = {}
    if len(text_ids) > 1:
        _, nb = cKDTree(T_c).query(T_c, k=min(TEXT_KNN + 1, len(text_ids)))
        for a in range(len(text_ids)):
            text_nb[text_ids[a]] = [int(text_ids[b]) for b in nb[a][1:]]
        # also the next text on the same line, edge to edge: a long text ('600(WD)x1125/825(DP)') has its
        # centre far from the name beside it, so centre-distance neighbours can miss it
        for a in text_ids:
            na = nodes[a]; best = None
            for b in text_ids:
                nb_ = nodes[b]
                if b == a or abs(na["c"][1] - nb_["c"][1]) > 0.5 * max(na["size"], nb_["size"]):
                    continue
                gap = nb_["bbox"][0] - na["bbox"][2]
                if -0.2 * UNIT <= gap <= 3 * UNIT and (best is None or gap < best[0]):
                    best = (gap, int(b))
            if best and best[1] not in text_nb[a]:
                text_nb[a].append(best[1])
        for a, nbs in text_nb.items():
            for j in nbs:
                add_edge(int(a), j, 2, bbox_gap(nodes[a]["bbox"], nodes[j]["bbox"]))

    EDGE_TYPES = ["touch", "near", "text_knn"]
    keys = list(E)
    edge_index = np.array(keys, np.int64).T.reshape(2, -1)
    edge_type = np.array([E[k][0] for k in keys], np.int64)
    C = np.array([n["c"] for n in nodes])
    edge_attr = np.clip(np.column_stack([(C[edge_index[1]] - C[edge_index[0]]) / UNIT,
                                         [E[k][1] / UNIT for k in keys]]), -50, 50).astype(np.float32)
    deg = np.bincount(edge_index[0], minlength=len(nodes))
    print({t: int((edge_type == k).sum()) for k, t in enumerate(EDGE_TYPES)}, f"| isolated nodes: {(deg == 0).sum()}")

    adj = defaultdict(set)
    for (a, b), (t, _) in E.items():
        if t in (0, 1):                      # walk through touch/near only
            adj[a].add(b)

    cand = set()
    for i in text_ids:
        seen, frontier = {int(i)}, {int(i)}
        for _ in range(CAND_HOPS):
            frontier = {j for f in frontier for j in adj[f]} - seen
            seen |= frontier
            if len(seen) > 3000: break
        close = set(S_own[S_tree.query_ball_point(nodes[i]["c"], CAND_RADIUS * UNIT)].tolist())
        cand |= {(int(i), j) for j in seen | close if is_line[j]}
    cand |= {(min(a, b), max(a, b)) for a, nbs in text_nb.items() for b in nbs}

    # Beam membership (BeamID -> RebarSize) is long-range: pair every title-like text with every
    # rebar-like text nearby. Uses only the text itself, so the same pairs exist at inference time.
    # the letters before B may be lower case (Ramboll: 11aB1a, 1aB13), one letter may follow it (1aBM4)
    RE_TITLE_NAME = re.compile(r"^\(?\d{0,2}[A-Za-z]{0,3}B[A-Z]?\d{1,4}[A-Za-z]?\)?$")
    RE_REBAR_LIKE = re.compile(r"^\(?(?:\d+\s*[x×]\s*)?\d*\s*-?\s*[RTHYØ∅]\s*\d{1,2}", re.I)   # 6T20, 3xT12, 4-T20, Y16@200, Ø12
    # Sections: 'SECTION 15A-15A' is drawn for the cut marks '15A' in an elevation, or 'SECTION 7B229-7B229'
    # for beam 7B229 itself; Ramboll writes 'SECTION 13A' (cut marks '13A'). The title may be one text or several
    # words on a line.
    RE_SEC = re.compile(r"SECTION\s*([A-Z0-9]{1,8})\s*-\s*([A-Z0-9]{1,8})", re.I)
    RE_SEC1 = re.compile(r"SECTION\s*([A-Z]{0,3}\d[A-Z0-9]{0,6})\b", re.I)     # one id, with a digit: not 'SECTION DETAILS'
    sec_title = {}                                   # first title node -> (section ids, title nodes)
    for i in text_ids:
        if not re.match(r"(?i)^SECTION", nodes[i]["text"].strip()):
            continue
        line = sorted([int(j) for j in text_ids if abs(nodes[j]["c"][1] - nodes[i]["c"][1]) < 0.5 * UNIT
                       and 0 <= nodes[j]["bbox"][0] - nodes[i]["bbox"][0] <= 25 * UNIT], key=lambda j: nodes[j]["bbox"][0])
        txt, used_, one = "", [], None
        for j in line:
            txt = (txt + " " + nodes[j]["text"]).strip(); used_.append(j)
            m = RE_SEC.search(txt)
            if m:
                sec_title[int(i)] = ({m.group(1).upper(), m.group(2).upper()}, list(used_))
                break
            m1 = RE_SEC1.search(txt)
            if m1 and one is None:
                one = ({m1.group(1).upper()}, list(used_))
        else:
            if one:                                      # no 'X-X' on the line: the one-id title
                sec_title[int(i)] = one
    sec_ids = set().union(*[v[0] for v in sec_title.values()]) if sec_title else set()
    in_titles = {t for _, ts in sec_title.values() for t in ts}
    def sec_id_of(t):                                # '15A', or a beam title '6B246 (250x500)' for 'SECTION 6B246-6B246'
        t = t.strip().upper()
        return next((k for k in sec_ids if t == k or t.startswith(k + " ") or t.startswith(k + "(")), None)
    mark_like = [int(i) for i in text_ids if sec_id_of(nodes[i]["text"]) and int(i) not in in_titles]
    title_like = [int(i) for i in text_ids if RE_TITLE_NAME.match(nodes[i]["text"]) or nodes[i].get("font") == "ocr"] + \
                 [t for _, ts in sec_title.values() for t in ts]          # a section title groups its own labels too
    rebar_like = np.array([int(i) for i in text_ids if RE_REBAR_LIKE.match(nodes[i]["text"])] + mark_like)
    beam_cand = set()
    if title_like and len(rebar_like):
        rb_tree = cKDTree(np.array([nodes[i]["c"] for i in rebar_like]))
        for t in title_like:
            for k in rb_tree.query_ball_point(nodes[t]["c"], BEAM_RADIUS * UNIT):
                r = int(rebar_like[k])
                if r != t:
                    beam_cand.add((min(t, r), max(t, r)))
    cand |= beam_cand

    # Same element labelled twice: where a beam continues in the next row, the overlap is drawn twice (a
    # RebarSize -> RebarSize / Dimension -> Dimension relation links the repeat to the copy that is counted),
    # and a section's labels (6T20 at its top bars) name the bars of the elevation whose line crosses the cut
    # (section label -> elevation label; counted in the elevation). Candidates: every pair of texts with the
    # same rebar label or number, anywhere.
    RE_NUM_ONLY = re.compile(r"^\d{2,6}$")
    by_text = defaultdict(list)
    for i in text_ids:
        t = nodes[i]["text"].replace(" ", "")
        if RE_REBAR_LIKE.match(t) or RE_NUM_ONLY.match(t):
            by_text[t].append(int(i))
    same_cand = {(min(a, b), max(a, b)) for ids in by_text.values() for k, a in enumerate(ids) for b in ids[k + 1:]}
    cand |= same_cand

    # section reference: cut mark '15A' <-> 'SECTION 15A-15A';  link shape: a link label (T12-100) <-> the
    # lines of the section drawings (the closed links drawn there), above each SECTION title
    sec_cand = {(min(a, t), max(a, t)) for a in mark_like for ids, ts in sec_title.values()
                if sec_id_of(nodes[a]["text"]) in ids for t in ts}
    beam_titles = [int(i) for i in text_ids if (RE_TITLE_NAME.match(nodes[i]["text"]) or nodes[i].get("font") == "ocr") and int(i) not in in_titles]
    sec_cand |= {(min(b, t), max(b, t)) for b in beam_titles for _, ts in sec_title.values() for t in ts}   # few; OCR text may be off
    RE_LINK_LIKE = re.compile(r"[RTHYØ∅]\s*\d{1,2}\s*[-@]\s*\d{2,3}", re.I)              # T12-150, Y10@200
    def not_link(i):                                 # 'T20-150 SIDE BAR', 'T10-200 E/W', 'H10-300 (T)': bars at a spacing, not links
        n = nodes[i]
        if re.search(r"\((T|B)\)", n["text"]):
            return True
        return any(re.match(r"(?i)(side|bars?|slab|e/?w|e/?f|t&b|\((t|b)\))$", nodes[j]["text"].strip()) and abs(nodes[j]["c"][1] - n["c"][1]) < 0.5 * UNIT
                   and 0 <= nodes[j]["bbox"][0] - n["bbox"][2] <= 8 * UNIT for j in text_ids)
    link_like = [int(i) for i in text_ids if RE_LINK_LIKE.search(nodes[i]["text"]) and not not_link(i)]
    link_cand = set()
    for ids, ts in sec_title.values():
        tb = nodes[ts[-1]]["bbox"]; tcx = (nodes[ts[0]]["c"][0] + nodes[ts[-1]]["c"][0]) / 2
        in_sec = [int(j) for j in np.where(is_line)[0] if abs(nodes[j]["c"][0] - tcx) <= 30 * UNIT
                  and tb[1] - 45 * UNIT <= nodes[j]["c"][1] <= tb[1] and nodes[j].get("length", 0) >= 0.6 * UNIT]
        link_cand |= {(a, j) for a in link_like for j in in_sec}
    cand |= sec_cand | link_cand

    ent_nodes = defaultdict(list)
    for i, e in enumerate(node_ent):
        if e is not None:
            ent_nodes[e].append(i)

    pos, lost = set(), []
    for a, b in rels:
        if not ent_nodes.get(a) or not ent_nodes.get(b):
            lost.append((a, b)); continue
        for i in ent_nodes[a]:
            for j in ent_nodes[b]:
                ki, kj = kinds[i], kinds[j]
                if ki == "text" and kj == "line": pos.add((i, j))
                elif ki == "line" and kj == "text": pos.add((j, i))
                elif ki == kj == "text": pos.add((min(i, j), max(i, j)))

    pairs = sorted(cand)
    pair_index = np.array(pairs, np.int64).T.reshape(2, -1)
    pair_label = np.array([p in pos for p in pairs], np.int64)
    pair_kind = np.array([5 if (i, j) in link_cand else 0 if is_line[j] else 4 if (i, j) in sec_cand else 2 if (i, j) in beam_cand
                          else 3 if (i, j) in same_cand else 1 for i, j in pairs], np.int64)
    PAIR_KINDS = ["text-line", "text-text (near)", "group membership (beam / section)", "same element (repeat / section label)",
                  "section reference (cut mark - section)", "link shape (link label - section)"]

    found = [any((i, j) in cand or (j, i) in cand for i in ent_nodes.get(a, []) for j in ent_nodes.get(b, [])) for a, b in rels]
    print(f"candidates: {len(pairs)}  positives: {pair_label.sum()}  (1 : {len(pairs) / max(1, pair_label.sum()):.0f})")
    print(f"relation recall (LS relations with >=1 candidate pair): {sum(found)}/{len(rels)}")
    by_type = defaultdict(lambda: [0, 0])
    for (a, b), f in zip(rels, found):
        k = " - ".join(sorted((regions[a]["cls"], regions[b]["cls"])))
        by_type[k][0] += f; by_type[k][1] += 1
    for k, (f, n) in sorted(by_type.items()):
        print(f"   {k:<22} {f}/{n}")
    print("pairs per kind:", {PAIR_KINDS[k]: int((pair_kind == k).sum()) for k in range(len(PAIR_KINDS))},
          "| positives per kind:", {PAIR_KINDS[k]: int(pair_label[pair_kind == k].sum()) for k in range(len(PAIR_KINDS))})
    for (a, b), f in zip(rels, found):
        if not f and (a, b) not in lost:
            print("   missed:", regions[a]["cls"], "-", regions[b]["cls"], " -> raise CAND_RADIUS / CAND_HOPS / BEAM_RADIUS")
    if lost:
        print(f"WARNING: {len(lost)} relations lost because one end matched no node (see warnings above)")

    RE_PART  = re.compile(r"^(?:(?P<sets>\d+)x)?(?P<n>\d+)?(?P<g>[RTHY])(?P<d>\d{1,2})(?:\s*[-@]\s*(?P<s>\d{2,3}))?$", re.I)
    RE_SIZE  = re.compile(r"\d{2,4}\s*[xX×]\s*\d{2,4}")
    RE_NAME  = re.compile(r"^\(?\d?[A-Z]{1,3}\d{1,3}[A-Za-z]?(?:[/-]\d{1,3})?\)?$")
    RE_NUM   = re.compile(r"^\(?(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\)?$")

    def parse_rebar(s):
        # '2T16' -> bar;  'R10-200' -> link;  'H10-300+2H10-300', 'H10-200 (T)', '2x7H10' also handled
        s = re.sub(r"\s*([xX×+])\s*(?=\d*[HRTY]?\d)", lambda m: "+" if m[1] == "+" else "x", s.strip())  # '3 x T12-100'
        core = re.sub(r"\s*\((T|B)\)$", "", s.strip()).strip("()")
        parts = []
        for p in core.split("+"):
            m = RE_PART.match(p.strip())
            if not m: return None
            parts.append(dict(sets=int(m["sets"] or 1), count=int(m["n"] or 1), grade=m["g"].upper(),
                              dia=int(m["d"]), spacing=int(m["s"]) if m["s"] else None))
        links = [p["spacing"] for p in parts]
        if any(links) and not all(links): return None
        p0 = parts[0]
        out = dict(count=sum(p["sets"] * p["count"] for p in parts), grade=p0["grade"], dia=p0["dia"],
                   spacing=p0["spacing"], kind="link" if p0["spacing"] else "bar", parts=parts)
        if p0["spacing"] and p0["sets"] > 1:          # '3xT12-100' = 3 links per set, sets at 100 c/c
            out["links_per_set"] = p0["sets"]
        return out

    FEAT = ["is_text", "is_line", "is_mark", "cx", "cy", "log_w", "log_h",
            "t_size", "t_dir_x", "t_dir_y", "t_log_nchar", "t_frac_digit", "t_rebar", "t_rebar_link",
            "t_rebar_dia", "t_rebar_count", "t_number", "t_number_log", "t_beam_size", "t_beam_name", "t_alpha_word",
            "l_log_len", "l_log_span", "l_straight", "l_cos2", "l_sin2", "l_log_nseg", "l_rel_width",
            "l_dashed", "l_closed", "l_filled", "l_r", "l_g", "l_b",
            "m_curved", "m_log_nvert", "m_triangle", "m_log_area"]
    F = {f: k for k, f in enumerate(FEAT)}
    med_w = np.median([n["width"] for n in nodes if n["kind"] != "text" and n["width"] > 0] or [1.0])

    X = np.zeros((len(nodes), len(FEAT)), np.float32)
    for i, n in enumerate(nodes):
        x0, y0, x1, y1 = n["bbox"]
        X[i, F["is_" + n["kind"]]] = 1
        X[i, F["cx"]], X[i, F["cy"]] = n["c"][0] / PAGE_W, n["c"][1] / PAGE_H
        X[i, F["log_w"]], X[i, F["log_h"]] = np.log1p((x1 - x0) / UNIT), np.log1p((y1 - y0) / UNIT)
        if n["kind"] == "text":
            s = n["text"]; rb = parse_rebar(s); num = RE_NUM.match(s)
            X[i, F["t_size"]] = n["size"] / UNIT
            X[i, F["t_dir_x"]], X[i, F["t_dir_y"]] = n["dir"]
            X[i, F["t_log_nchar"]] = np.log1p(len(s))
            X[i, F["t_frac_digit"]] = sum(ch.isdigit() for ch in s) / len(s)
            if rb:
                X[i, F["t_rebar"]], X[i, F["t_rebar_link"]] = 1, rb["kind"] == "link"
                X[i, F["t_rebar_dia"]], X[i, F["t_rebar_count"]] = rb["dia"] / 40, rb["count"] / 10
            if num:
                X[i, F["t_number"]] = 1
                X[i, F["t_number_log"]] = np.log10(float(num.group(1).replace(",", "")) + 1) / 5
            X[i, F["t_beam_size"]] = bool(RE_SIZE.search(s))
            X[i, F["t_beam_name"]] = bool(RE_NAME.match(s)) and not rb
            X[i, F["t_alpha_word"]] = s.isalpha() and len(s) >= 3
        else:
            X[i, F["l_log_len"]], X[i, F["l_log_span"]] = np.log1p(n["length"] / UNIT), np.log1p(n["span"] / UNIT)
            X[i, F["l_straight"]] = n["span"] / n["length"] if n["length"] else 0
            X[i, F["l_cos2"]], X[i, F["l_sin2"]] = np.cos(2 * n["theta"]), np.sin(2 * n["theta"])
            X[i, F["l_log_nseg"]] = np.log1p(len(n["segs"]))
            X[i, F["l_rel_width"]] = np.log1p(n["width"] / med_w)
            X[i, F["l_dashed"]], X[i, F["l_closed"]], X[i, F["l_filled"]] = n["dashed"], n["closed"], n["filled"]
            X[i, F["l_r"]:F["l_b"] + 1] = n["color"][:3]
            if n["kind"] == "mark":
                X[i, F["m_curved"]] = n["curved"]
                X[i, F["m_log_nvert"]] = np.log1p(n["n_vertices"])
                X[i, F["m_triangle"]] = n["n_vertices"] == 3 and n["closed"]
                X[i, F["m_log_area"]] = np.log1p((x1 - x0) * (y1 - y0) / UNIT**2)

    print("X:", X.shape, "| class counts:", {c: int((node_cls == k).sum()) for k, c in enumerate(CLASSES)})

    k_sub = max(1, round(VIEW_GAP * UNIT / 2 / STEP))
    V_xy, V_own = np.vstack([S_xy[::k_sub], Q]), np.concatenate([S_own[::k_sub], Q_own])
    pv = cKDTree(V_xy).query_pairs(VIEW_GAP * UNIT, output_type="ndarray")
    g = coo_matrix((np.ones(len(pv)), (V_own[pv[:, 0]], V_own[pv[:, 1]])), shape=(len(nodes), len(nodes)))
    view = connected_components(g, directed=False)[1]
    view[text_ids] = view[S_own[S_tree.query(T_c)[1]]] if len(text_ids) else []   # text -> view of nearest line work

    lab_views = sorted({int(view[i]) for i in np.where(node_cls > 0)[0]})
    for v in lab_views:
        members = np.where((view == v) & (node_cls > 0))[0]
        print(f"view {v}: labelled", sorted({CLASSES[c] for c in node_cls[members]}))

    arrays = dict(x=X, y=node_cls, edge_index=edge_index, edge_type=edge_type, edge_attr=edge_attr,
                  pair_index=pair_index, pair_label=pair_label, pair_kind=pair_kind, pair_kinds=np.array(PAIR_KINDS),
                  unit=UNIT, page_wh=np.array([PAGE_W, PAGE_H]),
                  feature_names=np.array(FEAT), class_names=np.array(CLASSES), edge_types=np.array(EDGE_TYPES))

    def node_json(i, n):
        d = dict(id=i, kind=n["kind"], cls=CLASSES[node_cls[i]], entity=node_ent[i], view=int(view[i]),
                 bbox=[round(v, 2) for v in n["bbox"]])
        if node_ent[i] in cancelled_ents:
            d["cancelled"] = True                       # part of a cancelled beam: leave it out of the takeoff
        if n["kind"] == "text":
            d.update(text=n["text"], size=round(n["size"], 2), dir=n["dir"])
        else:
            d.update(length=round(n["length"], 2), span=round(n["span"], 2), layer=n["layer"],
                     segs=np.round(n["segs"], 2).tolist())
        return d

    tmeta = task.get("meta") or {}
    meta = dict(pdf=str(PDF_PATH), page=PAGE_NO, ls_task=task.get("id"), unit=UNIT,
                consultant=str(tmeta.get("consultant") or ""), split=str(tmeta.get("split") or ""),
                relations=[[a, b] for a, b in rels], nodes=[node_json(i, n) for i, n in enumerate(nodes)],
                laps=laps, cancelled=list(cancels.values()),
                cancelled_beams=[dict(id=b, name=" ".join(nodes[i]["text"] for i in range(len(nodes)) if node_ent[i] == b))
                                 for c in cancels.values() for b in c["beams"]], **sheet)
    stats = dict(nodes=len(nodes), edges=int(edge_index.shape[1]), pairs=len(pairs), positives=int(pair_label.sum()),
                 relations=len(rels), relations_found=int(sum(found)), lost_relations=len(lost), unmatched_boxes=len(unmatched_boxes),
                 cancelled_beams=len(meta["cancelled_beams"]))
    print(f"graph: {stats['nodes']} nodes, {stats['edges']} edges, {stats['pairs']} candidate pairs, "
          f"{stats['positives']} true pairs ({stats['relations_found']}/{stats['relations']} relations reachable)")

    doc.close()
    return dict(arrays=arrays, meta=meta, stats=stats)


def save_graph(g, out_dir, stem=None):
    """Write <out_dir>/<pdf>_p<page>.npz + .json (the notebook's files). Returns the stem."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    stem = stem or f"{Path(g['meta']['pdf']).stem}_p{g['meta']['page']}"
    np.savez_compressed(out_dir / f"{stem}.npz", **g["arrays"])
    (out_dir / f"{stem}.json").write_text(json.dumps(g["meta"]), encoding="utf-8")
    return stem


# ---- which PDF page a Label Studio task shows (the same rules as the backend) ----
_UPLOAD_PREFIX = re.compile(r"^[0-9a-f]{8}-")                 # Label Studio adds 'xxxxxxxx-' to uploads
_PAGE_PATTERNS = [(re.compile(r"^(?P<stem>.+?)__p(?P<n>\d+)$"), 1),            # render_pages.py: drawing__p001.png
                  (re.compile(r"^(?P<stem>.+?)[-_]page[-_]?(?P<n>\d+)$", re.I), 1),
                  (re.compile(r"^(?P<stem>.+?)-(?P<n>\d+)$"), 1),              # pdf2image: drawing-01.png
                  (re.compile(r"^(?P<stem>.+?)_p(?P<n>\d+)$"), 0)]             # drawing_p0.png


def image_name(url):
    u = urlparse(url)
    raw = unquote(parse_qs(u.query)["d"][0]) if "d" in parse_qs(u.query) else unquote(u.path)
    return re.split(r"[\\/]", raw)[-1]


def pdf_page_for_task(task, pdf_dir):
    """(pdf_path, page_index) for a task, from data.pdf/page or the image file name; None if not found."""
    data = task.get("data", {})
    if data.get("pdf"):
        p = Path(data["pdf"]); p = p if p.is_absolute() else Path(pdf_dir) / p
        return p, int(data.get("page", 1)) - 1
    norm = lambda s: re.sub(r"[\s_\-]+", "_", s.lower()).strip("_")
    pdfs = {norm(p.stem): p for p in Path(pdf_dir).rglob("*") if p.suffix.lower() == ".pdf"}
    stem = _UPLOAD_PREFIX.sub("", Path(image_name(str(data.get("image") or next(iter(data.values()), "")))).stem)
    for pat, base in _PAGE_PATTERNS:
        m = pat.match(stem)
        if m and norm(m["stem"]) in pdfs:
            return pdfs[norm(m["stem"])], int(m["n"]) - base
    return (pdfs[norm(stem)], 0) if norm(stem) in pdfs else None


def is_not_beam(task):
    """Ticked "Not a beam drawing" in a review page (meta.page_type), or skipped in Label Studio (latest
    annotation cancelled): no graph for the rebar model."""
    pt = str((task.get("meta") or {}).get("page_type", "")).lower()
    if pt in ("not a beam drawing", "beam drawing"):
        return pt == "not a beam drawing"
    anns = sorted(task.get("annotations", []), key=lambda a: a.get("updated_at") or a.get("created_at") or "")
    return bool(anns) and bool(anns[-1].get("was_cancelled"))


def export_to_graphs(export_path, pdf_dir, out_dir, replace=True, verbose=True):
    """Every task of a Label Studio JSON export that has an annotation -> a graph in out_dir. Returns the stems.
    replace=False keeps pages that are already in out_dir (e.g. corrected later in the review page).
    Pages marked not a beam drawing (or skipped in Label Studio) are left out."""
    tasks = json.loads(Path(export_path).read_text(encoding="utf-8"))
    tasks = tasks if isinstance(tasks, list) else [tasks]
    stems = []
    for t in tasks:
        if is_not_beam(t):
            builtins.print(f"task {t.get('id')}: not a beam drawing (ticked in a review page / skipped in Label Studio) - left out")
            continue
        if not [a for a in t.get("annotations", []) if not a.get("was_cancelled")]:
            continue
        where = pdf_page_for_task(t, pdf_dir)
        if not where:
            builtins.print(f"task {t.get('id')}: no PDF found for {t.get('data', {}).get('image')} - skipped")
            continue
        if not replace and (Path(out_dir) / f"{Path(where[0]).stem}_p{where[1]}.npz").exists():
            builtins.print(f"task {t.get('id')}: {Path(where[0]).stem}_p{where[1]} is already in the training set - kept")
            continue
        g = build_graph(*where, t)
        stems.append(save_graph(g, out_dir))
        if verbose:
            s = g["stats"]
            builtins.print(f"task {t.get('id')}: {stems[-1]}  {s['nodes']} nodes, {s['pairs']} pairs, "
                           f"{s['positives']} true pairs, {s['relations_found']}/{s['relations']} relations reachable")
    return stems
