"""
Relation review page for Label Studio beam annotations.

    python review_relations.py export.json --images "D:\\ls_images" --pdf-dir "D:\\drawings"
    ->  review_<task>_<image>.html   (one page per task, opens in any browser, works offline)

Reads a Label Studio JSON export (your submitted annotation, or the prediction if a task has none),
and shows ONE label at a time with only the regions it is related to: the label box, its bar line,
its beam title, joined by dashed links. Everything else is dimmed. Step through with the keyboard,
mark each item right/wrong; the page also lists problems it can find on its own:
labels without a bar line, bar lines without a label, labels without a beam, dimensions whose
number does not match the drawn length (scale), beam titles without a size.

Cancelled (superseded) beams: a Cancelled box over the CANCELLED stamp / cloud, linked to each BeamID it
covers. Those beams are shown struck through ("cancelled, not counted"); only their label <-> line checks
are listed as problems (they are still training data), not the count checks.

Each beam (and section, and the whole page) also has an 'All annotations' view: everything that
belongs to it at once, counts per label, and PDF texts that look like a label/dimension but have no box.

--pdf-dir (optional) lets the page show the label texts (6T20, 2600 ...) read from the PDF and
enables the 'not labelled?' check.
"""
import argparse
from datetime import datetime, timezone, timedelta
from html import escape as html_escape, unescape as html_unescape
import base64
import hashlib
import io
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np
from PIL import Image

UPLOAD_PREFIX = re.compile(r"^[0-9a-f]{8}-")
PAGE_RE = re.compile(r"^(?P<stem>.+?)__p(?P<n>\d+)$")


def norm(name):
    return re.sub(r"[\s_\-]+", "_", name.lower()).strip("_")


def image_name(url):
    u = urlparse(url)
    raw = unquote(parse_qs(u.query)["d"][0]) if "d" in parse_qs(u.query) else unquote(u.path)
    return re.split(r"[\\/]", raw)[-1]                 # Windows (G:\...) or URL paths


def find_file(folder, name, exts):
    if not folder:
        return None
    want = norm(Path(name).stem)
    for p in Path(folder).rglob("*"):
        if p.suffix.lower() in exts and norm(p.stem) == want:
            return p
    return None


def label_of(r):
    v = r.get("value", {})
    for k in ("rectanglelabels", "vectorlabels", "polygonlabels", "labels"):
        if v.get(k):
            return v[k][0]
    return "?"


NOT_BEAM = "not a beam drawing"
_by_time = lambda a: a.get("updated_at") or a.get("created_at") or ""


def epoch_ms(v):
    """'2026-10-08T14:43:59.306012Z' (Label Studio, review page) -> milliseconds since 1970 (UTC), else None."""
    m = re.match(r"\s*(\d{4})-(\d\d)-(\d\d)[T ](\d\d):(\d\d)(?::(\d\d)(?:\.(\d+))?)?\s*(Z|[+-]\d\d:?\d\d)?\s*$", str(v or ""))
    if not m:
        return None
    y, mo, d, hh, mi, ss, frac, tz = m.groups()
    off = timezone.utc
    if tz and tz != "Z":
        sign = -1 if tz[0] == "-" else 1
        off = timezone(sign * timedelta(hours=int(tz[1:3]), minutes=int(tz[-2:])))
    try:
        t = datetime(int(y), int(mo), int(d), int(hh), int(mi), int(ss or 0), int(((frac or "") + "000000")[:6]), tzinfo=off)
    except ValueError:
        return None
    return int(t.timestamp() * 1000)


def last_edited(task):
    """When this task's annotation was last changed: in Label Studio (an annotation's updated_at, a skip included)
    or in a review page whose saved JSON this is (meta.edited_at). (ms, where) or (None, '')."""
    best = (None, "")
    for a in task.get("annotations") or []:
        if isinstance(a, dict):
            t = epoch_ms(a.get("updated_at") or a.get("created_at"))
            if t and (best[0] is None or t > best[0]):
                best = (t, "Label Studio")
    t = epoch_ms((task.get("meta") or {}).get("edited_at"))
    if t and (best[0] is None or t > best[0]):
        best = (t, "review page (saved JSON)")
    return best


def pick_result(task):
    """Your latest submitted annotation, else the latest prediction: (result, source, that annotation/prediction)."""
    all_anns = [a for a in task.get("annotations") or [] if isinstance(a, dict)]
    anns = [a for a in all_anns if not a.get("was_cancelled") and a.get("result")]
    if anns:
        a = sorted(anns, key=_by_time)[-1]
        return a["result"], "annotation", a
    preds = [p for p in task.get("predictions") or [] if isinstance(p, dict) and p.get("result")]   # an export lists them as ids
    if preds:
        return preds[-1]["result"], "prediction", preds[-1]
    skipped = [a for a in all_anns if a.get("result")]                      # only a skipped annotation has regions
    if skipped:
        a = sorted(skipped, key=_by_time)[-1]
        return a["result"], "skipped annotation", a
    return [], "none", {}


REGION_TYPES = ("rectanglelabels", "vectorlabels", "polygonlabels")


def fill_sizes(result, w=None, h=None):
    """Regions written by another tool (a script, an edited file) can lack original_width / original_height.
    Give them w x h, or the size the other regions of the page carry. In place; returns how many were filled."""
    if not (w and h):
        sized = next((r for r in result if r.get("original_width") and r.get("original_height")), None)
        if not sized:
            return 0
        w, h = sized["original_width"], sized["original_height"]
    n = 0
    for r in result:
        if r.get("type") in REGION_TYPES and not (r.get("original_width") and r.get("original_height")):
            r["original_width"], r["original_height"] = w, h
            r.setdefault("image_rotation", 0)
            n += 1
    return n


def unshrink_vectors(result):
    """Label Studio (1.2x) can save pre-annotated lines you didn't touch shrunk into the top-left corner: their
    points divided by the image size (x% / width, sometimes x% * 100 / width), boxes untouched. Such lines are
    put back here, in place, with the factor that lays them over the same part of the drawing as the boxes.
    Returns how many lines were put back."""
    vec = [r for r in result if r.get("type") == "vectorlabels" and (r.get("value") or {}).get("vertices")
           and (r.get("original_width") or 0) > 200 and (r.get("original_height") or 0) > 200]
    small = [r for r in vec if max(p.get("x", 0) for p in r["value"]["vertices"]) <= 1e4 / r["original_width"] * 1.001
             and max(p.get("y", 0) for p in r["value"]["vertices"]) <= 1e4 / r["original_height"] * 1.001]
    if not small:
        return 0
    W, H = small[0]["original_width"], small[0]["original_height"]
    bx = [r["value"] for r in result if r.get("type") == "rectanglelabels" and "x" in (r.get("value") or {})]
    ref = (min(v["x"] for v in bx), min(v["y"] for v in bx), max(v["x"] + v["width"] for v in bx),
           max(v["y"] + v["height"] for v in bx)) if bx else (0.0, 0.0, 100.0, 100.0)
    pts = [p for r in small for p in r["value"]["vertices"]]

    def overlap(kx, ky):                              # how well the scaled lines cover the boxes' area (0..1)
        x0, x1 = min(p["x"] for p in pts) * kx, max(p["x"] for p in pts) * kx
        y0, y1 = min(p["y"] for p in pts) * ky, max(p["y"] for p in pts) * ky
        if x1 > 100.5 or y1 > 100.5:
            return -1.0
        iw, ih = max(0.0, min(x1, ref[2]) - max(x0, ref[0])), max(0.0, min(y1, ref[3]) - max(y0, ref[1]))
        union = (x1 - x0) * (y1 - y0) + (ref[2] - ref[0]) * (ref[3] - ref[1]) - iw * ih
        return iw * ih / union if union > 0 else 0.0
    score, kx, ky = max((overlap(kx, ky), kx, ky) for kx, ky in ((W / 100, H / 100), (W, H), ((W / 100) ** 2, (H / 100) ** 2)))
    if score < 0.05:
        return 0                                      # can't tell how they were shrunk: leave them
    for r in small:
        for p in r["value"]["vertices"]:
            p["x"], p["y"] = p["x"] * kx, p["y"] * ky
            for c in ("controlPoint1", "controlPoint2"):
                if isinstance(p.get(c), dict):
                    p[c]["x"], p[c]["y"] = p[c]["x"] * kx, p[c]["y"] * ky
    return len(small)


def not_beam_of(task):
    """Was this page marked 'not a beam drawing'?  'file': ticked in a review page and saved (meta.page_type);
    'label studio': its latest annotation is cancelled (skipped in Label Studio); '': no."""
    pt = str((task.get("meta") or {}).get("page_type", "")).lower()
    if pt in (NOT_BEAM, "beam drawing"):
        return "file" if pt == NOT_BEAM else ""
    anns = [a for a in task.get("annotations", []) if isinstance(a, dict)]
    return "label studio" if anns and sorted(anns, key=_by_time)[-1].get("was_cancelled") else ""


def pdf_words(pdf_dir, img_name):
    """Words of the PDF page the image was rendered from (render_pages.py naming), in % of the page."""
    try:
        from pdf_backend import pymupdf  # pypdfium2 behind PyMuPDF's API (BEAM_PDF=pymupdf: PyMuPDF itself)
    except ImportError:
        return None
    stem = UPLOAD_PREFIX.sub("", Path(img_name).stem)
    m = PAGE_RE.match(stem)
    pdf_stem, page_no = (m["stem"], int(m["n"]) - 1) if m else (stem, 0)
    pdf = find_file(pdf_dir, pdf_stem + ".pdf", {".pdf"})
    if not pdf:
        return None
    page = pymupdf.open(pdf)[page_no]
    if page.rotation:
        page.remove_rotation()
    W, H = page.rect.width, page.rect.height
    words = [((w[0] + w[2]) / 2 / W * 100, (w[1] + w[3]) / 2 / H * 100, w[4],
              w[0] / W * 100, w[1] / H * 100, w[2] / W * 100, w[3] / H * 100) for w in page.get_text("words")]
    return words, W * 25.4 / 72, pdf_segments(page, W, H)       # words, page width in mm, line pieces


MAX_SEGS = 150000          # line pieces kept for snapping (text drawn as strokes can make many more)


def pdf_segments(page, W, H):
    """The PDF's vector lines as straight pieces (curves in 8), in % of the page: what Edit mode snaps to."""
    segs = []
    for path in page.get_drawings():
        for it in path["items"]:
            op = it[0]
            if op == "l":
                segs.append((*it[1], *it[2]))
            elif op == "c":
                p0, p1, p2, p3 = (tuple(q) for q in it[1:5]); prev = p0
                for k in range(1, 9):
                    t = k / 8; u = 1 - t
                    q = tuple(u**3 * p0[i] + 3 * u * u * t * p1[i] + 3 * u * t * t * p2[i] + t**3 * p3[i] for i in (0, 1))
                    segs.append((*prev, *q)); prev = q
            elif op in ("re", "qu"):
                q = it[1].quad if op == "re" else it[1]
                c = [tuple(q.ul), tuple(q.ur), tuple(q.lr), tuple(q.ll)]
                segs += [(*c[i], *c[(i + 1) % 4]) for i in range(4)]
    if len(segs) > MAX_SEGS:
        print(f"  {len(segs):,} PDF line pieces (text drawn as lines?): the longest {MAX_SEGS:,} kept for snapping")
        segs = sorted(segs, key=lambda s: -abs(s[2] - s[0]) - abs(s[3] - s[1]))[:MAX_SEGS]
    return [(x0 / W * 100, y0 / H * 100, x1 / W * 100, y1 / H * 100) for x0, y0, x1, y1 in segs]


def build_task(task, images, pdf_dir, max_w):
    result, source, src_obj = pick_result(task)
    # the page's fingerprint (unsaved edits kept in the browser are restored only when it matches): taken as
    # earlier versions took it - lines put back in place, before image sizes are filled in - so updating
    # these tools doesn't orphan edits you haven't saved yet
    _fp = json.loads(json.dumps(result)); unshrink_vectors(_fp)
    stamp = hashlib.md5(json.dumps(_fp, sort_keys=True).encode()).hexdigest()[:12]; del _fp
    data = task.get("data") or {}
    img_val = data.get("image") or next((v for v in data.values() if isinstance(v, str) and
                                         re.search(r"\.(png|jpe?g)(\?|$)", v, re.I)), "")   # the image field, whatever its name
    img_name = UPLOAD_PREFIX.sub("", image_name(str(img_val)))
    img_path = find_file(images, img_name, {".png", ".jpg", ".jpeg"})
    if not img_path:
        print(f"  task {task.get('id')}: image '{img_name}' not found under {images} - skipped")
        return None
    im = Image.open(img_path).convert("RGB")
    W0, H0 = im.size
    n_size = fill_sizes(result, W0, H0)                      # saved again from the page, they carry it in the file
    if n_size:
        print(f"  task {task.get('id')}: {n_size} region(s) had no image size (original_width/height) - given {W0} x {H0}")
    n_fix = unshrink_vectors(result)                         # saved again from the page, they are right in the file
    if n_fix:
        print(f"  task {task.get('id')}: {n_fix} line(s) were saved by Label Studio shrunk into the top-left corner "
              "(its vector bug) - put back in place; Save JSON + push to correct them in Label Studio too")
    if W0 > max_w:
        im = im.resize((max_w, round(H0 * max_w / W0)), Image.LANCZOS)
    W, H = im.size
    buf = io.BytesIO(); im.save(buf, "JPEG", quality=82)
    img_b64 = base64.b64encode(buf.getvalue()).decode()
    pw = pdf_words(pdf_dir, img_name) if pdf_dir else None
    words, page_w_mm, pdf_segs = pw if pw else (None, None, None)

    # the Label Studio items as they are, so the page can save your edits in the same format
    raw_main, raw_side, raw_rels, extras = {}, defaultdict(list), [], []
    for r in result:
        if r.get("type") == "relation":
            raw_rels.append(r)
        elif r.get("type") in ("rectanglelabels", "vectorlabels", "polygonlabels") and "value" in r and r.get("id") not in raw_main:
            raw_main[r["id"]] = r
        elif r.get("id") in raw_main:
            raw_side[r["id"]].append(r)                     # e.g. a per-region choice or text on that region
        else:
            extras.append(r)
    names = {}
    for r in raw_main.values():
        names.setdefault(r["type"], [r.get("from_name"), r.get("to_name")])

    regions, order = {}, []
    for r in result:
        if r.get("type") == "relation" or "value" not in r or r.get("id") in regions:
            continue
        v, lab = r["value"], label_of(r)
        reg = dict(id=r["id"], n=len(order) + 1, label=lab, note=" ".join((r.get("meta") or {}).get("text", [])))
        if r["type"] == "rectanglelabels":
            reg.update(kind="box", x=v["x"] / 100 * W, y=v["y"] / 100 * H, w=v["width"] / 100 * W, h=v["height"] / 100 * H)
            if words:
                x0, y0, x1, y1 = v["x"], v["y"], v["x"] + v["width"], v["y"] + v["height"]
                inside = [w for w in words if x0 <= w[0] <= x1 and y0 <= w[1] <= y1]
                inside.sort(key=lambda w: (round(w[1] / 0.4), w[0]))
                reg["text"] = " ".join(w[2] for w in inside)
        elif r["type"] in ("vectorlabels", "polygonlabels"):
            pts = [[p["x"], p["y"]] for p in v.get("vertices", [])] if r["type"] == "vectorlabels" else v.get("points", [])
            pts = [[x / 100 * W, y / 100 * H] for x, y in pts]
            if r["type"] == "polygonlabels" or v.get("closed"):
                pts = pts + pts[:1]
            reg.update(kind="line", pts=pts, len_px=sum(math.dist(a, b) for a, b in zip(pts[:-1], pts[1:])),
                       closed=bool(r["type"] == "polygonlabels" or v.get("closed")))
        else:
            continue
        regions[r["id"]] = reg
        order.append(r["id"])
    rels = [(r["from_id"], r["to_id"]) for r in result
            if r.get("type") == "relation" and r.get("from_id") in regions and r.get("to_id") in regions]
    nb = defaultdict(set)
    for a, b in rels:
        nb[a].add(b); nb[b].add(a)

    def is_(rid, *names):
        return regions[rid]["label"].lower().replace(" ", "") in [n.lower() for n in names]
    BEAM, SIZE, REBAR, DIM = ("BeamID", "BeamName", "BeamMark"), ("BeamSize",), ("RebarSize", "Rebar"), ("Dimension", "Dim")
    RLINE, DLINE = ("RebarLine", "RebarCoordinate"), ("DimensionLine", "DimLine")
    SMARK, STITLE = ("SectionMark", "SectionCut", "CutMark"), ("SectionTitle", "Section")
    CANCEL = ("Cancelled", "Canceled", "Superseded")
    S = 1 / W * W0                                   # displayed px -> original image px

    # scale from Dimension -> DimensionLine pairs whose number is readable (mm per original pixel)
    dim_scale = {}
    for rid in order:
        if is_(rid, *DIM):
            lines = [o for o in nb[rid] if is_(o, *DLINE)]
            num = re.fullmatch(r"\d{2,6}", (regions[rid].get("text") or "").strip())
            if lines and num:
                L = sum(regions[o]["len_px"] for o in lines) * S
                if L > 5:
                    dim_scale[rid] = float(num.group()) / L
    def agree(a, b):
        return abs(a / b - 1) <= 0.03
    def local_scale(cx, cy):
        """Scale of the view a point is in: the most common scale among the nearest dimensions."""
        if not dim_scale:
            return None
        near = sorted(dim_scale, key=lambda d: math.dist((cx, cy), centre(regions[d])))[:7]
        vals = [dim_scale[d] for d in near]
        best = max(vals, key=lambda v: (sum(agree(v, o) for o in vals), -vals.index(v)))
        return float(np.median([o for o in vals if agree(best, o)]))
    def drawing_scale(mm_per_px):      # mm per original image px -> 1:N (needs the PDF page width)
        return round(mm_per_px * W0 / page_w_mm) if page_w_mm else None
    def centre(g):
        if g["kind"] == "box":
            return (g["x"] + g["w"] / 2, g["y"] + g["h"] / 2)
        p = np.array(g["pts"]); return tuple(p.mean(0))

    beam_rel_used = any(is_(a, *BEAM) and is_(b, *REBAR) or is_(b, *BEAM) and is_(a, *REBAR) for a, b in rels)

    # drawn twice (a beam continued in the next row): RebarSize -> RebarSize / Dimension -> Dimension
    # relations link a repeat to the copy that is counted (the one in a beam; else from -> to)
    repeat_of, same_bars = {}, {}
    sec_label = lambda r: any(is_(o, *STITLE) for o in nb[r]) and not any(is_(o, *BEAM) for o in nb[r])
    for a, b in rels:
        if is_(a, *REBAR) and is_(b, *REBAR) and sec_label(a) != sec_label(b):
            s_, e_ = (a, b) if sec_label(a) else (b, a)       # a section label and the elevation label of the same bars
            same_bars[s_] = e_
            continue
        if is_(a, *REBAR) and is_(b, *REBAR) or is_(a, *DIM) and is_(b, *DIM):
            ha, hb = any(is_(o, *BEAM) for o in nb[a]), any(is_(o, *BEAM) for o in nb[b])
            rep, cnt = (b, a) if ha and not hb else (a, b)
            repeat_of[rep] = cnt
    nb2 = {r: {o for o in nb[r] if repeat_of.get(r) != o and repeat_of.get(o) != r} for r in order}   # without repeat links
    for s_, e_ in same_bars.items():
        regions[s_]["same_as"] = e_
        regions[e_].setdefault("sec_twins", []).append(s_)
    for rep, cnt in repeat_of.items():
        regions[rep]["repeat_of"] = cnt
        regions[cnt].setdefault("repeated_by", []).append(rep)
        for o in nb2[rep]:
            if regions[o]["kind"] == "line" and all(t in repeat_of for t in nb2[o] if regions[t]["kind"] == "box"):
                regions[o]["rep_line"] = True     # the bar / dimension line of a repeat
    for rid in order:
        g = regions[rid]; probs = []
        if is_(rid, *REBAR):
            beams = [o for o in nb[rid] if is_(o, *BEAM)]
            has_line = any(is_(o, *RLINE) for o in nb[rid])
            in_sec = [o for o in nb[rid] if is_(o, *STITLE)]
            if in_sec and not beams:
                g["section"] = regions[in_sec[0]].get("text") or "section"   # bars are dots in a section: no line expected
                if rid not in same_bars:
                    probs.append("not linked to its bars in the elevation (only labelled in the section?)")
            elif rid in repeat_of:
                if beams and any(is_(o, *BEAM) for o in nb[repeat_of[rid]]):
                    probs.append("counted twice: this repeat and its other copy are both in a beam")
                if not has_line: probs.append("no bar line")
            elif not has_line and beam_rel_used and not beams:
                probs.append("no relations (section label?)")
            else:
                if not has_line: probs.append("no bar line")
                if beam_rel_used and not beams: probs.append("no beam")
            if len(beams) > 1: probs.append(f"{len(beams)} beams")
        elif is_(rid, *RLINE):
            if not any(is_(o, *REBAR) for o in nb[rid]): probs.append("line has no label")
            m = re.search(r"closed link (\d+)x(\d+) mm", g.get("note") or "")
            if m:                                   # a link drawn in a section: its size is in the note (section scale)
                g["len_mm"] = 2 * (int(m.group(1)) + int(m.group(2)))
        elif is_(rid, *DIM):
            if not any(is_(o, *DLINE) for o in nb[rid]): probs.append("no dimension line")
            elif rid in dim_scale:
                v = dim_scale[rid]
                others = [dim_scale[d] for d in dim_scale if d != rid]
                n = drawing_scale(v)
                g["scale"] = f"1:{n}" if n else f"{v:.2f} mm/px"
                if others and not any(agree(v, o) for o in others):      # matches no other dimension's scale
                    ref = local_scale(*centre(g)) if len(others) > 1 else others[0]
                    dev = v / ref - 1
                    g["_scaleprob"] = f"number vs drawn length: no other dimension at this scale ({dev * 100:+.0f}% from the nearby views)"
                    probs.append(g["_scaleprob"])
        elif is_(rid, *DLINE):
            if not any(is_(o, *DIM) for o in nb[rid]): probs.append("line has no dimension")
        elif is_(rid, *SMARK):
            if not any(is_(o, *STITLE) for o in nb[rid]): probs.append("cut mark not linked to a section")
            if beam_rel_used and not any(is_(o, *BEAM) for o in nb[rid]): probs.append("no beam")
        elif is_(rid, *STITLE):
            if not any(is_(o, *SMARK) or is_(o, *BEAM) for o in nb[rid]): probs.append("no cut mark or beam linked (which beam?)")
        elif is_(rid, *BEAM):
            if not any(is_(o, *SIZE) for o in nb[rid]): probs.append("no beam size")
            if beam_rel_used and not any(is_(o, *REBAR) for o in nb[rid]): probs.append("no rebar labels")
        g["problems"] = probs

    # No BeamID->RebarSize relations in the export (backend run without --beam-relations, or not
    # labelled yet): group the labels by position instead - the beam title under the drawing - and
    # show that grouping as 'by position', not as a relation.
    titles = []
    for b in [r for r in order if is_(r, *BEAM)]:
        boxes = [regions[b]] + [regions[o] for o in nb[b] if is_(o, *SIZE)]
        titles.append((b, min(g["x"] for g in boxes), regions[b]["y"], max(g["x"] + g["w"] for g in boxes)))
    hts = [regions[r]["h"] for r in order if is_(r, *REBAR)]
    th = float(np.median(hts)) if hts else 10.0
    sections = {}                                   # "sec:SECTION 15B-15B" -> box (px)
    RE_SEC_ID = r"[\w]+-[\w]+|[A-Z]{0,3}\d[A-Z0-9]{0,6}"     # 'SECTION 15B-15B', or Ramboll's 'SECTION 13A' / 'SECTION 11'
    if words:
        wl = [(x0 / 100 * W, y0 / 100 * H, x1 / 100 * W, y1 / 100 * H, t) for _, _, t, x0, y0, x1, y1 in words]
        for a in wl:
            if a[4].upper() != "SECTION":
                continue
            nxt = [b for b in wl if b is not a and 0 <= b[0] - a[2] < 3 * (a[3] - a[1]) and abs(b[1] - a[1]) < 0.5 * (a[3] - a[1])
                   and re.fullmatch(RE_SEC_ID, b[4], re.I)]
            if nxt:
                b = min(nxt, key=lambda b: b[0])
                sections[f"sec:SECTION {b[4].upper()}"] = (a[0], min(a[1], b[1]), b[2], max(a[3], b[3]))
    # every SectionTitle you labelled is a section too (its drawing is above it, like a beam's above its title),
    # also where the PDF text gave none (no PDF, or a title style not recognised above)
    for r in order:
        if is_(r, *STITLE):
            g = regions[r]; cx, cy = g["x"] + g["w"] / 2, g["y"] + g["h"] / 2
            if not any(x0 - th <= cx <= x1 + th and y0 - th <= cy <= y1 + th for x0, y0, x1, y1 in sections.values()):
                t = " ".join((g.get("text") or "").split()).upper()
                k = f"sec:{t}" if t.startswith("SECTION") else f"sec:SECTION #{g['n']}"
                if k in sections:
                    k = f"{k} #{g['n']}"
                sections[k] = (g["x"], g["y"], g["x"] + g["w"], g["y"] + g["h"])
    for k, (x0, y0, x1, y1) in sections.items():
        titles.append((k, x0, y0, x1))

    def beam_below(x0, y1, x1):
        """The title under a drawing: titles below it, scored 2 x vertical gap + sideways distance
        (drawings run far past their title sideways, but never past the next row of titles)."""
        below = [(2 * max(0.0, ty - y1) + max(0.0, tx0 - x1, x0 - tx1), b) for b, tx0, ty, tx1 in titles if ty >= y1 - th]
        return min(below)[1] if below else None

    def ext(rid):
        g = regions[rid]
        if g["kind"] == "box":
            return g["x"], g["y"], g["x"] + g["w"], g["y"] + g["h"]
        P = np.array(g["pts"]); return P[:, 0].min(), P[:, 1].min(), P[:, 0].max(), P[:, 1].max()

    inferred = {}
    if not beam_rel_used:
        for r in order:
            if not is_(r, *REBAR) or r in repeat_of:
                continue
            E = np.array([ext(o) for o in nb[r] if is_(o, *RLINE)] or [ext(r)])     # the bar lines, else the label
            b = beam_below(E[:, 0].min(), E[:, 3].max(), E[:, 2].max())
            if b and b in regions:
                inferred[r] = b

    # which beam every region belongs to (for the per-beam 'all annotations' view):
    # title + size; rebar labels by relation (or position) with their bar lines; the rest by position
    owner = {}
    for b, *_ in titles:
        if b not in regions:
            continue
        owner[b] = b
        for o in nb[b]:
            if is_(o, *SIZE): owner.setdefault(o, b)
    for r in order:
        if is_(r, *REBAR):
            bs = [o for o in nb[r] if is_(o, *BEAM)] if beam_rel_used else ([inferred[r]] if r in inferred else [])
            if bs:
                owner[r] = bs[0]
                for o in nb[r]:
                    if is_(o, *RLINE) and not regions[o].get("closed"): owner.setdefault(o, bs[0])   # links drawn in a section stay there
        elif is_(r, *SMARK):
            bs = [o for o in nb[r] if is_(o, *BEAM)]
            if bs: owner[r] = bs[0]
    for r in order:
        if is_(r, *STITLE):
            g = regions[r]; cx, cy = g["x"] + g["w"] / 2, g["y"] + g["h"] / 2
            k = min(sections, key=lambda k: math.dist((cx, cy), ((sections[k][0] + sections[k][2]) / 2, (sections[k][1] + sections[k][3]) / 2)), default=None)
            if k and math.dist((cx, cy), ((sections[k][0] + sections[k][2]) / 2, (sections[k][1] + sections[k][3]) / 2)) < 3 * th:
                owner[r] = k; g["sec_key"] = k
                for o in nb[r]:
                    if is_(o, *REBAR): owner.setdefault(o, k)
    for r in order:                                   # a Cancelled box sits with (the first of) the beams it cancels
        bs = [o for o in nb[r] if is_(o, *BEAM)] if is_(r, *CANCEL) else []
        if bs: owner[r] = bs[0]
    SHEET = ("DrawingNo", "DrawingNumber", "DwgNo", "DrawingRef", "DrawingTitle", "DwgTitle")
    for r in order:
        if r in owner or is_(r, *SHEET):
            continue
        grp = [r] + [o for o in nb2[r] if o not in owner and not is_(o, *BEAM)]
        E = np.array([ext(o) for o in grp])
        b = beam_below(E[:, 0].min(), E[:, 3].max(), E[:, 2].max())
        if b:
            for o in grp: owner.setdefault(o, b)

    # Cancelled (superseded) beams: Cancelled box -> BeamID. Labelled as usual (training data for reading labels
    # and bars), left out of the takeoff: their regions get cx = the beam, and only the label <-> line checks stay.
    CANCEL_FOLLOW = [(BEAM, SIZE + REBAR + SMARK + STITLE), (SMARK, STITLE), (STITLE, REBAR), (REBAR, RLINE)]
    cxof = {b: b for a in order if is_(a, *CANCEL) for b in nb[a] if is_(b, *BEAM)}
    todo = list(cxof)
    while todo:
        a = todo.pop()
        for frm, to in CANCEL_FOLLOW:
            if is_(a, *frm):
                for o in nb[a]:
                    if o not in cxof and is_(o, *to):
                        cxof[o] = cxof[a]; todo.append(o)
    for r in order:                                   # no relation to the beam: by position (the title below it)
        if r not in cxof and owner.get(r) in cxof and not is_(r, *CANCEL):
            cxof[r] = owner[r]
    CANCEL_KEEP = ("no bar line", "line has no label", "no dimension line", "line has no dimension")
    for r, b in cxof.items():
        g = regions[r]; g["cx"] = b
        g["problems"] = [p for p in g["problems"] if p in CANCEL_KEEP or p == g.get("_scaleprob")]
    for r in order:
        if is_(r, *CANCEL) and not any(is_(o, *BEAM) for o in nb[r]):
            regions[r]["problems"] = ["not linked to a beam: link it to each BeamID it cancels"]

    def beam_name(rid):
        b = [o for o in nb[rid] if is_(o, *BEAM)] or ([inferred[rid]] if rid in inferred else []) or ([owner[rid]] if rid in owner else [])
        b = b[0] if b else None
        return (regions[b].get("text") or f"BeamID #{regions[b]['n']}") if b in regions else (b[4:] if b else None)
    for rep, cnt in repeat_of.items():
        regions[rep]["counted_in"] = beam_name(cnt)

    # bar line lengths at the scale of their own drawing: the dimensions of the same beam (or section), else nearby
    for rid in order:
        g = regions[rid]
        if is_(rid, *RLINE) and "len_mm" not in g:
            same = [dim_scale[d] for d in dim_scale if owner.get(d) is not None and owner.get(d) == owner.get(rid)]
            if same:
                best = max(same, key=lambda v: sum(agree(v, o) for o in same))
                sc = float(np.median([o for o in same if agree(best, o)]))
            else:
                sc = local_scale(*centre(g))
            if sc: g["len_mm"] = round(g["len_px"] * S * sc)

    # texts in the PDF that look like a label/dimension/beam name but no box covers them
    RE_REBAR_W = re.compile(r"^\(?(?:\d{1,3}[xX×])?\d{0,3}[HRTY](?:\d{2}|[68](?=[-@]))")
    RE_DIM_W = re.compile(r"^\d{3,6}$")
    RE_BEAM_W = re.compile(r"^\d*[A-Za-z]{0,3}B[A-Z]?\d+[A-Za-z]?$")      # lower case too: 11aB1a
    missing = []
    if words:
        bxs = [regions[r] for r in order if regions[r]["kind"] == "box"]
        for cx, cy, t, x0, y0, x1, y1 in words:
            kind = "rebar label" if RE_REBAR_W.match(t) else "dimension" if RE_DIM_W.match(t) else "beam name" if RE_BEAM_W.match(t) else None
            if not kind:
                continue
            px, py = cx / 100 * W, cy / 100 * H
            if any(g["x"] - 2 <= px <= g["x"] + g["w"] + 2 and g["y"] - 2 <= py <= g["y"] + g["h"] + 2 for g in bxs):
                continue
            X0, Y0, X1, Y1 = x0 / 100 * W, y0 / 100 * H, x1 / 100 * W, y1 / 100 * H
            missing.append(dict(x=X0, y=Y0, w=X1 - X0, h=Y1 - Y0, text=t, kind=kind, beam=beam_below(X0, Y1, X1)))

    # repeats the backend left out (drawn a second time where a beam continues in the next row): the
    # counted copy's note says 'also drawn in the row below/above'; its repeat is the unboxed PDF text
    # with the same label, shifted by the offset that all copies of one continuation share
    rep_marks = []
    cnt_ids = [r for r in order if regions[r]["kind"] == "box" and re.search(r"also drawn in the row (below|above)", regions[r].get("note") or "")]
    if cnt_ids and missing:
        nt = lambda t: re.sub(r"\s+", "", t or "")
        cand = []                                   # (counted region, missing index, offset upper -> lower row)
        for r in cnt_ids:
            g = regions[r]; cx, cy = g["x"] + g["w"] / 2, g["y"] + g["h"] / 2
            down = "row below" in g["note"]
            for k, m in enumerate(missing):
                mx, my = m["x"] + m["w"] / 2, m["y"] + m["h"] / 2
                if nt(m["text"]) == nt(g.get("text")) and (my > cy) == down:
                    cand.append((r, k, (mx - cx, my - cy) if down else (cx - mx, cy - my)))
        tol, taken_r, taken_k = 0.6 * th, set(), set()
        while True:
            free = [c for c in cand if c[0] not in taken_r and c[1] not in taken_k]
            if not free:
                break
            sup = lambda v: len({c[0] for c in free if math.dist(v, c[2]) < tol})
            v = max((c[2] for c in free), key=sup)
            if sup(v) < 2 and len(cnt_ids) > 1:
                break
            for r, k, v2 in sorted(free, key=lambda c: math.dist(c[2], v)):
                if r not in taken_r and k not in taken_k and math.dist(v, v2) < tol:
                    taken_r.add(r); taken_k.add(k)
                    rep_marks.append(dict(missing[k], counted=r, counted_n=regions[r]["n"], counted_in=beam_name(r)))
        missing = [m for k, m in enumerate(missing) if k not in taken_k]
    area = {}
    for b, *_ in titles:
        E = [ext(r) for r in order if owner.get(r) == b] + ([sections[b]] if b in sections else [])
        if E:
            E = np.array(E); m = 2 * th
            area[b] = (E[:, 0].min() - m, E[:, 1].min() - m, E[:, 2].max() + m, E[:, 3].max() + m)
    def in_area(mm):
        a = area.get(mm["beam"])
        return a and a[0] <= mm["x"] and mm["x"] + mm["w"] <= a[2] and a[1] <= mm["y"] and mm["y"] + mm["h"] <= a[3]
    missing = [mm for mm in missing if in_area(mm)]          # title block, notes, grid bubbles etc. are left out
    ov = {b: dict(ids=[r for r in order if owner.get(r) == b], missing=[m for m in missing if m["beam"] == b],
                  repeats=[m for m in rep_marks if m["beam"] == b]) for b, *_ in titles}
    ov["page"] = dict(ids=list(order), missing=missing, repeats=rep_marks)
    for r in order:                                   # link shape from another beam's section (same label: precedent)
        m = re.search(r"pattern of (SECTION\s*\S+)", regions[r].get("note") or "")
        b = owner.get(r)
        if m and b in ov and m.group(1) not in ov[b].setdefault("patterns", []):
            ov[b]["patterns"].append(m.group(1))
    for r in order:                                   # a section named after its beam: BeamID -> SectionTitle
        if is_(r, *STITLE):
            for b in [o for o in nb[r] if is_(o, *BEAM)]:
                if b in ov:
                    t = regions[r].get("text") or f"SectionTitle #{regions[r]['n']}"
                    if t not in ov[b].setdefault("sections", []): ov[b]["sections"].append(t)
                    k = regions[r].get("sec_key")
                    if k in ov and regions[b].get("text") and regions[b]["text"] not in ov[k].setdefault("cut_in", []):
                        ov[k]["cut_in"].append(regions[b]["text"])
    for r in order:
        if is_(r, *SMARK):
            ttl = [o for o in nb[r] if is_(o, *STITLE)]
            b = owner.get(r)
            if ttl and b in ov:
                ov[b].setdefault("sections", [])
                t = regions[ttl[0]].get("text") or f"SectionTitle #{regions[ttl[0]]['n']}"
                if t not in ov[b]["sections"]: ov[b]["sections"].append(t)
                k = regions[ttl[0]].get("sec_key")
                bn = regions[b].get("text") if b in regions else None
                if k in ov and bn and bn not in ov[k].setdefault("cut_in", []): ov[k]["cut_in"].append(bn)

    # items to step through: beam by beam, then labels without beam, dimensions, loose lines
    items, used = [], set()
    def item(kind, rid, title, **kw):
        items.append(dict(kind=kind, id=rid, title=title, **kw)); used.add(rid)
    item("ov", "ov:page", "Whole page - all annotations", key="page")
    for r in [r for r in order if is_(r, *SHEET)]:
        item("sheet", r, regions[r].get("text") or f"{regions[r]['label']} #{regions[r]['n']}")
    for r in [r for r in order if is_(r, *CANCEL)]:
        bn = [regions[o].get("text") or f"BeamID #{regions[o]['n']}" for o in nb[r] if is_(o, *BEAM)]
        item("cancel", r, f"{regions[r]['label']} → {', '.join(bn)}" if bn else f"{regions[r]['label']} (no beam linked)")
    groups = [dict(name="Page", start=0, end=len(items))]
    beams = sorted([r for r in order if is_(r, *BEAM)], key=lambda r: (regions[r]["y"] // (H / 6), regions[r]["x"]))
    for b in beams:
        members = sorted([o for o in nb[b] if is_(o, *REBAR)] if beam_rel_used else [o for o, t in inferred.items() if t == b],
                         key=lambda o: (regions[o]["x"], regions[o]["y"]))
        start = len(items)
        name = (regions[b].get("text") or f"BeamID #{regions[b]['n']}") + (" (cancelled)" if b in cxof else "")
        item("ov", "ov:" + b, "All annotations", key=b, name=name)
        item("beam", b, name)
        for m in sorted([o for o in order if is_(o, *SMARK) and owner.get(o) == b], key=lambda o: (regions[o]["y"], regions[o]["x"])):
            if m not in used:
                item("mark", m, "cut " + (regions[m].get("text") or f"#{regions[m]['n']}"))
        for m in members:
            if m not in used:
                item("rebar", m, regions[m].get("text") or f"RebarSize #{regions[m]['n']}")
        groups.append(dict(name=name, start=start, end=len(items)))
    for k in sorted(sections, key=lambda k: (sections[k][1] // (H / 6), sections[k][0])):
        if not ov[k]["ids"]:
            continue
        start = len(items)
        item("ov", "ov:" + k, "All annotations", key=k, name=k[4:])
        for t in [r for r in order if is_(r, *STITLE) and regions[r].get("sec_key") == k and r not in used]:
            item("section", t, regions[t].get("text") or f"SectionTitle #{regions[t]['n']}")
        for m in sorted([o for o in ov[k]["ids"] if is_(o, *REBAR) and o not in used], key=lambda o: (regions[o]["x"], regions[o]["y"])):
            item("rebar", m, regions[m].get("text") or f"RebarSize #{regions[m]['n']}")
        groups.append(dict(name=k[4:], start=start, end=len(items)))
    reps = sorted([r for r in order if r in repeat_of and r not in used], key=lambda r: (regions[r]["y"], regions[r]["x"]))
    if reps:
        start = len(items)
        for r in reps:
            item("repeat", r, regions[r].get("text") or f"{regions[r]['label']} #{regions[r]['n']}")
        groups.append(dict(name="Drawn twice - repeats (not counted)", start=start, end=len(items)))
    start = len(items)
    for r in sorted([r for r in order if (is_(r, *REBAR) or is_(r, *SMARK) or is_(r, *STITLE)) and r not in used], key=lambda r: (regions[r]["y"], regions[r]["x"])):
        item("rebar" if is_(r, *REBAR) else "mark" if is_(r, *SMARK) else "section", r, regions[r].get("text") or f"{regions[r]['label']} #{regions[r]['n']}")
    if len(items) > start: groups.append(dict(name="Labels without a beam", start=start, end=len(items)))
    start = len(items)
    for r in sorted([r for r in order if is_(r, *DIM) and r not in used], key=lambda r: (regions[r]["y"], regions[r]["x"])):
        item("dim", r, regions[r].get("text") or f"Dimension #{regions[r]['n']}")
    if len(items) > start: groups.append(dict(name="Dimensions", start=start, end=len(items)))
    start = len(items)
    for r in order:
        if r not in used and regions[r]["problems"] and regions[r]["kind"] == "line":
            item("line", r, f"{regions[r]['label']} #{regions[r]['n']}")
    if len(items) > start: groups.append(dict(name="Lines without a label", start=start, end=len(items)))

    sheet = {regions[r]["label"]: regions[r].get("text", "") for r in order if is_(r, *SHEET)}
    seen = defaultdict(list)
    for r in raw_main.values():
        k = "box" if r["type"] == "rectanglelabels" else "line"
        if label_of(r) not in seen[k]: seen[k].append(label_of(r))
    labels = {"box": seen["box"] + [l for l in ("BeamID", "BeamSize", "RebarSize", "Dimension", "SectionMark", "SectionTitle",
                                                 "DrawingNo", "DrawingTitle", "Cancelled") if l not in seen["box"]],
              "line": seen["line"] + [l for l in ("RebarLine", "DimensionLine") if l not in seen["line"]],
              "seen": seen["box"] + seen["line"]}
    skip = not_beam_of(task)
    ann_id = src_obj.get("id") if source in ("annotation", "skipped annotation") else None
    if ann_id is None and skip == "label studio":               # update the skipped annotation, not add one
        ann_id = sorted([a for a in task.get("annotations", []) if isinstance(a, dict)], key=_by_time)[-1].get("id")
    ed_t, ed_src = last_edited(task)
    tmeta = task.get("meta") or {}
    task_meta = dict(id=task.get("id"), image=img_val, source=source, edited=ed_t, edited_src=ed_src,
                     ann_id=ann_id, W0=W0, H0=H0, names=names, not_beam=bool(skip), skip_src=skip,
                     consultant=str(tmeta.get("consultant") or ""), test=str(tmeta.get("split") or "") == "test",
                     extras=extras, stamp=stamp)
    return dict(task=task.get("id"), image=img_name, sheet=sheet, task_meta=task_meta, labels=labels,
                raw={k: dict(main=raw_main[k], side=raw_side.get(k, [])) for k in raw_main}, raw_rels=raw_rels,
                words=[[round(x0 / 100 * W, 1), round(y0 / 100 * H, 1), round(x1 / 100 * W, 1), round(y1 / 100 * H, 1), t]
                       for _, _, t, x0, y0, x1, y1 in (words or [])],
                segs=[round(v / 100 * (H if k % 2 else W), 1) for sg in dict.fromkeys(
                      tuple(round(v, 4) for v in sg) for sg in (pdf_segs or [])) for k, v in enumerate(sg)],   # x0 y0 x1 y1 ... (px)
                beam_order=beams, owner=owner, sec_keys=sorted(sections), source=source, W=W, H=H, img=img_b64,
                regions=regions, rels=rels, items=items, groups=groups,
                has_text=bool(words), inferred=inferred, beam_rel_used=beam_rel_used, ov=ov,
                n_repeats=len(repeat_of) + len(rep_marks), rep_marks=rep_marks, n_vec_fixed=n_fix)


# light / dark theme, the same for every review page and the index (kept in this browser; first time: as the computer is set)
THEME_HEAD = r"""<script>(function () { let t = null; try { t = localStorage.getItem("relreview:theme"); } catch (e) {}
  document.documentElement.dataset.theme = t || (window.matchMedia && matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light"); })();
function toggleTheme() { const t = document.documentElement.dataset.theme === "dark" ? "light" : "dark"; document.documentElement.dataset.theme = t;
  try { localStorage.setItem("relreview:theme", t); } catch (e) {} }</script>"""

PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>Relations - __TITLE__</title><meta name="review-info" content="__INFO__">
__THEME_HEAD__
<style>
:root{--bg:#f6f6f4;--panel:#fff;--ink:#222;--mute:#777;--line:#ddd;--ok:#1e8e3e;--bad:#d93025;--sel:#1a73e8}
*{box-sizing:border-box}body{margin:0;font:13px/1.4 system-ui,Segoe UI,Arial,sans-serif;color:var(--ink);background:var(--bg);height:100vh;overflow:hidden;display:grid;grid-template-columns:330px minmax(0,1fr);grid-template-rows:auto minmax(0,1fr)}
header{grid-column:1/3;display:flex;gap:6px 14px;align-items:center;min-width:0;padding:8px 12px;background:var(--panel);border-bottom:1px solid var(--line)}
header b{font-size:14px}header .sp{flex:1}button{font:inherit;padding:4px 10px;border:1px solid var(--line);background:#fff;border-radius:6px;cursor:pointer}button:hover{border-color:#aaa}
.tabs button.on{background:var(--ink);color:#fff;border-color:var(--ink)}
aside{overflow:auto;background:var(--panel);border-right:1px solid var(--line);grid-column:1;grid-row:2}main{grid-column:2;grid-row:2}
body.nolist{grid-template-columns:0 minmax(0,1fr)}body.nolist aside{visibility:hidden;border:0}
#listbtn{padding:1px 9px;font-size:16px;line-height:20px}#homebtn{margin-right:2px}#listbtn.on{background:var(--ink);color:#fff;border-color:var(--ink)}
.grp{display:flex;align-items:center;gap:6px;padding:8px 12px 5px;font-weight:600;color:var(--mute);text-transform:uppercase;font-size:11px;letter-spacing:.04em;position:sticky;top:0;background:var(--panel);cursor:pointer;user-select:none;z-index:1}
.grp:hover{color:var(--ink)}.grp .car{width:10px;flex:none;font-size:10px}.grp .gn{flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.grp .gc{text-transform:none;letter-spacing:0;font-weight:600;color:var(--mute)}.grp .gc.done{color:var(--ok)}.grp .gbad{color:var(--bad)}
.grp.shut{border-bottom:1px solid #f0f0f0}
.ltools{display:flex;gap:6px;padding:7px 12px;border-bottom:1px solid var(--line);flex-wrap:wrap}.ltools button{padding:1px 8px;font-size:12px}
.ck{flex:none;width:20px;height:20px;padding:0;border:1.5px solid #c4c4c4;border-radius:4px;background:#fff;color:#fff;font-weight:700;font-size:13px;line-height:16px;text-align:center}
.ck:hover{border-color:var(--ok);background:#f1f8f3}.ck.ok{background:var(--ok);border-color:var(--ok)}.ck.bad{background:var(--bad);border-color:var(--bad)}
.it{display:flex;align-items:center;gap:8px;padding:5px 12px;cursor:pointer;border-left:3px solid transparent}
.it:hover{background:var(--hov)}.it.on{background:var(--selbg);border-left-color:var(--sel)}
.it .t{flex:1;min-width:0}.it .t div{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.it small{color:var(--mute)}
.chip{width:10px;height:10px;border-radius:2px;flex:none}.pb{color:var(--bad);font-size:12px}
#ileg{position:absolute;left:10px;top:10px;z-index:5;background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:6px 8px;font-size:12px;max-width:340px;box-shadow:0 2px 8px #0002}
#ileg .ih{display:flex;align-items:center;gap:6px;font-weight:600;cursor:move;user-select:none}#ileg .grip{color:var(--mute);font-size:13px}#ileg .ir{display:flex;align-items:center;gap:6px;padding:2px 0;cursor:pointer;user-select:none}
#ileg .ir.off{opacity:.4;text-decoration:line-through}#ileg .sw{width:12px;height:12px;border-radius:3px;flex:none;border:2px solid}#ileg .ic{margin-left:auto;font-variant-numeric:tabular-nums;color:var(--mute)}
#ileg.min .ir{display:none}.pb .dot{display:inline-block;width:8px;height:8px;border-radius:4px;margin-right:4px;vertical-align:middle}
.st{width:18px;text-align:center;font-weight:700}.st.ok{color:var(--ok)}.st.bad{color:var(--bad)}
main{position:relative;overflow:hidden}svg{width:100%;height:100%;display:block;cursor:grab;background:#fff}
#info{position:absolute;left:12px;bottom:12px;max-width:min(600px,70%);max-height:48vh;overflow:auto;background:rgba(255,255,255,.95);border:1px solid var(--line);border-radius:8px;padding:10px 12px;box-shadow:0 2px 10px rgba(0,0,0,.08)}
#info h3{margin:0 0 4px;font-size:15px;display:flex;align-items:flex-start;gap:8px;cursor:move;user-select:none}
#info .grip{flex:none;color:var(--mute);font-size:14px;line-height:20px;margin-left:-2px}
#info h3 .hx{flex:1;min-width:0}#info .mini{flex:none;margin:-2px -4px 0 0;padding:0 7px;font-size:14px;line-height:20px}
#info.min{max-height:none;overflow:hidden;padding:6px 10px}#info.min>:not(h3){display:none}#info.min h3{margin:0;font-size:13px}
#info.min h3 .hx{white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:420px}
#banner{position:absolute;left:12px;top:12px;right:12px;background:#fff8e1;border:1px solid #f0d58a;border-radius:8px;padding:8px 12px;color:#6b5400}
#alllinks.on{background:var(--ink);color:#fff;border-color:var(--ink)}
header{flex-wrap:wrap;white-space:nowrap}header>*{flex:none}header #sum{flex:0 1 auto;min-width:0;overflow:hidden;text-overflow:ellipsis}html{overflow:hidden}#info .row{color:var(--mute)}#info .pb{margin-top:4px}
#info table{border-collapse:collapse;margin:4px 0}#info td{padding:1px 10px 1px 0;vertical-align:top}#info td:first-child{white-space:nowrap}#info .ms{color:var(--bad);cursor:pointer;text-decoration:underline dotted}
.cxt{color:#a000a0;font-weight:600}.it.cx .t>div:first-child{text-decoration:line-through;text-decoration-color:#e600e6}
.chip.ov{background:conic-gradient(#8e44ad 0 25%,#e74c3c 0 50%,#f39c12 0 75%,#2471a3 0)}.it.ovit .t>div:first-child{font-weight:600}
[hidden]{display:none!important}
#edbar{position:absolute;left:12px;top:12px;right:12px;display:flex;gap:6px;align-items:center;flex-wrap:wrap;background:rgba(255,255,255,.97);border:1px solid var(--line);border-radius:8px;padding:6px 8px;box-shadow:0 2px 10px rgba(0,0,0,.1);z-index:3}
#edbar .sp{flex:1}#edbar button.on{background:var(--sel);color:#fff;border-color:var(--sel)}#edbar select{font:inherit;padding:3px 2px;max-width:130px}#edbar button:disabled{opacity:.4;cursor:default}
#edsave{background:#1e8e3e;color:#fff;border-color:#1e8e3e}#edcount{color:var(--mute)}
#editbtn.on{background:var(--sel);color:#fff;border-color:var(--sel)}body.editing #banner{top:64px}
#trainp{position:absolute;right:12px;top:12px;width:min(560px,calc(100% - 24px));max-height:72vh;overflow:auto;background:#fff;border:1px solid var(--line);border-radius:8px;padding:10px 14px;box-shadow:0 4px 18px rgba(0,0,0,.16);z-index:5}
#trainp .tph{display:flex;align-items:center;gap:8px;margin-bottom:8px;font-size:14px}#trainp .tph .sp{flex:1}
#trainp input{font:inherit;padding:3px 6px;border:1px solid var(--line);border-radius:5px;width:230px}
#trainp .pri{background:var(--sel);color:#fff;border-color:var(--sel)}#trainp .row{margin:6px 0}
#trainp table{border-collapse:collapse;width:100%;font-size:12px;margin-top:4px}#trainp td,#trainp th{padding:2px 6px;border-bottom:1px solid #eee;text-align:left;font-weight:normal}
#trainp th{color:var(--mute)}#trainp .ok{color:var(--ok)}#trainp .err{color:var(--bad)}#trainp details{margin-top:6px}#trainp summary{cursor:pointer;color:var(--mute)}
#restore{position:absolute;left:12px;right:12px;top:12px;background:#e8f0fe;border:1px solid #9bb8f0;border-radius:8px;padding:8px 12px;z-index:6}
#edmsg{position:absolute;left:50%;transform:translateX(-50%);top:70px;background:#222;color:#fff;border-radius:6px;padding:6px 12px;z-index:5;font-size:12px}
.hd{cursor:move}.hd.nw,.hd.se{cursor:nwse-resize}.hd.ne,.hd.sw{cursor:nesw-resize}#info select{font:inherit;padding:2px}
kbd{border:1px solid #ccc;border-bottom-width:2px;border-radius:4px;padding:0 4px;font-size:11px;background:#fafafa}
#nbwrap{display:inline-flex;align-items:center;gap:6px;padding:3px 10px;border:1px solid var(--line);border-radius:6px;cursor:pointer;user-select:none}
#cowrap,#testwrap{display:inline-flex;align-items:center;gap:6px;padding:3px 8px;border:1px solid var(--line);border-radius:6px;user-select:none}
#cowrap input{font:inherit;padding:1px 4px;border:1px solid var(--line);border-radius:4px}
#cowrap.empty{border-color:#e6a23c;background:#fdf6ec}#testwrap{cursor:pointer}#testwrap input{margin:0;cursor:pointer}
body.testpg #testwrap{background:#eaf4fe;border-color:#2471a3;color:#2471a3;font-weight:600}
#nbwrap:hover{border-color:#aaa}#nbwrap input{margin:0;width:15px;height:15px;cursor:pointer;accent-color:var(--bad)}
body.notbeam #nbwrap{background:#fdecea;border-color:var(--bad);color:var(--bad);font-weight:600}
body.notbeam #all,body.notbeam #links,body.notbeam #hl,body.notbeam #info,body.notbeam #banner,body.notbeam .tabs,body.notbeam #alllinks,body.notbeam #copy{display:none}
body.notbeam #img{filter:grayscale(.7);opacity:.75}body.notbeam #list,body.notbeam .ltools{opacity:.35;pointer-events:none;filter:grayscale(1)}
#editbtn:disabled{opacity:.4;cursor:default}
#nbbar{position:absolute;left:12px;right:12px;top:12px;display:flex;gap:10px;align-items:center;flex-wrap:wrap;background:#fdecea;border:1px solid #f1a9a0;border-radius:8px;padding:8px 12px;z-index:4;box-shadow:0 2px 10px rgba(0,0,0,.08)}
#nbbar .tx{flex:1 1 420px}#nbbar .bt{display:flex;gap:6px;flex:none;margin-left:auto}#nbbar .pri{background:var(--ok);color:#fff;border-color:var(--ok)}#nbbar .ok{color:var(--ok)}
body.notbeam #edmsg{top:auto;bottom:24px}
#aip{position:absolute;right:12px;top:12px;width:min(540px,calc(100% - 24px));max-height:calc(100% - 24px);overflow:auto;background:#fff;border:1px solid var(--line);border-radius:8px;padding:10px 14px;box-shadow:0 4px 18px rgba(0,0,0,.16);z-index:5}
#aip .tph{display:flex;align-items:center;gap:8px;margin-bottom:6px;font-size:14px}#aip .tph .sp{flex:1}#aip .row{margin:6px 0}
#aip .mute{color:var(--mute);font-size:12px}#aip .ok{color:var(--ok)}#aip .err{color:var(--bad)}#aip input{font:inherit;padding:3px 6px;border:1px solid var(--line);border-radius:5px;width:230px}
#aip details{margin-top:8px;border-top:1px solid #eee;padding-top:6px}#aip summary{cursor:pointer;font-weight:600;margin-bottom:2px}
#aip .big{font-size:22px;font-weight:700}
.ai-it{display:flex;gap:8px;align-items:center;padding:5px 6px;border-radius:6px;cursor:pointer;border-left:3px solid transparent}
.ai-it:hover{background:#f2f5fb}.ai-it.on{background:#e8f0fe;border-left-color:var(--sel)}.ai-it.done{opacity:.5}
.ai-it .tx{flex:1;min-width:0}.ai-it small{color:var(--mute)}.ai-it .bt{display:flex;gap:4px;flex:none;align-items:center}.ai-it .bt button{padding:2px 8px;font-size:12px}
#aibtn.on{background:var(--ink);color:#fff;border-color:var(--ink)}#aibtn:disabled{opacity:.4;cursor:default}body.notbeam #ai{display:none}
@media (max-width:1650px){header .keys{display:none}}
#aip:not([hidden]) ~ #edbar{right:calc(min(540px, 100% - 24px) + 24px)}#trainp:not([hidden]) ~ #edbar{right:calc(min(560px, 100% - 24px) + 24px)}
#nbnote{margin:10px 12px;padding:8px 10px;border:1px solid #f1a9a0;background:#fdecea;border-radius:6px;color:#8a1c12}
#findwrap{position:relative;flex:none}#find{font:inherit;width:120px;padding:3px 8px;border:1px solid var(--line);border-radius:6px;background:var(--panel);color:var(--ink)}
#find:focus{width:220px;outline:2px solid var(--sel);outline-offset:-1px}
#findres{position:absolute;left:0;top:calc(100% + 4px);width:420px;max-height:62vh;overflow:auto;background:var(--panel);border:1px solid var(--line);border-radius:8px;box-shadow:0 6px 22px rgba(0,0,0,.2);z-index:30;white-space:normal}
#findres .fr{display:flex;gap:8px;align-items:center;padding:5px 10px;cursor:pointer}#findres .fr.on,#findres .fr:hover{background:var(--hov)}
#findres .fr .tx{flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}#findres small{color:var(--mute)}#findres .none{padding:8px 10px;color:var(--mute)}
#edbar .tg{padding:2px 7px;font-size:12px}#edbar .tg.on{background:#0a8a3a;color:#fff;border-color:#0a8a3a}#edbar .tg b{font-weight:600;opacity:.75;margin-left:3px}
#themebtn{padding:1px 8px;font-size:15px;line-height:20px}
#coordbtn{padding:1px 7px}#coordbtn b{font-weight:600;opacity:.75;margin-left:3px;font-size:11px}#coordbtn.on{background:#0a8a3a;color:#fff;border-color:#0a8a3a}
#coordtip{position:fixed;pointer-events:none;z-index:30;background:#222;color:#fff;font:12px/1.35 Consolas,Menlo,monospace;padding:3px 7px;border-radius:4px;white-space:nowrap;box-shadow:0 1px 4px #0006}#coordtip small{display:block;color:#ffd27f;font:11px/1.3 system-ui,sans-serif}
/* dark theme: the drawing is shown as dark paper (colours turned round), the annotation colours stay */
html[data-theme=dark]{color-scheme:dark;--bg:#1b1d21;--panel:#25282d;--ink:#e3e5e8;--mute:#9ba1a9;--line:#3b4047;--ok:#4cc26d;--bad:#ff6f61;--sel:#5b9dff;--hov:#2f343b;--selbg:#253552}
:root{--hov:#f2f5fb;--selbg:#e8f0fe}
html[data-theme=dark] button{background:#2d3138;color:var(--ink)}html[data-theme=dark] button:hover{border-color:#6b727c}
html[data-theme=dark] .tabs button.on,html[data-theme=dark] #listbtn.on,html[data-theme=dark] #alllinks.on,html[data-theme=dark] #aibtn.on{background:var(--ink);color:var(--bg)}
html[data-theme=dark] .grp.shut{border-bottom-color:var(--line)}html[data-theme=dark] .ck{background:#1f2226;border-color:#59606a}
html[data-theme=dark] .it:hover,html[data-theme=dark] .ai-it:hover{background:var(--hov)}html[data-theme=dark] .it.on,html[data-theme=dark] .ai-it.on{background:var(--selbg)}
html[data-theme=dark] svg{background:#15171a}html[data-theme=dark] #img{filter:invert(.92) hue-rotate(180deg)}
html[data-theme=dark] #info{background:rgba(37,40,45,.96)}html[data-theme=dark] #edbar{background:rgba(37,40,45,.97)}
html[data-theme=dark] #trainp,html[data-theme=dark] #aip{background:var(--panel)}html[data-theme=dark] #trainp td,html[data-theme=dark] #trainp th{border-bottom-color:var(--line)}
html[data-theme=dark] #banner{background:#3a3220;border-color:#6b5a2a;color:#f0d58a}html[data-theme=dark] #restore{background:#1f2c44;border-color:#3d5a8c}
html[data-theme=dark] #nbbar,html[data-theme=dark] #nbnote,html[data-theme=dark] body.notbeam #nbwrap{background:#3d2220;border-color:#7a3a33;color:#ffb3aa}
html[data-theme=dark] #cowrap.empty{background:#3a3020;border-color:#8a6a2a}html[data-theme=dark] body.testpg #testwrap{background:#1f3044;border-color:#5b9dff;color:#8fbaff}
html[data-theme=dark] kbd{background:#2d3138;border-color:#555}html[data-theme=dark] #aip details{border-top-color:var(--line)}
html[data-theme=dark] .cxt{color:#e08ae0}
</style></head><body>
<header><button id="homebtn" title="Back to the list of review pages (review_index.html, in this folder)" onclick="location.href='review_index.html'">◀ Review pages</button><button id="listbtn" title="Hide / show the list (L)">☰</button><b>__TITLE__</b><span id="sum"></span><span class="sp"></span>
<span id="findwrap"><input id="find" placeholder="🔍 Find (Ctrl+F)" autocomplete="off" spellcheck="false" title="Find a label, a beam, a text on the drawing: type part of it (T12, 2T16, B12, 250x600 ...), pick one from the list. ↑ ↓ Enter, Esc closes"><div id="findres" hidden></div></span>
<span class="tabs"><button data-f="all" class="on">All</button> <button data-f="prob">Problems</button> <button data-f="todo">Unchecked</button></span>
<label id="nbwrap" title="Tick when this image is not a beam detail drawing (a plan, a sketch, a cover sheet ...): its annotation is cancelled, and Save JSON keeps it as a page to skip (Label Studio: skipped; the AI: an example of what to skip)"><input type="checkbox" id="notbeam"> Not a beam drawing</label>
<label id="cowrap" title="Who drew this sheet. Used to check the AI per consultant. Names you typed before are offered; Save JSON keeps it, Push corrections writes it to Label Studio">Consultant <input id="consultant" list="colist" size="11" placeholder="type name" autocomplete="off"></label><datalist id="colist"></datalist>
<label id="testwrap" title="Tick for the pages of the fixed test set: the AI never trains on them, they are its exam. Correct these pages fully"><input type="checkbox" id="testpage"> Test page</label>
<button id="hdsave" hidden title="Save as Label Studio JSON (Ctrl+S)">💾 Save JSON</button><button id="loadjson" title="Open an edited annotation JSON (a saved JSON or a Label Studio export) and show it on this page. Undo (Ctrl+Z in Edit) goes back">📂 Load JSON</button><button id="htmlsave" title="Save this review page with your edits in it (Ctrl+Shift+S). The first time the browser asks where: pick this same file to overwrite it; after that it saves straight to it">💾 Save HTML</button><button id="htmlsaveas" title="Save this review page with your edits in it as a new HTML file">Save HTML as…</button><input type="file" id="loadfile" accept=".json,application/json" hidden>
<button id="editbtn" title="Add / remove / amend the annotation here (E)">✎ Edit</button> <button id="trainbtn" title="Add this page to the AI model's training set (the model is trained in Colab)">🧠 Training set</button> <button id="aibtn" title="Rules and AI side by side: what the AI model (trained in Colab) thinks of this page - the links it doubts or would add, regions it reads differently, labels without a box">🤖 AI check</button> <button id="alllinks">All links</button> <button id="fit">Fit</button><button id="copy">Copy wrong</button><button id="coordbtn" title="Coordinates beside the cursor (F6), like AutoCAD: % across and % down the sheet from its top-left, as in the annotation check lists, plus the region under the cursor">⌗ XY<b>F6</b></button><button id="themebtn" title="Light / dark theme (T), kept for every review page">◐</button>
<span class="keys" style="color:var(--mute)" title="N/P next/prev · Space right · X wrong · F fit · A all links · O this beam's all-annotations view · M minimise the info panel · I hide it · L hide/show the list · E edit · T light/dark · Ctrl+F find"><kbd>N</kbd><kbd>P</kbd> <kbd>Space</kbd> <kbd>X</kbd> <kbd>F</kbd> <kbd>A</kbd> <kbd>O</kbd> <kbd>M</kbd> <kbd>L</kbd> <kbd>E</kbd> <kbd>T</kbd></span></header>
<aside id="side"><div id="nbnote" hidden></div><div class="ltools"><button id="expall" title="Expand every group">▾ Expand all</button><button id="colall" title="Collapse every group">▸ Collapse all</button><button id="coldone" title="Collapse the groups whose items are all ticked ✓">Collapse ticked</button></div><div id="list"></div></aside>
<main><div id="banner" hidden></div><div id="nbbar" hidden></div><div id="restore" hidden></div><div id="edmsg" hidden></div><div id="coordtip" hidden></div><div id="trainp" hidden></div><div id="aip" hidden></div>
<div id="edbar" hidden><b>Edit</b>
<button data-t="select" title="Select, move, resize (Esc)">⬚ Select</button>
<button data-t="box" title="Draw a new box (B)">▭ Box</button><select id="lbbox" title="Label for new boxes"></select>
<button data-t="line" title="Draw a new line: click the points, double-click to finish (D). With a label selected, the new line is linked to it">╱ Line</button><select id="lbline" title="Label for new lines"></select>
<button data-t="extend" title="Extend the selected line from its nearer end: click the new points, double-click to finish (G)">⤳ Extend</button>
<button data-t="redraw" title="Draw the selected line again; it keeps its label and links (W)">✎ Redraw</button>
<button id="edlink2" title="Link the selected region to another one (R); or Shift+click it">↔ Link</button>
<button id="eddel2" title="Delete the selected region (Del)">🗑 Delete</button>
<button id="undo" title="Undo (Ctrl+Z)">↶</button><button id="redo" title="Redo (Ctrl+Y)">↷</button>
<button class="tg" id="snapbtn" title="Snap to the corners and ends of the drawing's lines, and where lines cross (F3)">⌖ Snap<b>F3</b></button><button class="tg" id="perpbtn" title="Perpendicular snap: the point where your line meets a line or curve at 90° (F4)">⊥ Perp<b>F4</b></button><button class="tg" id="orthobtn" title="Ortho: lines only horizontal or vertical, like AutoCAD (F8; hold Shift to flip it for a moment)">∟ Ortho<b>F8</b></button>
<span id="edcount"></span><span class="sp"></span>
<button id="edsave" title="Save the annotation as Label Studio JSON (Ctrl+S)">💾 Save JSON</button></div><svg id="svg"><image id="img" href="data:image/jpeg;base64,__IMG__"/><g id="all"></g><g id="links"></g><g id="hl"></g><g id="ai"></g><g id="findg" pointer-events="none"></g><g id="snapg" pointer-events="none"></g><g id="issg" pointer-events="none"></g></svg><div id="ileg"></div><div id="info"></div></main>
<script id="baked" type="application/json">null</script>
<script>
const SRC0 = "<!doctype html>\n" + document.documentElement.outerHTML;   // this page as it was opened (for Save HTML)
const D = __DATA__;
const COL = {BeamID:"#8e44ad",BeamName:"#8e44ad",BeamSize:"#16a085",RebarSize:"#e74c3c",Dimension:"#2471a3",Dim:"#2471a3",RebarLine:"#f39c12",DimensionLine:"#00a3d9",DrawingNo:"#5d6d7e",DrawingTitle:"#a04000",SectionMark:"#d35400",SectionTitle:"#7f8c8d",Cancelled:"#e600e6",Canceled:"#e600e6",Superseded:"#e600e6"};
const col = l => COL[l] || "#555";
const R = D.regions, KEY = "relreview:" + D.task + ":" + D.image;
let notBeam = !!D.task_meta.not_beam;        // ticked "Not a beam drawing": the page's annotation is cancelled
let consultant = D.task_meta.consultant || "", testPage = !!D.task_meta.test;   // page tags: who drew it, test-set page
let marks = {}; try { marks = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch(e) {}
const save = () => { try { localStorage.setItem(KEY, JSON.stringify(marks)); } catch(e) {} };
const esc = v => String(v).replace(/[&<>"]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"})[c]);
const GKEY = "relreview-groups:" + D.task + ":" + D.image;          // collapsed groups in the list
let shut = new Set(); try { shut = new Set(JSON.parse(localStorage.getItem(GKEY) || "[]")); } catch(e) {}
const saveShut = () => { try { localStorage.setItem(GKEY, JSON.stringify([...shut])); } catch(e) {} };
const groupOf = i => D.groups.find(g => i >= g.start && i < g.end);
const isShut = i => { const g = groupOf(i); return !!g && shut.has(g.name); };
function toggleGroup(name) { shut.has(name) ? shut.delete(name) : shut.add(name); saveShut(); list(); }
function reveal(i) { const g = groupOf(i); if (g && shut.has(g.name)) { shut.delete(g.name); saveShut(); list(); } }   // open the group of a selected item
function setMark(id, v) {           // the tick box of an item: ✓ checked, ✗ wrong, or not checked
  if (v) marks[id] = v; else delete marks[id];
  save(); list(); if (!editing && cur >= 0 && D.items[cur] && D.items[cur].id === id) select(cur, false); }
const ckBtn = id => `<button class="ck ${marks[id] || ""}" title="Click: ticked ✓ (checked) · right-click: ✗ wrong · keys: Space ✓ and next, X ✗ and next">${marks[id] === "ok" ? "✓" : marks[id] === "bad" ? "✗" : ""}</button>`;
function wireCk(d, id) { const b = d.querySelector(".ck");
  b.onclick = ev => { ev.stopPropagation(); setMark(id, marks[id] ? null : "ok"); };
  b.oncontextmenu = ev => { ev.preventDefault(); ev.stopPropagation(); setMark(id, marks[id] === "bad" ? null : "bad"); }; }
let nb = {}; D.rels.forEach(([a,b]) => { (nb[a] = nb[a] || new Set()).add(b); (nb[b] = nb[b] || new Set()).add(a); });
const svg = document.getElementById("svg"), NS = "http://www.w3.org/2000/svg";
document.getElementById("img").setAttribute("width", D.W); document.getElementById("img").setAttribute("height", D.H);
let vb = [0, 0, D.W, D.H];
const setVB = () => svg.setAttribute("viewBox", vb.join(" "));
const el = (tag, at, parent) => { const e = document.createElementNS(NS, tag); for (const k in at) e.setAttribute(k, at[k]); parent.appendChild(e); return e; };
const sw = f => (vb[2] / svg.clientWidth) * f;               // stroke width in screen px
function shape(g, parent, st) {
  if (g.kind === "box") return el("rect", {x:g.x, y:g.y, width:g.w, height:g.h, fill:st.fill||"none", stroke:st.c, "stroke-width":st.w, opacity:st.o, "data-id":g.id}, parent);
  return el("polyline", {points:g.pts.map(p => p.join(",")).join(" "), fill:"none", stroke:st.c, "stroke-width":st.w, opacity:st.o, "stroke-linejoin":"round", "stroke-linecap":"round", "data-id":g.id}, parent);
}
function centre(g) { if (g.kind === "box") return [g.x + g.w/2, g.y + g.h/2]; let sx=0, sy=0; g.pts.forEach(p => {sx+=p[0]; sy+=p[1];}); return [sx/g.pts.length, sy/g.pts.length]; }
function nearest(g, p) {          // point of region g nearest to p (for the dashed link)
  if (g.kind === "box") return [Math.min(Math.max(p[0], g.x), g.x+g.w), Math.min(Math.max(p[1], g.y), g.y+g.h)];
  let best = null;
  for (let i = 0; i + 1 < g.pts.length; i++) { const a = g.pts[i], b = g.pts[i+1], dx = b[0]-a[0], dy = b[1]-a[1], L = dx*dx+dy*dy;
    const t = L ? Math.max(0, Math.min(1, ((p[0]-a[0])*dx + (p[1]-a[1])*dy) / L)) : 0, q = [a[0]+t*dx, a[1]+t*dy], d = Math.hypot(q[0]-p[0], q[1]-p[1]);
    if (!best || d < best[0]) best = [d, q]; }
  return best ? best[1] : centre(g);
}
function bbox(ids) { let x0=1e9,y0=1e9,x1=-1e9,y1=-1e9; ids.forEach(id => { const g = R[id]; const ps = g.kind==="box" ? [[g.x,g.y],[g.x+g.w,g.y+g.h]] : g.pts;
  ps.forEach(p => { x0=Math.min(x0,p[0]); y0=Math.min(y0,p[1]); x1=Math.max(x1,p[0]); y1=Math.max(y1,p[1]); }); }); return [x0,y0,x1,y1]; }
function drawAll(dim) {
  const g = document.getElementById("all"); g.innerHTML = "";
  Object.values(R).forEach(r => { const e = shape(r, g, {c:col(r.label), w:sw(dim ? 1 : 1.5), o:dim ? 0.18 : 0.75});
    if (r.cx) e.setAttribute("stroke-dasharray", `${sw(4)} ${sw(3)}`); });      // part of a cancelled beam
  drawIssues();
}
const cxBeams = () => [...new Set(Object.values(R).filter(g => g.cx && R[g.cx]).map(g => g.cx))];
const cxName = b => R[b] ? (R[b].text || `BeamID #${R[b].n}`) : "?";
function drawCancelled(ids, parent) {     // a cancelled beam: magenta dashed frame round its regions, 'CANCELLED - not counted'
  cxBeams().forEach(b => { const mine = ids.filter(id => R[id] && R[id].cx === b); if (!mine.length) return;
    const [x0, y0, x1, y1] = bbox(mine), m = sw(8);
    el("rect", {x:x0-m, y:y0-m, width:x1-x0+2*m, height:y1-y0+2*m, fill:"#e600e60a", stroke:"#e600e6", "stroke-width":sw(1.8), "stroke-dasharray":`${sw(8)} ${sw(5)}`, "pointer-events":"none"}, parent);
    const t = el("text", {x:x0-m, y:y0-m-sw(4), fill:"#a000a0", "font-size":sw(12), "font-weight":"700", "paint-order":"stroke", stroke:"#fff", "stroke-width":sw(3), "pointer-events":"none"}, parent);
    t.textContent = `${cxName(b)} CANCELLED - not counted`; });
}
const cxRow = g => g && g.cx && R[g.cx] ? `<div class="row cxt">✕ ${g.cx === g.id ? "This beam is" : "Part of beam " + esc(cxName(g.cx)) + ", which is"} cancelled (superseded): labelled for training, left out of the takeoff. Only the label ↔ line checks apply.</div>` : "";
let cur = -1, filter = "all", showAll = false;
const INF = D.inferred || {};
const relCol = (a, b) => { const l = [R[a].label, R[b].label]; return R[a].repeat_of === b || R[b].repeat_of === a || R[a].same_as === b || R[b].same_as === a ? "#888" : l.includes("RebarLine") ? "#e67e22" : l.includes("DimensionLine") ? "#2e86c1" : "#8e44ad"; };
const isRep = g => !!(g.repeat_of || g.rep_line);
function drawLinks(parent) {           // every relation, thin (like Label Studio, but quieter)
  D.rels.forEach(([a, b]) => { if (!R[a] || !R[b]) return; const ga = R[a], gb = R[b], q = nearest(gb, centre(ga)), p = nearest(ga, q);
    el("line", {x1:p[0], y1:p[1], x2:q[0], y2:q[1], stroke:relCol(a, b), "stroke-width":sw(1.2), opacity:0.75}, parent); });
  Object.entries(INF).forEach(([a, b]) => { if (!R[a] || !R[b]) return; const ga = R[a], gb = R[b], q = nearest(gb, centre(ga)), p = nearest(ga, q);
    el("line", {x1:p[0], y1:p[1], x2:q[0], y2:q[1], stroke:"#999", "stroke-width":sw(1), "stroke-dasharray":`${sw(2)} ${sw(3)}`, opacity:0.8}, parent); });
}
function members(it) {           // regions shown for an item: the item and its relations
  const ids = new Set([it.id]); (nb[it.id] || []).forEach(o => ids.add(o));
  return ids;
}
function byPos(it) {             // beam grouping by position (no BeamID->RebarSize relation in the export)
  if (it.kind === "rebar" && INF[it.id] && R[INF[it.id]]) return [INF[it.id]];
  if (it.kind === "beam") return Object.keys(INF).filter(k => INF[k] === it.id && R[k]);
  return [];
}
let infoMin = false; try { infoMin = localStorage.getItem("relreview:infoMin") === "1"; } catch(e) {}
function decorateInfo() {          // minimise button + drag grip on the info panel (state kept between items)
  const inf = document.getElementById("info"), h = inf.querySelector("h3"); if (!h) return;
  h.innerHTML = `<span class="grip" title="Drag the title bar to move this panel · double-click it to put the panel back">⠿</span><span class="hx">${h.innerHTML}</span><button class="mini" title="Minimise / expand (M)">${infoMin ? "▴" : "▾"}</button>`;
  h.title = "Drag to move · click to minimise / expand · double-click to put the panel back";
  inf.classList.toggle("min", infoMin);
  h.onclick = e => { e.stopPropagation(); if (infoDragged) { infoDragged = false; return; } toggleMin(); };
  h.ondblclick = e => { e.stopPropagation(); infoPos = null; try { localStorage.removeItem("relreview:infoPos"); } catch (_) {} placeInfo(); };
  h.onmousedown = e => {
    if (e.button !== 0 || e.target.closest("button,select,a,input")) return;
    const r = inf.getBoundingClientRect(), m = inf.parentElement.getBoundingClientRect();
    infoDrag = {x0: e.clientX, y0: e.clientY, l: r.left - m.left, t: r.top - m.top, moved: false}; e.preventDefault(); e.stopPropagation();
  };
  placeInfo();
}
// where the info panel sits (dragged by its title bar; kept in this browser for every page)
let infoPos = null, infoDrag = null, infoDragged = false;
try { infoPos = JSON.parse(localStorage.getItem("relreview:infoPos") || "null"); } catch (e) {}
function placeInfo() {
  const inf = document.getElementById("info"), main = inf.parentElement;
  if (!infoPos) { inf.style.left = inf.style.top = inf.style.bottom = inf.style.maxHeight = ""; return; }
  const W = main.clientWidth, H = main.clientHeight;
  const x = Math.max(0, Math.min(infoPos.x, W - Math.min(inf.offsetWidth, 160))), y = Math.max(0, Math.min(infoPos.y, H - 36));
  inf.style.left = x + "px"; inf.style.top = y + "px"; inf.style.bottom = "auto";
  inf.style.maxHeight = Math.max(60, Math.min(H * 0.6, H - y - 8)) + "px";        // scrolls inside, never runs off the bottom
}
window.addEventListener("mousemove", e => {
  if (!infoDrag) return;
  const dx = e.clientX - infoDrag.x0, dy = e.clientY - infoDrag.y0;
  if (!infoDrag.moved && Math.hypot(dx, dy) < 4) return;
  infoDrag.moved = true; infoPos = {x: infoDrag.l + dx, y: infoDrag.t + dy}; placeInfo();
});
window.addEventListener("mouseup", () => {
  if (!infoDrag) return;
  if (infoDrag.moved) { infoDragged = true; const inf = document.getElementById("info");
    infoPos = {x: parseFloat(inf.style.left) || 0, y: parseFloat(inf.style.top) || 0};          // where it ended up (kept on screen)
    try { localStorage.setItem("relreview:infoPos", JSON.stringify(infoPos)); } catch (e) {} }
  infoDrag = null;
});
window.addEventListener("resize", () => placeInfo());
function toggleMin() { infoMin = !infoMin; try { localStorage.setItem("relreview:infoMin", infoMin ? "1" : "0"); } catch(e) {}
  const inf = document.getElementById("info"); inf.classList.toggle("min", infoMin); const b = inf.querySelector(".mini"); if (b) b.textContent = infoMin ? "▴" : "▾"; }
const OV = D.ov || {};
const ORDER = ["DrawingNo","DrawingTitle","BeamID","BeamName","BeamSize","SectionMark","SectionTitle","RebarSize","RebarLine","Dimension","DimensionLine","Cancelled"];
function ovCounts(o) { const c = {}; o.ids.forEach(id => { if (isRep(R[id])) return; const l = R[id].label; c[l] = (c[l] || 0) + 1; }); return c; }
const ovReps = o => o.ids.filter(id => R[id].repeat_of);
function drawRepMark(m, parent, link) {     // a repeat left out of the annotation: grey dashed, 'not labelled'
  const pad = Math.max(m.h * 0.35, sw(3));
  el("rect", {x:m.x-pad, y:m.y-pad, width:m.w+2*pad, height:m.h+2*pad, rx:pad, fill:"#88888822", stroke:"#888", "stroke-width":sw(1.6), "stroke-dasharray":`${sw(4)} ${sw(3)}`}, parent);
  const t = el("text", {x:m.x-pad, y:m.y-pad-sw(3), fill:"#666", "font-size":sw(11), "font-weight":"700", "paint-order":"stroke", stroke:"#fff", "stroke-width":sw(3)}, parent);
  t.textContent = "repeat - not labelled";
  if (link && R[m.counted]) { const g = R[m.counted], q = nearest(g, [m.x + m.w/2, m.y + m.h/2]);
    el("line", {x1:m.x + m.w/2, y1:m.y + m.h/2, x2:q[0], y2:q[1], stroke:"#888", "stroke-width":sw(1.2), "stroke-dasharray":`${sw(5)} ${sw(4)}`}, parent); }
}
function ovProblems(o) {
  const c = ovCounts(o), p = [], n = l => c[l] || 0;
  if (o.missing.length) p.push(`${o.missing.length} text${o.missing.length > 1 ? "s" : ""} not labelled?`);
  const bad = o.ids.filter(id => (R[id].problems || []).length).length; if (bad) p.push(`${bad} region${bad > 1 ? "s" : ""} with problems`);
  if (n("Dimension") !== n("DimensionLine")) p.push(`${n("Dimension")} Dimension vs ${n("DimensionLine")} DimensionLine`);
  if (n("RebarSize") && !n("RebarLine")) p.push("no RebarLine");
  if (o.key === "page" || o === OV.page) { if (!n("DrawingNo")) p.push("no DrawingNo"); if (!n("DrawingTitle")) p.push("no DrawingTitle"); }
  return p;
}
function infoFrac() { const inf = document.getElementById("info"); return inf.hidden ? 0 : Math.min(0.55, (inf.offsetHeight + 16) / svg.clientHeight); }
function zoomTo(x0, y0, x1, y1, k = 1.25) {      // fit the box in the part of the view between the edit bar and the info panel
  const bar = document.getElementById("edbar"), t = bar && !bar.hidden ? Math.min(0.3, (bar.offsetHeight + 24) / svg.clientHeight) : 0;
  const w = Math.max(x1-x0, D.W*0.04), h = Math.max(y1-y0, D.H*0.03), cx = (x0+x1)/2, cy = (y0+y1)/2, ar = svg.clientWidth / svg.clientHeight, f = infoFrac();
  const ap = document.getElementById("aip"), rt = ap && !ap.hidden ? Math.min(0.6, (ap.offsetWidth + 24) / svg.clientWidth) : 0;   // AI panel on the right
  let W2 = w*k/(1-rt), H2 = h*k/(1-f-t); if (W2/H2 < ar) W2 = H2*ar; else H2 = W2/ar; vb = [cx - W2*(1-rt)/2, cy - H2*(t + (1-f-t)/2), W2, H2]; setVB(); }
function selectOv(i, zoom) {
  const it = D.items[i], o = ovOf(it.key), ids = new Set(o.ids);
  drawAll(true);
  const hl = document.getElementById("hl"), lk = document.getElementById("links"); hl.innerHTML = ""; lk.innerHTML = "";
  if (showAll) drawLinks(lk);
  [...ids].sort((a, b) => (R[a].kind === "box") - (R[b].kind === "box")).forEach(id => { const g = R[id];
    if (isRep(g)) { const e = shape(g, hl, {c:"#888", w:sw(2), o:0.9, fill: g.kind === "box" ? "#88888833" : "none"}); e.setAttribute("stroke-dasharray", `${sw(4)} ${sw(3)}`);
      if (g.repeat_of && g.kind === "box") { const t = el("text", {x:g.x, y:g.y - sw(3), fill:"#666", "font-size":sw(11), "font-weight":"700", "paint-order":"stroke", stroke:"#fff", "stroke-width":sw(3)}, hl); t.textContent = "repeat"; }
      return; }
    shape(g, hl, {c:"#fff", w:sw(5), o:0.85}); shape(g, hl, {c:col(g.label), w:sw(2.5), o:1, fill: g.kind === "box" ? col(g.label) + "22" : "none"});
    if ((g.problems || []).length) { const [cx, cy] = g.kind === "box" ? [g.x + g.w, g.y] : g.pts[g.pts.length - 1];
      el("circle", {cx, cy, r:sw(5), fill:(catOf(g.problems[0]) || [0,0,"#d93025"])[2], stroke:"#fff", "stroke-width":sw(1.5)}, hl); } });
  (o.repeats || []).forEach(m => drawRepMark(m, hl, false));
  drawCancelled([...ids], hl);
  o.missing.forEach((m, k) => { const pad = Math.max(m.h * 0.35, sw(3));
    el("rect", {x:m.x-pad, y:m.y-pad, width:m.w+2*pad, height:m.h+2*pad, rx:pad, fill:"#d9302518", stroke:"#d93025", "stroke-width":sw(2), "stroke-dasharray":`${sw(5)} ${sw(3)}`, "data-m":k}, hl);
    const t = el("text", {x:m.x-pad, y:m.y-pad-sw(3), fill:"#d93025", "font-size":sw(12), "font-weight":"700", "paint-order":"stroke", stroke:"#fff", "stroke-width":sw(3)}, hl); t.textContent = "not labelled?"; });
  document.querySelectorAll(".it").forEach(e => e.classList.toggle("on", +e.dataset.i === i));
  const e = document.querySelector(`.it[data-i="${i}"]`); if (e) e.scrollIntoView({block:"nearest"});
  const c = ovCounts(o), labs = [...ORDER.filter(l => c[l]), ...Object.keys(c).filter(l => !ORDER.includes(l))];
  const reb = o.ids.filter(id => R[id].label === "RebarSize"), noLine = reb.filter(id => ![...(nb[id] || [])].some(x => R[x].kind === "line")).length;
  const dims = o.ids.filter(id => R[id].label === "Dimension"), noDl = dims.filter(id => ![...(nb[id] || [])].some(x => R[x].kind === "line")).length;
  const extra = {DrawingNo: (D.sheet || {}).DrawingNo || "", DrawingTitle: (D.sheet || {}).DrawingTitle || "", RebarSize: noLine ? `<span class="pb">${noLine} without a bar line</span>` : (reb.length ? "all with a bar line" : ""),
                 Dimension: noDl ? `<span class="pb">${noDl} without a dimension line</span>` : (dims.length ? "all with a line" : "")};
  const nrep = ovReps(o).length, nrepm = (o.repeats || []).length;
  const probs = o.ids.filter(id => (R[id].problems || []).length).map(id => `<div class="pb">⚠ ${R[id].label} #${R[id].n}${R[id].text ? " “" + R[id].text + "”" : ""}: ${R[id].problems.join(", ")}</div>`);
  const ms = o.missing.map((m, k) => `<span class="ms" data-k="${k}">${m.text}</span> <small>(${m.kind})</small>`);
  const title = it.key === "page" ? "Whole page" : it.name;
  document.getElementById("info").innerHTML = `<h3>${title} — all annotations</h3>` +
    `<div class="row">In colour: everything ${it.key === "page" ? "on the page" : "that belongs to this beam (by relation, or the title below it)"}. Red dot = region with a problem. Red dashed = text in the PDF that no box covers.</div>` +
    (it.key === "page" ? (cxBeams().length ? `<div class="row cxt">✕ Cancelled (not counted): ${cxBeams().map(b => esc(cxName(b))).join(", ")}</div>` : "") : cxRow(R[it.key])) +
    `<table>${labs.map(l => `<tr><td><span class="chip" style="display:inline-block;background:${col(l)}"></span> ${l}</td><td><b>${c[l]}</b></td><td>${extra[l] || ""}</td></tr>`).join("")}` +
    (R[it.key] && !R[it.key].cx && !c.BeamSize ? `<tr><td colspan="3" class="pb">⚠ no BeamSize</td></tr>` : "") +
    (it.key === "page" ? ["DrawingNo", "DrawingTitle"].filter(l => !c[l]).map(l => `<tr><td colspan="3" class="pb">⚠ no ${l} box</td></tr>`).join("") : "") +
    ((o.sections || []).length ? `<tr><td><span class="chip" style="display:inline-block;background:#7f8c8d"></span> Section</td><td colspan="2">${o.sections.join(", ")} - shape of the links</td></tr>` : "") +
    ((o.cut_in || []).length ? `<tr><td><span class="chip" style="display:inline-block;background:#d35400"></span> Cut in</td><td colspan="2">${o.cut_in.join(", ")}</td></tr>` : "") +
    ((o.patterns || []).length ? `<tr><td><span class="chip" style="display:inline-block;background:#7f8c8d"></span> Link pattern</td><td colspan="2">${o.patterns.join(", ")} (same label as a beam with that section: precedent)</td></tr>` : "") +
    (nrep ? `<tr><td><span class="chip" style="display:inline-block;background:#888"></span> Repeats</td><td><b>${nrep}</b></td><td>drawn twice, not counted (grey dashed)</td></tr>` : "") +
    (nrepm ? `<tr><td><span class="chip" style="display:inline-block;background:#888"></span> Repeats</td><td><b>${nrepm}</b></td><td>drawn a second time where the beam continues: left out on purpose, counted in the other row (grey dashed)</td></tr>` : "") + `</table>` +
    (D.has_text ? (ms.length ? `<div class="pb">Possibly not labelled (click to zoom): ${ms.join(", ")}</div>` : `<div class="row">✓ every rebar label, dimension number and beam name in the PDF text here has a box</div>`)
                : `<div class="row">Unlabelled-text check needs the PDF text (--pdf-dir, and a PDF whose labels are real text).</div>`) +
    probs.join("") + `<div class="row" style="margin-top:6px">${marks[it.id]==="ok" ? "✓ marked right" : marks[it.id]==="bad" ? "✗ marked wrong" : "not checked yet"}</div>`;
  decorateInfo();
  document.querySelectorAll("#info .ms").forEach(sp => sp.onclick = () => { const m = o.missing[+sp.dataset.k]; zoomTo(m.x - m.w*4, m.y - m.h*6, m.x + m.w*5, m.y + m.h*7); selectOv(i, false); });
  if (zoom) { const ps = [...ids].length ? bbox([...ids]) : [1e9, 1e9, -1e9, -1e9];
    [...o.missing, ...(o.repeats || [])].forEach(m => { ps[0] = Math.min(ps[0], m.x); ps[1] = Math.min(ps[1], m.y); ps[2] = Math.max(ps[2], m.x + m.w); ps[3] = Math.max(ps[3], m.y + m.h); });
    if (ps[0] > ps[2]) { fit(); return; } zoomTo(...ps, 1.12); selectOv(i, false); }
}
function select(i, zoom = true) {
  if (i < 0 || i >= D.items.length) return;
  reveal(i);
  if (editing) return editSelectItem(i, zoom);
  if (D.items[i].kind === "ov") { cur = i; return selectOv(i, zoom); }
  cur = i; const it = D.items[i], g0 = R[it.id], ids = members(it);
  drawAll(true);
  const hl = document.getElementById("hl"), lk = document.getElementById("links"); hl.innerHTML = ""; lk.innerHTML = "";
  if (showAll) drawLinks(lk);
  const c0 = centre(g0), pos = byPos(it);
  pos.forEach(id => { const g = R[id], q = nearest(g, c0), p = nearest(g0, q);
    el("line", {x1:p[0], y1:p[1], x2:q[0], y2:q[1], stroke:"#888", "stroke-width":sw(1.6), "stroke-dasharray":`${sw(2)} ${sw(4)}`}, lk);
    shape(g, hl, {c:col(g.label), w:sw(2), o:0.8}); });
  (D.rep_marks || []).filter(m => m.counted === it.id).forEach(m => drawRepMark(m, hl, true));
  ids.forEach(id => { if (id === it.id) return; const g = R[id], q = nearest(g, c0), p = nearest(g0, q);
    el("line", {x1:p[0], y1:p[1], x2:q[0], y2:q[1], stroke:"#1a73e8", "stroke-width":sw(1.6), "stroke-dasharray":`${sw(6)} ${sw(4)}`}, lk); });
  if (g0.cx) drawCancelled([...ids], hl);
  ids.forEach(id => { const g = R[id]; shape(g, hl, {c:"#fff", w:sw(id===it.id?7:6), o:0.9}); shape(g, hl, {c:col(g.label), w:sw(id===it.id?4:3), o:1, fill: g.kind==="box" ? col(g.label)+"22" : "none"}); });
  document.querySelectorAll(".it").forEach(e => e.classList.toggle("on", +e.dataset.i === i));
  const e = document.querySelector(`.it[data-i="${i}"]`); if (e) e.scrollIntoView({block:"nearest"});
  // info panel
  const rel = [...ids].filter(id => id !== it.id).map(id => R[id]);
  const fmt = g => `${g.label} #${g.n}${g.text ? " “" + g.text + "”" : ""}${g.repeat_of ? " (repeat)" : ""}${g.len_mm && !g.closed ? ` ≈ ${g.len_mm.toLocaleString()} mm` : ""}${g.scale ? ` (scale ${g.scale})` : ""}${g.note ? " — " + g.note : ""}`;
  const probs = [...(g0.problems||[]), ...rel.filter(g => (g.problems||[]).length && g.kind==="line").map(g => `${g.label} #${g.n}: ${g.problems.join(", ")}`)];
  const posTxt = pos.length ? `<div class="row">⋯ ${it.kind === "beam" ? `${pos.length} rebar labels` : fmt(R[pos[0]])} by position (beam title below the drawing), <b>not a relation</b></div>` : "";
  const repTxt = g0.repeat_of ? `<div class="row">⧉ Drawn twice: this copy is a <b>repeat</b> of #${R[g0.repeat_of].n} “${R[g0.repeat_of].text || ""}”${g0.counted_in ? ", counted in " + g0.counted_in : ""}. Not counted here.</div>`
    : (g0.repeated_by || []).length ? `<div class="row">⧉ Also drawn as ${g0.repeated_by.map(r => "#" + R[r].n).join(", ")} (repeat, not counted). This copy is counted.</div>`
    : (D.rep_marks || []).some(m => m.counted === it.id) ? `<div class="row">⧉ Also drawn in the row ${/row above/.test(g0.note || "") ? "above" : "below"} (grey dashed): that copy is a repeat and is left out on purpose. This copy is counted.</div>` : "";
  const twinTxt = g0.same_as && R[g0.same_as] ? `<div class="row">⧉ Section label: the same bars as #${R[g0.same_as].n} “${R[g0.same_as].text || ""}” in the elevation. Counted there, not here.</div>`
    : (g0.sec_twins || []).filter(t => R[t]).length ? `<div class="row">⧉ Also labelled in ${g0.sec_twins.filter(t => R[t]).map(t => `${esc(R[t].section || "a section")} (#${R[t].n})`).join(", ")}. Counted here.</div>` : "";
  document.getElementById("info").innerHTML = `<h3>${fmt(g0)}</h3><div class="row">ID ${g0.id}</div>` + cxRow(g0) + repTxt + twinTxt +
    (rel.length ? rel.map(g => `<div>↔ ${fmt(g)}</div>`).join("") : it.kind === "sheet" ? `<div class="row">sheet-level label: no relations needed</div>` : `<div class="pb">no relations</div>`) + posTxt +
    (probs.length ? `<div class="pb">⚠ ${probs.join(" · ")}</div>` : "") +
    `<div class="row" style="margin-top:6px">${marks[it.id]==="ok" ? "✓ marked right" : marks[it.id]==="bad" ? "✗ marked wrong" : "not checked yet"}</div>`;
  decorateInfo();
  if (zoom) { const [x0,y0,x1,y1] = bbox([...ids, ...pos]); const w = Math.max(x1-x0, D.W*0.06), h = Math.max(y1-y0, D.H*0.05), cx = (x0+x1)/2, cy = (y0+y1)/2;
    zoomTo(cx - w*0.8, cy - h*0.9, cx + w*0.8, cy + h*0.9, 1); select(i, false); }
}
function visible(i) { const it = D.items[i], g = R[it.id];
  if (it.kind === "ov") return filter === "prob" ? ovProblems(ovOf(it.key)).length > 0 : filter === "todo" ? !marks[it.id] : true;
  if (filter === "prob") return (g.problems||[]).length > 0 || [...(nb[it.id]||[])].some(o => R[o].kind==="line" && (R[o].problems||[]).length);
  if (filter === "todo") return !marks[it.id]; return true; }
function step(d) { if (notBeam) return; let i = cur; for (let k = 0; k < D.items.length; k++) { i += d; if (i < 0 || i >= D.items.length) return; if (visible(i) && !isShut(i)) return select(i); } }
function mark(v) { if (cur < 0 || notBeam) return; const id = D.items[cur].id; marks[id] = marks[id] === v ? undefined : v; save(); list(); select(cur, false); if (v) step(1); }
function list() {
  const A = document.getElementById("side"), top = A.scrollTop;
  const L = document.getElementById("list"); L.innerHTML = ""; let np = 0, nok = 0, nbad = 0;
  D.groups.forEach(gr => { let any = false, gok = 0, gbad = 0; const n = gr.end - gr.start, closed = shut.has(gr.name);
    for (let i = gr.start; i < gr.end; i++) { const m = marks[D.items[i].id]; if (m === "ok") gok++; if (m === "bad") gbad++; }
    const head = () => { if (any) return; any = true;
      const h = document.createElement("div"); h.className = "grp" + (closed ? " shut" : ""); h.title = closed ? "Expand" : "Collapse";
      h.innerHTML = `<span class="car">${closed ? "▸" : "▾"}</span><span class="gn">${esc(gr.name)}</span>` +
        `<span class="gc${gok === n ? " done" : ""}">${gok}/${n} ✓${gbad ? ` <span class="gbad">${gbad} ✗</span>` : ""}</span>`;
      h.onclick = () => toggleGroup(gr.name); L.appendChild(h); };
    for (let i = gr.start; i < gr.end; i++) { const it = D.items[i], g = R[it.id];
      if (it.kind === "ov") { if (marks[it.id]==="ok") nok++; if (marks[it.id]==="bad") nbad++;
        if (!visible(i)) continue; head(); if (closed) continue; const o = ovOf(it.key), p = ovProblems(o);
        const d = document.createElement("div"); d.className = "it ovit"; d.dataset.i = i;
        d.innerHTML = `<span class="chip ov"></span><span class="t"><div>${it.title}</div><small>${o.ids.length} annotations</small>${p.length ? `<div class="pb">⚠ ${p.join(" · ")}</div>` : ""}</span>${ckBtn(it.id)}`;
        wireCk(d, it.id); d.onclick = () => select(i); L.appendChild(d); continue; }
      const p = (g.problems||[]).concat([...(nb[it.id]||[])].filter(o => R[o].kind==="line" && (R[o].problems||[]).length).map(o => R[o].problems[0]));
      if (p.length) np++; if (marks[it.id]==="ok") nok++; if (marks[it.id]==="bad") nbad++;
      if (!visible(i)) continue;
      head(); if (closed) continue;
      const lines = [...(nb[it.id]||[])].map(o => R[o]).filter(o => o.kind === "line");
      const sub0 = it.kind === "mark" ? `cut mark${[...(nb[it.id]||[])].filter(o => R[o].label === "SectionTitle").map(o => " → " + (R[o].text || "section")).join("")}`
                : it.kind === "section" ? `${[...(nb[it.id]||[])].filter(o => R[o].label === "SectionMark").length} cut marks${g.note ? " · " + g.note.replace(/^closed links( as drawn)?: /, "") : ""}`
                : it.kind === "repeat" ? `repeat of #${R[g.repeat_of].n}${g.counted_in ? " · counted in " + g.counted_in : ""}` : it.kind === "sheet" ? g.label : it.kind === "rebar" ? (lines.length ? lines.filter(o => !o.closed).map(o => o.len_mm ? `≈ ${o.len_mm.toLocaleString()} mm` : o.label).join(", ") +
                    (lines.some(o => o.closed) ? ` · ${lines.filter(o => o.closed).length} links drawn in the section` : "") + ((g.sec_twins || []).some(t => R[t]) ? " · also in the section" : "")
                    : (g.section ? "in " + g.section : "") + (g.same_as && R[g.same_as] ? ` · same bars as #${R[g.same_as].n} ${R[g.same_as].text || ""}` : ""))
                : it.kind === "beam" ? (D.beam_rel_used ? `${[...(nb[it.id]||[])].filter(o => R[o].label==="RebarSize").length} rebar labels` : `${byPos(it).length} rebar labels by position`) : it.kind === "dim" ? (g.scale ? `scale ${g.scale}` : "")
                : it.kind === "cancel" ? "beams under a CANCELLED stamp: not counted" : "";
      const sub = g.cx ? `<span class="cxt">✕ cancelled, not counted</span>${sub0 ? " · " + sub0 : ""}` : sub0;
      const d = document.createElement("div"); d.className = "it" + (g.cx ? " cx" : ""); d.dataset.i = i;
      d.innerHTML = `<span class="chip" style="background:${col(g.label)}"></span><span class="t"><div>${it.title} <small>#${g.n}</small></div>${sub ? `<small>${sub}</small>` : ""}${p.length ? `<div class="pb">⚠ ${p.join(" · ")}</div>` : ""}</span>${ckBtn(it.id)}`;
      wireCk(d, it.id); d.onclick = () => select(i); L.appendChild(d); } });
  const nreg = Object.keys(R).length, note = document.getElementById("nbnote"); note.hidden = !notBeam;
  if (notBeam) note.innerHTML = `<b>Not a beam drawing</b>: this page's annotation is cancelled.${nreg ? ` Its ${nreg} regions below are kept (greyed) in case you untick.` : ""}`;
  if (notBeam) { document.getElementById("sum").innerHTML = ` · <span class="pb"><b>not a beam drawing</b></span>`; A.scrollTop = top; return; }
  document.getElementById("sum").innerHTML = (D.sheet && Object.keys(D.sheet).length ? `<b>${[D.sheet.DrawingNo, D.sheet.DrawingTitle].filter(Boolean).join(" · ")}</b> · ` : "") + `${D.items.filter(it => it.kind !== "ov").length} items${D.n_repeats ? ` · ${D.n_repeats} repeats` : ""}${cxBeams().length ? ` · <span class="cxt">${cxBeams().length} cancelled</span>` : ""} · <span class="pb">${np} with problems</span> · ${nok} ✓ · ${nbad} ✗ · from ${D.source}${D.has_text ? "" : " · (no PDF: texts not shown)"}`;
  document.querySelectorAll(".it").forEach(e => e.classList.toggle("on", +e.dataset.i === cur));
  A.scrollTop = top;
}
document.getElementById("expall").onclick = () => { shut.clear(); saveShut(); list(); };
document.getElementById("colall").onclick = () => { D.groups.forEach(g => shut.add(g.name)); saveShut(); list(); };
document.getElementById("coldone").onclick = () => { D.groups.forEach(g => { let all = true; for (let i = g.start; i < g.end; i++) if (marks[D.items[i].id] !== "ok") all = false; if (all) shut.add(g.name); }); saveShut(); list(); };
// pan / zoom
let drag = null;
// the view stays on the drawing: smaller than the page it stays inside it, larger it keeps the whole page in view
// (lo/hi: a view already outside when you start panning is not pulled back with a jump)
function keepOnPage(v, from) {
  const lim = (o, s, S, f) => { let lo = Math.min(0, S - s), hi = Math.max(0, S - s); if (f != null) { lo = Math.min(lo, f); hi = Math.max(hi, f); } return Math.max(lo, Math.min(hi, o)); };
  return [lim(v[0], v[2], D.W, from && from[0]), lim(v[1], v[3], D.H, from && from[1]), v[2], v[3]];
}
svg.addEventListener("mousedown", e => { if (e.button === 1) e.preventDefault();          // no auto-scroll on the middle button
  if (editing && e.button === 0 && edDown(e)) { e.preventDefault(); return; }
  drag = [e.clientX, e.clientY, vb.slice()]; svg.style.cursor = "grabbing"; });
svg.addEventListener("auxclick", e => { if (e.button === 1) e.preventDefault(); });
window.addEventListener("mouseup", e => { const panned = drag && Math.hypot(e.clientX - drag[0], e.clientY - drag[1]) > 3, was = drag; if (editing) edUp(e); drag = null;
  if (!editing) svg.style.cursor = "grab"; else if (panned) drawEdit();
  if (!editing && was && !panned && e.button === 0) reviewClick(e); });
window.addEventListener("mousemove", e => { if (editing && edMove(e)) return; if (!drag) return; const k = vb[2] / svg.clientWidth; vb = keepOnPage([drag[2][0] - (e.clientX-drag[0])*k, drag[2][1] - (e.clientY-drag[1])*k, vb[2], vb[3]], drag[2]); setVB(); });
svg.addEventListener("wheel", e => { e.preventDefault(); const r = svg.getBoundingClientRect(); let k = e.deltaY > 0 ? 1.2 : 1/1.2;
  const px = vb[0] + (e.clientX - r.left) / r.width * vb[2], py = vb[1] + (e.clientY - r.top) / r.height * vb[3];
  const fitW = Math.max(D.W, D.H * svg.clientWidth / svg.clientHeight); if (k > 1 && vb[2] * k > fitW) k = Math.max(1, fitW / vb[2]);   // no smaller than Fit
  vb = keepOnPage([px - (px - vb[0]) * k, py - (py - vb[1]) * k, vb[2] * k, vb[3] * k]); setVB(); if (editing) drawEdit(); else if (cur >= 0) select(cur, false); else drawAll(false); }, {passive:false});
svg.addEventListener("dblclick", e => {
  if (editing) return;                // edit mode detects its own double-clicks (the drawing is redrawn between clicks)
  const t = e.target.dataset && e.target.dataset.id; if (!t) return; const i = D.items.findIndex(it => it.id === t); if (i >= 0) select(i); });
function fit() {                    // whole page; below the 'Not a beam drawing' bar when it is shown
  const bar = document.getElementById("nbbar"), t = bar && !bar.hidden ? Math.min(0.4, (bar.offsetHeight + 20) / svg.clientHeight) : 0;
  const ar = svg.clientWidth / (svg.clientHeight * (1 - t)); let W2 = D.W, H2 = D.H; if (W2/H2 < ar) W2 = H2*ar; else H2 = W2/ar;
  const Hv = H2 / (1 - t); vb = [(D.W-W2)/2, (D.H-H2)/2 - t*Hv, W2, Hv]; setVB(); if (editing) drawEdit(); else if (cur >= 0) select(cur, false); else drawAll(false); }
document.getElementById("fit").onclick = fit;
function toggleAll() { showAll = !showAll; document.getElementById("alllinks").classList.toggle("on", showAll);
  if (editing) { drawEdit(); return; }
  if (cur >= 0) select(cur, false); else { drawAll(false); const lk = document.getElementById("links"); lk.innerHTML = ""; if (showAll) drawLinks(lk); } }
document.getElementById("alllinks").onclick = toggleAll;
if (!D.beam_rel_used && Object.keys(INF).length) { const b = document.getElementById("banner"); b.hidden = false;
  b.innerHTML = `This export has <b>no BeamID → RebarSize relations</b> (the backend was started without <code>--beam-relations</code>, or they were not labelled). ` +
    `Labels are grouped under the beam title below their drawing, shown as grey dotted links — that grouping is <b>not</b> in your annotation. <a href="#" onclick="this.parentNode.hidden=true;return false">hide</a>`; }
if (D.n_vec_fixed) { const b = document.getElementById("banner"); b.hidden = false;
  b.innerHTML = (b.innerHTML ? b.innerHTML + "<br>" : "") + `<b>${D.n_vec_fixed} line${D.n_vec_fixed > 1 ? "s were" : " was"}</b> saved by Label Studio shrunk into the top-left corner ` +
    `(its bug with pre-annotated lines you didn't touch). Shown in place here. <button id="vfsave">💾 Save JSON</button> and push it (step 7) to correct them in Label Studio too. ` +
    `<a href="#" onclick="this.parentNode.hidden=true;return false">hide</a>`;
  document.getElementById("vfsave").onclick = saveJSON; }
document.querySelectorAll(".tabs button").forEach(b => b.onclick = () => { filter = b.dataset.f; document.querySelectorAll(".tabs button").forEach(x => x.classList.toggle("on", x === b)); list(); });
document.getElementById("copy").onclick = () => { const bad = D.items.filter(it => marks[it.id] === "bad").map(it => it.kind === "ov" ? `${it.key === "page" ? "Whole page" : it.name}: all annotations` : `#${R[it.id].n} ${R[it.id].label} ${it.title}  (ID ${it.id})`).join("\n") || "(none marked wrong)";
  navigator.clipboard && navigator.clipboard.writeText(bad); alert(bad); };
window.addEventListener("keydown", e => { if (["INPUT", "SELECT", "TEXTAREA"].includes(e.target.tagName)) return; const k = e.key.toLowerCase();
  if (k === "n" || k === "arrowdown" || k === "arrowright") { e.preventDefault(); step(1); } else if (k === "p" || k === "arrowup" || k === "arrowleft") { e.preventDefault(); step(-1); }
  else if (k === " ") { e.preventDefault(); mark("ok"); } else if (k === "x") mark("bad"); else if (k === "f") fit(); else if (k === "a") toggleAll();
  else if (k === "m") toggleMin();
  else if (k === "l") toggleList();
  else if (k === "i") { const inf = document.getElementById("info"); inf.hidden = !inf.hidden; }
  else if (k === "o") { const gr = D.groups.find(g => cur >= g.start && cur < g.end); if (gr && D.items[gr.start].kind === "ov") select(gr.start); } });
window.addEventListener("resize", () => { if (editing) drawEdit(); else if (cur >= 0) select(cur); else fit(); });
let listMin = false; try { listMin = localStorage.getItem("relreview:listMin") === "1"; } catch(e) {}
function applyList() { document.body.classList.toggle("nolist", listMin); document.getElementById("listbtn").classList.toggle("on", listMin); }
function toggleList() {           // hide / show the list on the left (the drawing gets the whole width)
  listMin = !listMin; try { localStorage.setItem("relreview:listMin", listMin ? "1" : "0"); } catch(e) {}
  applyList();
  if (cur >= 0) { select(cur); if (!listMin) { const e = document.querySelector(`.it[data-i="${cur}"]`); if (e) e.scrollIntoView({block:"nearest"}); } } else fit();
}
document.getElementById("listbtn").onclick = toggleList;
applyList();
// ================= Edit mode: add / remove / amend the annotation, save it as Label Studio JSON =================
const LBL = s => (s || "").toLowerCase().replace(/ /g, "");
const SETS = {BEAM: ["beamid", "beamname", "beammark"], SIZE: ["beamsize"], REBAR: ["rebarsize", "rebar"], DIM: ["dimension", "dim"],
  RLINE: ["rebarline", "rebarcoordinate"], DLINE: ["dimensionline", "dimline"], SMARK: ["sectionmark", "sectioncut", "cutmark"],
  STITLE: ["sectiontitle", "section"], SHEET: ["drawingno", "drawingnumber", "dwgno", "drawingref", "drawingtitle", "dwgtitle"],
  CANCEL: ["cancelled", "canceled", "superseded"]};
const isA = (id, set) => !!R[id] && SETS[set].includes(LBL(R[id].label));
const EKEY = "relreview-edit:" + D.task + ":" + D.image;
let lastTool = null;                    // AutoCAD style: Space / Enter with no command running repeats the last one
let editing = false, tool = "select", selId = null, hist = [], fut = [], nEdits = 0, ver = 0, savedVer = null, savedFile = "", ed = null, draft = null;
let ORD = Object.keys(R);
const newLbl = {box: (D.labels && D.labels.box[0]) || "RebarSize", line: (D.labels && D.labels.line[0]) || "RebarLine"};
Object.values(R).forEach(g => { if (g.kind === "line" && g.len_mm && g.len_px) g._mmpp = g.len_mm / g.len_px; });

function rebuildNb() { nb = {}; D.rels.forEach(([a, b]) => { (nb[a] = nb[a] || new Set()).add(b); (nb[b] = nb[b] || new Set()).add(a); }); }
const nbs = id => [...(nb[id] || [])];
const P = pts => pts.map(p => p[0] + "," + p[1]).join(" ");
function toSvg(e) { const pt = svg.createSVGPoint(); pt.x = e.clientX; pt.y = e.clientY; return pt.matrixTransform(svg.getScreenCTM().inverse()); }
function newId() { const c = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"; let s;
  do { s = ""; for (let i = 0; i < 10; i++) s += c[Math.floor(Math.random() * c.length)]; } while (R[s]); return s; }
const nextN = () => Math.max(0, ...Object.values(R).map(g => g.n || 0)) + 1;
const openPts = g => g.closed ? g.pts.slice(0, -1) : g.pts.slice();
const setPts = (g, pts) => { g.pts = g.closed ? [...pts, pts[0]] : pts; };

function boxText(g) {                 // the PDF words inside a box (as the page reads them for existing boxes)
  const ws = (D.words || []).filter(w => { const cx = (w[0] + w[2]) / 2, cy = (w[1] + w[3]) / 2; return cx >= g.x && cx <= g.x + g.w && cy >= g.y && cy <= g.y + g.h; });
  const row = 0.004 * D.H;
  ws.sort((a, b) => Math.round((a[1] + a[3]) / 2 / row) - Math.round((b[1] + b[3]) / 2 / row) || a[0] - b[0]);
  return ws.map(w => w[4]).join(" ");
}
function lineDerived(g) {
  g.len_px = 0; for (let i = 0; i + 1 < g.pts.length; i++) g.len_px += Math.hypot(g.pts[i + 1][0] - g.pts[i][0], g.pts[i + 1][1] - g.pts[i][1]);
  if (!g._mmpp) {                     // a new line: the scale of the nearest line that has one
    const c = centre(g); let best = null;
    Object.values(R).forEach(o => { if (o !== g && o._mmpp && o.kind === "line") { const d = Math.hypot(centre(o)[0] - c[0], centre(o)[1] - c[1]); if (!best || d < best[0]) best = [d, o._mmpp]; } });
    if (best) g._mmpp = best[1];
  }
  const m = /closed link (\d+)x(\d+) mm/.exec(g.note || "");
  if (m && !g._dirty) g.len_mm = 2 * (+m[1] + +m[2]); else if (g._mmpp) g.len_mm = Math.round(g.len_px * g._mmpp); else delete g.len_mm;
}
function derived(g) { if (g.kind === "box") { if (D.words && D.words.length) g.text = boxText(g); } else lineDerived(g); }

const secLabel = id => nbs(id).some(o => isA(o, "STITLE")) && !nbs(id).some(o => isA(o, "BEAM"));
function recomputeRepeats() {
  Object.values(R).forEach(g => { delete g.repeat_of; delete g.repeated_by; delete g.rep_line; delete g.same_as; delete g.sec_twins; });
  D.rels.forEach(([a, b]) => {
    if (isA(a, "REBAR") && isA(b, "REBAR") && secLabel(a) !== secLabel(b)) {      // section label -> the same bars in the elevation
      const [s, e] = secLabel(a) ? [a, b] : [b, a]; R[s].same_as = e; (R[e].sec_twins = R[e].sec_twins || []).push(s); return; }
    if ((isA(a, "REBAR") && isA(b, "REBAR")) || (isA(a, "DIM") && isA(b, "DIM"))) {
      const ha = nbs(a).some(o => isA(o, "BEAM")), hb = nbs(b).some(o => isA(o, "BEAM"));
      const [rep, cnt] = ha && !hb ? [b, a] : [a, b];
      R[rep].repeat_of = cnt; (R[cnt].repeated_by = R[cnt].repeated_by || []).push(rep);
    }
  });
  Object.values(R).forEach(g => { if (g.kind === "line") { const bx = nbs(g.id).filter(o => R[o].kind === "box"); if (bx.length && bx.every(o => R[o].repeat_of)) g.rep_line = true; } });
}
const aliasOf = id => nbs(id).filter(o => isA(o, "BEAM"));          // BeamID -> BeamID: the same beam written two ways
const canonBeam = b => [b, ...aliasOf(b)].sort()[0];
const NOTED = /query log|\bQ\d+\b|only labelled here|count them from the section/i;   // a note that already explains it (query log, section-only bars)
function problemsOf(id) {             // the same checks as the page builder (scale check kept from the build)
  const g = R[id], has = set => nbs(id).some(o => isA(o, set)), p = [];
  if (isA(id, "CANCEL")) return has("BEAM") ? [] : ["not linked to a beam: link it to each BeamID it cancels"];
  if (isA(id, "REBAR")) {
    const beams = nbs(id).filter(o => isA(o, "BEAM")), nbm = new Set(beams.map(canonBeam)).size, line = has("RLINE"), sec = has("STITLE");
    if (sec && !beams.length) { if (!g.same_as && !NOTED.test(g.note || "")) p.push("not linked to its bars in the elevation (only labelled in the section?)"); }
    else if (g.repeat_of && R[g.repeat_of]) { if (beams.length && nbs(g.repeat_of).some(o => isA(o, "BEAM"))) p.push("counted twice: this repeat and its other copy are both in a beam"); if (!line && !nbs(g.repeat_of).some(o => isA(o, "RLINE"))) p.push("no bar line"); }
    else if (!line && D.beam_rel_used && !beams.length) p.push(has("REBAR") ? [] : "no relations (section label?)");
    else { if (!line && !has("REBAR")) p.push("no bar line"); if (D.beam_rel_used && !beams.length && !has("REBAR")) p.push("no beam"); }
    if (sec && beams.length) p.push("counted twice: a section label that is also in a beam");
    if (nbm > 1) p.push(`${nbm} beams`);
  } else if (isA(id, "RLINE")) {
    const labs = nbs(id).filter(o => isA(o, "REBAR"));
    if (!labs.length) { if (!/UNKNOWN/i.test(g.note || "")) p.push("line has no label"); }
    else { const bs = new Set(); labs.forEach(l => nbs(l).filter(o => isA(o, "BEAM")).forEach(b => bs.add(canonBeam(b))));
      if (bs.size > 1 && !g.cx) p.push(`one bar line used by ${bs.size} beams (${[...bs].map(b => R[b].text || "#" + R[b].n).join(", ")}): split it at the support?`); }
  }
  else if (isA(id, "DIM")) { if (!has("DLINE")) p.push("no dimension line"); else if (g._scaleprob && !g._dirty) p.push(g._scaleprob); }
  else if (isA(id, "DLINE")) { if (!has("DIM")) p.push("line has no dimension"); }
  else if (isA(id, "SMARK")) { if (!has("STITLE")) p.push("cut mark not linked to a section"); }
  else if (isA(id, "STITLE")) { if (!nbs(id).some(o => isA(o, "SMARK") || isA(o, "BEAM"))) p.push("no cut mark or beam linked (which beam?)"); }
  else if (isA(id, "BEAM")) { if (!has("SIZE") && !aliasOf(id).length && !/DETAIL/i.test(g.text || "")) p.push("no beam size");      // a detail title used as its own BeamID has no size
    if (D.beam_rel_used && !has("REBAR") && !aliasOf(id).some(a => nbs(a).some(o => isA(o, "REBAR")))) p.push("no rebar labels"); }
  return (g.cx ? p.filter(x => CANCEL_KEEP.includes(x) || x === g._scaleprob) : p).flat();     // a cancelled beam: not counted, so no count checks
}
function infoOf(id) {                 // not problems, but worth seeing: bars with no size on the drawing (in the query log), beams written two ways
  const g = R[id];
  if (isA(id, "RLINE") && !nbs(id).some(o => isA(o, "REBAR")) && /UNKNOWN/i.test(g.note || "")) return ["unknown bar size: in the query log"];
  if (isA(id, "REBAR") && NOTED.test(g.note || "") && !nbs(id).some(o => isA(o, "BEAM"))) return ["noted: " + (/\bQ\d+\b/.test(g.note) ? "see query log " + g.note.match(/\bQ\d+\b/)[0] : "section-only bars, count from the section")];
  if (isA(id, "BEAM") && aliasOf(id).length) return ["same beam as " + aliasOf(id).map(b => R[b].text || "#" + R[b].n).join(", ")];
  return [];
}
// ---- issue colours: each kind of problem has its own colour on the drawing (legend bottom-left, click to hide a kind) ----
const PCAT = [  ["note", "Noted (query log / section only)", "#26a69a", p => /^noted:/.test(p)],
  ["dbl", "Counted twice / in 2 beams", "#e53935", p => /beams$|counted twice/.test(p)],
  ["own", "Label not linked (no beam)", "#fb8c00", p => /no beam$|no relations|not linked to its bars/.test(p)],
  ["line", "Label with no bar line", "#9e9d24", p => p === "no bar line"],
  ["orph", "Bar line with no label", "#d81b60", p => p === "line has no label"],
  ["share", "One bar line used by 2 beams", "#3949ab", p => /^one bar line used by/.test(p)],
  ["sec", "Section title / cut mark", "#00acc1", p => /cut mark|section/.test(p)],
  ["beam", "Beam name / size", "#8e24aa", p => /no rebar labels|no beam size/.test(p)],
  ["cx", "Cancelled box not linked", "#5d4037", p => /cancels/.test(p)],
  ["dim", "Dimensions", "#9e9e9e", p => /dimension|scale|number vs/.test(p)],
  ["unk", "Unknown bar size (query)", "#1e88e5", p => /^unknown bar size/.test(p)],
  ["alias", "Same beam, two names", "#546e7a", p => /^same beam as/.test(p)],
  ["oth", "Other problem", "#d93025", p => true]];
let iHide = new Set(["dim"]), iMin = false;
try { const v = JSON.parse(localStorage.getItem("relreview:ihide") || "null"); if (v) iHide = new Set(v); iMin = localStorage.getItem("relreview:imin") === "1"; } catch (e) {}
const catOf = p => PCAT.find(c => c[3](p));
function issueOf(g) {                 // the first shown kind of problem (or info) of a region, or null
  if (g.cx && !(g.problems || []).length) return null;
  for (const p of [...(g.problems || []), ...(g.info || [])]) { const c = catOf(p); if (c && !iHide.has(c[0])) return c; }
  return null;
}
function drawIssues() {
  const G = document.getElementById("issg"); if (!G) return; G.innerHTML = "";
  if (notBeam) return;
  Object.values(R).forEach(g => { const c = issueOf(g); if (!c) return;
    if (g.kind === "box") { const m = sw(4);
      el("rect", {x:g.x - m, y:g.y - m, width:g.w + 2*m, height:g.h + 2*m, rx:sw(3), fill:c[2] + "26", stroke:c[2], "stroke-width":sw(3)}, G); }
    else { shape(g, G, {c:"#fff", w:sw(7), o:0.8}); shape(g, G, {c:c[2], w:sw(4), o:0.95}); }
    const [cx, cy] = g.kind === "box" ? [g.x + g.w + sw(4), g.y - sw(4)] : g.pts[g.pts.length - 1];
    el("circle", {cx, cy, r:sw(6), fill:c[2], stroke:"#fff", "stroke-width":sw(1.5)}, G); });
  legend();
}
// where the Issues panel sits: dragged by its title bar (kept in this browser), else top-left, below the edit bar when editing
let ilegPos = null, ilegDrag = null, ilegDragged = false;
try { ilegPos = JSON.parse(localStorage.getItem("relreview:ilegPos") || "null"); } catch (e) {}
function placeIleg() {
  const L = document.getElementById("ileg"); if (!L) return; const main = L.parentElement, bar = document.getElementById("edbar");
  if (!ilegPos) { L.style.left = "10px"; L.style.top = (bar && !bar.hidden ? bar.offsetTop + bar.offsetHeight + 8 : 10) + "px"; return; }
  L.style.left = Math.max(0, Math.min(ilegPos.x, main.clientWidth - Math.min(L.offsetWidth, 120))) + "px";
  L.style.top = Math.max(0, Math.min(ilegPos.y, main.clientHeight - 30)) + "px";
}
window.addEventListener("mousemove", e => {
  if (!ilegDrag) return; const dx = e.clientX - ilegDrag.x0, dy = e.clientY - ilegDrag.y0;
  if (!ilegDrag.moved && Math.hypot(dx, dy) < 4) return;
  ilegDrag.moved = true; ilegPos = {x: ilegDrag.l + dx, y: ilegDrag.t + dy}; placeIleg();
});
window.addEventListener("mouseup", () => {
  if (!ilegDrag) return;
  if (ilegDrag.moved) { ilegDragged = true; const L = document.getElementById("ileg");
    ilegPos = {x: parseFloat(L.style.left) || 0, y: parseFloat(L.style.top) || 0};
    try { localStorage.setItem("relreview:ilegPos", JSON.stringify(ilegPos)); } catch (e) {} }
  ilegDrag = null;
});
window.addEventListener("resize", () => placeIleg());
function legend() {
  const L = document.getElementById("ileg"); if (!L) return; const cnt = {};
  Object.values(R).forEach(g => { if (g.cx && !(g.problems || []).length) return; const seen = new Set();
    [...(g.problems || []), ...(g.info || [])].forEach(p => { const c = catOf(p); if (c && !seen.has(c[0])) { seen.add(c[0]); cnt[c[0]] = (cnt[c[0]] || 0) + 1; } }); });
  const rows = PCAT.filter(c => cnt[c[0]]);
  L.hidden = notBeam;
  const shown = rows.filter(c => !iHide.has(c[0]) && !["unk", "alias", "dim", "note"].includes(c[0])).reduce((a, c) => a + cnt[c[0]], 0);
  L.className = iMin ? "min" : "";
  L.innerHTML = `<div class="ih" title="Click to show / hide this list · drag to move it · double-click to put it back"><span class="grip">⠿</span>${iMin ? "▸" : "▾"} Issues on this page: ${shown || "none ✓"}</div>` +
    rows.map(c => `<div class="ir${iHide.has(c[0]) ? " off" : ""}" data-k="${c[0]}" title="Click to hide / show this kind"><span class="sw" style="border-color:${c[2]};background:${c[2]}33"></span>${c[1]}<span class="ic">${cnt[c[0]]}</span></div>`).join("");
  const ih = L.querySelector(".ih");
  ih.onclick = () => { if (ilegDragged) { ilegDragged = false; return; } iMin = !iMin; try { localStorage.setItem("relreview:imin", iMin ? "1" : "0"); } catch (e) {} legend(); };
  ih.ondblclick = e => { e.stopPropagation(); ilegPos = null; try { localStorage.removeItem("relreview:ilegPos"); } catch (_) {} placeIleg(); };
  ih.onmousedown = e => { if (e.button !== 0) return; const r = L.getBoundingClientRect(), m = L.parentElement.getBoundingClientRect();
    ilegDrag = {x0: e.clientX, y0: e.clientY, l: r.left - m.left, t: r.top - m.top, moved: false}; e.preventDefault(); e.stopPropagation(); };
  placeIleg();
  L.querySelectorAll(".ir").forEach(r => r.onclick = () => { const k = r.dataset.k; iHide.has(k) ? iHide.delete(k) : iHide.add(k);
    try { localStorage.setItem("relreview:ihide", JSON.stringify([...iHide])); } catch (e) {} drawIssues(); });
}
// Cancelled (superseded) beams: Cancelled box -> BeamID; their regions get cx = the beam (as the page builder does)
const CANCEL_KEEP = ["no bar line", "line has no label", "no dimension line", "line has no dimension"];
const CANCEL_FOLLOW = [["BEAM", ["SIZE", "REBAR", "SMARK", "STITLE"]], ["SMARK", ["STITLE"]], ["STITLE", ["REBAR"]], ["REBAR", ["RLINE"]]];
function recomputeCancelled() {
  Object.values(R).forEach(g => { delete g.cx; });
  const todo = [];
  ORD.forEach(a => { if (isA(a, "CANCEL")) nbs(a).forEach(b => { if (isA(b, "BEAM") && !R[b].cx) { R[b].cx = b; todo.push(b); } }); });
  while (todo.length) { const a = todo.pop();
    CANCEL_FOLLOW.forEach(([frm, to]) => { if (isA(a, frm)) nbs(a).forEach(o => { if (!R[o].cx && to.some(s => isA(o, s))) { R[o].cx = R[a].cx; todo.push(o); } }); }); }
  ORD.forEach(r => { const b = (D.owner || {})[r];             // no relation to the beam: by position (as built)
    if (R[r] && !R[r].cx && b && R[b] && R[b].cx === b && !isA(r, "CANCEL")) R[r].cx = b; });
}
const byYX = (a, b) => (R[a].y ?? centre(R[a])[1]) - (R[b].y ?? centre(R[b])[1]) || (R[a].x ?? centre(R[a])[0]) - (R[b].x ?? centre(R[b])[0]);
const byXY = (a, b) => (R[a].x ?? centre(R[a])[0]) - (R[b].x ?? centre(R[b])[0]) || (R[a].y ?? centre(R[a])[1]) - (R[b].y ?? centre(R[b])[1]);
function rebuildItems() {             // the list on the left, from the current regions and relations
  const items = [], groups = [], used = new Set(), ids = ORD.filter(id => R[id]);
  const add = (kind, id, title, x = {}) => { items.push(Object.assign({kind, id, title}, x)); used.add(id); };
  const ttl = id => R[id].text || `${R[id].label} #${R[id].n}`;
  const relBeam = id => nbs(id).filter(o => isA(o, "BEAM"));
  add("ov", "ov:page", "Whole page - all annotations", {key: "page"});
  ids.filter(id => isA(id, "SHEET")).forEach(id => add("sheet", id, ttl(id)));
  ids.filter(id => isA(id, "CANCEL")).forEach(id => { const bn = relBeam(id).map(ttl);
    add("cancel", id, bn.length ? `${R[id].label} → ${bn.join(", ")}` : `${R[id].label} (no beam linked)`); });
  groups.push({name: "Page", start: 0, end: items.length});
  const beams = [...(D.beam_order || []).filter(b => R[b] && isA(b, "BEAM")), ...ids.filter(id => isA(id, "BEAM") && !(D.beam_order || []).includes(id))];
  for (const b of beams) {
    const start = items.length, name = ttl(b) + (R[b].cx ? " (cancelled)" : "");
    if (OV[b]) add("ov", "ov:" + b, "All annotations", {key: b, name});
    add("beam", b, name);
    ids.filter(o => isA(o, "SMARK") && !used.has(o) && (relBeam(o).includes(b) || (!relBeam(o).length && D.owner[o] === b))).sort(byYX).forEach(m => add("mark", m, "cut " + ttl(m)));
    const mem = D.beam_rel_used ? nbs(b).filter(o => isA(o, "REBAR")) : Object.keys(INF).filter(k => INF[k] === b && R[k]);
    mem.sort(byXY).forEach(m => { if (!used.has(m)) add("rebar", m, ttl(m)); });
    groups.push({name, start, end: items.length});
  }
  for (const k of D.sec_keys || []) {
    if (!OV[k]) continue;
    const start = items.length;
    add("ov", "ov:" + k, "All annotations", {key: k, name: k.slice(4)});
    ids.filter(t => isA(t, "STITLE") && R[t].sec_key === k && !used.has(t)).forEach(t => add("section", t, ttl(t)));
    ids.filter(o => isA(o, "REBAR") && !used.has(o) && !relBeam(o).length && D.owner[o] === k).sort(byXY).forEach(m => add("rebar", m, ttl(m)));
    groups.push({name: k.slice(4), start, end: items.length});
  }
  const grp = (name, list, kind) => { const start = items.length; list.forEach(r => add(typeof kind === "function" ? kind(r) : kind, r, ttl(r))); if (items.length > start) groups.push({name, start, end: items.length}); };
  grp("Drawn twice - repeats (not counted)", ids.filter(r => R[r].repeat_of && R[R[r].repeat_of] && !used.has(r)).sort(byYX), "repeat");
  grp("Labels without a beam", ids.filter(r => (isA(r, "REBAR") || isA(r, "SMARK") || isA(r, "STITLE")) && !used.has(r)).sort(byYX),
      r => isA(r, "REBAR") ? "rebar" : isA(r, "SMARK") ? "mark" : "section");
  grp("Dimensions", ids.filter(r => isA(r, "DIM") && !used.has(r)).sort(byYX), "dim");
  grp("Lines without a label", ids.filter(r => !used.has(r) && R[r].kind === "line" && (R[r].problems || []).length), "line");
  grp("New - not placed yet", ids.filter(r => !used.has(r) && R[r]._new), r => R[r].kind === "box" ? "rebar" : "line");
  D.items = items; D.groups = groups;
}
function ovOf(key) {                  // regions of a beam/section/page overview, as they are now
  const o = OV[key] || {ids: [], missing: [], repeats: []};
  if (key === "page") return Object.assign({}, o, {key, ids: ORD.filter(id => R[id])});
  const ids = new Set(o.ids.filter(id => R[id] && !(isA(id, "REBAR") && nbs(id).some(b => isA(b, "BEAM") && b !== key) && R[key])));
  if (R[key]) nbs(key).forEach(m => { if (isA(m, "REBAR") || isA(m, "SMARK") || isA(m, "SIZE")) { ids.add(m); nbs(m).forEach(x => { if (R[x].kind === "line" && !R[x].closed) ids.add(x); }); } });
  return Object.assign({}, o, {key, ids: [...ids]});
}
function recompute() {
  rebuildNb(); recomputeRepeats(); recomputeCancelled();
  D.beam_rel_used = D.rels.some(([a, b]) => (isA(a, "BEAM") && isA(b, "REBAR")) || (isA(b, "BEAM") && isA(a, "REBAR")));
  Object.keys(R).forEach(id => { R[id].problems = problemsOf(id); R[id].info = infoOf(id); });
  const curId = cur >= 0 && D.items[cur] ? D.items[cur].id : null;
  rebuildItems();
  cur = curId ? D.items.findIndex(it => it.id === curId) : -1;
}

// ---- history (undo / redo) and autosave in this browser ----
const editable = g => ({id: g.id, n: g.n, label: g.label, kind: g.kind, x: g.x ?? null, y: g.y ?? null, w: g.w ?? null, h: g.h ?? null,
  pts: g.pts || null, closed: !!g.closed, note: g.note || "", text: g.text ?? null, _new: !!g._new, _dirty: !!g._dirty, _noteSet: !!g._noteSet, _mmpp: g._mmpp ?? null});
const snapshot = () => JSON.stringify({ord: ORD, regs: ORD.filter(id => R[id]).map(id => editable(R[id])), rels: D.rels, sel: selId, notBeam, consultant, testPage});
function restore(s) {
  const st = JSON.parse(s), keep = {};
  Object.keys(R).forEach(id => { keep[id] = R[id]; delete R[id]; });
  st.regs.forEach(e => { const g = Object.assign(keep[e.id] || {}, e); ["x", "y", "w", "h", "pts", "text", "_mmpp"].forEach(k => { if (g[k] === null) delete g[k]; }); R[e.id] = g; if (g.kind === "line") lineDerived(g); });
  ORD = st.ord.filter(id => R[id]); D.rels = st.rels; selId = st.sel && R[st.sel] ? st.sel : null;
  notBeam = st.notBeam === undefined ? !!D.task_meta.not_beam : !!st.notBeam;
  consultant = st.consultant === undefined ? (D.task_meta.consultant || "") : st.consultant;
  testPage = st.testPage === undefined ? !!D.task_meta.test : !!st.testPage;
}
const isSaved = () => savedVer === ver;
function autosave() { try { if (nEdits) localStorage.setItem(EKEY, JSON.stringify({stamp: D.task_meta.stamp, t: Date.now(), n: nEdits, saved: isSaved(), file: savedFile, state: snapshot()})); else localStorage.removeItem(EKEY); } catch (e) {}
  saveTaskJSON(); }
// this page's annotation as Label Studio JSON, kept in this browser while it has edits: the review index exports it (tick the pages, Export)
const TKEY = "relreview-task:" + D.task + ":" + D.image;
function saveTaskJSON() {
  try { if (!nEdits) { localStorage.removeItem(TKEY); return; }
    localStorage.setItem(TKEY, JSON.stringify({stamp: D.task_meta.stamp, t: Date.now(), file: D.image.replace(/\.[^.]+$/, "") + "_task" + D.task_meta.id + "_edited.json", task: currentTask()})); }
  catch (e) { try { localStorage.removeItem(TKEY); } catch (_) {}
    if (!saveTaskJSON.warned) { saveTaskJSON.warned = true; flash("This browser's storage is full: the index can't export this page - use Save JSON here"); } }
}
function afterEdit() { recompute(); autosave(); applyNotBeam(); applyTags();
  if (notBeam && editing) { setEditing(false); return; }             // an undo brought the tick back
  list(); if (editing) drawEdit(); }
function edit(fn) { const before = snapshot(); fn(); hist.push(before); if (hist.length > 80) hist.shift(); fut = []; nEdits++; ver++; afterEdit(); }
function undo() { if (!hist.length) return; fut.push(snapshot()); restore(hist.pop()); nEdits = Math.max(0, nEdits - 1); ver++; afterEdit(); }
function redo() { if (!fut.length) return; hist.push(snapshot()); restore(fut.pop()); nEdits++; ver++; afterEdit(); }

// ---- edits ----
function addBox(x, y, w, h) { edit(() => { const id = newId(); R[id] = {id, n: nextN(), label: newLbl.box, kind: "box", x, y, w, h, note: "", _new: true, _dirty: true, problems: []}; derived(R[id]); ORD.push(id); selId = id; }); }
function addLine(pts, linkTo) { edit(() => { const id = newId(); R[id] = {id, n: nextN(), label: newLbl.line, kind: "line", pts, closed: false, note: "", _new: true, _dirty: true, problems: []}; derived(R[id]); ORD.push(id);
  if (linkTo && R[linkTo]) D.rels.push(rankOf(linkTo) > rankOf(id) ? [id, linkTo] : [linkTo, id]); selId = id; }); }
function delRegion(id) { if (!R[id]) return; edit(() => { D.rels = D.rels.filter(([a, b]) => a !== id && b !== id); delete R[id]; ORD = ORD.filter(x => x !== id); if (selId === id) selId = null; }); }
const RANK = {CANCEL: -1, BEAM: 0, SIZE: 1, SMARK: 2, STITLE: 3, REBAR: 4, DIM: 4, RLINE: 5, DLINE: 5};
const rankOf = id => { for (const k in RANK) if (isA(id, k)) return RANK[k]; return 9; };
function addRel(a, b) {
  if (!R[a] || !R[b] || a === b) return;
  if (D.rels.some(([x, y]) => (x === a && y === b) || (x === b && y === a))) { flash("These two are linked already"); return; }
  if (rankOf(a) > rankOf(b)) [a, b] = [b, a];     // same direction as the backend's (BeamID→RebarSize, RebarSize→RebarLine ...)
  edit(() => D.rels.push([a, b]));
}
function delRel(a, b) { edit(() => { D.rels = D.rels.filter(([x, y]) => !((x === a && y === b) || (x === b && y === a))); }); }
function setLabel(id, lab) { if (R[id] && R[id].label !== lab) edit(() => { R[id].label = lab; R[id]._dirty = true; }); }

// ---- review mode: click a box or line on the drawing to see it and change its label (e.g. RebarLine -> DimensionLine) ----
function regionAt(p) {               // the region under a click: inside a box, or within a few screen px of a line
  const tol = sw(7); let best = null;
  for (const id of ORD) {
    const g = R[id]; if (!g) continue; let d;
    if (g.kind === "box") d = Math.hypot(Math.max(g.x - p.x, 0, p.x - g.x - g.w), Math.max(g.y - p.y, 0, p.y - g.y - g.h));
    else { const o = openPts(g), P = g.closed ? [...o, o[0]] : o; d = Infinity;
      for (let i = 0; i + 1 < P.length; i++) { const a = P[i], b = P[i + 1], dx = b[0] - a[0], dy = b[1] - a[1], L = dx * dx + dy * dy;
        const t = L ? Math.max(0, Math.min(1, ((p.x - a[0]) * dx + (p.y - a[1]) * dy) / L)) : 0;
        d = Math.min(d, Math.hypot(a[0] + t * dx - p.x, a[1] + t * dy - p.y)); } }
    if (d > tol) continue;
    const area = g.kind === "box" ? g.w * g.h : Infinity;          // boxes inside boxes: the smaller one
    if (!best || d < best.d - 1e-9 || (Math.abs(d - best.d) <= 1e-9 && area < best.area)) best = {id, d, area};
  }
  return best ? best.id : null;
}
function reviewClick(e) { if (notBeam) return; const id = regionAt(toSvg(e)); if (id) showRegion(id); }
function showRegion(id) {
  const g = R[id]; if (!g) return;
  const hl = document.getElementById("hl"), lk = document.getElementById("links"); hl.innerHTML = ""; lk.innerHTML = "";
  nbs(id).forEach(o => { if (R[o]) shape(R[o], hl, {c: col(R[o].label), w: sw(2.2), o: 0.85, fill: R[o].kind === "box" ? col(R[o].label) + "22" : "none"}).setAttribute("pointer-events", "none"); });
  shape(g, hl, {c: "#fff", w: sw(7), o: 0.9}).setAttribute("pointer-events", "none");
  shape(g, hl, {c: col(g.label), w: sw(3.2), o: 1, fill: g.kind === "box" ? col(g.label) + "30" : "none"}).setAttribute("pointer-events", "none");
  const opts = (g.kind === "box" ? D.labels.box : D.labels.line).slice(); if (!opts.includes(g.label)) opts.unshift(g.label);
  const inf = document.getElementById("info");
  inf.innerHTML = `<h3>${esc(g.label)} #${g.n}${g.text ? " “" + esc(g.text) + "”" : ""}${g._new ? " · new" : g._dirty ? " · changed" : ""}</h3>` +
    `<div class="row" style="margin:4px 0">Label <select id="rvlab">${opts.map(l => `<option${l === g.label ? " selected" : ""}>${esc(l)}</option>`).join("")}</select> ` +
    `<button id="rved" title="Edit this region: move, redraw, link, delete (E)">✎ Edit mode</button>` +
    (nEdits ? ` <button id="rvsave" title="Save the annotation as Label Studio JSON">💾 Save JSON</button>` : "") + `</div>` +
    (nbs(id).length ? nbs(id).map(o => `<div>↔ ${esc(R[o].label)} #${R[o].n}${R[o].text ? " “" + esc(R[o].text) + "”" : ""}${R[o].len_mm && !R[o].closed ? " ≈ " + R[o].len_mm.toLocaleString() + " mm" : ""}</div>`).join("") : `<div class="row">no relations</div>`) +
    ((g.problems || []).length ? `<div class="pb">⚠ ${g.problems.map(esc).join(" · ")}</div>` : "") +
    `<div class="row" style="margin-top:6px">Clicked on the drawing. A new label is an edit: pick the old one to undo it, ` +
    `<b>💾 Save JSON</b> to keep it in a file${nEdits ? ` (${nEdits} edit${nEdits > 1 ? "s" : ""} on this page${isSaved() ? ", saved" : ", not saved yet"})` : ""}.</div>`;
  inf.hidden = false; decorateInfo(); placeInfo();
  document.getElementById("rvlab").onchange = ev => {
    const lab = ev.target.value; setLabel(id, lab);
    if (cur >= 0 && cur < D.items.length) select(cur, false); else drawAll(false);
    showRegion(id); flash(`${R[id] ? "#" + R[id].n : "Region"} is now ${lab} - Save JSON to keep it`);
  };
  document.getElementById("rved").onclick = () => { setEditing(true); selId = id; syncList(); drawEdit(); };
  const sv = document.getElementById("rvsave"); if (sv) sv.onclick = () => { saveJSON(); showRegion(id); };
}
function insertVertex(id, p) {
  const g = R[id], pts = openPts(g), seq = g.closed ? [...pts, pts[0]] : pts; let best = null;
  for (let i = 0; i + 1 < seq.length; i++) { const a = seq[i], b = seq[i + 1], dx = b[0] - a[0], dy = b[1] - a[1], L = dx * dx + dy * dy;
    const t = L ? Math.max(0, Math.min(1, ((p.x - a[0]) * dx + (p.y - a[1]) * dy) / L)) : 0, d = Math.hypot(a[0] + t * dx - p.x, a[1] + t * dy - p.y);
    if (!best || d < best[0]) best = [d, i]; }
  if (best) edit(() => { pts.splice(best[1] + 1, 0, [p.x, p.y]); setPts(g, pts); g._dirty = true; derived(g); });
}
function delVertex(id, k) { const g = R[id], pts = openPts(g); if (pts.length <= 2) { flash("A line needs at least 2 points"); return; }
  edit(() => { pts.splice(k, 1); setPts(g, pts); g._dirty = true; derived(g); }); }
function flash(msg) { const f = document.getElementById("edmsg"); if (!f) return; f.textContent = msg; f.hidden = false; clearTimeout(flash.t); flash.t = setTimeout(() => { f.hidden = true; }, 2200); }

// ---- drawing in edit mode ----
function drawEdit() {
  const all = document.getElementById("all"), hl = document.getElementById("hl"), lk = document.getElementById("links");
  all.innerHTML = ""; hl.innerHTML = ""; lk.innerHTML = "";
  ORD.forEach(id => { const g = R[id]; if (!g) return;
    if (g.kind === "box") el("rect", {x: g.x, y: g.y, width: g.w, height: g.h, fill: col(g.label) + "14", stroke: col(g.label), "stroke-width": sw(1.4), "data-id": id, class: "ed"}, all);
    else { el("polyline", {points: P(g.pts), fill: "none", stroke: col(g.label), "stroke-width": sw(1.8), "stroke-linejoin": "round", "data-id": id, class: "ed"}, all);
           el("polyline", {points: P(g.pts), fill: "none", stroke: "transparent", "stroke-width": sw(11), "data-id": id, class: "ed"}, all); } });
  if (showAll) drawLinks(lk);
  if (selId && R[selId]) {
    const g = R[selId], c0 = centre(g), hs = sw(9);
    nbs(selId).forEach(o => { const q = nearest(R[o], c0), p = nearest(g, q); el("line", {x1: p[0], y1: p[1], x2: q[0], y2: q[1], stroke: "#1a73e8", "stroke-width": sw(1.6), "stroke-dasharray": `${sw(6)} ${sw(4)}`}, lk); });
    shape(g, hl, {c: "#fff", w: sw(6), o: 0.9}).setAttribute("pointer-events", "none");
    const s = shape(g, hl, {c: col(g.label), w: sw(3), o: 1, fill: g.kind === "box" ? col(g.label) + "30" : "none"}); s.setAttribute("data-id", selId); s.setAttribute("class", "ed sel");
    if (g.kind === "box") [["nw", g.x, g.y], ["ne", g.x + g.w, g.y], ["sw", g.x, g.y + g.h], ["se", g.x + g.w, g.y + g.h]].forEach(([h, x, y]) =>
      el("rect", {x: x - hs / 2, y: y - hs / 2, width: hs, height: hs, fill: "#fff", stroke: "#1a73e8", "stroke-width": sw(1.5), "data-h": h, "data-id": selId, class: "hd " + h}, hl));
    else openPts(g).forEach((p, k) => el("circle", {cx: p[0], cy: p[1], r: hs / 2, fill: "#fff", stroke: "#1a73e8", "stroke-width": sw(1.5), "data-h": k, "data-id": selId, class: "hd"}, hl));
  }
  if (ed && ed.kind === "newbox") { const x0 = Math.min(ed.p0.x, ed.p1.x), y0 = Math.min(ed.p0.y, ed.p1.y);
    el("rect", {x: x0, y: y0, width: Math.abs(ed.p1.x - ed.p0.x), height: Math.abs(ed.p1.y - ed.p0.y), fill: col(newLbl.box) + "22", stroke: col(newLbl.box), "stroke-width": sw(2), "stroke-dasharray": `${sw(5)} ${sw(3)}`, "pointer-events": "none"}, hl); }
  if (draft && draft.pts.length) { let pts = draft.hover ? [...draft.pts, [draft.hover.x, draft.hover.y]] : draft.pts;
    const g0 = draft.id && R[draft.id], lc = col(g0 ? g0.label : newLbl.line);
    if (draft.mode === "extend" && g0) { const o = openPts(g0); pts = [draft.atStart ? o[0] : o[o.length - 1], ...pts]; }
    if (draft.mode === "redraw" && g0) el("polyline", {points: P(g0.pts), fill: "none", stroke: "#888", "stroke-width": sw(2), "stroke-dasharray": `${sw(2)} ${sw(3)}`, "pointer-events": "none"}, hl);
    el("polyline", {points: P(pts), fill: "none", stroke: "#fff", "stroke-width": sw(5), "pointer-events": "none"}, hl);
    el("polyline", {points: P(pts), fill: "none", stroke: lc, "stroke-width": sw(2.5), "stroke-dasharray": `${sw(5)} ${sw(3)}`, "pointer-events": "none"}, hl);
    draft.pts.forEach(p => el("circle", {cx: p[0], cy: p[1], r: sw(4), fill: lc, stroke: "#fff", "stroke-width": sw(1.2), "pointer-events": "none"}, hl)); }
  svg.style.cursor = ["box", "line", "extend", "redraw"].includes(tool) ? DRAW_CURSOR : tool === "link" ? "alias" : "default";
  document.querySelectorAll("#edbar [data-t]").forEach(b => b.classList.toggle("on", b.dataset.t === tool));
  document.getElementById("undo").disabled = !hist.length; document.getElementById("redo").disabled = !fut.length;
  document.getElementById("edcount").textContent = nEdits ? `${nEdits} edit${nEdits > 1 ? "s" : ""}${isSaved() ? " · saved" : ""}` : "";
  edInfo();
}
function edInfo() {
  const inf = document.getElementById("info"), g = selId && R[selId]; let h;
  if (tool === "box") h = `<h3>New box</h3><div class="row">Drag a rectangle on the drawing. Label: <b>${newLbl.box}</b> (choose it in the bar above). Esc cancels.</div>`;
  else if (tool === "line") h = `<h3>New line</h3><div class="row">Click the points of the line; double-click or Enter / Space to finish (${draft ? draft.pts.length : 0} points so far). Label: <b>${newLbl.line}</b>` +
    (drawLink && R[drawLink] ? `, linked to <b>${R[drawLink].label} #${R[drawLink].n}${R[drawLink].text ? " “" + R[drawLink].text + "”" : ""}</b>` : "") + `. Esc cancels.</div>`;
  else if (tool === "extend") h = `<h3>Extend ${g ? g.label + " #" + g.n : ""}</h3><div class="row">Click the new points: the line goes on from its end nearer your first click. Double-click or Enter / Space to finish (${draft ? draft.pts.length : 0} points so far). Esc cancels.</div>`;
  else if (tool === "redraw") h = `<h3>Redraw ${g ? g.label + " #" + g.n : ""}</h3><div class="row">Click the points of the line as it should be (the old one is shown grey dotted). Double-click or Enter / Space to finish (${draft ? draft.pts.length : 0} points so far). It keeps its label and links. Esc cancels.</div>`;
  else if (tool === "link") h = `<h3>Link</h3><div class="row">Click the region to link <b>${g ? g.label + " #" + g.n : ""}</b> to. Esc cancels.</div>`;
  else if (g) {
    const opts = g.kind === "box" ? D.labels.box : D.labels.line;
    h = `<h3>${g.label} #${g.n}${g.text ? " “" + g.text + "”" : ""}${g._new ? " · new" : g._dirty ? " · changed" : ""}</h3>` +
      `<div class="row" style="margin:4px 0">Label <select id="edlab">${opts.map(l => `<option${l === g.label ? " selected" : ""}>${l}</option>`).join("")}</select> ` +
      `<button id="edlk">↔ Link to… (R)</button> ` + (g.kind === "line" ? `<button id="edext">⤳ Extend (G)</button> <button id="edrd">✎ Redraw (W)</button> ` : "") +
      `<button id="eddl">Delete (Del)</button></div>` +
      (nbs(selId).length ? nbs(selId).map(o => `<div>↔ ${R[o].label} #${R[o].n}${R[o].text ? " “" + R[o].text + "”" : ""}${R[o].len_mm && !R[o].closed ? " ≈ " + R[o].len_mm.toLocaleString() + " mm" : ""} <a href="#" class="unlink" data-o="${o}" title="Remove this relation">✕ remove</a></div>`).join("") : `<div class="row">no relations</div>`) +
      ((g.problems || []).length ? `<div class="pb">⚠ ${g.problems.join(" · ")}</div>` : "") +
      `<div class="row" style="margin-top:4px">${g.kind === "box" ? "Drag to move · drag a corner to resize · <b>D</b> draws a new line linked to it" : "Drag the line to move it · drag a point · double-click the line to add a point · Alt+click a point to remove it · <b>G</b> extend from an end · <b>W</b> redraw (keeps label and links) · wrong label: pick another above"} · Shift+click another region to link it</div>`;
  } else h = `<h3>Edit mode</h3><div class="row">Click a region to select it. <b>B</b> new box · <b>D</b> new line · <b>G</b> extend a line · <b>W</b> redraw a line · <b>R</b> link · <b>Space</b>/<b>Enter</b> finish, or repeat the last command · <b>Del</b> delete · <b>Ctrl+Z</b> undo · <b>Ctrl+S</b> save JSON · <b>E</b> back to review. Drag empty space to pan; wheel to zoom.</div>`;
  h += `<div class="row" style="margin-top:6px">${nEdits ? `${nEdits} edit${nEdits > 1 ? "s" : ""}, kept in this browser${isSaved() ? " and saved to " + esc(savedFile) : " - <b>Save JSON</b> to keep them in a file"}` : "No edits yet."}</div>`;
  inf.hidden = false; inf.innerHTML = h; decorateInfo();
  const s = document.getElementById("edlab"); if (s) s.onchange = () => setLabel(selId, s.value);
  const lkb = document.getElementById("edlk"); if (lkb) lkb.onclick = () => setTool("link");
  const db = document.getElementById("eddl"); if (db) db.onclick = () => delRegion(selId);
  const xb = document.getElementById("edext"); if (xb) xb.onclick = () => setTool("extend");
  const rb = document.getElementById("edrd"); if (rb) rb.onclick = () => setTool("redraw");
  inf.querySelectorAll(".unlink").forEach(a => a.onclick = ev => { ev.preventDefault(); delRel(selId, a.dataset.o); });
}
// a bold crosshair for drawing: black with a white outline, seen on any part of the drawing
const DRAW_CURSOR = `url("data:image/svg+xml,${encodeURIComponent('<svg xmlns="http://www.w3.org/2000/svg" width="40" height="40"><g fill="none" stroke-linecap="square">' +
  '<path d="M20 2v13M20 25v13M2 20h13M25 20h13" stroke="#fff" stroke-width="5"/><path d="M20 2v13M20 25v13M2 20h13M25 20h13" stroke="#d00" stroke-width="2.2"/>' +
  '<circle cx="20" cy="20" r="3.2" stroke="#fff" stroke-width="3"/><circle cx="20" cy="20" r="3.2" stroke="#d00" stroke-width="1.4"/></g></svg>')}") 20 20, crosshair`;
let drawLink = null;                   // the label a new line gets linked to (selected when D was pressed)
const isLine = id => !!(id && R[id] && R[id].kind === "line");
function setTool(t) {
  if (t === "link" && !selId) { flash("Select a region first"); return; }
  if ((t === "extend" || t === "redraw") && !isLine(selId)) { flash("Select a line first"); return; }
  if (t === "extend" && R[selId].closed) { flash("A closed line can't be extended: Redraw it (W)"); return; }
  if (t === "line") drawLink = selId && R[selId] && R[selId].kind === "box" ? selId : null;
  if (t !== "select") lastTool = t;
  tool = t; draft = null; ed = null; clearSnap(); drawEdit(); }
function finishDraft() {
  if (!draft) return;
  const pts = []; draft.pts.forEach(p => { if (!pts.length || Math.hypot(p[0] - pts[pts.length - 1][0], p[1] - pts[pts.length - 1][1]) > sw(3)) pts.push(p); });
  const d = draft, g = d.id && R[d.id]; draft = null; tool = "select"; clearSnap();
  if (d.mode === "extend" && g && pts.length) {
    edit(() => { const o = openPts(g); setPts(g, d.atStart ? [...pts.reverse(), ...o] : [...o, ...pts]); g._dirty = true; derived(g); selId = g.id; }); return; }
  if (d.mode === "redraw" && g && pts.length >= 2) {
    edit(() => { g.closed = false; g.pts = pts; g._dirty = true; derived(g); selId = g.id; }); return; }
  if (d.mode === "line" && pts.length >= 2) { addLine(pts, d.link); return; }
  drawEdit();
}
function applyDrag(e2, p) {
  const g = R[e2.id], o = e2.orig, dx = p.x - e2.p0.x, dy = p.y - e2.p0.y;
  if (e2.kind === "move") { if (g.kind === "box") { g.x = o.x + dx; g.y = o.y + dy; } else g.pts = o.pts.map(q => [q[0] + dx, q[1] + dy]); }
  else if (g.kind === "box") {
    let x0 = o.x, y0 = o.y, x1 = o.x + o.w, y1 = o.y + o.h;
    if (e2.h.includes("w")) x0 = p.x; else x1 = p.x;
    if (e2.h.includes("n")) y0 = p.y; else y1 = p.y;
    g.x = Math.min(x0, x1); g.y = Math.min(y0, y1); g.w = Math.abs(x1 - x0); g.h = Math.abs(y1 - y0);
  } else { const pts = (o.closed ? o.pts.slice(0, -1) : o.pts.slice()).map(q => q.slice()); pts[+e2.h] = [p.x, p.y]; setPts(g, pts); }
}
let lastClick = null;
function edDown(e) {                  // true: the edit tools took the click (no panning)
  const p = toSvg(e), t = e.target, id = t.dataset ? t.dataset.id : null, h = t.dataset ? t.dataset.h : undefined, now = Date.now();
  const dbl = lastClick && now - lastClick.t < 450 && Math.hypot(e.clientX - lastClick.x, e.clientY - lastClick.y) < 6;
  lastClick = dbl ? null : {t: now, x: e.clientX, y: e.clientY};
  if (tool === "box") { ed = {kind: "newbox", p0: p, p1: p}; return true; }
  if (tool === "line" || tool === "extend" || tool === "redraw") {
    if (dbl && draft) { finishDraft(); return true; }
    if (!draft) { draft = {pts: [], mode: tool, id: tool === "line" ? null : selId, link: tool === "line" && R[drawLink] ? drawLink : null};
      if (tool === "extend") { const o = openPts(R[selId]), d0 = Math.hypot(p.x - o[0][0], p.y - o[0][1]), d1 = Math.hypot(p.x - o[o.length - 1][0], p.y - o[o.length - 1][1]); draft.atStart = d0 < d1; } }
    const q = snapPt(p, draftFrom(), null); draft.pts.push([q.x, q.y]); draft.hover = q; drawEdit(); drawSnap(); return true; }
  if (dbl && id && id === selId && R[id] && R[id].kind === "line" && h === undefined && tool === "select") { insertVertex(id, p); return true; }
  if (tool === "link") { if (id && R[id] && selId && id !== selId) { tool = "select"; addRel(selId, id); } else { tool = "select"; drawEdit(); } return true; }
  if (h !== undefined && id === selId && R[id]) {
    if (e.altKey && R[id].kind === "line") { delVertex(id, +h); return true; }
    ed = {kind: "handle", id, h, p0: p, before: snapshot(), orig: JSON.parse(JSON.stringify(editable(R[id]))), moved: false}; return true;
  }
  if (id && R[id]) {
    if (e.shiftKey && selId && id !== selId) { addRel(selId, id); return true; }
    if (selId !== id) { selId = id; syncList(); drawEdit(); }
    ed = {kind: "move", id, p0: p, before: snapshot(), orig: JSON.parse(JSON.stringify(editable(R[id]))), moved: false}; return true;
  }
  ed = {kind: "empty", sx: e.clientX, sy: e.clientY}; return false;
}
function edMove(e) {
  if (ed && (ed.kind === "move" || ed.kind === "handle")) { let p = toSvg(e);
    if (!ed.moved && Math.hypot(p.x - ed.p0.x, p.y - ed.p0.y) < sw(3)) return true;
    if (ed.kind === "move") { if (orthoNow()) { if (Math.abs(p.x - ed.p0.x) >= Math.abs(p.y - ed.p0.y)) p.y = ed.p0.y; else p.x = ed.p0.x; } snapHit = null; }
    else if (ed.orig.kind === "line") { const o = ed.orig.closed ? ed.orig.pts.slice(0, -1) : ed.orig.pts, k = +ed.h;      // a line's point: from its neighbour
      p = snapPt(p, k > 0 ? o[k - 1] : o.length > 1 ? (ed.orig.closed ? o[o.length - 1] : o[1]) : null, ed.id); }
    else p = snapPt(p, null, ed.id);    // a box corner: ends and crossings only
    ed.moved = true; applyDrag(ed, p); drawEdit(); drawSnap(); return true; }
  if (ed && ed.kind === "newbox") { ed.p1 = toSvg(e); drawEdit(); return true; }
  if (DRAW_TOOLS.includes(tool)) {      // drawing: the snap marker follows the cursor, before the first click too
    let from = draftFrom();
    if (!draft && tool === "extend" && isLine(selId)) { const o = openPts(R[selId]), p0 = toSvg(e);
      from = Math.hypot(p0.x - o[0][0], p0.y - o[0][1]) < Math.hypot(p0.x - o[o.length - 1][0], p0.y - o[o.length - 1][1]) ? o[0] : o[o.length - 1]; }
    const q = snapPt(toSvg(e), from, null);
    if (draft) { draft.hover = q; drawEdit(); }
    drawSnap();
  }
  return false;
}
function edUp(e) {
  if (!ed) return;
  const e2 = ed; ed = null; if (e2.kind === "move" || e2.kind === "handle") clearSnap();
  if (e2.kind === "newbox") { const x0 = Math.min(e2.p0.x, e2.p1.x), y0 = Math.min(e2.p0.y, e2.p1.y), w = Math.abs(e2.p1.x - e2.p0.x), h = Math.abs(e2.p1.y - e2.p0.y);
    tool = "select"; if (w > sw(4) && h > sw(4)) addBox(x0, y0, w, h); else drawEdit(); }
  else if ((e2.kind === "move" || e2.kind === "handle") && e2.moved) {
    const g = R[e2.id]; g._dirty = true; derived(g); hist.push(e2.before); if (hist.length > 80) hist.shift(); fut = []; nEdits++; ver++; afterEdit(); }
  else if (e2.kind === "empty" && Math.hypot(e.clientX - e2.sx, e.clientY - e2.sy) < 4 && selId) { selId = null; syncList(); drawEdit(); }
}
function syncList() {
  const i = selId ? D.items.findIndex(it => it.id === selId) : -1;
  if (i >= 0) { cur = i; reveal(i); }
  document.querySelectorAll(".it").forEach(x => x.classList.toggle("on", +x.dataset.i === cur && i >= 0));
  const e = i >= 0 && document.querySelector(`.it[data-i="${i}"]`); if (e) e.scrollIntoView({block: "nearest"});
}
function editSelectItem(i, zoom) {
  cur = i; const it = D.items[i];
  if (it.kind === "ov") { selId = null;
    if (zoom) { const o = ovOf(it.key); if (o.ids.length) { const ps = bbox(o.ids); zoomTo(...ps, 1.12); } } }
  else if (R[it.id]) { selId = it.id;
    if (zoom) { const near = [it.id, ...nbs(it.id).filter(o => !R[o].closed && Math.hypot(centre(R[o])[0] - centre(R[it.id])[0], centre(R[o])[1] - centre(R[it.id])[1]) < 0.25 * D.W)];
      const [x0, y0, x1, y1] = bbox(near), w = Math.max(x1 - x0, D.W * 0.05), h = Math.max(y1 - y0, D.H * 0.04), cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
      zoomTo(cx - w * 0.8, cy - h * 0.9, cx + w * 0.8, cy + h * 0.9, 1); } }
  document.querySelectorAll(".it").forEach(x => x.classList.toggle("on", +x.dataset.i === i));
  const e = document.querySelector(`.it[data-i="${i}"]`); if (e) e.scrollIntoView({block: "nearest"});
  drawEdit();
}
function setEditing(on) {
  editing = on; tool = "select"; draft = null; ed = null; clearSnap();
  document.body.classList.toggle("editing", on); document.getElementById("editbtn").classList.toggle("on", on);
  document.getElementById("edbar").hidden = !on;
  if (on) { selId = cur >= 0 && D.items[cur] && R[D.items[cur].id] ? D.items[cur].id : null; drawEdit(); }
  else { recompute(); list(); svg.style.cursor = "grab"; if (cur >= 0 && cur < D.items.length) select(cur, false); else { cur = -1; drawAll(false); document.getElementById("hl").innerHTML = ""; document.getElementById("links").innerHTML = ""; document.getElementById("info").innerHTML = ""; } }
}

// ---- save as Label Studio JSON (the same format as Export > JSON) ----
function buildResult() {
  const W = D.W, H = D.H, tm = D.task_meta, out = [];
  const nm = t => (tm.names && tm.names[t]) || (t === "rectanglelabels" ? ["label", "image"] : ["vec", "image"]);
  for (const id of ORD) {
    const g = R[id]; if (!g) continue;
    const raw = D.raw && D.raw[id];
    if (raw && !g._dirty) { out.push(raw.main, ...raw.side); continue; }
    const base = raw ? JSON.parse(JSON.stringify(raw.main)) : {};
    let item;
    if (g.kind === "box") {
      const t = "rectanglelabels";
      item = Object.assign(base, {id, type: t, from_name: base.from_name || nm(t)[0], to_name: base.to_name || nm(t)[1], original_width: tm.W0, original_height: tm.H0, image_rotation: 0,
        value: {x: g.x / W * 100, y: g.y / H * 100, width: g.w / W * 100, height: g.h / H * 100, rotation: 0, rectanglelabels: [g.label]}});
    } else if (raw && raw.main.type === "polygonlabels") {
      item = Object.assign(base, {value: {points: openPts(g).map(p => [p[0] / W * 100, p[1] / H * 100]), polygonlabels: [g.label]}});
    } else {
      const t = "vectorlabels"; let prev = null;
      const verts = openPts(g).map(p => { const vid = newId(); const v = {id: vid, x: p[0] / W * 100, y: p[1] / H * 100, prevPointId: prev, isBezier: false}; prev = vid; return v; });
      item = Object.assign(base, {id, type: t, from_name: base.from_name || nm(t)[0], to_name: base.to_name || nm(t)[1], original_width: tm.W0, original_height: tm.H0, image_rotation: 0,
        value: {vertices: verts, closed: !!g.closed, vectorlabels: [g.label]}});
    }
    delete item.score;
    if (!raw && g.note) item.meta = {text: [g.note]};          // a new region's note (e.g. 'added from the AI check')
    else if (raw && g._noteSet) { if (g.note) item.meta = Object.assign({}, item.meta, {text: [g.note]}); else if (item.meta) delete item.meta.text; }   // a note from a loaded JSON
    out.push(item, ...(raw ? raw.side : []));
  }
  const rawRel = {}; (D.raw_rels || []).forEach(r => { rawRel[r.from_id + ">" + r.to_id] = r; });
  D.rels.forEach(([a, b]) => out.push(rawRel[a + ">" + b] || {type: "relation", from_id: a, to_id: b, direction: "right"}));
  (tm.extras || []).forEach(x => out.push(x));
  return out;
}
function currentTask() {             // this page's annotation as it is now (your edits included), in Label Studio JSON
  const tm = D.task_meta;
  const ann = notBeam ? {result: [], was_cancelled: true, ground_truth: false}    // not a beam drawing: cancelled, no regions
                      : {result: buildResult(), was_cancelled: false, ground_truth: false};
  if (tm.ann_id != null) ann.id = tm.ann_id;
  return {id: tm.id, data: {image: tm.image}, annotations: [ann], predictions: [],
          meta: {page_type: notBeam ? "not a beam drawing" : "beam drawing", consultant: consultant.trim(), split: testPage ? "test" : "train",
                 edited_in: "review_relations page", edited_at: new Date().toISOString(), based_on: tm.source, edits: nEdits}};
}
function saveJSON() {
  const tm = D.task_meta, task = currentTask();
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([JSON.stringify([task], null, 1)], {type: "application/json"}));
  a.download = `${D.image.replace(/\.[^.]+$/, "")}_task${tm.id}_edited.json`;
  document.body.appendChild(a); a.click(); setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 1500);
  savedVer = ver; savedFile = a.download; autosave(); flash("Saved " + a.download + " - see your browser's downloads"); if (editing) drawEdit();
  applyNotBeam(); applyTags();
}

// ---- "Not a beam drawing": tick it for images that are not beam detail drawings ----
function applyNotBeam() {
  document.body.classList.toggle("notbeam", notBeam);
  document.getElementById("notbeam").checked = notBeam;
  const eb = document.getElementById("editbtn"); eb.disabled = notBeam;
  eb.title = notBeam ? "Untick 'Not a beam drawing' first to edit the annotation" : "Add / remove / amend the annotation here (E)";
  const ab = document.getElementById("aibtn"); ab.disabled = notBeam;
  ab.title = notBeam ? "Untick 'Not a beam drawing' first" : "Rules and AI side by side: what the AI model (trained in Colab) thinks of this page";
  if (notBeam && !document.getElementById("aip").hidden) closeAI();
  const bar = document.getElementById("nbbar"); bar.hidden = !notBeam;
  if (!notBeam) return;
  const src = D.task_meta.skip_src, after = " Untick if it is a beam drawing after all.";
  const st = isSaved() && savedFile ? `annotation cancelled. <span class="ok">✓ saved to <b>${esc(savedFile)}</b></span>`
    : nEdits ? `annotation cancelled. <b>Save JSON</b> to keep it: pushed to Label Studio (step 7) the task becomes <i>skipped</i>, and the file tells the AI this is a drawing to skip.`
    : src === "label studio" ? "skipped in Label Studio, so this page's annotation is cancelled." + after
    : src === "file" ? "marked so in the file this page was made from; its annotation is cancelled." + after : "annotation cancelled.";
  bar.innerHTML = `<span class="tx">🚫 <b>Not a beam drawing</b> - ${st}</span><span class="bt">` +
    `<button id="nbsave" class="pri" title="Save as Label Studio JSON: the annotation cancelled (was_cancelled), no regions, page_type 'not a beam drawing' (Ctrl+S)">💾 Save JSON</button>` +
    `<button id="nbtrain" title="Keep this page in the AI's training set as an example of a drawing to skip">🧠 Training set</button>` +
    `<button id="nbun" title="It is a beam drawing: bring its annotation back">Untick</button></span>`;
  document.getElementById("nbsave").onclick = saveJSON;
  document.getElementById("nbtrain").onclick = () => document.getElementById("trainbtn").click();
  document.getElementById("nbun").onclick = () => setNotBeam(false);
}
function setNotBeam(on) {
  if (on === notBeam) return;
  if (on && editing) setEditing(false);
  const before = snapshot(); notBeam = on;
  hist.push(before); if (hist.length > 80) hist.shift(); fut = []; nEdits++; ver++;
  autosave(); applyNotBeam(); list();
  if (on) fit(); else if (cur >= 0 && cur < D.items.length) select(cur, false); else fit();
}
document.getElementById("notbeam").onchange = e => { e.target.blur(); setNotBeam(e.target.checked); };   // blur: Space stays 'right and next'

// ---- page tags: consultant and test page (saved in the file's meta, pushed to the Label Studio task) ----
const CKEY = "relreview-consultants";        // names typed on any review page in this browser
function knownConsultants() {
  let mine = []; try { mine = JSON.parse(localStorage.getItem(CKEY) || "[]"); } catch (e) {}
  return [...new Set([...(D.consultants || []), ...mine].filter(Boolean))].sort();
}
function applyTags() {
  const ci = document.getElementById("consultant");
  if (document.activeElement !== ci) ci.value = consultant;
  document.getElementById("cowrap").classList.toggle("empty", !consultant.trim());
  document.getElementById("colist").innerHTML = knownConsultants().map(n => `<option value="${esc(n)}">`).join("");
  document.getElementById("testpage").checked = testPage;
  document.body.classList.toggle("testpg", testPage);
  document.getElementById("hdsave").hidden = !(nEdits && !isSaved()) || editing || notBeam;
}
function setTag(fn) {
  const before = snapshot(); fn();
  hist.push(before); if (hist.length > 80) hist.shift(); fut = []; nEdits++; ver++;
  autosave(); applyTags(); if (editing) drawEdit();
}
const coIn = document.getElementById("consultant");
coIn.onchange = () => { const v = coIn.value.trim(); if (v === consultant) return;
  setTag(() => { consultant = v; });
  if (v) { const names = knownConsultants(); if (!names.includes(v)) { try { localStorage.setItem(CKEY, JSON.stringify([...names, v])); } catch (e) {} } applyTags(); }
  flash(v ? `Consultant: ${v} - Save JSON to keep it` : "Consultant cleared"); };
coIn.onkeydown = e => { if (e.key === "Enter") { e.preventDefault(); coIn.blur(); } };
document.getElementById("testpage").onchange = e => { e.target.blur(); const on = e.target.checked;
  setTag(() => { testPage = on; }); flash(on ? "Test page: the AI will not train on it - correct it fully, then Save JSON" : "Not a test page"); };
document.getElementById("hdsave").onclick = saveJSON;
applyTags();

// ---- wiring ----
document.getElementById("editbtn").onclick = () => { if (!notBeam) setEditing(!editing); };
document.querySelectorAll("#edbar [data-t]").forEach(b => b.onclick = () => setTool(b.dataset.t === tool && b.dataset.t !== "select" ? "select" : b.dataset.t));
document.getElementById("undo").onclick = undo; document.getElementById("redo").onclick = redo;
document.getElementById("edsave").onclick = saveJSON;
document.getElementById("eddel2").onclick = () => { if (selId) delRegion(selId); else flash("Select a region first"); };
document.getElementById("edlink2").onclick = () => setTool("link");
[["lbbox", "box"], ["lbline", "line"]].forEach(([sid, k]) => { const s = document.getElementById(sid);
  s.innerHTML = D.labels[k].map(l => `<option${l === newLbl[k] ? " selected" : ""}>${l}</option>`).join(""); s.onchange = () => { newLbl[k] = s.value; if (editing) drawEdit(); }; });
window.addEventListener("keydown", e => {
  if (["INPUT", "SELECT", "TEXTAREA"].includes(e.target.tagName)) return;
  const k = e.key.toLowerCase(), stop = () => { e.preventDefault(); e.stopImmediatePropagation(); };
  if (k === "e" && !e.ctrlKey && !e.metaKey) { stop(); if (notBeam) flash("Untick 'Not a beam drawing' first to edit"); else setEditing(!editing); return; }
  if ((notBeam || (nEdits && !isSaved())) && (e.ctrlKey || e.metaKey) && k === "s" && !e.shiftKey) { stop(); saveJSON(); return; }
  if (!editing) return;
  if (e.ctrlKey || e.metaKey) { if (k === "z") { stop(); e.shiftKey ? redo() : undo(); } else if (k === "y") { stop(); redo(); } else if (k === "s" && !e.shiftKey) { stop(); saveJSON(); } return; }
  if (k === "delete" || k === "backspace") { stop(); if (selId) delRegion(selId); }
  else if (k === "escape") { stop(); if (draft || tool !== "select") { tool = "select"; draft = null; ed = null; clearSnap(); drawEdit(); } else if (selId) { selId = null; syncList(); drawEdit(); } }
  else if (k === "enter" || k === " ") {     // AutoCAD style: finish the command; with none running, repeat the last one
    stop();
    if (draft) finishDraft();
    else if (tool !== "select") { tool = "select"; ed = null; clearSnap(); drawEdit(); }
    else if (lastTool) { setTool(lastTool); flash(`Repeat: ${{box: "new box (B)", line: "new line (D)", link: "link (R)", extend: "extend (G)", redraw: "redraw (W)"}[lastTool] || lastTool}`); }
  }
  else if (k === "b") { stop(); setTool("box"); } else if (k === "d") { stop(); setTool("line"); } else if (k === "r") { stop(); setTool("link"); }
  else if (k === "g") { stop(); setTool("extend"); } else if (k === "w") { stop(); setTool("redraw"); }
}, true);
window.addEventListener("beforeunload", e => { if (nEdits && !isSaved() && htmlSavedVer !== ver) { e.preventDefault(); e.returnValue = ""; } });

// ---- Save HTML: this page with your edits in it (they open as the page's own annotation) ----
let htmlSavedVer = null, htmlHandle = null;
const BAKED = (() => { try { return JSON.parse((document.getElementById("baked") || {}).textContent || "null"); } catch (e) { return null; } })();
if (BAKED && BAKED.state) {            // a page saved with Save HTML: start from the edits saved in it
  restore(BAKED.state); nEdits = BAKED.n || 0; hist = []; fut = []; ver = 0; selId = null;
  savedVer = BAKED.jsonSaved ? 0 : null; savedFile = BAKED.file || ""; htmlSavedVer = 0;
}
const pageFile = () => { try { const f = decodeURIComponent(location.pathname.split("/").pop() || ""); if (/\.html?$/i.test(f)) return f; } catch (e) {}
  return "review_" + D.image.replace(/\.[^.]+$/, "") + ".html"; };
function bakedHTML() {
  const st = {saved_at: new Date().toISOString(), n: nEdits, jsonSaved: isSaved(), file: savedFile, state: snapshot()};
  const tag = '<script id="baked" type="application/json">' + JSON.stringify(st).replace(/</g, "\\u003c") + "</" + "script>";
  let h = SRC0.replace(/<html[^>]*>/i, "<html>");                // not the light / dark choice of this browser
  const re = /<script id="baked" type="application\/json">[\s\S]*?<\/script>/;
  h = re.test(h) ? h.replace(re, () => tag) : h.replace(/<script>\s*const SRC0/, m => tag + "\n" + m);
  return h;
}
async function saveHTML(as) {
  const html = bakedHTML(), stem = pageFile().replace(/\.html?$/i, "");
  const name = as ? stem.replace(/_edited(_\d+)?$/, "") + "_edited.html" : pageFile();
  const done = f => { htmlSavedVer = ver; flash("Saved " + f + (nEdits ? ` (${nEdits} edit${nEdits > 1 ? "s" : ""} in it)` : "")); };
  if (window.showSaveFilePicker) {
    try {
      let h = as ? null : htmlHandle;
      if (!h) h = await window.showSaveFilePicker({suggestedName: name, id: "relreview-html", types: [{description: "Review page", accept: {"text/html": [".html"]}}]});
      const w = await h.createWritable(); await w.write(new Blob([html], {type: "text/html"})); await w.close();
      htmlHandle = h; done(h.name); return;
    } catch (e) { if (e && e.name === "AbortError") return; htmlHandle = null; }      // no file picker: download it instead
  }
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([html], {type: "text/html"})); a.download = as ? name : stem + "_edited.html";
  document.body.appendChild(a); a.click(); setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 1500);
  done(a.download + " - see your browser's downloads");
}
document.getElementById("htmlsave").onclick = () => saveHTML(false);
document.getElementById("htmlsaveas").onclick = () => saveHTML(true);
window.addEventListener("keydown", e => { if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key.toLowerCase() === "s") { e.preventDefault(); e.stopImmediatePropagation(); saveHTML(false); } }, true);

// ---- Load JSON: an edited annotation (Save JSON, a Label Studio export) shown on this page, as one edit (undo goes back) ----
function pickLoaded(js) {              // the task for this page in a file: [task, ...], a task, {result: [...]}, or [region, ...]
  if (Array.isArray(js) && js.length && js.every(r => r && r.type && !r.annotations && !r.data)) return {result: js};
  const tasks = Array.isArray(js) ? js : [js];
  return tasks.find(t => t && t.id === D.task_meta.id) || tasks.find(t => t && t.data && imgBase(t.data.image) === imgBase(D.image)) || (tasks.length === 1 ? tasks[0] : null);
}
function imgBase(s) {                  // '/data/local-files/?d=261002_Image%5Cname.png', 'G:\\...\\name.png', 'ab12cd34-name.png' -> 'name'
  s = String(s || ""); const q = /[?&]d=([^&]*)/.exec(s); if (q) s = q[1];
  try { s = decodeURIComponent(s); } catch (e) {}
  return s.split(/[\\/]/).pop().replace(/^[0-9a-f]{8}-/i, "").replace(/\.[^.]+$/, "").toLowerCase();
}
function resultOf(t) {                 // as the page builder picks it: the latest annotation, else the latest prediction
  if (t.result) return {result: t.result, cancelled: false};
  const byT = (a, b) => String(a.updated_at || a.created_at || "").localeCompare(String(b.updated_at || b.created_at || ""));
  const anns = (t.annotations || []).filter(a => a && typeof a === "object").sort(byT);
  const ok = anns.filter(a => !a.was_cancelled && (a.result || []).length);
  if (ok.length) return {result: ok[ok.length - 1].result, cancelled: false};
  const pr = (t.predictions || []).filter(p => p && (p.result || []).length);
  if (pr.length) return {result: pr[pr.length - 1].result, cancelled: false};
  const last = anns[anns.length - 1];
  return {result: (last && last.result) || [], cancelled: !!(last && last.was_cancelled)};
}
function loadTask(t, fname) {
  const {result, cancelled} = resultOf(t), W = D.W, H = D.H;
  const r6 = v => Math.round(v * 1e6) / 1e6, geo = r => { const v = r.value || {};
    return JSON.stringify([r.type, (v.rectanglelabels || v.vectorlabels || v.polygonlabels || [])[0],
      r.type === "rectanglelabels" ? [v.x, v.y, v.width, v.height].map(r6) : r.type === "vectorlabels" ? [(v.vertices || []).map(p => [r6(p.x), r6(p.y)]), !!v.closed] : (v.points || []).map(p => p.map(r6))]); };
  const noteOf = r => ((r.meta || {}).text || []).join(" ");
  const regs = [], ord = [], seen = new Set(); let n = Math.max(0, ...Object.values(R).map(g => g.n || 0));
  for (const r of result) {
    if (!r || !["rectanglelabels", "vectorlabels", "polygonlabels"].includes(r.type) || !r.value || seen.has(r.id)) continue;
    const v = r.value, label = (v.rectanglelabels || v.vectorlabels || v.polygonlabels || [])[0]; if (!label) continue;
    seen.add(r.id);
    const raw = D.raw && D.raw[r.id], old = R[r.id], same = raw && geo(raw.main) === geo(r) && noteOf(raw.main) === noteOf(r);
    const g = {id: r.id, n: old ? old.n : ++n, label, x: null, y: null, w: null, h: null, pts: null, closed: false, note: noteOf(r), text: null,
               _new: !raw, _dirty: !same, _noteSet: !!raw, _mmpp: old && old._mmpp != null ? old._mmpp : null};
    if (r.type === "rectanglelabels") {
      Object.assign(g, {kind: "box", x: v.x / 100 * W, y: v.y / 100 * H, w: v.width / 100 * W, h: v.height / 100 * H});
      g.text = D.words && D.words.length ? boxText(g) : old && old.text != null ? old.text : null;
    } else {
      let pts = (r.type === "vectorlabels" ? (v.vertices || []).map(p => [p.x, p.y]) : (v.points || [])).map(([x, y]) => [x / 100 * W, y / 100 * H]);
      if (pts.length < 2) continue;
      g.closed = r.type === "polygonlabels" || !!v.closed; if (g.closed) pts = [...pts, pts[0]];
      Object.assign(g, {kind: "line", pts});
    }
    regs.push(g); ord.push(r.id);
  }
  const ids = new Set(ord), rels = [];
  result.forEach(r => { if (r && r.type === "relation" && ids.has(r.from_id) && ids.has(r.to_id) && r.from_id !== r.to_id &&
    !rels.some(([a, b]) => a === r.from_id && b === r.to_id)) rels.push([r.from_id, r.to_id]); });
  const meta = t.meta || {}, pt = String(meta.page_type || "").toLowerCase();
  const st = {ord, regs, rels, sel: null,
    notBeam: pt === "not a beam drawing" ? true : pt === "beam drawing" ? false : cancelled && !regs.length ? true : notBeam,
    consultant: meta.consultant != null ? String(meta.consultant) : consultant,
    testPage: meta.split ? meta.split === "test" : testPage};
  if (editing && st.notBeam) setEditing(false);
  edit(() => restore(JSON.stringify(st)));
  applyNotBeam(); applyTags();
  if (!editing) { if (cur >= 0 && cur < D.items.length) select(cur, false); else drawAll(false); }
  flash(`Loaded ${fname}: ${regs.length} regions, ${rels.length} links${regs.some(g => g._dirty) ? "" : " (the same as this page)"} - Save JSON / Save HTML to keep it`);
}
document.getElementById("loadjson").onclick = () => { const f = document.getElementById("loadfile"); f.value = ""; f.click(); };
document.getElementById("loadfile").onchange = e => {
  const file = e.target.files && e.target.files[0]; if (!file) return;
  file.text().then(txt => {
    let js; try { js = JSON.parse(txt); } catch (err) { alert(file.name + " is not a JSON file."); return; }
    const t = pickLoaded(js);
    if (!t) { alert(`${file.name} has no task for this page (task ${D.task_meta.id}, ${D.image}).`); return; }
    const img = t.data && t.data.image ? imgBase(t.data.image) : "";
    if (img && img !== imgBase(D.image) && !confirm(`${file.name} is for ${img}, this page is ${D.image}. Load it anyway?`)) return;
    loadTask(t, file.name);
  });
};

(function offerRestore() {             // edits kept in this browser from an earlier visit
  let st = null; try { st = JSON.parse(localStorage.getItem(EKEY) || "null"); } catch (e) {}
  if (!st || !st.n || st.stamp !== D.task_meta.stamp) return;
  const plain = s => { try { const o = JSON.parse(s); delete o.sel; return JSON.stringify(o); } catch (e) { return s; } };
  if (BAKED && BAKED.state && plain(st.state) === plain(BAKED.state)) return;      // the same edits this saved page opened with
  const b = document.getElementById("restore"), n = `${st.n} edit${st.n > 1 ? "s" : ""}`; b.hidden = false;
  const nbst = (() => { try { return JSON.parse(st.state).notBeam; } catch (e) { return undefined; } })();
  const what = nbst ? ` (this page ticked <b>Not a beam drawing</b>)` : nbst === false && D.task_meta.not_beam ? ` (this page unticked: it is a beam drawing)` : "";
  b.innerHTML = st.saved
    ? `Your <b>${n}</b>${what} on this page from ${new Date(st.t).toLocaleString()} ${st.n > 1 ? "were" : "was"} saved as <b>${esc(st.file || "a JSON file")}</b>. ` +
      `<button id="rsyes">Keep editing them</button> <button id="rsno">Start from the original</button>`
    : `You have <b>${n} not saved to a file yet</b>${what} on this page from ${new Date(st.t).toLocaleString()}. ` +
      `<button id="rsyes">Restore them</button> <button id="rsno">Discard</button>`;
  document.getElementById("rsyes").onclick = () => { restore(st.state); nEdits = st.n; hist = []; fut = []; ver = 1;
    savedVer = st.saved ? 1 : null; savedFile = st.file || ""; b.hidden = true; recompute(); autosave(); applyNotBeam(); list();
    if (notBeam) fit(); else setEditing(true); };
  document.getElementById("rsno").onclick = () => { try { localStorage.removeItem(EKEY); } catch (e) {} b.hidden = true; };
})();

// ---- training set for the AI model (the backend builds the graph; Colab trains on the folder) ----
let backendURL = "http://localhost:9090"; try { backendURL = localStorage.getItem("relreview:backend") || backendURL; } catch (e) {}
async function backend(path, body) {
  const url = backendURL.replace(/\/+$/, "") + path; let r;
  try { r = await fetch(url, body === undefined ? {} : {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)}); }
  catch (e) { throw new Error(`Can't reach the backend at ${esc(backendURL)}. Is server.py running? (Anaconda Prompt: <code>python server.py --pdf-dir ...</code>)`); }
  if (r.status === 404) throw new Error("This backend doesn't know this yet: replace server.py (and its new files: ai_check.py, gnn.py, graph_builder.py) from the new zip, then restart it.");
  if (!r.ok) throw new Error(`The backend said ${r.status}: ${esc((await r.text()).slice(0, 300))}`);
  return r.json();
}
const pct = v => v == null ? "–" : (100 * v).toFixed(0) + "%";
function renderTraining(s, note) {
  const P = document.getElementById("trainp"), mine = (s.pages || []).find(p => p.image === D.task_meta.image);
  const m = s.model && s.model.metrics;
  let h = `<div class="tph"><b>🧠 Training set for the AI model</b><span class="sp"></span><button id="tpx" title="Close">✕</button></div>` +
    `<div class="row">Backend <input id="tpurl" value="${esc(backendURL)}"> <button id="tpref" title="Refresh">↻</button></div>` +
    `<div class="row"><button id="tpadd" class="pri" title="${notBeam ? "Keeps this page as an example of a drawing to skip (not used to train the rebar model)" : "Builds this page's graph from the PDF and your annotation as it is now (edits included)"}">➕ ${mine ? "Update" : "Add"} this page${notBeam ? " as not a beam drawing" : ""}</button> ` +
    (mine ? `<span class="ok">in the training set${mine.page_type === "not a beam drawing" ? " as not a beam drawing" : ""} since ${esc(mine.added || "?")}</span>` : `<span style="color:var(--mute)">not in the training set yet</span>`) + `</div>`;
  if (note) h += `<div class="row">${note}</div>`;
  const pages = s.pages || [], nNot = pages.filter(p => p.page_type === "not a beam drawing").length;
  h += `<details${pages.length ? "" : " open"}><summary>${pages.length} page${pages.length === 1 ? "" : "s"} in the training set${nNot ? ` (${nNot} not a beam drawing: examples of pages to skip)` : ""}</summary>` +
    (pages.length ? `<table><tr><th>page</th><th>added</th><th>from</th><th>relations</th></tr>` + pages.map(p => `<tr><td>${esc(p.page)}</td><td>${esc(p.added || "")}</td><td>${esc(p.source || "")}</td><td>${p.page_type === "not a beam drawing" ? `<span class="err">not a beam drawing</span>` : p.relations != null ? `${p.relations_found}/${p.relations}` : ""}</td></tr>`).join("") + `</table>` : "") +
    `<div style="color:var(--mute);margin-top:4px">Folder: ${esc(s.train_dir || "")}</div></details>`;
  if (m) {
    h += `<div class="row" style="margin-top:8px"><b>Latest model</b> (trained in Colab ${esc(s.model.saved || "")}${m.trained_on ? ` on ${m.trained_on.length} page${m.trained_on.length === 1 ? "" : "s"}` : ""}), checked on ${esc(m.checked_on || "?")}:<br>` +
      `node classes F1 <b>${pct(m.node_f1)}</b> · relations F1 <b>${pct(m.relation_f1)}</b></div>` +
      `<details><summary>per relation kind / class</summary><table><tr><th>relations</th><th>precision</th><th>recall</th><th>F1</th><th>true</th></tr>` +
      Object.entries(m.per_kind || {}).map(([k, v]) => `<tr><td>${esc(k)}</td><td>${pct(v.precision)}</td><td>${pct(v.recall)}</td><td>${pct(v.f1)}</td><td>${v.true}</td></tr>`).join("") +
      `<tr><th>node class</th><th></th><th></th><th></th><th></th></tr>` +
      Object.entries(m.per_class || {}).map(([k, v]) => `<tr><td>${esc(k)}</td><td>${pct(v.precision)}</td><td>${pct(v.recall)}</td><td>${pct(v.f1)}</td><td>${v.true}</td></tr>`).join("") + `</table></details>`;
  } else h += `<div class="row" style="margin-top:8px;color:var(--mute)">No trained model in this folder yet.</div>`;
  h += `<div class="row" style="margin-top:8px">To train: open <b>02_train_gnn.ipynb</b> (in the backend folder on your Google Drive) in Colab and run all. ` +
       `It trains on the pages above and saves the model into this folder; its scores show here once Drive has synced.</div>`;
  P.innerHTML = h;
  document.getElementById("tpx").onclick = () => { P.hidden = true; };
  document.getElementById("tpref").onclick = () => openTraining();
  document.getElementById("tpurl").onchange = e => { backendURL = e.target.value.trim() || "http://localhost:9090"; try { localStorage.setItem("relreview:backend", backendURL); } catch (_) {} openTraining(); };
  document.getElementById("tpadd").onclick = addToTraining;
}
function trainError(e) {
  const P = document.getElementById("trainp");
  P.innerHTML = `<div class="tph"><b>🧠 Training set for the AI model</b><span class="sp"></span><button id="tpx">✕</button></div>` +
    `<div class="row">Backend <input id="tpurl" value="${esc(backendURL)}"> <button id="tpref">↻</button></div><div class="row err">${e.message}</div>`;
  document.getElementById("tpx").onclick = () => { P.hidden = true; };
  document.getElementById("tpref").onclick = () => openTraining();
  document.getElementById("tpurl").onchange = ev => { backendURL = ev.target.value.trim() || "http://localhost:9090"; try { localStorage.setItem("relreview:backend", backendURL); } catch (_) {} openTraining(); };
}
async function openTraining() {
  const P = document.getElementById("trainp"); P.hidden = false;
  if (!P.innerHTML) P.innerHTML = `<div class="row">Asking the backend…</div>`;
  try { renderTraining(await backend("/gnn/status")); } catch (e) { trainError(e); }
}
async function addToTraining() {
  const btn = document.getElementById("tpadd"); btn.disabled = true; btn.textContent = notBeam ? "Adding…" : "Building this page's graph…";
  try {
    const r = await backend("/gnn/add", {task: currentTask()}), a = (r.added || [])[0] || {};
    const oldBackend = notBeam && (a.error ? /has no annotation/.test(a.error) : a.page_type === undefined);
    const note = oldBackend ? `<span class="err">This backend doesn't know 'not a beam drawing' yet: replace server.py from the new zip and restart it.</span>`
      : a.error ? `<span class="err">Not added: ${esc(a.error)}</span>`
      : a.page_type === "not a beam drawing" ? `<span class="ok">✓ ${esc(a.page)}: kept as an example of a drawing to skip (not used to train the rebar model).</span>`
      : `<span class="ok">✓ ${esc(a.page)}: ${(a.nodes || 0).toLocaleString()} nodes, ${(a.pairs || 0).toLocaleString()} candidate pairs, ` +
        `${a.relations_found}/${a.relations} of your relations reachable${a.relations_found < a.relations ? " (the rest can't be learned from this graph)" : ""}.</span>`;
    renderTraining(r, note);
  } catch (e) { trainError(e); }
}
document.getElementById("trainbtn").onclick = () => { const P = document.getElementById("trainp"); if (!P.hidden) { P.hidden = true; return; }
  if (!document.getElementById("aip").hidden) closeAI(); openTraining(); };

// ---- AI check: rules and AI side by side. The backend's AI model (trained in Colab) looks at this page next to
// its annotation (/gnn/check, ai_check.py): links it doubts or would add, regions it reads differently, labels it
// finds without a box. You decide: Apply (in Edit mode, undo with Ctrl+Z) or ✓ (the annotation is right).
const AKEY = "relreview-ai:" + D.task + ":" + D.image;
let aiRep = null, aiSel = -1, aiDone = {};
try { aiDone = JSON.parse(localStorage.getItem(AKEY) || "{}"); } catch (e) {}
const aiSave = () => { try { localStorage.setItem(AKEY, JSON.stringify(aiDone)); } catch (e) {} };
const aiSig = it => [it.type, it.regions.map(r => r.id).sort().join(","), it.text || "", (it.bbox || []).map(v => v.toFixed(1)).join(",")].join("|");
const KIND = k => ({"text-line": "label – line", "text-text (near)": "side by side", "group membership (beam / section)": "beam membership",
  "same element (repeat / section label)": "same bars", "section reference (cut mark - section)": "cut mark – section",
  "link shape (link label - section)": "link – section"})[k] || k || "";
const AIT = {doubt: "Links the AI doubts", suggest: "Links the AI would make", class: "Read differently", undrawn: "Bar / dimension line not drawn?", unboxed: "Labels without a box?"};
const pc = p => Math.round(100 * p) + "%";
function rname(r) { const g = R[r.id];
  return g ? `${esc(g.label)} #${g.n}${g.text ? " “" + esc(g.text) + "”" : ""}` : `${esc(r.label)}${r.text ? " “" + esc(r.text) + "”" : ""} <i>(gone)</i>`; }
function aiText(it) {
  const [a, b] = it.regions;
  if (it.type === "doubt") return `${rname(a)} ↔ ${rname(b)}<br><small>${it.unreachable ? "no path between them in the drawing" : "AI " + pc(it.p) + " for this link"} · ${KIND(it.kind)}</small>`;
  if (it.type === "suggest") return `${rname(a)} → ${rname(b)}<br><small>AI ${pc(it.p)} · ${KIND(it.kind)}` +
    (it.replaces.length ? ` · instead of ${it.replaces.map(r => rname(r) + " (" + pc(r.p) + ")").join(", ")}` : "") +
    (it.alternatives ? ` · ${it.alternatives} other${it.alternatives > 1 ? "s" : ""} nearly as likely` : "") + (it.fix ? " · <b>the AI is sure</b>" : "") + `</small>`;
  if (it.type === "class") return `${rname(a)}<br><small>AI reads it as <b>${esc(it.ai)}</b> (${pc(it.p)}; ${esc(it.rules)} ${pc(it.p_rules)})</small>`;
  if (it.type === "undrawn") return `${rname(a)}<br><small>its ${esc(it.ai)} may be the PDF line shown in green, which has none · AI ${pc(it.p)}</small>`;
  return `“${esc(it.text)}”<br><small>looks like a <b>${esc(it.ai)}</b> (AI ${pc(it.p)}), but has no box</small>`;
}
function aiAction(it) {
  if (it.type === "doubt") return "Remove link";
  if (it.type === "suggest") return it.replaces.length ? "Move link" : "Add link";
  if (it.type === "class") return it.ai_cls === "Other" ? "Delete it" : "Make it " + esc(it.ai);
  if (it.type === "undrawn") return "Draw " + esc(it.ai);
  return "Add box";
}
function aiHead() { return `<div class="tph"><b>🤖 AI check</b><span class="sp"></span><button id="airef" title="Check again (e.g. after your edits)">↻</button><button id="aix" title="Close">✕</button></div>`; }
function aiWireHead() {
  document.getElementById("aix").onclick = closeAI;
  document.getElementById("airef").onclick = () => openAI(true);
}
function closeAI() { document.getElementById("aip").hidden = true; document.getElementById("aibtn").classList.remove("on"); drawAI(null); }
function renderAI() {
  const P = document.getElementById("aip"), r = aiRep;
  if (!r.ok) {
    P.innerHTML = aiHead() + `<div class="row">Backend <input id="aiurl" value="${esc(backendURL)}"></div>` +
      `<div class="row err">${r.html || esc(r.reason || "no answer")}</div>` +
      (r.model && r.model.path ? `<div class="row mute">The backend looks for the model here: ${esc(r.model.path)}. It gets there when 02_train_gnn.ipynb has run in Colab and Google Drive has synced.</div>` : "");
    aiWireHead();
    document.getElementById("aiurl").onchange = e => { backendURL = e.target.value.trim() || "http://localhost:9090"; try { localStorage.setItem("relreview:backend", backendURL); } catch (_) {} openAI(true); };
    return;
  }
  const m = r.model || {}, n = r.checked.regions + r.checked.relations, ok = r.agree.regions + r.agree.relations;
  let h = aiHead() + `<div class="row"><span class="big">${r.agreement == null ? "–" : pc(r.agreement)}</span> agreement with this page <span class="mute">(${ok} of ${n} regions and links)</span></div>` +
    `<div class="row mute">Model of ${esc(m.saved || "?")}, trained on ${m.trained_on || "?"} page(s)` +
    (m.relation_f1 != null ? `; on pages it didn't train on: links F1 ${m.relation_f1.toFixed(2)}, regions F1 ${(m.node_f1 || 0).toFixed(2)}` : "") + ". " +
    ((r.trusted || []).length ? `Proven well enough to fix by itself (backend --ai fix): ${r.trusted.map(KIND).join(", ")}.` : "Not proven enough yet to fix anything by itself: you decide.") +
    ` Checked in ${r.seconds} s.</div>`;
  ["doubt", "suggest", "class", "undrawn", "unboxed"].forEach(t => {
    const its = r.items.map((it, k) => [it, k]).filter(([it]) => it.type === t);
    if (!its.length) return;
    const left = its.filter(([it]) => !aiDone[aiSig(it)]).length;
    h += `<details${left ? " open" : ""}><summary>${AIT[t]} · ${its.length}${left < its.length ? ` <span class="ok">(${its.length - left} done)</span>` : ""}</summary>` +
      its.map(([it, k]) => { const st = aiDone[aiSig(it)];
        return `<div class="ai-it${k === aiSel ? " on" : ""}${st ? " done" : ""}" data-k="${k}"><div class="tx">${aiText(it)}</div><div class="bt">` +
          (st ? `<span class="ok">${st === "applied" ? "✓ applied" : "✓ checked"}</span><button data-re="${k}" title="Open it again">↺</button>`
              : `<button data-apply="${k}" title="Edit mode: does it for you (undo with Ctrl+Z)">${aiAction(it)}</button><button data-ok="${k}" title="The annotation is right here: mark it checked">✓</button>`) +
          `</div></div>`; }).join("") + `</details>`;
  });
  if (!r.items.length) h += `<div class="row ok">Nothing to check: the AI agrees with all it can judge on this page.</div>`;
  P.innerHTML = h; aiWireHead();
  P.querySelectorAll(".ai-it").forEach(d => d.onclick = () => aiShow(+d.dataset.k));
  P.querySelectorAll("[data-apply]").forEach(b => b.onclick = e => { e.stopPropagation(); aiApply(+b.dataset.apply); });
  P.querySelectorAll("[data-ok]").forEach(b => b.onclick = e => { e.stopPropagation(); aiDone[aiSig(r.items[+b.dataset.ok])] = "checked"; aiSave(); renderAI(); });
  P.querySelectorAll("[data-re]").forEach(b => b.onclick = e => { e.stopPropagation(); delete aiDone[aiSig(r.items[+b.dataset.re])]; aiSave(); renderAI(); });
}
const aiPx = (x, y) => [x / 100 * D.W, y / 100 * D.H];
function aiGeo(it) {                       // what to zoom to: the regions, the box or the PDF line it is about
  const ps = [];
  it.regions.concat(it.replaces || []).forEach(r => { if (R[r.id]) { const b = bbox([r.id]); ps.push([b[0], b[1]], [b[2], b[3]]); } });
  if (it.bbox) ps.push(aiPx(it.bbox[0], it.bbox[1]), aiPx(it.bbox[2], it.bbox[3]));
  (it.segs || []).forEach(sg => ps.push(aiPx(sg[0], sg[1]), aiPx(sg[2], sg[3])));
  return ps;
}
function drawAI(it) {
  const G = document.getElementById("ai"); G.innerHTML = "";
  if (!it) return;
  const NSV = {"vector-effect": "non-scaling-stroke"};
  const outline = id => { const g = R[id]; if (!g) return;
    shape(g, G, {c: "#fff", w: 7, o: 0.9}).setAttribute("vector-effect", "non-scaling-stroke");
    shape(g, G, {c: col(g.label), w: 3.5, o: 1, fill: g.kind === "box" ? col(g.label) + "33" : "none"}).setAttribute("vector-effect", "non-scaling-stroke"); };
  const link = (a, b, c) => { const ga = R[a], gb = R[b]; if (!ga || !gb) return; const q = nearest(gb, centre(ga)), p = nearest(ga, q);
    el("line", Object.assign({x1: p[0], y1: p[1], x2: q[0], y2: q[1], stroke: c, "stroke-width": 3, "stroke-dasharray": "9 5"}, NSV), G); };
  const [a, b] = it.regions.map(r => r.id);
  it.regions.forEach(r => outline(r.id));
  if (it.type === "doubt") link(a, b, "#d93025");
  if (it.type === "suggest") { link(a, b, "#1e8e3e"); (it.replaces || []).forEach(r => { outline(r.id); link(a, r.id, "#d93025"); }); }
  if (it.type === "unboxed") { const [x0, y0] = aiPx(it.bbox[0], it.bbox[1]), [x1, y1] = aiPx(it.bbox[2], it.bbox[3]), pd = (y1 - y0) * 0.35;
    el("rect", Object.assign({x: x0 - pd, y: y0 - pd, width: x1 - x0 + 2 * pd, height: y1 - y0 + 2 * pd, fill: "#f39c1222", stroke: "#e67e22", "stroke-width": 3, "stroke-dasharray": "7 4"}, NSV), G); }
  if (it.type === "undrawn") {
    it.segs.forEach(sg => { const [x0, y0] = aiPx(sg[0], sg[1]), [x1, y1] = aiPx(sg[2], sg[3]);
      el("line", Object.assign({x1: x0, y1: y0, x2: x1, y2: y1, stroke: "#1e8e3e", "stroke-width": 4, opacity: 0.85, "stroke-dasharray": "10 4"}, NSV), G); });
    const g = R[a]; if (g) { const c0 = centre(g); let best = null;
      it.segs.forEach(sg => [aiPx(sg[0], sg[1]), aiPx(sg[2], sg[3])].forEach(q => { const d = Math.hypot(q[0] - c0[0], q[1] - c0[1]); if (!best || d < best[0]) best = [d, q]; }));
      if (best) { const p = nearest(g, best[1]); el("line", Object.assign({x1: p[0], y1: p[1], x2: best[1][0], y2: best[1][1], stroke: "#1e8e3e", "stroke-width": 2, "stroke-dasharray": "4 4"}, NSV), G); } }
  }
}
function aiShow(k) {
  aiSel = k; const it = aiRep.items[k];
  document.querySelectorAll("#aip .ai-it").forEach(d => d.classList.toggle("on", +d.dataset.k === k));
  drawAI(it);
  const ps = aiGeo(it); if (!ps.length) return;
  const xs = ps.map(p => p[0]), ys = ps.map(p => p[1]), x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
  const w = Math.max(x1 - x0, D.W * 0.05), h = Math.max(y1 - y0, D.H * 0.04), cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
  zoomTo(cx - w * 0.8, cy - h * 0.9, cx + w * 0.8, cy + h * 0.9, 1);
  if (editing) drawEdit(); else if (cur >= 0) select(cur, false); else drawAll(false);
}
function chainSegs(segs) {                 // PDF line pieces (in %) -> one polyline (px), from the first piece on
  const P = segs.map(sg => [aiPx(sg[0], sg[1]), aiPx(sg[2], sg[3])]), pts = [P[0][0], P[0][1]], used = new Set([0]), tol = D.W * 0.002;
  const d = (p, q) => Math.hypot(p[0] - q[0], p[1] - q[1]);
  for (let grew = true; grew;) { grew = false;
    for (let i = 0; i < P.length; i++) { if (used.has(i)) continue; const [a, b] = P[i], e = pts[pts.length - 1], s0 = pts[0];
      if (d(a, e) < tol) pts.push(b); else if (d(b, e) < tol) pts.push(a); else if (d(b, s0) < tol) pts.unshift(a); else if (d(a, s0) < tol) pts.unshift(b); else continue;
      used.add(i); grew = true; } }
  return pts;
}
const ordered = (a, b) => rankOf(a) > rankOf(b) ? [b, a] : [a, b];
function aiApply(k) {
  const it = aiRep.items[k], ids = it.regions.map(r => r.id);
  if (ids.some(id => !R[id])) { flash("A region of this item is gone - check again (↻)"); return; }
  if (!editing) setEditing(true);
  if (it.type === "doubt") delRel(ids[0], ids[1]);
  else if (it.type === "suggest") { const [sid, oid] = ids, gone = (it.replaces || []).map(r => r.id);
    edit(() => { D.rels = D.rels.filter(([x, y]) => !gone.some(g => (x === sid && y === g) || (x === g && y === sid)));
      if (!D.rels.some(([x, y]) => (x === sid && y === oid) || (x === oid && y === sid))) D.rels.push(ordered(sid, oid)); selId = sid; }); }
  else if (it.type === "class") { if (it.ai_cls === "Other") delRegion(ids[0]); else setLabel(ids[0], it.ai); }
  else if (it.type === "unboxed") { const [x0, y0] = aiPx(it.bbox[0], it.bbox[1]), [x1, y1] = aiPx(it.bbox[2], it.bbox[3]), pd = (y1 - y0) * 0.15;
    edit(() => { const id = newId(); R[id] = {id, n: nextN(), label: it.ai, kind: "box", x: x0 - pd, y: y0 - pd, w: x1 - x0 + 2 * pd, h: y1 - y0 + 2 * pd,
      note: "added from the AI check", _new: true, _dirty: true, problems: []}; derived(R[id]); ORD.push(id); selId = id; }); }
  else if (it.type === "undrawn") { const pts = chainSegs(it.segs);
    edit(() => { const id = newId(); R[id] = {id, n: nextN(), label: it.ai, kind: "line", pts, closed: false, note: "added from the AI check",
      _new: true, _dirty: true, problems: []}; derived(R[id]); ORD.push(id); D.rels.push(ordered(ids[0], id)); selId = id; }); }
  aiDone[aiSig(it)] = "applied"; aiSave(); renderAI(); drawAI(it);
}
async function openAI(fresh) {
  const P = document.getElementById("aip"); document.getElementById("trainp").hidden = true;
  P.hidden = false; document.getElementById("aibtn").classList.add("on");
  if (aiRep && !fresh) { renderAI(); return; }
  P.innerHTML = aiHead() + `<div class="row">Asking the AI… the backend builds this page's graph from the PDF and your annotation as it is now (10–30 s).</div>`;
  aiWireHead(); drawAI(null); aiSel = -1;
  try { aiRep = await backend("/gnn/check", {task: currentTask()}); } catch (e) { aiRep = {ok: false, html: e.message}; }
  if (!P.hidden) renderAI();
}
document.getElementById("aibtn").onclick = () => { if (notBeam) return; const P = document.getElementById("aip"); if (!P.hidden) closeAI(); else openAI(false); };

// ================= Snap (F3: line ends, corners, crossings · F4: perpendicular) and Ortho (F8), for drawing and editing lines =================
const SKEY = "relreview:snap";
let SNAP = {on: true, perp: false, ortho: false}; try { Object.assign(SNAP, JSON.parse(localStorage.getItem(SKEY) || "{}")); } catch (e) {}
let shiftHeld = false, snapHit = null;
const SG = D.segs || [], NSEG = SG.length / 4, CELL = Math.max(D.W, D.H) / 256;   // the PDF's line pieces (px), in a grid
let grid = null;
function segGrid() {
  if (grid) return grid; grid = new Map();
  for (let s = 0; s < NSEG; s++) {
    const x0 = SG[4 * s], y0 = SG[4 * s + 1], x1 = SG[4 * s + 2], y1 = SG[4 * s + 3], n = Math.ceil(Math.hypot(x1 - x0, y1 - y0) / (CELL / 2)) + 1, seen = new Set();
    for (let k = 0; k <= n; k++) { const key = Math.floor((x0 + (x1 - x0) * k / n) / CELL) + "," + Math.floor((y0 + (y1 - y0) * k / n) / CELL);
      if (seen.has(key)) continue; seen.add(key); let a = grid.get(key); if (!a) grid.set(key, a = []); a.push(s); } }
  return grid;
}
function nearSegs(p, r) {               // [x0, y0, x1, y1] pieces near p: the PDF's and the annotation's (but not region excl)
  const out = [], G = segGrid(), seen = new Set();
  for (let i = Math.floor((p.x - r) / CELL); i <= Math.floor((p.x + r) / CELL); i++)
    for (let j = Math.floor((p.y - r) / CELL); j <= Math.floor((p.y + r) / CELL); j++)
      (G.get(i + "," + j) || []).forEach(s => { if (!seen.has(s)) { seen.add(s); out.push([SG[4 * s], SG[4 * s + 1], SG[4 * s + 2], SG[4 * s + 3]]); } });
  return out;
}
function annSegs(p, r, excl) {
  const out = [], near = s => Math.min(s[0], s[2]) - r <= p.x && p.x <= Math.max(s[0], s[2]) + r && Math.min(s[1], s[3]) - r <= p.y && p.y <= Math.max(s[1], s[3]) + r;
  ORD.forEach(id => { const g = R[id]; if (!g || id === excl) return;
    const P = g.kind === "box" ? [[g.x, g.y], [g.x + g.w, g.y], [g.x + g.w, g.y + g.h], [g.x, g.y + g.h], [g.x, g.y]] : g.pts;
    for (let k = 0; k + 1 < P.length; k++) { const s = [P[k][0], P[k][1], P[k + 1][0], P[k + 1][1]]; if (near(s)) out.push(s); } });
  return out;
}
function segDist(p, s) { const dx = s[2] - s[0], dy = s[3] - s[1], L = dx * dx + dy * dy, t = L ? Math.max(0, Math.min(1, ((p.x - s[0]) * dx + (p.y - s[1]) * dy) / L)) : 0;
  return Math.hypot(s[0] + t * dx - p.x, s[1] + t * dy - p.y); }
function crossing(a, b) {               // where pieces a and b cross (on both), or null
  const r = [a[2] - a[0], a[3] - a[1]], q = [b[2] - b[0], b[3] - b[1]], den = r[0] * q[1] - r[1] * q[0];
  if (Math.abs(den) < 1e-9 * (Math.hypot(...r) * Math.hypot(...q) + 1e-9)) return null;
  const t = ((b[0] - a[0]) * q[1] - (b[1] - a[1]) * q[0]) / den, u = ((b[0] - a[0]) * r[1] - (b[1] - a[1]) * r[0]) / den;
  return t >= -1e-6 && t <= 1 + 1e-6 && u >= -1e-6 && u <= 1 + 1e-6 ? [a[0] + t * r[0], a[1] + t * r[1]] : null;
}
const orthoNow = () => SNAP.ortho !== shiftHeld;
// the point to use for cursor p: ortho from the previous point 'from', then the nearest snap within 10 screen px (a snap wins over ortho, as in AutoCAD)
function snapPt(p, from, excl) {
  const q = {x: p.x, y: p.y}; snapHit = null;
  if (from && orthoNow()) { if (Math.abs(p.x - from[0]) >= Math.abs(p.y - from[1])) q.y = from[1]; else q.x = from[0]; }
  if (!SNAP.on && !(SNAP.perp && from)) return q;
  const tol = sw(10), cand = [...nearSegs(p, tol), ...annSegs(p, tol, excl)];
  let best = null;
  const consider = (x, y, kind, w) => { const d = Math.hypot(x - p.x, y - p.y); if (d <= tol && (!best || d * w < best.d)) best = {x, y, kind, d: d * w}; };
  if (SNAP.on) {
    cand.forEach(s => { consider(s[0], s[1], "end", 0.6); consider(s[2], s[3], "end", 0.6); });
    const close = cand.filter(s => segDist(p, s) <= tol).slice(0, 80);
    for (let i = 0; i < close.length; i++) for (let j = i + 1; j < close.length; j++) { const X = crossing(close[i], close[j]); if (X) consider(X[0], X[1], "cross", 0.8); }
  }
  if (SNAP.perp && from) cand.forEach(s => {        // foot of the perpendicular from the previous point onto the piece
    const dx = s[2] - s[0], dy = s[3] - s[1], L = dx * dx + dy * dy; if (!L) return;
    const t = ((from[0] - s[0]) * dx + (from[1] - s[1]) * dy) / L; if (t < -0.02 || t > 1.02) return;
    const fx = s[0] + t * dx, fy = s[1] + t * dy; if (Math.hypot(fx - from[0], fy - from[1]) < sw(4)) return;
    consider(fx, fy, "perp", 0.7); });
  if (best) { snapHit = best; return {x: best.x, y: best.y}; }
  return q;
}
function drawSnap() {                   // the snap marker: □ end / corner, ✕ crossing, ⊥ perpendicular
  const G = document.getElementById("snapg"); G.innerHTML = ""; if (!editing || !snapHit) return;
  const {x, y, kind} = snapHit, a = sw(7), st = c => ({fill: "none", stroke: c, "stroke-width": sw(2.2)});
  const mark = (c, w) => { if (kind === "end") el("rect", Object.assign({x: x - a, y: y - a, width: 2 * a, height: 2 * a}, st(c), {"stroke-width": w}), G);
    else if (kind === "cross") el("path", Object.assign({d: `M${x - a} ${y - a}L${x + a} ${y + a}M${x - a} ${y + a}L${x + a} ${y - a}`}, st(c), {"stroke-width": w}), G);
    else el("path", Object.assign({d: `M${x - a} ${y + a}H${x + a}M${x} ${y + a}V${y - a}`}, st(c), {"stroke-width": w}), G); };
  mark("#fff", sw(5)); mark("#00a650", sw(2.2));
}
function clearSnap() { snapHit = null; const G = document.getElementById("snapg"); if (G) G.innerHTML = ""; }
const DRAW_TOOLS = ["line", "extend", "redraw"];
function draftFrom() {                  // the point the next segment starts from
  if (draft && draft.pts.length) return draft.pts[draft.pts.length - 1];
  if (draft && draft.mode === "extend" && R[draft.id]) { const o = openPts(R[draft.id]); return draft.atStart ? o[0] : o[o.length - 1]; }
  return null;
}
function applySnapUI() {
  [["snapbtn", SNAP.on], ["perpbtn", SNAP.perp], ["orthobtn", SNAP.ortho]].forEach(([id, on]) => document.getElementById(id).classList.toggle("on", !!on));
}
function toggleSnap(k) {
  SNAP[k] = !SNAP[k]; try { localStorage.setItem(SKEY, JSON.stringify(SNAP)); } catch (e) {}
  applySnapUI(); clearSnap();
  flash({on: "Snap to line ends, corners and crossings", perp: "Perpendicular snap", ortho: "Ortho (horizontal / vertical only)"}[k] + (SNAP[k] ? " ON" : " OFF") + (editing ? "" : " - used in Edit mode (E)"));
}
document.getElementById("snapbtn").onclick = () => toggleSnap("on");
document.getElementById("perpbtn").onclick = () => toggleSnap("perp");
document.getElementById("orthobtn").onclick = () => toggleSnap("ortho");
applySnapUI();
window.addEventListener("keyup", e => { if (e.key === "Shift") shiftHeld = false; });
window.addEventListener("blur", () => { shiftHeld = false; });

// ================= Find (Ctrl+F): labels, beams, texts on the drawing =================
const fIn = document.getElementById("find"), fRes = document.getElementById("findres");
let fHits = [], fSel = 0;
const fNorm = v => String(v || "").toLowerCase().replace(/\s+/g, " ").trim(), fComp = v => fNorm(v).replace(/[^a-z0-9]/g, "");
function bigrams(v) { const m = new Map(); for (let i = 0; i + 1 < v.length; i++) { const b = v.slice(i, i + 2); m.set(b, (m.get(b) || 0) + 1); } return m; }
function dice(a, b) { if (a.length < 2 || b.length < 2) return a === b ? 1 : 0; const A = bigrams(a), B = bigrams(b); let n = 0;
  A.forEach((c, k) => { n += Math.min(c, B.get(k) || 0); }); return 2 * n / (a.length + b.length - 2); }
function fScore(q, text) {              // how well text matches the query: whole, start, part, every word, letters in order, or similar
  const t = fNorm(text), tc = fComp(text), qc = fComp(q); if (!qc) return 0;
  if (tc === qc) return 100;
  if (tc.startsWith(qc)) return 90 - Math.min(10, tc.length - qc.length);
  const at = tc.indexOf(qc); if (at >= 0) return 80 - Math.min(10, at) - Math.min(5, (tc.length - qc.length) / 4);
  const words = fNorm(q).split(" ").filter(Boolean); if (words.length > 1 && words.every(w => t.includes(w) || tc.includes(fComp(w)))) return 70;
  let k = 0, gaps = 0, last = -1; for (let i = 0; i < tc.length && k < qc.length; i++) if (tc[i] === qc[k]) { if (last >= 0) gaps += i - last - 1; last = i; k++; }
  if (k === qc.length && gaps <= 2 * qc.length) return 55 - Math.min(15, gaps);
  let d = 0; for (const w of [tc, ...t.split(" ").map(fComp)]) d = Math.max(d, dice(qc, w));
  return d >= 0.5 ? 20 + 30 * d : 0;
}
function findWhere(id) {               // where a region is: its beam or section (a line: that of its label)
  const g = R[id], beamOf = r => { const b = nbs(r).find(o => isA(o, "BEAM")) || ((D.owner || {})[r]);
    return b && R[b] ? (R[b].text || `${R[b].label} #${R[b].n}`) : typeof b === "string" && b.startsWith("sec:") ? b.slice(4) : ""; };
  let w = beamOf(id);
  if (!w && g.kind === "line") { const lab = nbs(id).find(o => R[o].kind === "box"); if (lab) w = (R[lab].text ? `“${R[lab].text}” ` : "") + beamOf(lab); }
  return w;
}
function findCands() {                  // what can be found: every region (label, #n, text) and the PDF words no box covers
  const out = ORD.filter(id => R[id]).map(id => { const g = R[id]; return {id, text: g.text || "", label: g.label, n: g.n, kind: g.kind}; });
  const boxes = ORD.map(id => R[id]).filter(g => g && g.kind === "box");
  (D.words || []).forEach((w, k) => { const cx = (w[0] + w[2]) / 2, cy = (w[1] + w[3]) / 2;
    if (!boxes.some(g => cx >= g.x && cx <= g.x + g.w && cy >= g.y && cy <= g.y + g.h)) out.push({word: k, text: w[4]}); });
  return out;
}
function runFind() {
  const q = fIn.value; if (!fComp(q)) { fRes.hidden = true; fHits = []; return; }
  const qn = fNorm(q);
  fHits = findCands().map(c => { let s = Math.max(fScore(q, c.text), c.label ? fScore(q, c.label) - 15 : 0);
      if (c.label && /^#?\d+$/.test(qn) && String(c.n) === qn.replace("#", "")) s = Math.max(s, qn.startsWith("#") ? 101 : 95);   // #12: region number 12 (first, before a text "12")
      if (c.id && q.trim() === c.id) s = 102;                                       // a region ID (as in Label Studio / the check lists): never changes
      return [s, c]; })
    .filter(([s]) => s > 0).sort((a, b) => b[0] - a[0] || String(a[1].text).length - String(b[1].text).length).slice(0, 40).map(([, c]) => c);
  fSel = 0; showFind();
}
function showFind() {
  fRes.hidden = false;
  if (!fHits.length) { fRes.innerHTML = `<div class="none">Nothing like “${esc(fIn.value)}” on this page</div>`; return; }
  fRes.innerHTML = fHits.map((c, k) => c.word !== undefined
      ? `<div class="fr${k === fSel ? " on" : ""}" data-k="${k}"><span class="chip" style="background:none;border:1.5px dashed #e67e22"></span><span class="tx">${esc(c.text)}</span><small>text on the drawing, no box</small></div>`
      : `<div class="fr${k === fSel ? " on" : ""}" data-k="${k}"><span class="chip" style="background:${col(c.label)}"></span><span class="tx">${c.text ? esc(c.text) : `<i>${esc(c.label)}</i>`}</span><small>${esc(c.label)} #${c.n}${findWhere(c.id) ? " · " + esc(findWhere(c.id)) : ""}</small></div>`).join("");
  fRes.querySelectorAll(".fr").forEach(d => d.onmousedown = ev => { ev.preventDefault(); pickFind(+d.dataset.k); });
  const on = fRes.querySelector(".fr.on"); if (on) on.scrollIntoView({block: "nearest"});
}
function pickFind(k) {
  const c = fHits[k]; if (!c) return;
  fRes.hidden = true; fIn.blur();
  const G = document.getElementById("findg"); G.innerHTML = "";
  if (c.word !== undefined) {           // a PDF word without a box: zoom to it and ring it
    const w = D.words[c.word], ww = w[2] - w[0], hh = w[3] - w[1], pd = Math.max(hh * 0.4, 2);
    zoomTo(w[0] - ww * 3, w[1] - hh * 8, w[2] + ww * 3, w[3] + hh * 8, 1);
    if (editing) drawEdit(); else if (cur >= 0) select(cur, false); else drawAll(false);
    el("rect", {x: w[0] - pd, y: w[1] - pd, width: ww + 2 * pd, height: hh + 2 * pd, rx: pd, fill: "#f39c1222", stroke: "#e67e22", "stroke-width": 3, "stroke-dasharray": "7 4", "vector-effect": "non-scaling-stroke"}, G);
    flash(`“${w[4]}”: text on the drawing that no box covers`); return; }
  const id = c.id; if (!R[id]) return;
  if (editing) { selId = id; syncList(); const [x0, y0, x1, y1] = bbox([id]), w = Math.max(x1 - x0, D.W * 0.05), h = Math.max(y1 - y0, D.H * 0.04);
    zoomTo((x0 + x1) / 2 - w * 0.8, (y0 + y1) / 2 - h * 0.9, (x0 + x1) / 2 + w * 0.8, (y0 + y1) / 2 + h * 0.9, 1); drawEdit(); return; }
  const i = D.items.findIndex(it => it.id === id);
  if (i >= 0 && !notBeam) { select(i); return; }
  const [x0, y0, x1, y1] = bbox([id]), w = Math.max(x1 - x0, D.W * 0.05), h = Math.max(y1 - y0, D.H * 0.04);
  zoomTo((x0 + x1) / 2 - w * 0.8, (y0 + y1) / 2 - h * 0.9, (x0 + x1) / 2 + w * 0.8, (y0 + y1) / 2 + h * 0.9, 1);
  drawAll(true); showRegion(id);
}
fIn.oninput = runFind;
fIn.onfocus = () => { if (fComp(fIn.value)) runFind(); };
fIn.onblur = () => { setTimeout(() => { fRes.hidden = true; }, 150); };
fIn.onkeydown = e => {
  if (e.key === "ArrowDown" || e.key === "ArrowUp") { e.preventDefault(); if (!fHits.length) return; fSel = (fSel + (e.key === "ArrowDown" ? 1 : fHits.length - 1)) % fHits.length; showFind(); }
  else if (e.key === "Enter") { e.preventDefault(); pickFind(fSel); }
  else if (e.key === "Escape") { e.preventDefault(); fRes.hidden = true; fIn.blur(); }
};

// ---- coordinates beside the cursor (F6), like AutoCAD: % across / % down the sheet, as the annotation check lists give them ----
const XYKEY = "relreview:coords", cTip = document.getElementById("coordtip"), cBtn = document.getElementById("coordbtn");
let COORDS = false; try { COORDS = localStorage.getItem(XYKEY) === "1"; } catch (e) {}
function toggleCoords() {
  COORDS = !COORDS; try { localStorage.setItem(XYKEY, COORDS ? "1" : "0"); } catch (e) {}
  cBtn.classList.toggle("on", COORDS); if (!COORDS) cTip.hidden = true;
  flash("Coordinates beside the cursor " + (COORDS ? "ON" : "OFF") + " (F6)");
}
cBtn.onclick = toggleCoords; cBtn.classList.toggle("on", COORDS);
svg.addEventListener("mousemove", e => {
  if (!COORDS) return;
  const p = toSvg(e), x = p.x / D.W * 100, y = p.y / D.H * 100;
  if (x < 0 || y < 0 || x > 100 || y > 100) { cTip.hidden = true; return; }
  const id = regionAt(p), g = id && R[id];
  cTip.innerHTML = `${x.toFixed(1)}% across, ${y.toFixed(1)}% down` + (g ? `<small>${esc(g.label)} #${g.n}${g.text ? " “" + esc(g.text) + "”" : ""}</small>` : "");
  cTip.hidden = false;
  const r = cTip.getBoundingClientRect(), left = e.clientX + 16 + r.width > innerWidth ? e.clientX - 12 - r.width : e.clientX + 16,
        top = e.clientY + 18 + r.height > innerHeight ? e.clientY - 10 - r.height : e.clientY + 18;
  cTip.style.left = left + "px"; cTip.style.top = top + "px";
});
svg.addEventListener("mouseleave", () => { cTip.hidden = true; });

// ---- keys for the tools above, and the theme ----
document.getElementById("themebtn").onclick = toggleTheme;
window.addEventListener("keydown", e => {
  if (e.key === "Shift") shiftHeld = true;
  if ((e.ctrlKey || e.metaKey) && !e.altKey && e.key.toLowerCase() === "f") { e.preventDefault(); e.stopImmediatePropagation(); fIn.focus(); fIn.select(); return; }
  if (["INPUT", "SELECT", "TEXTAREA"].includes(e.target.tagName)) return;
  const stop = () => { e.preventDefault(); e.stopImmediatePropagation(); };
  if (e.key === "F6") { stop(); toggleCoords(); return; }
  if (e.key === "F3") { stop(); toggleSnap("on"); } else if (e.key === "F4") { stop(); toggleSnap("perp"); } else if (e.key === "F8") { stop(); toggleSnap("ortho"); }
  else if (e.key.toLowerCase() === "t" && !e.ctrlKey && !e.metaKey && !e.altKey) { stop(); toggleTheme(); }
}, true);

recompute(); applyNotBeam(); applyTags(); list(); fit(); drawIssues(); if (D.items.length && !notBeam) select(0);
new MutationObserver(() => placeIleg()).observe(document.getElementById("edbar"), {attributes: true, attributeFilter: ["hidden"]});
</script></body></html>"""


INDEX_COLS = [("task", "Task", "num"), ("image", "Image", "txt"), ("consultant", "Consultant", "cat"), ("test", "Test", "cat"),
              ("items", "Items", "num"), ("problems", "Regions with problems", "num"), ("marks", "Your marks", "txt"),
              ("source", "From", "cat"), ("edited", "Last edited", "txt"), ("export", "Export", "cat"), ("made", "Made", "txt")]

INDEX_CSS = """
:root{--bg:#fff;--ink:#222;--mute:#666;--line:#e3e3e3;--hov:#f4f7fd;--link:#1a56c4;--bad:#b00020;--ok:#1e8e3e;--panel:#fff;--sel:#1a73e8}
html[data-theme=dark]{color-scheme:dark;--bg:#1b1d21;--ink:#e3e5e8;--mute:#9ba1a9;--line:#3b4047;--hov:#262a30;--link:#7fb0ff;--bad:#ff6f61;--ok:#4cc26d;--panel:#25282d;--sel:#5b9dff}
body{font:14px system-ui,-apple-system,'Segoe UI',sans-serif;margin:24px;color:var(--ink);background:var(--bg)}h2{margin:0 0 4px}
p{color:var(--mute);margin:0 0 10px}table{border-collapse:collapse}td,th{padding:6px 12px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}
th{color:var(--mute);font-weight:600}a{font-weight:600;color:var(--link);text-decoration:none}a:hover{text-decoration:underline}.p{color:var(--bad)}
tr.new td[data-c=task]::after{content:' new';color:var(--ok);font-size:11px;font-weight:600}tbody tr:hover{background:var(--hov)}.mk{color:var(--mute)}.src{color:var(--mute);font-size:12px}
th .ttl{cursor:pointer;user-select:none}th .ttl:hover{color:var(--ink)}th.on[data-dir=asc] .ttl::after{content:' ▲'}th.on[data-dir=desc] .ttl::after{content:' ▼'}
tr.flt th{padding:2px 6px 8px;font-weight:normal}tr.flt input,tr.flt select{font:inherit;font-size:12px;width:100%;min-width:60px;padding:2px 5px;border:1px solid var(--line);border-radius:4px;background:var(--panel);color:var(--ink)}
tr.flt .act{outline:2px solid var(--sel);outline-offset:-1px}
td.tk,th.tk{width:28px;padding-right:2px}td.tk input,th.tk input{width:16px;height:16px;cursor:pointer;margin:0}tr.ticked{background:color-mix(in srgb,var(--sel) 10%,transparent)}
.bar{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin:0 0 10px}.bar .sp{flex:1}
button{font:inherit;padding:4px 10px;border:1px solid var(--line);background:var(--panel);color:var(--ink);border-radius:6px;cursor:pointer}button:hover{border-color:#999}
button.pri{background:var(--sel);border-color:var(--sel);color:#fff}button:disabled{opacity:.45;cursor:default}
#msg{margin:0 0 10px;padding:6px 10px;border-radius:6px;background:var(--hov);white-space:normal;max-width:1100px}#count{color:var(--mute)}
.js{color:var(--ok);font-size:11px;font-weight:600;margin-left:4px}
"""

INDEX_JS = r"""<script>
const rows = [...document.querySelectorAll("tbody tr[data-k]")], tbody = document.querySelector("tbody");
const cell = (tr, c) => tr.querySelector(`td[data-c="${c}"]`);
const read = k => { try { return JSON.parse(localStorage.getItem(k) || "null"); } catch (e) { return null; } };
rows.forEach(tr => {
  // "Your marks": what you did on each page in this browser (ticks, edits, Not a beam drawing), as the pages keep it
  const k = tr.dataset.k, mk = cell(tr, "marks"), parts = [], st = read("relreview-edit:" + k), marks = read("relreview:" + k) || {};
  if (st && st.n && st.stamp === tr.dataset.stamp) {
    let nb; try { nb = JSON.parse(st.state).notBeam; } catch (e) {}
    const saved = st.saved ? "saved" : "<b>not saved to a file yet</b>";
    parts.push(nb ? "🚫 not a beam drawing · " + saved : nb === false && mk.textContent ? "unticked: a beam drawing · " + saved
                  : "✎ " + st.n + " edit" + (st.n > 1 ? "s" : "") + " · " + saved);
  } else if (mk.innerHTML) parts.push(mk.innerHTML);
  const v = Object.values(marks), ok = v.filter(x => x === "ok").length, bad = v.filter(x => x === "bad").length;
  if (ok || bad) parts.push(ok + " ✓" + (bad ? " " + bad + " ✗" : ""));
  mk.innerHTML = parts.join(" · ");
  // Last edited: the newer of the annotation's own time (Label Studio / saved review JSON) and this browser's edits
  const ed = cell(tr, "edited"); let t = +ed.dataset.t || 0, src = ed.dataset.src;
  if (st && st.n && st.stamp === tr.dataset.stamp && st.t > t) { t = st.t; src = "this browser" + (st.saved ? "" : ", not saved to a file yet"); }
  const p2 = n => String(n).padStart(2, "0"), d = new Date(t);
  ed.dataset.sort = t;
  ed.innerHTML = t ? `${d.getFullYear()}-${p2(d.getMonth() + 1)}-${p2(d.getDate())} ${p2(d.getHours())}:${p2(d.getMinutes())} <span class="src">${src}</span>` : `<span class="src">not edited</span>`;
  // the edited JSON this browser keeps for the page (what Export takes)
  const tj = read("relreview-task:" + k), ready = !!(tj && tj.task && tj.stamp === tr.dataset.stamp);
  tr.dataset.json = ready ? "1" : "";
  const box = tr.querySelector(".tk input");
  box.title = ready ? "Tick to export this page's edited JSON" : "No edits kept in this browser for this page: nothing to export (open the page; if it offers to restore edits, restore them)";
  if (ready) cell(tr, "task").insertAdjacentHTML("beforeend", `<span class="js" title="Edited JSON kept in this browser: tick and Export">JSON</span>`);
});

// ---- filter and sort on every column, kept in this browser (they come back next time) ----
const VKEY = "relreview-index:view", heads = [...document.querySelectorAll("tr.hd th[data-c]")];
let view = {sort: null, f: {}}; Object.assign(view, read(VKEY) || {}); view.f = view.f || {};
const saveView = () => { try { localStorage.setItem(VKEY, JSON.stringify(view)); } catch (e) {} };
const txt = (tr, c) => [...cell(tr, c).childNodes].filter(n => !(n.classList && n.classList.contains("js"))).map(n => n.textContent).join("").trim();
const num = (tr, c) => { const e = cell(tr, c); return e.dataset.sort !== undefined ? +e.dataset.sort : parseFloat(e.textContent.replace(/[^\d.\-]+/g, " ")) || 0; };
const flt = document.querySelector("tr.flt");
heads.forEach(th => {
  const c = th.dataset.c, type = th.dataset.type, f = document.createElement("th"); f.dataset.c = c; flt.appendChild(f);
  if (type === "cat") {
    const vals = [...new Set(rows.map(tr => txt(tr, c)))].sort((a, b) => a.localeCompare(b, undefined, {numeric: true}));
    f.innerHTML = `<select><option value="">all</option>${vals.map(v => `<option value="${v.replace(/"/g, "&quot;")}">${v || "(blank)"}</option>`).join("")}</select>`;
  } else f.innerHTML = `<input placeholder="${type === "num" ? "e.g. >0" : "filter"}" title="${type === "num" ? "A number, or >5, <10, >=1, <=3, =0, !=0 - or any text it contains" : "Text it contains (any case) · start with ! for rows that don't contain it"}">`;
  const inp = f.firstElementChild; inp.value = view.f[c] || "";
  if (inp.value !== (view.f[c] || "")) { delete view.f[c]; inp.value = ""; }       // a value no row has any more
  inp.oninput = inp.onchange = () => { if (inp.value) view.f[c] = inp.value; else delete view.f[c]; saveView(); apply(); };
  th.querySelector(".ttl").onclick = () => { const s = view.sort;
    view.sort = !s || s.c !== c ? {c, dir: "asc"} : s.dir === "asc" ? {c, dir: "desc"} : null; saveView(); apply(); };
});
function match(tr, c, q) {
  const type = heads.find(h => h.dataset.c === c).dataset.type, t = txt(tr, c).toLowerCase();
  if (type === "cat") return txt(tr, c) === q;
  q = q.trim().toLowerCase(); if (!q) return true;
  const m = type === "num" && /^(<=|>=|!=|<|>|=)?\s*(-?\d+(?:\.\d+)?)$/.exec(q);
  if (m) { const x = num(tr, c), y = +m[2]; return {"<": x < y, ">": x > y, "<=": x <= y, ">=": x >= y, "!=": x !== y}[m[1]] ?? x === y; }
  return q[0] === "!" ? !t.includes(q.slice(1).trim()) : t.includes(q);
}
function apply() {
  const shown = rows.filter(tr => { const ok = Object.entries(view.f).every(([c, q]) => !cell(tr, c) || match(tr, c, q)); tr.hidden = !ok; return ok; });
  heads.forEach(th => { const on = view.sort && view.sort.c === th.dataset.c; th.classList.toggle("on", !!on); if (on) th.dataset.dir = view.sort.dir; else delete th.dataset.dir; });
  flt.querySelectorAll("th[data-c]").forEach(f => f.firstElementChild.classList.toggle("act", !!view.f[f.dataset.c]));
  const s = view.sort, h = s && heads.find(x => x.dataset.c === s.c);
  const order = h ? rows.slice().sort((a, b) => {
      const ty = h.dataset.type, x = ty === "num" || s.c === "edited" ? num(a, s.c) : txt(a, s.c).toLowerCase(), y = ty === "num" || s.c === "edited" ? num(b, s.c) : txt(b, s.c).toLowerCase();
      return (x < y ? -1 : x > y ? 1 : 0) * (s.dir === "asc" ? 1 : -1) || rows.indexOf(a) - rows.indexOf(b); }) : rows;
  order.forEach(r => tbody.appendChild(r));
  document.getElementById("clear").disabled = !s && !Object.keys(view.f).length;
  ticks(); return shown;
}
document.getElementById("clear").onclick = () => { view = {sort: null, f: {}}; saveView(); flt.querySelectorAll("input,select").forEach(i => { i.value = ""; }); apply(); };

// ---- tick boxes: pick pages, export their edited JSON (one Label Studio JSON file: push_to_ls.py takes it as it is) ----
const all = document.getElementById("tickall");
function ticks() {
  const vis = rows.filter(tr => !tr.hidden), on = rows.filter(tr => tr.querySelector(".tk input").checked), von = vis.filter(tr => tr.querySelector(".tk input").checked);
  rows.forEach(tr => tr.classList.toggle("ticked", tr.querySelector(".tk input").checked));
  all.checked = vis.length > 0 && von.length === vis.length; all.indeterminate = von.length > 0 && von.length < vis.length;
  const hid = on.length - von.length;
  document.getElementById("count").textContent = `${vis.length} of ${rows.length} page${rows.length === 1 ? "" : "s"} shown · ${on.length} ticked${hid ? ` (${hid} hidden by the filters)` : ""}`;
  const ex = document.getElementById("export"); ex.disabled = !on.length; ex.textContent = `⬇ Export ticked JSON${on.length ? ` (${on.length})` : ""}`;
}
rows.forEach(tr => tr.querySelector(".tk input").onchange = ticks);
all.onchange = () => { rows.filter(tr => !tr.hidden).forEach(tr => { tr.querySelector(".tk input").checked = all.checked; }); ticks(); };
document.getElementById("export").onclick = () => {
  const on = rows.filter(tr => tr.querySelector(".tk input").checked), tasks = [], files = [], skipped = [];
  on.forEach(tr => { const tj = read("relreview-task:" + tr.dataset.k);
    if (tj && tj.task && tj.stamp === tr.dataset.stamp) { tasks.push(tj.task); files.push(tj.file); } else skipped.push("task " + tr.dataset.task); });
  const msg = document.getElementById("msg"); msg.hidden = false;
  if (!tasks.length) { msg.innerHTML = `Nothing exported: ${skipped.join(", ")} ${skipped.length > 1 ? "have" : "has"} no edits kept in this browser. Open the page; if it offers to restore your edits, restore them, then export again.`; return; }
  const p2 = n => String(n).padStart(2, "0"), d = new Date();
  const name = tasks.length === 1 ? files[0] : `review_export_${d.getFullYear()}${p2(d.getMonth() + 1)}${p2(d.getDate())}-${p2(d.getHours())}${p2(d.getMinutes())}_${tasks.length}tasks.json`;
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([JSON.stringify(tasks, null, 1)], {type: "application/json"})); a.download = name;
  document.body.appendChild(a); a.click(); setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 1500);
  msg.innerHTML = `Exported <b>${tasks.length}</b> task${tasks.length > 1 ? "s" : ""} (${tasks.map(t => t.id).join(", ")}) to <b>${name}</b>, in your browser's downloads.` +
    (skipped.length ? ` Skipped ${skipped.join(", ")}: no edits kept in this browser.` : "") +
    ` To write it into Label Studio: <code>python push_to_ls.py "${name}" --url http://localhost:8080 --token ...</code>`;
};
document.getElementById("themebtn").onclick = toggleTheme;
apply();
</script>"""


def write_index(out, new=()):
    """review_index.html: a link to every review page in the folder (this run's and earlier ones), by task."""
    rows = []
    for f in Path(out).glob("review_*.html"):
        if f.name == "review_index.html":
            continue
        head = f.open(encoding="utf-8", errors="ignore").read(6000)
        m = re.search(r'<meta name="review-info" content="([^"]*)"', head)
        info = json.loads(html_unescape(m.group(1))) if m else {}
        if not info:                                   # a page made before the index existed: from its name
            n = re.match(r"review_(\d+)_(.+)\.html$", f.name)
            info = dict(task=int(n.group(1)), image=n.group(2)) if n else dict(task=f.stem, image="")
        rows.append((info, f))
    def key(r):
        t = str(r[0].get("task", ""))
        return (0, int(t), "") if t.isdigit() else (1, 0, t)
    cells = []
    for info, f in sorted(rows, key=key):
        when = datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        npb = info.get("problems", "")
        mk = {"label studio": "🚫 not a beam drawing (skipped in Label Studio)", "file": "🚫 not a beam drawing"}.get(info.get("not_beam"), "")
        k = f'{info.get("task")}:{info.get("image", "")}'
        co = html_escape(str(info.get("consultant") or ""))
        cells.append(f'<tr{" class=new" if f.name in new else ""} data-k="{html_escape(k)}" data-task="{html_escape(str(info.get("task")))}" '
                     f'data-stamp="{html_escape(str(info.get("stamp", "")))}"><td class="tk"><input type="checkbox"></td>'
                     f'<td data-c="task"><a href="{html_escape(f.name)}">task {html_escape(str(info.get("task")))}</a></td>'
                     f'<td data-c="image">{html_escape(str(info.get("image", "")))}</td>'
                     f'<td data-c="consultant">{co}</td>'
                     f'<td data-c="test">{"test" if info.get("test") else ""}</td><td data-c="items">{info.get("items", "")}</td>'
                     f'<td data-c="problems" class="{"p" if npb else ""}">{npb}</td><td data-c="marks" class="mk">{html_escape(mk)}</td>'
                     f'<td data-c="source">{html_escape(str(info.get("source", "")))}</td>'
                     f'<td data-c="edited" data-t="{info.get("edited") or ""}" data-src="{html_escape(str(info.get("edited_src", "")))}"></td>'
                     f'<td data-c="export">{html_escape(str(info.get("export", "")))}</td><td data-c="made">{when}</td></tr>')
    heads = "".join(f'<th data-c="{c}" data-type="{t}"><span class="ttl" title="Sort by {h} (click again: the other way, a third time: no sort)">{h}</span></th>'
                    for c, h, t in INDEX_COLS)
    idx = Path(out) / "review_index.html"
    idx.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>Review pages</title>" + THEME_HEAD + "<style>" + INDEX_CSS + "</style></head><body>"
        f"<h2>Review pages</h2><p>{len(rows)} page(s) in this folder · click a task to open it · "
        "<b>◀ Review pages</b> on each page comes back here · <b>Your marks</b>: what you did on each page in this "
        "browser · <b>Last edited</b>: when you last changed the annotation (in Label Studio, a saved review JSON, or "
        "this browser) · click a column title to sort, type or pick under it to filter (both kept for next time) · "
        "tick pages and <b>Export</b> their edited JSON in one file (<span class='js'>JSON</span>: edits kept in this browser)</p>"
        "<div class='bar'><button id='export' class='pri' disabled>⬇ Export ticked JSON</button><button id='clear' title='Show every page, in task order'>"
        "✕ Clear filters &amp; sort</button><span id='count'></span><span class='sp'></span>"
        "<button id='themebtn' title='Light / dark theme, kept for every review page'>◐ Theme</button></div><div id='msg' hidden></div>"
        "<table><thead><tr class='hd'><th class='tk'><input type='checkbox' id='tickall' title='Tick / untick every page shown'></th>" + heads +
        "</tr><tr class='flt'><th class='tk'></th></tr></thead><tbody>" + "".join(cells) +
        "</tbody></table>" + INDEX_JS + "</body></html>", encoding="utf-8")
    return idx


def main():
    ap = argparse.ArgumentParser(description="One-label-at-a-time relation review page for a Label Studio JSON export")
    ap.add_argument("export", help="Label Studio export (JSON)")
    ap.add_argument("--images", required=True, help="folder with the page images you uploaded")
    ap.add_argument("--pdf-dir", help="optional: folder with the PDFs (shows label texts)")
    ap.add_argument("--task", type=int, action="append", help="only these task ids (repeatable)")
    ap.add_argument("--out", default=".", help="output folder")
    ap.add_argument("--max-width", type=int, default=3600, help="page image width in the review page (px)")
    a = ap.parse_args()
    try:
        tasks = json.loads(Path(a.export).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        sys.exit(f"{a.export}: not a Label Studio JSON export ({e}). In Label Studio: Export > JSON, then pick that "
                 f"file (project-<n>-at-<date>.json).")
    if isinstance(tasks, dict):
        tasks = [tasks]
    tasks = [t for t in tasks if isinstance(t, dict)]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    print(f"{a.export}: {len(tasks)} task(s)" + (f", of which {sum(t.get('id') in a.task for t in tasks)} asked for" if a.task else ""))
    written, made = 0, []
    names = sorted({str((t.get("meta") or {}).get("consultant") or "").strip() for t in tasks} - {""})
    for t in tasks:
        if a.task and t.get("id") not in a.task:
            continue
        data = build_task(t, a.images, a.pdf_dir, a.max_width)
        if not data:
            continue
        written += 1
        data["consultants"] = names
        title = f"task {data['task']} · {data['image']}"
        img = data.pop("img")
        n_prob = sum(bool(r["problems"]) for r in data["regions"].values())
        skip = data["task_meta"]["skip_src"]
        info = dict(task=data["task"], image=data["image"], items=len(data["items"]), problems=n_prob, source=data["source"],
                    export=Path(a.export).name, not_beam=skip, stamp=data["task_meta"]["stamp"],
                    edited=data["task_meta"]["edited"], edited_src=data["task_meta"]["edited_src"],
                    consultant=data["task_meta"]["consultant"], test=data["task_meta"]["test"])
        html = PAGE.replace("__THEME_HEAD__", THEME_HEAD).replace("__TITLE__", title).replace("__INFO__", html_escape(json.dumps(info), quote=True)) \
                   .replace("__IMG__", img).replace("__DATA__", json.dumps(data).replace("</", "<\\/"))
        f = out / f"review_{data['task']}_{Path(data['image']).stem}.html"
        f.write_text(html, encoding="utf-8")
        nb = {"label studio": ", skipped in Label Studio: not a beam drawing", "file": ", marked not a beam drawing"}.get(skip, "")
        print(f"{f.resolve()}  ({len(data['items'])} items, {n_prob} regions with problems, from {data['source']}{nb})")
        made.append((data["task"], data["image"], f.name, len(data["items"]), n_prob, data["source"]))
    if made:
        print(f"index of all review pages in this folder: {write_index(out, {m[2] for m in made}).resolve()}")
    print(f"{written} review page(s) written to {out.resolve()}" if written else
          "No review page written: see the reasons above (no tasks in the export, or their images not in the image folder).")


if __name__ == "__main__":
    main()
