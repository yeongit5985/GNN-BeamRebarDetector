"""
PDF access for the beam tools. pypdfium2 (Chrome's PDF engine, Apache-2.0 / BSD-3 licences) behind the small
part of PyMuPDF's API that beam_detector, graph_builder, review_relations and render_pages use, so their code
stays as it was:

    from pdf_backend import pymupdf
    page = pymupdf.open("drawing.pdf")[0]
    page.remove_rotation()            # nothing to do here: coordinates are always as the page is displayed
    page.get_drawings()               # vector paths: items "l" line, "c" curve, "re" rectangle, "qu" quad
    page.get_text("rawdict")          # characters with boxes, grouped in lines and spans (size, font, flags, color)
    page.get_text("dict" | "words" | "text"),  page.get_textbox(rect)
    page.get_pixmap(dpi=300, clip=pymupdf.Rect(x0, y0, x1, y1), colorspace=pymupdf.csGRAY)

What PyMuPDF does is reproduced, so the detector and the graphs come out the same (checked on three sheets):
coordinates in single precision, from the top-left corner of the page as displayed (its /Rotate applied, as
after PyMuPDF's remove_rotation()); paths as PyMuPDF lists them (fill + stroke merged to "fs", four lines that
close = "qu", three lines + close forming an upright rectangle = "re", a line to the same point dropped unless
it is a dot); characters in drawing order, boxed from the origin by the font's advance width and from
ascender to descender scaled to the font size; lines and spaces by MuPDF's rules (same direction, < 0.8 x size
off the baseline and along it; a gap of 0.15 x size = a space; a character printed twice on itself = once);
clipped images on the same pixel grid as the whole page.
Differences that remain: text printed twice on top of itself as a whole string is kept once (PyMuPDF kept
both; the detector removed the copy anyway), repeated spaces are one, colours are 8-bit (1/255 steps), a
rectangle drawn as four lines + close is "re" (PyMuPDF: "qu"), and rendering is Chrome's (OCR of outlined
text can read a character differently). PDF layers are not provided ("layer" is always "").

Set the environment variable BEAM_PDF=pymupdf to use PyMuPDF itself (AGPL licence: not for a product you sell
or host unless you buy Artifex's commercial licence).
"""
import ctypes
import math
import os
from types import SimpleNamespace

import numpy as np

BACKEND = "pymupdf" if os.environ.get("BEAM_PDF", "").strip().lower() == "pymupdf" else "pypdfium2"

if BACKEND == "pymupdf":
    try:
        import pymupdf
    except ImportError:
        import fitz as pymupdf
else:
    import pypdfium2 as pdfium
    import pypdfium2.raw as R

    class Rect:
        def __init__(self, *a):
            if len(a) == 1:
                a = tuple(a[0])
            if len(a) == 2:
                a = (*a[0], *a[1])
            self.x0, self.y0, self.x1, self.y1 = (float(v) for v in a)

        width = property(lambda s: s.x1 - s.x0)
        height = property(lambda s: s.y1 - s.y0)

        def __iter__(self):
            return iter((self.x0, self.y0, self.x1, self.y1))

        def __getitem__(self, i):
            return (self.x0, self.y0, self.x1, self.y1)[i]

        def __eq__(self, o):
            return isinstance(o, Rect) and tuple(self) == tuple(o)

        def __repr__(self):
            return f"Rect({self.x0}, {self.y0}, {self.x1}, {self.y1})"

        @property
        def quad(self):
            return Quad((self.x0, self.y0), (self.x1, self.y0), (self.x0, self.y1), (self.x1, self.y1))

    class Point(tuple):
        """(x, y) that also has .x and .y, like PyMuPDF's Point."""
        def __new__(cls, x, y):
            return super().__new__(cls, (x, y))

        x = property(lambda s: s[0])
        y = property(lambda s: s[1])

    class Quad:
        """Four corners, like PyMuPDF's Quad (ul, ur, ll, lr)."""
        def __init__(self, ul, ur, ll, lr):
            self.ul, self.ur, self.ll, self.lr = Point(*ul), Point(*ur), Point(*ll), Point(*lr)

        def __iter__(self):
            return iter((self.ul, self.ur, self.ll, self.lr))

        def __eq__(self, o):
            return isinstance(o, Quad) and tuple(self) == tuple(o)

        rect = property(lambda s: Rect(min(p[0] for p in s), min(p[1] for p in s), max(p[0] for p in s), max(p[1] for p in s)))

        def __repr__(self):
            return f"Quad({self.ul}, {self.ur}, {self.ll}, {self.lr})"

    class Pixmap:
        def __init__(self, arr):
            arr = np.ascontiguousarray(arr)
            self.height, self.width = arr.shape[:2]
            self.n = 1 if arr.ndim == 2 else arr.shape[2]
            self._arr = arr
            self.samples = arr.tobytes()

        def save(self, path):
            from PIL import Image
            Image.fromarray(self._arr if self.n != 1 or self._arr.ndim == 2 else self._arr[..., 0]).save(path)

    def _mul(A, B):                       # PDF matrices, row-vector convention: p * A * B
        a, b, c, d, e, f = A
        a2, b2, c2, d2, e2, f2 = B
        return (a * a2 + b * c2, a * b2 + b * d2, c * a2 + d * c2, c * b2 + d * d2, e * a2 + f * c2 + e2, e * b2 + f * d2 + f2)

    def _expansion(M):                    # how much a matrix scales lengths (MuPDF's fz_matrix_expansion)
        return math.sqrt(abs(M[0] * M[3] - M[1] * M[2]))

    _F32 = ctypes.c_float

    def _f32(v):                          # MuPDF works in single precision: the same point gives the same number
        return _F32(v).value

    class Page:
        def __init__(self, doc, number):
            self._doc, self.number = doc, number
            self._page = doc._pdf[number]
            self.rotation = self._page.get_rotation() % 360
            self._rot = self.rotation
            self._box = tuple(float(v) for v in self._page.get_cropbox())
            x0, y0, x1, y1 = self._box
            self.rect = Rect(0, 0, y1 - y0, x1 - x0) if self._rot in (90, 270) else Rect(0, 0, x1 - x0, y1 - y0)
            self._chars = None

        def remove_rotation(self):
            """Coordinates here are always as displayed (the rotation applied), as after PyMuPDF's remove_rotation()."""
            self.rotation = 0

        # ---- PDF space -> displayed page (top-left origin, y down)
        def _xy(self, x, y):
            x0, y0, x1, y1 = self._box
            r = self._rot
            if r == 90:
                X, Y = y - y0, x - x0
            elif r == 180:
                X, Y = x1 - x, y - y0
            elif r == 270:
                X, Y = y1 - y, x1 - x
            else:
                X, Y = x - x0, y1 - y
            return _f32(X), _f32(Y)

        def _dir(self, dx, dy):
            r = self._rot
            v = (dy, dx) if r == 90 else (-dx, dy) if r == 180 else (-dy, -dx) if r == 270 else (dx, -dy)
            n = math.hypot(*v)
            return (v[0] / n, v[1] / n) if n else (1.0, 0.0)

        # ---- vector paths
        def get_drawings(self):
            out = []

            def walk(parent, M, nested):
                count = R.FPDFFormObj_CountObjects(parent) if nested else R.FPDFPage_CountObjects(parent)
                for k in range(count):
                    obj = R.FPDFFormObj_GetObject(parent, k) if nested else R.FPDFPage_GetObject(parent, k)
                    mt = R.FS_MATRIX()
                    R.FPDFPageObj_GetMatrix(obj, ctypes.byref(mt))
                    Mo = _mul((mt.a, mt.b, mt.c, mt.d, mt.e, mt.f), M)
                    typ = R.FPDFPageObj_GetType(obj)
                    if typ == R.FPDF_PAGEOBJ_FORM:
                        walk(obj, Mo, True)
                    elif typ == R.FPDF_PAGEOBJ_PATH:
                        for p in self._path(obj, Mo):
                            prev = out[-1] if out else None
                            # a fill, then a stroke with the same items = one "fs" path (as PyMuPDF merges them)
                            if prev and prev["type"] == "f" and p["type"] == "s" and prev["items"] == p["items"]:
                                for k, v in p.items():
                                    if prev.get(k) is None:
                                        prev[k] = v
                                prev["type"] = "fs"
                            else:
                                out.append(p)

            walk(self._page.raw, (1, 0, 0, 1, 0, 0), False)
            return out

        @staticmethod
        def _commands(segs):
            """PDFium path segments [(type, point, close)] -> the path as MuPDF builds it: a line to where the
            pen already is is dropped (unless it follows a move: a dot), a curve whose control points sit on its
            ends is a line, a repeated close is dropped, a move after a move replaces it. PDFium writes every
            close as a line back to the start flagged 'close': that becomes the close itself."""
            cmds, cur, start, last, bez = [], None, None, None, []
            for typ, p, close in segs:
                if typ == R.FPDF_SEGMENT_MOVETO:
                    if last == "M":
                        cmds.pop()
                    cmds.append(("M", p)); cur = start = p; last = "M"; bez = []
                    continue
                if cur is None:
                    continue
                if typ == R.FPDF_SEGMENT_LINETO:
                    if not (close and p == start):
                        if last == "M" or p != cur:
                            cmds.append(("L", p)); last = "L"
                        cur = p
                elif typ == R.FPDF_SEGMENT_BEZIERTO:
                    bez.append(p)
                    if len(bez) < 3:
                        continue
                    c1, c2, p3 = bez; bez = []
                    if (cur == c1 and (c2 == p3 or c1 == c2)) or (c2 == p3 and c1 == c2):
                        if not (cur == c1 == c2 == p3 and last != "M"):
                            cmds.append(("L", p3)); last = "L"
                    else:
                        cmds.append(("C", c1, c2, p3)); last = "C"
                    cur = p3
                if close and last != "H":
                    cmds.append(("H",)); last = "H"; cur = start
            return cmds

        @staticmethod
        def _items(cmds, fill_only):
            """MuPDF path -> PyMuPDF's item list: lines 'l', curves 'c'; four lines that end where they began
            become one quad 'qu' (not for a pure fill); three lines + close that form an upright rectangle
            become 're'; a close adds the line back to the start. -> (items, closePath)"""
            items, last = [], None
            first, havemove, n_lines, closed = None, False, 0, False
            for cmd in cmds:
                op = cmd[0]
                if op == "M":
                    last = first = Point(*cmd[1]); havemove, n_lines = True, 0
                elif op == "L":
                    p = Point(*cmd[1])
                    items.append(("l", last, p)); last = p; n_lines += 1
                    if n_lines == 4 and not fill_only and p == items[-4][1]:
                        q = [it[1] for it in items[-4:]]
                        items[-4:] = [("qu", Quad(q[0], q[3], q[1], q[2]))]; n_lines = 0
                elif op == "C":
                    p = Point(*cmd[3])
                    items.append(("c", last, Point(*cmd[1]), Point(*cmd[2]), p)); last = p; n_lines = 0
                else:                                                     # close
                    if n_lines == 3:
                        n_lines = 0
                        (_, ll, lr), (_, ur, ul) = items[-3], items[-1]
                        if ll[1] == lr[1] and ll[0] == ul[0] and ur[1] == ul[1] and ur[0] == lr[0]:
                            o = 1 if ul[1] < lr[1] else -1
                            items[-3:] = [("re", Rect(min(ll[0], ur[0]), min(ll[1], ur[1]), max(ll[0], ur[0]), max(ll[1], ur[1])), o)]
                            continue
                    n_lines = 0
                    if havemove:
                        if last != first:
                            items.append(("l", last, first)); last = first
                        havemove = False
                    else:
                        closed = True
            return items, closed

        def _path(self, obj, M):
            """A PDFium path object -> PyMuPDF path dict(s): one, or a fill and a stroke when PyMuPDF's
            fill and stroke item lists differ (a quad is only made for the stroke)."""
            fm, st = ctypes.c_int(), ctypes.c_int()
            R.FPDFPath_GetDrawMode(obj, ctypes.byref(fm), ctypes.byref(st))
            fill, stroke = fm.value != 0, bool(st.value)
            if not fill and not stroke:
                return []
            a, b, c, d, e, f = M
            segs = []
            x, y = ctypes.c_float(), ctypes.c_float()
            for k in range(R.FPDFPath_CountSegments(obj)):
                sg = R.FPDFPath_GetPathSegment(obj, k)
                R.FPDFPathSegment_GetPoint(sg, ctypes.byref(x), ctypes.byref(y))
                segs.append((R.FPDFPathSegment_GetType(sg),
                             self._xy(x.value * a + y.value * c + e, x.value * b + y.value * d + f),
                             bool(R.FPDFPathSegment_GetClose(sg))))
            cmds = self._commands(segs)
            f_items, f_closed = self._items(cmds, True) if fill else (None, False)
            s_items, s_closed = self._items(cmds, False) if stroke else (None, False)
            if not (f_items or s_items):
                return []
            rgba = [ctypes.c_uint() for _ in range(4)]
            def color(getter):
                getter(obj, *(ctypes.byref(v) for v in rgba))
                return tuple(v.value / 255 for v in rgba[:3]), rgba[3].value / 255
            fill_c, fill_a = color(R.FPDFPageObj_GetFillColor) if fill else (None, 1.0)
            stroke_c, stroke_a = color(R.FPDFPageObj_GetStrokeColor) if stroke else (None, 1.0)
            width = dashes = None
            if stroke:
                w = ctypes.c_float()
                R.FPDFPageObj_GetStrokeWidth(obj, ctypes.byref(w))
                ex = _expansion(M)
                width = _f32(w.value * ex)
                n = R.FPDFPageObj_GetDashCount(obj)
                if n > 0:
                    arr = (ctypes.c_float * n)()
                    R.FPDFPageObj_GetDashArray(obj, arr, n)
                    ph = ctypes.c_float()
                    R.FPDFPageObj_GetDashPhase(obj, ctypes.byref(ph))
                    dashes = "[ " + " ".join(f"{v * ex:g}" for v in arr) + f" ] {ph.value * ex:g}"
                else:
                    dashes = "[] 0"

            def bounds(items):
                pts = [q for it in items for v in it[1:] if not isinstance(v, int)
                       for q in (v.quad if isinstance(v, Rect) else v if isinstance(v, Quad) else (v,))]
                return Rect(min(p[0] for p in pts), min(p[1] for p in pts), max(p[0] for p in pts), max(p[1] for p in pts))

            even_odd = fm.value == R.FPDF_FILLMODE_ALTERNATE
            fill_d = dict(items=f_items, type="f", closePath=f_closed or None, even_odd=even_odd, fill=fill_c,
                          fill_opacity=fill_a, color=None, width=None, dashes=None, stroke_opacity=None,
                          lineCap=None, lineJoin=None, layer="", rect=bounds(f_items)) if f_items else None
            stroke_d = dict(items=s_items, type="s", closePath=s_closed, even_odd=None, fill=None, fill_opacity=None,
                            color=stroke_c, width=width, dashes=dashes, stroke_opacity=stroke_a,
                            lineCap=(R.FPDFPageObj_GetLineCap(obj),) * 3, lineJoin=R.FPDFPageObj_GetLineJoin(obj),
                            layer="", rect=bounds(s_items)) if s_items else None
            if fill_d and stroke_d and f_items == s_items:            # PyMuPDF: one "fs" path
                for k, v in stroke_d.items():
                    if fill_d.get(k) is None:
                        fill_d[k] = v
                fill_d["type"] = "fs"
                return [fill_d]
            return [p for p in (fill_d, stroke_d) if p]

        # ---- text
        def _load_chars(self):
            if self._chars is not None:
                return self._chars
            tp = self._page.get_textpage()
            raw, chars = tp.raw, []
            box, mt = R.FS_RECTF(), R.FS_MATRIX()
            buf, fl = ctypes.create_string_buffer(256), ctypes.c_int()
            ox, oy = ctypes.c_double(), ctypes.c_double()
            fw, fa, fd = ctypes.c_float(), ctypes.c_float(), ctypes.c_float()
            rgba = [ctypes.c_uint() for _ in range(4)]
            for i in range(tp.count_chars()):
                code = R.FPDFText_GetUnicode(raw, i)
                ch = "-" if code == 2 else chr(code) if code else ""     # PDFium marks a line-end hyphen as 0x02
                if R.FPDFText_IsGenerated(raw, i):
                    chars.append(dict(gen=ch))          # a space or line break the engine inferred
                    continue
                if not ch or ch in "\r\n":
                    chars.append(dict(gen="\n"))
                    continue
                R.FPDFText_GetMatrix(raw, i, ctypes.byref(mt))
                fs = R.FPDFText_GetFontSize(raw, i)
                size = fs * math.sqrt(abs(mt.a * mt.d - mt.b * mt.c) or 1.0)
                R.FPDFText_GetCharOrigin(raw, i, ctypes.byref(ox), ctypes.byref(oy))
                o = self._xy(ox.value, oy.value)
                u = self._dir(mt.a, mt.b) if (mt.a or mt.b) else (1.0, 0.0)
                v = (-u[1], u[0])                                   # across the baseline, towards the descenders
                # Character box as PyMuPDF makes it: from the origin along the baseline by the glyph's advance
                # (the PDF font's widths), across it from ascender to descender scaled to the font size.
                R.FPDFText_GetLooseCharBox(raw, i, ctypes.byref(box))
                lb = [self._xy(box.left, box.bottom), self._xy(box.right, box.top)]
                lx, ly = (lb[0][0], lb[1][0]), (lb[0][1], lb[1][1])
                pa = [(x - o[0]) * u[0] + (y - o[1]) * u[1] for x in lx for y in ly]
                pv = [(x - o[0]) * v[0] + (y - o[1]) * v[1] for x in lx for y in ly]
                a1 = max(pa)                                        # PDFium's loose box: fallbacks
                asc, dsc = (-min(pv) / size, -max(pv) / size) if size else (0.8, -0.2)
                obj = R.FPDFText_GetTextObject(raw, i)
                fnt = R.FPDFTextObj_GetFont(obj) if obj else None
                if fnt and fs > 0:
                    if R.FPDFFont_GetGlyphWidth(fnt, ord(ch), fs, ctypes.byref(fw)):
                        adv = fw.value * math.hypot(mt.a, mt.b)
                        straight = abs(u[0]) > 0.999 or abs(u[1]) > 0.999    # loose box = glyph box only then
                        if 0 < adv and (a1 <= 0 or ((0.5 * a1 <= adv or not straight) and adv <= 1.05 * a1 + 0.01)):
                            a1 = adv                                # else a glyph the font can't map: PDFium's box
                    if R.FPDFFont_GetAscent(fnt, fs, ctypes.byref(fa)) and R.FPDFFont_GetDescent(fnt, fs, ctypes.byref(fd)) \
                            and fa.value > fd.value:
                        asc, dsc = fa.value / fs, fd.value / fs
                if asc < 1e-3:
                    asc, dsc = 0.9, -0.1
                if asc - dsc < 1:                                   # PyMuPDF: box height = font size
                    asc, dsc = asc / (asc - dsc), dsc / (asc - dsc)
                corners = [(o[0] + a * u[0] + t * v[0], o[1] + a * u[1] + t * v[1])
                           for a in (0.0, max(a1, 0.0)) for t in (-asc * size, -dsc * size)]
                xs, ys = [p[0] for p in corners], [p[1] for p in corners]
                n = R.FPDFText_GetFontInfo(raw, i, buf, 256, ctypes.byref(fl))
                font = buf.value.decode("utf-8", "ignore") if n else ""
                weight = R.FPDFText_GetFontWeight(raw, i)
                bold = "bold" in font.lower() or weight >= 700 or bool(fl.value & 0x40000)
                flags = (16 if bold else 0) | (2 if fl.value & 64 or "italic" in font.lower() else 0) | \
                        (4 if fl.value & 2 else 0) | (8 if fl.value & 1 else 0)
                R.FPDFText_GetFillColor(raw, i, *(ctypes.byref(c) for c in rgba))
                chars.append(dict(c=ch, bbox=(_f32(min(xs)), _f32(min(ys)), _f32(max(xs)), _f32(max(ys))),
                                  origin=o, size=_f32(size), font=font, flags=flags,
                                  color=(rgba[0].value << 16) | (rgba[1].value << 8) | rgba[2].value, dir=u,
                                  adv=max(a1, 0.0), ext=(-asc * size, -dsc * size),
                                  _obj=ctypes.cast(obj, ctypes.c_void_p).value if obj else None))
            tp.close()
            # PDFium puts text that shares a baseline in reading order; PyMuPDF keeps the order it is drawn in.
            # Sort the characters back into drawing order: by their text object's place in the page content.
            order = self._content_order()
            real = sorted((k for k, c in enumerate(chars) if "c" in c),
                          key=lambda k: (order.get(chars[k]["_obj"], len(order)), k))
            chars = [chars[k] for k in real]
            for c in chars:
                c.pop("_obj", None)
            self._chars = chars
            return chars

        def _content_order(self):
            """Page objects (form contents included) -> their place in the page's drawing order."""
            order = {}

            def walk(parent, nested):
                count = R.FPDFFormObj_CountObjects(parent) if nested else R.FPDFPage_CountObjects(parent)
                for k in range(count):
                    obj = R.FPDFFormObj_GetObject(parent, k) if nested else R.FPDFPage_GetObject(parent, k)
                    if R.FPDFPageObj_GetType(obj) == R.FPDF_PAGEOBJ_FORM:
                        walk(obj, True)
                    else:
                        order[ctypes.cast(obj, ctypes.c_void_p).value] = len(order)

            walk(self._page.raw, False)
            return order

        def _lines(self):
            """Characters (in drawing order) -> lines -> spans (same font, size, colour, flags), by the rules PyMuPDF
            follows: a character stays on the line when it runs the same way and starts near where the previous one
            ended (less than 0.8 x font size off the baseline, less than 0.8 x font size further along); a gap of
            0.15 x font size or more becomes a space; the same character printed again on itself is dropped.
            PDFium's own line breaks and inferred spaces are not used."""
            lines, line = [], None
            pen = lag = last = None
            W, H = self.rect.width, self.rect.height
            for ch in self._load_chars():
                if "gen" in ch:
                    continue
                x0, y0, x1, y1 = ch["bbox"]
                if x1 < 0 or y1 < 0 or x0 > W or y0 > H:            # entirely off the page (PyMuPDF ignores it)
                    continue
                u, o, s, c = ch["dir"], ch["origin"], ch["size"] or 1.0, ch["c"]
                new_line, space = True, False
                if line is not None and u[0] * line["dir"][0] + u[1] * line["dir"][1] >= 0.999:
                    if c == last and math.hypot(o[0] - lag[0], o[1] - lag[1]) < 0.1 * s:
                        continue                                    # printed twice on the same spot (fake bold)
                    dx, dy = o[0] - pen[0], o[1] - pen[1]
                    along, across = (dx * u[0] + dy * u[1]) / s, (dx * -u[1] + dy * u[0]) / s
                    if abs(across) < 0.8 and -0.8 < along < 0.8:
                        new_line = False
                        space = along >= 0.15 and c != " " and last != " " and (ord(last) < 0x700 or 0x2000 <= ord(last) <= 0x20CF)
                if new_line:
                    if line and line["spans"]:
                        lines.append(line)
                    line = dict(dir=u, spans=[])
                sp = line["spans"][-1] if line["spans"] else None
                if sp is None or sp["font"] != ch["font"] or abs(sp["size"] - ch["size"]) > 0.01 or sp["color"] != ch["color"] \
                        or sp["flags"] != ch["flags"]:
                    sp = dict(size=ch["size"], flags=ch["flags"], font=ch["font"], color=ch["color"], origin=o, chars=[])
                    line["spans"].append(sp)
                if space:                                           # from where the previous character ended to here
                    v = (-u[1], u[0])
                    pts = [(p[0] + t * v[0], p[1] + t * v[1]) for p in (pen, o) for t in ch["ext"]]
                    sp["chars"].append(dict(c=" ", origin=pen, bbox=(_f32(min(q[0] for q in pts)), _f32(min(q[1] for q in pts)),
                                                                     _f32(max(q[0] for q in pts)), _f32(max(q[1] for q in pts)))))
                sp["chars"].append(dict(c=c, bbox=ch["bbox"], origin=o))
                pen, lag, last = (o[0] + ch["adv"] * u[0], o[1] + ch["adv"] * u[1]), o, c
            if line and line["spans"]:
                lines.append(line)
            for ln in lines:
                for sp in ln["spans"]:
                    bb = np.array([c["bbox"] for c in sp["chars"]]) if sp["chars"] else np.zeros((1, 4))
                    sp["bbox"] = (bb[:, 0].min(), bb[:, 1].min(), bb[:, 2].max(), bb[:, 3].max())
                ln["spans"] = [sp for sp in ln["spans"] if sp["chars"]]
                bb = np.array([sp["bbox"] for sp in ln["spans"]]) if ln["spans"] else np.zeros((1, 4))
                ln["bbox"] = (bb[:, 0].min(), bb[:, 1].min(), bb[:, 2].max(), bb[:, 3].max())
            return [ln for ln in lines if ln["spans"]]

        def get_text(self, kind="text", clip=None, **kw):
            lines = self._lines()
            if clip is not None:
                clip = Rect(clip)
                inside = lambda c: clip.x0 <= (c["bbox"][0] + c["bbox"][2]) / 2 <= clip.x1 and clip.y0 <= (c["bbox"][1] + c["bbox"][3]) / 2 <= clip.y1
                kept = []
                for ln in lines:
                    spans = [dict(sp, chars=[c for c in sp["chars"] if inside(c)]) for sp in ln["spans"]]
                    spans = [sp for sp in spans if sp["chars"]]
                    if spans:
                        kept.append(dict(ln, spans=spans))
                lines = kept
            if kind == "rawdict":
                blocks = [dict(type=0, number=k, bbox=ln["bbox"], lines=[dict(wmode=0, dir=ln["dir"], bbox=ln["bbox"],
                          spans=[dict(sp, chars=[dict(c) for c in sp["chars"]]) for sp in ln["spans"]])]) for k, ln in enumerate(lines)]
                return dict(width=self.rect.width, height=self.rect.height, blocks=blocks)
            if kind == "dict":
                blocks = [dict(type=0, number=k, bbox=ln["bbox"], lines=[dict(wmode=0, dir=ln["dir"], bbox=ln["bbox"],
                          spans=[dict({k2: v for k2, v in sp.items() if k2 != "chars"}, text="".join(c["c"] for c in sp["chars"]))
                                 for sp in ln["spans"]])]) for k, ln in enumerate(lines)]
                return dict(width=self.rect.width, height=self.rect.height, blocks=blocks)
            if kind == "words":
                out = []
                for k, ln in enumerate(lines):
                    cur, n = [], 0
                    for c in [c for sp in ln["spans"] for c in sp["chars"]] + [dict(c=" ")]:
                        if c["c"].isspace():
                            if cur:
                                bb = np.array([q["bbox"] for q in cur])
                                out.append((bb[:, 0].min(), bb[:, 1].min(), bb[:, 2].max(), bb[:, 3].max(),
                                            "".join(q["c"] for q in cur), k, 0, n)); n += 1; cur = []
                        else:
                            cur.append(c)
                return out
            if kind == "text":
                return "\n".join("".join(c["c"] for sp in ln["spans"] for c in sp["chars"]) for ln in lines) + ("\n" if lines else "")
            raise ValueError(f"get_text('{kind}') is not provided by pdf_backend")

        def get_textbox(self, rect):
            return self.get_text("text", clip=rect).rstrip("\n")

        # ---- rendering
        def get_pixmap(self, dpi=None, clip=None, colorspace=None, matrix=None, alpha=False, **kw):
            """The page as displayed, at dpi (or the scale of matrix), RGB or grey. With clip, the pixels of
            that part of the full-page image (same pixel grid, edges rounded outwards, as PyMuPDF does)."""
            s = (dpi or 72) / 72
            if matrix is not None:
                s = getattr(matrix, "a", s)
            W, H = self.rect.width, self.rect.height
            x0, y0, x1, y1 = (0.0, 0.0, W, H) if clip is None else Rect(clip)
            x0, y0, x1, y1 = max(x0, 0.0), max(y0, 0.0), min(x1, W), min(y1, H)
            X0, Y0 = math.floor(x0 * s + 0.001), math.floor(y0 * s + 0.001)
            w, h = max(1, math.ceil(x1 * s - 0.001) - X0), max(1, math.ceil(y1 * s - 0.001) - Y0)
            gray = colorspace == "gray"
            bm = pdfium.PdfBitmap.new_native(w, h, R.FPDFBitmap_Gray if gray else R.FPDFBitmap_BGR, rev_byteorder=not gray)
            bm.fill_rect((255, 255, 255, 255), 0, 0, w, h)
            # PDFium first maps the page onto whole points (page size truncated), then applies this matrix:
            # undo the truncation so the scale is exactly dpi / 72, and shift the clip to the bitmap's corner
            raw = self._page.raw
            Wi, Hi = int(R.FPDF_GetPageWidthF(raw)) or 1, int(R.FPDF_GetPageHeightF(raw)) or 1
            m = R.FS_MATRIX(s * W / Wi, 0, 0, s * H / Hi, -X0, -Y0)
            flags = R.FPDF_ANNOT | (R.FPDF_GRAYSCALE if gray else R.FPDF_REVERSE_BYTE_ORDER)
            R.FPDF_RenderPageBitmapWithMatrix(bm.raw, raw, ctypes.byref(m), ctypes.byref(R.FS_RECTF(0, 0, w, h)), flags)
            arr = bm.to_numpy()
            if gray and arr.ndim == 3:
                arr = arr[..., 0]
            elif not gray and arr.ndim == 3 and arr.shape[2] == 4:
                arr = arr[..., :3]
            return Pixmap(arr)

    class Document:
        def __init__(self, path):
            self.name = str(path)
            self._pdf = pdfium.PdfDocument(self.name)

        def __len__(self):
            return len(self._pdf)

        page_count = property(__len__)

        def __getitem__(self, i):
            if i < 0:
                i += len(self)
            if not 0 <= i < len(self):
                raise IndexError(f"page {i} not in document ({len(self)} pages)")
            return Page(self, i)

        def __iter__(self):
            return (self[i] for i in range(len(self)))

        def close(self):
            self._pdf.close()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            self.close()

    def _open(path):
        return Document(path)

    try:
        _ver = pdfium.PYPDFIUM_INFO.version
    except AttributeError:
        _ver = getattr(pdfium, "__version__", "")
    pymupdf = SimpleNamespace(open=_open, Rect=Rect, csGRAY="gray", csRGB="rgb", VersionBind=f"pypdfium2 {_ver}",
                              Document=Document, Page=Page)
