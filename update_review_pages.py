"""Give existing review pages the page code of the current review_relations.py, keeping each page's own
drawing, annotation and info. Pages saved with Save HTML keep the edits saved in them.

    python update_review_pages.py review_relations.py OUT_DIR page1.html page2.html ...
"""
import ast
import re
import sys
from pathlib import Path


def template(script):
    """PAGE and THEME_HEAD from review_relations.py, without importing it (no PIL / numpy needed)."""
    vals = {}
    for node in ast.parse(Path(script).read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and getattr(node.targets[0], "id", None) in ("PAGE", "THEME_HEAD"):
            vals[node.targets[0].id] = ast.literal_eval(node.value)
    return vals["PAGE"], vals["THEME_HEAD"]


def one(m, what, name):
    if not m:
        raise ValueError(f"{name}: {what} not found")
    return m.group(1)


def rebuild(html, page, theme_head, name="page"):
    title = one(re.search(r"<title>Relations - (.*?)</title>", html), "title", name)
    info = one(re.search(r'<meta name="review-info" content="(.*?)">', html), "review-info", name)
    img = one(re.search(r'<image id="img" href="data:image/jpeg;base64,([^"]*)"', html), "image", name)
    data = one(re.search(r"^const D = (.*);$", html, re.M), "data", name)
    baked = re.search(r'<script id="baked" type="application/json">([\s\S]*?)</script>', html)
    out = (page.replace("__THEME_HEAD__", theme_head).replace("__TITLE__", title).replace("__INFO__", info)
               .replace("__IMG__", img).replace("__DATA__", data))
    if baked:                                       # edits saved in the page with Save HTML
        out = out.replace('<script id="baked" type="application/json">null</script>',
                          '<script id="baked" type="application/json">' + baked.group(1) + "</script>", 1)
    return out


if __name__ == "__main__":
    script, out_dir, *pages = sys.argv[1:]
    page, theme_head = template(script)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for p in map(Path, pages):
        html = p.read_text(encoding="utf-8")
        new = rebuild(html, page, theme_head, p.name)
        (Path(out_dir) / p.name).write_text(new, encoding="utf-8")
        print(f"{p.name}: {len(html):,} -> {len(new):,} chars")
