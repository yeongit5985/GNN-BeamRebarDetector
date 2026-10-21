"""
Graph neural network for beam-detail graphs (the .npz + .json files that 01_pdf_labelstudio_to_graph.ipynb
or graph_builder.py save). One network, two outputs:

  * node class: every word / line / mark -> BeamName, BeamSize, RebarSize, Dim, RebarLine, DimLine,
    SectionMark, SectionTitle or Other
  * relation: every candidate pair (label - its bar line, beam title - its labels, section label - the same
    bar in the elevation, ...) -> related or not, i.e. the relations you draw in Label Studio

How it works: each node starts from its features (size, line length, angle, width, ...); a word also from
its text, read character by character by a small learned encoder.
A few rounds of message passing let every node take in its neighbours along the graph's edges (lines that
touch, text next to a line, nearby texts), one message function per edge type. The node head classifies
the result; the pair head scores a candidate pair from both nodes and how they sit (distance, offset,
same text, pair kind).

Plain PyTorch, no PyTorch Geometric:  pip install torch

    import gnn
    graphs = gnn.load_graphs("graphs")
    train, val = gnn.split(graphs)
    model, history, metrics = gnn.train(train, val, epochs=150)
    print(gnn.metrics_text(metrics))
    gnn.save_model(model, "model/beam_gnn.pt", metrics=metrics, trained_on=[g["name"] for g in train])
"""
import json
import math
import os
import random
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

THRESHOLD = 0.5            # a pair counts as a relation when its probability is above this (until thresholds are fitted)
TENSORS = ("x", "y", "edge_index", "edge_type", "edge_attr", "pair_index", "pair_label", "pair_kind", "pair_x",
           "text_idx", "chars", "shapes")

# The text of each word, read by the model itself (characters + their shape), so new label styles
# ('Y16@200', '4-T20', 'T20 @ 150 c/c') are learned from your annotations instead of fixed patterns.
MAX_CHARS = 24
N_CHARS = 97                   # 0 = padding, 1..95 = printable ASCII, 96 = any other character
PUNCT = "-@/().,+&%:*#"
N_SHAPES = 5 + len(PUNCT) + 1  # 0 pad, 1 digit, 2 capital, 3 small letter, 4 space, punctuation, other
# the hand-made text pattern features of graph_builder (off by default: the model reads the text instead)
PATTERN_FEATURES = ("t_rebar", "t_rebar_link", "t_rebar_dia", "t_rebar_count", "t_number", "t_number_log",
                    "t_beam_size", "t_beam_name", "t_alpha_word")


def _char_ids(texts):
    ids = np.zeros((len(texts), MAX_CHARS), np.int64)
    shp = np.zeros((len(texts), MAX_CHARS), np.int64)
    for k, t in enumerate(texts):
        for m, ch in enumerate(t.strip()[:MAX_CHARS]):
            o = ord(ch)
            ids[k, m] = o - 31 if 32 <= o < 127 else 96
            shp[k, m] = (1 if ch.isdigit() else 2 if ch.isupper() else 3 if ch.islower() else 4 if ch.isspace()
                         else 5 + PUNCT.index(ch) if ch in PUNCT else N_SHAPES - 1)
    return ids, shp


def default_device():
    """The GPU when there is one (Colab: Runtime > Change runtime type > T4 GPU), else the CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def on(g, dev):
    """The graph with its tensors on a device (kept, so they move once)."""
    if g["x"].device == torch.device(dev):
        return g
    cache = g.setdefault("_on", {})
    if str(dev) not in cache:
        cache[str(dev)] = {**{k: v for k, v in g.items() if k != "_on"}, **{k: g[k].to(dev) for k in TENSORS}}
    return cache[str(dev)]


# ------------------------------------------------------------------------------------------- data
def _hops(n, edge_index, edge_type, sources, max_hops=4):
    """Steps from each source node to every node along touch / near edges (a label's leader -> arrowhead
    -> bar line is a few steps), up to max_hops; farther = max_hops + 1."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra
    m = edge_type <= 1
    A = csr_matrix((np.ones(int(m.sum())), (edge_index[0][m], edge_index[1][m])), shape=(n, n))
    D = dijkstra(A, indices=sources, unweighted=True, limit=max_hops + 0.5)
    D[~np.isfinite(D)] = max_hops + 1
    return D


def _leader_targets(meta, edge_index, edge_type, width, unit, sources):
    """Where each label's leader points, as a QS reads it. A leader starts at the label (a line with one
    end at the text) and may bend (more short pieces end to end); the line it ends on is the bar.
    {(label, line): 1 = the line at the leader's tip, 2 = the rest of that bar (pieces joined end to end
    with the same pen width: bends, legs), 3 = a leader piece itself}"""
    nodes, tol = meta["nodes"], 0.35 * unit
    touch, near = defaultdict(set), defaultdict(set)
    for a, b, t in zip(edge_index[0], edge_index[1], edge_type):
        if t == 0:
            touch[int(a)].add(int(b))
        elif t == 1:
            near[int(a)].add(int(b))
    seg_c, end_c = {}, {}
    def segs(k):
        if k not in seg_c:
            seg_c[k] = np.asarray(nodes[k]["segs"], float).reshape(-1, 2, 2)
        return seg_c[k]
    def ends(k):                                   # polyline ends: vertices used once
        if k not in end_c:
            u, c = np.unique(np.round(segs(k).reshape(-1, 2), 1), axis=0, return_counts=True)
            end_c[k] = u[c == 1]
        return end_c[k]
    def dseg(p, S):                                # point -> nearest segment distance
        a, ab = S[:, 0], S[:, 1] - S[:, 0]
        t = np.clip(((p - a) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-9), 0, 1)
        return float(np.hypot(*(a + t[:, None] * ab - p).T).min())
    def bdist(p, bb):
        return math.hypot(max(bb[0] - p[0], 0, p[0] - bb[2]), max(bb[1] - p[1], 0, p[1] - bb[3]))
    is_line = lambda k: nodes[k]["kind"] == "line"
    short = lambda k: nodes[k]["length"] <= 25 * unit
    out = {}
    for i in sources:
        bb, tg = nodes[i]["bbox"], {}
        for L in near[i]:                          # a leader: short, one end at the label, the other away from it
            if not is_line(L) or not short(L) or len(ends(L)) != 2:
                continue
            dn = sorted(ends(L), key=lambda e: bdist(e, bb))
            if bdist(dn[0], bb) > 0.6 * unit or bdist(dn[1], bb) < 1.0 * unit:
                continue
            path, tip, prev = [L], dn[1], L
            for _ in range(3):                     # follow the bends to the tip
                hits = [k for k in touch[prev] if k not in path and is_line(k) and dseg(tip, segs(k)) <= tol]
                if not hits:
                    break
                nxt = [k for k in hits if short(k) and len(ends(k)) == 2 and min(math.dist(tip, e) for e in ends(k)) <= tol
                       and nodes[k]["length"] < nodes[prev]["length"] * 4]
                if len(hits) == 1 and nxt:         # the leader goes on (a bend), unless it reached a line here
                    k = nxt[0]; path.append(k); prev = k
                    tip = max(ends(k), key=lambda e: math.dist(e, tip))
                    continue
                for k in hits:                     # the line(s) at the tip: the bar
                    tg[k] = 1
                break
            for k in path:
                tg.setdefault(k, 3)
        for k in [k for k, v in tg.items() if v == 1]:          # the rest of the bar, end to end, same pen
            frontier, seen = [k], {k}
            for _ in range(8):
                nxt = []
                for a in frontier:
                    for b in touch[a]:
                        if b in seen or not is_line(b) or abs(width[b] - width[a]) > 0.05 or not len(ends(a)) or not len(ends(b)):
                            continue
                        if np.hypot(*(ends(a)[:, None] - ends(b)[None]).reshape(-1, 2).T).min() <= tol:
                            seen.add(b); nxt.append(b); tg.setdefault(b, 2)
                frontier = nxt
        for j, lev in tg.items():
            out[(int(i), j)] = lev
    return out


def _pair_features(meta, pair_index, pair_kind, n_kinds, unit, is_text, is_mark, edge_index, edge_type, width):
    """How the two nodes of each candidate pair sit: kind (one-hot), offset, distance, gap, same text,
    steps between them along touch / near edges, whether the last step comes from a mark (an arrowhead:
    label -> leader -> arrowhead -> bar), whether an edge joins them, and how close the pair is compared
    with the label's other candidates (rank by steps, then gap)."""
    B = np.array([n["bbox"] for n in meta["nodes"]], np.float32)
    C = np.column_stack([(B[:, 0] + B[:, 2]) / 2, (B[:, 1] + B[:, 3]) / 2])
    i, j = pair_index
    d = (C[j] - C[i]) / unit
    dist = np.hypot(d[:, 0], d[:, 1])
    gap = np.hypot(np.maximum(0, np.maximum(B[i, 0] - B[j, 2], B[j, 0] - B[i, 2])),
                   np.maximum(0, np.maximum(B[i, 1] - B[j, 3], B[j, 1] - B[i, 3]))) / unit
    texts = [("".join(n.get("text", "").split()).upper() if n["kind"] == "text" else "") for n in meta["nodes"]]
    same = np.array([texts[a] != "" and texts[a] == texts[b] for a, b in zip(i, j)], np.float32)
    onehot = np.eye(n_kinds, dtype=np.float32)[pair_kind]
    both_text = (is_text[i] * is_text[j]).astype(np.float32)
    src = np.unique(i)
    D = _hops(len(B), edge_index, edge_type, src)
    hop = D[np.searchsorted(src, i), j].astype(np.int64)
    hop_1h = np.eye(6, dtype=np.float32)[np.clip(hop, 0, 5)][:, 1:]          # 1, 2, 3, 4 steps, farther
    n = len(B)
    keys = set((edge_index[0] * n + edge_index[1]).tolist())
    joined = np.array([a * n + b in keys for a, b in zip(i, j)], np.float32)
    touch = edge_type == 0                                          # marks touching each node
    mark_nb = [[] for _ in range(n)]
    for a, b in zip(edge_index[0][touch], edge_index[1][touch]):
        if is_mark[a]:
            mark_nb[b].append(a)
    si = np.searchsorted(src, i)
    via_mark = np.array([any(D[s_, m] == D[s_, b] - 1 for m in mark_nb[b]) if D[s_, b] <= 4 else False
                         for s_, b in zip(si, j)], np.float32)
    rank = np.zeros(len(i), np.float32)                             # 0 = the label's closest candidate
    order = np.lexsort((gap, hop, i))
    first = np.r_[True, i[order][1:] != i[order][:-1]]
    start = np.maximum.accumulate(np.where(first, np.arange(len(order)), 0))
    rank[order] = np.arange(len(order)) - start
    lead = _leader_targets(meta, edge_index, edge_type, width, unit, np.unique(i[pair_kind == 0]))
    lv = np.array([lead.get((int(a), int(b)), 0) for a, b in zip(i, j)])
    lead_1h = np.stack([lv == 1, lv == 2, lv == 3], 1).astype(np.float32)
    return np.column_stack([onehot, np.clip(d, -60, 60) / 20, np.log1p(dist) / 4, np.log1p(gap) / 4,
                            same, both_text, hop_1h, joined, via_mark, np.minimum(rank, 10) / 10,
                            (rank == 0).astype(np.float32), lead_1h]).astype(np.float32)


def load_graph(npz_path):
    """One saved graph -> dict of tensors (+ names). Needs the .json next to the .npz."""
    npz_path = Path(npz_path)
    A = np.load(npz_path, allow_pickle=False)
    meta = json.loads(npz_path.with_suffix(".json").read_text(encoding="utf-8"))
    return graph_from_arrays(A, meta, npz_path.stem)


def graph_from_arrays(A, meta, name="page"):
    """A graph from graph_builder's arrays + meta, in memory (what load_graph reads from the .npz + .json):
    the backend's AI check uses it for a page it has just built."""
    feat = [str(f) for f in A["feature_names"]]
    x = A["x"].astype(np.float32)
    is_text, is_mark = x[:, feat.index("is_text")], x[:, feat.index("is_mark")]
    kinds = [str(k) for k in A["pair_kinds"]]
    unit = float(A["unit"])
    ea = A["edge_attr"].astype(np.float32) / 10.0
    pi = A["pair_index"].astype(np.int64)
    text_idx = np.where(is_text > 0)[0]
    chars, shapes = _char_ids([meta["nodes"][k].get("text", "") for k in text_idx])
    return dict(
        name=name, meta=meta, has_hint="pair_hint" in getattr(A, "files", A),
        feature_names=feat, class_names=[str(c) for c in A["class_names"]], pair_kinds=kinds,
        edge_types=[str(t) for t in A["edge_types"]],
        x=torch.from_numpy(x), y=torch.from_numpy(_ignore_unpaired_dims(A, meta, kinds)),
        edge_index=torch.from_numpy(A["edge_index"].astype(np.int64)),
        edge_type=torch.from_numpy(A["edge_type"].astype(np.int64)), edge_attr=torch.from_numpy(ea),
        pair_index=torch.from_numpy(pi), pair_label=torch.from_numpy(A["pair_label"].astype(np.float32)),
        pair_kind=torch.from_numpy(A["pair_kind"].astype(np.int64)),
        pair_x=torch.from_numpy(np.column_stack([
            _pair_features(meta, pi, A["pair_kind"], len(kinds), unit, is_text, is_mark,
                           A["edge_index"], A["edge_type"], x[:, feat.index("l_rel_width")]),
            # the rule-based detector's same-bars link (graph_builder: pair_hint); 0 in graphs built before it
            A["pair_hint"].astype(np.float32) if "pair_hint" in getattr(A, "files", A) else np.zeros(pi.shape[1], np.float32),
        ]).astype(np.float32)),
        text_idx=torch.from_numpy(text_idx.astype(np.int64)), chars=torch.from_numpy(chars), shapes=torch.from_numpy(shapes),
    )


IGNORE = -1      # node label left out of the loss and the scores


def _ignore_unpaired_dims(A, meta, kinds):
    """Node labels, with every Dim that has no text-line link to a DimLine set to IGNORE: the Dim - DimLine
    pair is kept only to read the drawing scale (scanned drawings later), so a Dim without its line teaches
    nothing (YS, 18 Oct). Checked per region: one linked piece keeps the whole Dim."""
    y = A["y"].astype(np.int64)
    classes = [str(c) for c in A["class_names"]]
    if "Dim" not in classes or "DimLine" not in classes or "text-line" not in kinds:
        return y
    dim, dline = classes.index("Dim"), classes.index("DimLine")
    ent = [n.get("entity") for n in meta["nodes"]]
    pi, lab, pk = A["pair_index"], A["pair_label"] > 0, A["pair_kind"] == kinds.index("text-line")
    paired = set()
    for a, b in zip(pi[0][lab & pk], pi[1][lab & pk]):
        for u, v in ((a, b), (b, a)):
            if y[u] == dim and y[v] == dline:
                paired.add(ent[u] if ent[u] is not None else f"n{u}")
    for i in np.where(y == dim)[0]:
        if (ent[i] if ent[i] is not None else f"n{i}") not in paired:
            y[i] = IGNORE
    return y


def load_graphs(folder, verbose=True):
    """Every graph (.npz with its .json) in a folder."""
    out = []
    for p in sorted(Path(folder).glob("*.npz")):
        if not p.with_suffix(".json").exists():
            continue
        g = load_graph(p)
        if out and g["feature_names"] != out[0]["feature_names"]:
            raise ValueError(f"{p.name} was built with other node features than {out[0]['name']}: rebuild it")
        out.append(g)
        if verbose:
            print(f"{g['name']}: {len(g['y'])} nodes ({int((g['y'] > 0).sum())} labelled, {int((g['y'] == IGNORE).sum())} ignored), {g['edge_index'].shape[1]} edges, "
                  f"{len(g['pair_label'])} candidate pairs, {int(g['pair_label'].sum())} true")
    return out


def split(graphs, val_frac=0.2, seed=0):
    """Hold out whole pages for validation (at least 1 when there are 3 or more pages)."""
    if len(graphs) < 3:
        return list(graphs), []
    idx = list(range(len(graphs)))
    random.Random(seed).shuffle(idx)
    n_val = max(1, round(val_frac * len(graphs)))
    return [graphs[i] for i in idx[n_val:]], [graphs[i] for i in idx[:n_val]]


# ------------------------------------------------------------------------------------------- augmentation
# More training examples from the same pages (AUGMENT in the notebook). The model sees words and lines, not
# the image, so these change the graph: other numbers in the labels, part of the sheet, another drawing
# scale, less clutter. Your relations stay as they are, so every changed copy is still correctly labelled.
DIAS = [8, 10, 12, 16, 20, 25, 32]
SPACINGS = [75, 100, 125, 150, 175, 200, 225, 250, 300]
_NUM = re.compile(r"\d+")


def swap_text(texts, rng):
    """The page's words with other valid numbers, the same way everywhere on the page (so a label and its
    repeat in a section still match): bar sizes swapped among the usual sizes ('3H12' -> '3H20'), counts,
    spacings ('T10-200' -> 'T10-150'), dimensions and beam sizes scaled ('600' -> '450'), other numbers
    ('11aB1a', '1B245') replaced by others of the same length. Letters stay: they are the consultant's style."""
    dia = dict(zip(DIAS, rng.permutation(DIAS).tolist()))
    spc = dict(zip(SPACINGS, rng.permutation(SPACINGS).tolist()))
    cnt = dict(zip(range(1, 13), (rng.permutation(12) + 1).tolist()))
    f = float(rng.uniform(0.6, 1.6))
    other = {}

    def new(m):
        s, v = m.string, int(m.group())
        before, after = s[:m.start()], s[m.end():]
        if re.search(r"[TYRH]$", before) and len(m.group()) <= 2:
            return str(dia.get(v, v))
        if re.match(r"\s*[TYRH]\d", after) and not re.search(r"[A-Za-z0-9]$", before):
            return str(cnt.get(v, v))
        if re.search(r"[TYRH]\d{1,2}\s*[-@/]\s*$|@\s*$", before):
            return str(spc.get(v, v))
        if v >= 100 and len(m.group()) <= 5:
            return str(max(25, int(round(v * f / 5)) * 5))
        if m.group() not in other:
            n = len(m.group())
            other[m.group()] = str(rng.integers(1, 10)) if n == 1 else str(rng.integers(10 ** (n - 1), 10 ** n))
        return other[m.group()]
    return [_NUM.sub(new, t) if t else t for t in texts]


def _aug_info(g):
    """What augment() needs of a page, worked out once."""
    if "_aug" not in g:
        nodes, feat = g["meta"]["nodes"], g["feature_names"]
        B = np.array([n["bbox"] for n in nodes], np.float32)
        y, ei, et = g["y"].numpy(), g["edge_index"].numpy(), g["edge_type"].numpy()
        pi, lab = g["pair_index"].numpy(), g["pair_label"].numpy() > 0
        labelled = y > 0
        keep = labelled.copy()                          # never dropped as clutter: labelled nodes,
        keep[pi[:, lab].ravel()] = True                 # nodes of a true pair,
        near = (et <= 1) & labelled[ei[0]]              # and whatever touches / is next to a labelled node
        keep[ei[1][near]] = True                        # (leaders, arrowheads, bar ends)
        keep |= g["x"][:, feat.index("is_text")].numpy() > 0
        g["_aug"] = dict(C=np.column_stack([(B[:, 0] + B[:, 2]) / 2, (B[:, 1] + B[:, 3]) / 2]),
                         texts=[n.get("text", "") if n["kind"] == "text" else "" for n in nodes],
                         clutter=np.where(~keep)[0], col={f: k for k, f in enumerate(feat)})
    return g["_aug"]


def _subgraph(g, keep):
    """The graph with only the nodes in keep (bool), their edges and their candidate pairs."""
    new = np.full(len(keep), -1, np.int64); new[keep] = np.arange(int(keep.sum()))
    new_t = torch.from_numpy(new)
    ei = g["edge_index"]; em = torch.from_numpy(keep)[ei].all(0)
    pi = g["pair_index"]; pm = torch.from_numpy(keep)[pi].all(0)
    tk = torch.from_numpy(keep)[g["text_idx"]]
    out = {**g, "x": g["x"][keep], "y": g["y"][keep], "edge_index": new_t[ei[:, em]], "edge_type": g["edge_type"][em],
           "edge_attr": g["edge_attr"][em], "pair_index": new_t[pi[:, pm]], "pair_label": g["pair_label"][pm],
           "pair_kind": g["pair_kind"][pm], "pair_x": g["pair_x"][pm], "text_idx": new_t[g["text_idx"][tk]],
           "chars": g["chars"][tk], "shapes": g["shapes"][tk]}
    return out


def augment(g, rng, text=0.5, crop=0.5, scale=0.1, clutter=0.15):
    """A changed copy of a page for one training step (CPU tensors; the page itself is not touched):
    text: chance of other numbers in the words (swap_text); crop: chance of training on a part of the sheet
    only (40-80 % of its width and height, with at least one of your relations in it); scale: the drawing
    stretched by up to this share (+-10 %), text size unchanged, like a sheet drawn at another scale;
    clutter: share of unlabelled lines dropped (hatching, grid lines; never a line next to a label or bar).
    rng: numpy.random.Generator."""
    info = _aug_info(g)
    col = info["col"]
    n = len(g["y"])
    out = {k: v for k, v in g.items() if k not in ("_on", "_aug", "_ent")}

    if text and rng.random() < text:
        texts = swap_text(info["texts"], rng)
        ti = g["text_idx"].numpy()
        chars, shapes = _char_ids([texts[k] for k in ti])
        x = g["x"].clone()
        tt = [texts[k].strip() for k in ti]
        x[ti, col["t_log_nchar"]] = torch.tensor([math.log1p(len(t)) for t in tt])
        x[ti, col["t_frac_digit"]] = torch.tensor([sum(c.isdigit() for c in t) / max(len(t), 1) for t in tt])
        norm = ["".join(t.split()).upper() for t in texts]          # the pairs' 'same text' with the new words
        pi = g["pair_index"].numpy()
        same = torch.tensor([norm[a] != "" and norm[a] == norm[b] for a, b in zip(pi[0], pi[1])], dtype=torch.float32)
        px = g["pair_x"].clone(); px[:, len(g["pair_kinds"]) + 4] = same
        out.update(chars=torch.from_numpy(chars), shapes=torch.from_numpy(shapes), x=x, pair_x=px)

    if scale:
        s = float(rng.uniform(1 - scale, 1 + scale))
        lg = lambda v, k=s: torch.log1p(torch.expm1(v) * k)          # log1p(d) -> log1p(d * s)
        x = out["x"].clone()
        line = x[:, col["is_text"]] == 0
        for f in ("log_w", "log_h", "l_log_len", "l_log_span"):
            x[line, col[f]] = lg(x[line, col[f]])
        x[line, col["m_log_area"]] = lg(x[line, col["m_log_area"]], s * s)
        ea = out["edge_attr"] * s                                      # offsets and gaps, in text heights / 10
        nk = len(g["pair_kinds"])
        px = out["pair_x"].clone()
        px[:, nk:nk + 2] = (px[:, nk:nk + 2] * s).clamp(-3, 3)        # offset
        px[:, nk + 2:nk + 4] = lg(px[:, nk + 2:nk + 4] * 4) / 4        # log distance, log gap
        out.update(x=x, edge_attr=ea.clamp(-5, 5), pair_x=px)

    keep = np.ones(n, bool)
    if clutter and len(info["clutter"]):
        keep[info["clutter"][rng.random(len(info["clutter"])) < clutter]] = False
    if crop and rng.random() < crop:
        C = info["C"]; lo, hi = C.min(0), C.max(0)
        lab = g["pair_label"].numpy() > 0; pi = g["pair_index"].numpy()
        for _ in range(5):
            w = (hi - lo) * rng.uniform(0.4, 0.8, 2)
            a = lo + (hi - lo - w) * rng.random(2)
            inside = ((C >= a) & (C <= a + w)).all(1)
            if (inside[pi[0][lab]] & inside[pi[1][lab]]).any():
                keep &= inside
                break
    return _subgraph(out, keep) if not keep.all() else out


# ------------------------------------------------------------------------------------------- model
class EdgeTypeConv(nn.Module):
    """One round of message passing: a message function per edge type (touch / near / text_knn), the mean
    of the incoming messages, then a residual update of each node."""

    def __init__(self, hidden, n_edge_types, edge_dim):
        super().__init__()
        self.msg = nn.ModuleList([nn.Sequential(nn.Linear(hidden + edge_dim, hidden), nn.ReLU())
                                  for _ in range(n_edge_types)])
        self.upd = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.norm = nn.LayerNorm(hidden)

    def forward(self, h, edge_index, edge_type, edge_attr):
        src, dst = edge_index
        agg = torch.zeros_like(h)
        for t, mlp in enumerate(self.msg):
            m = edge_type == t
            if bool(m.any()):
                agg.index_add_(0, dst[m], mlp(torch.cat([h[src[m]], edge_attr[m]], 1)))
        deg = torch.bincount(dst, minlength=h.size(0)).clamp(min=1).unsqueeze(1).to(h.dtype)
        return self.norm(h + self.upd(torch.cat([h, agg / deg], 1)))


class TextEncoder(nn.Module):
    """Reads a word character by character: each character and its shape (digit, capital, '-', '@', ...) are
    embedded, two 1-D convolutions look at 3- and 5-character windows ('T20', '-150'), and the strongest
    response over the word becomes its text vector. While training, some characters are hidden (only their
    shape stays), so '6T20' and '4T25' look alike to it and new numbers are no surprise."""

    def __init__(self, hidden, dim=32, char_dropout=0.15):
        super().__init__()
        self.ch = nn.Embedding(N_CHARS, dim, padding_idx=0)
        self.sh = nn.Embedding(N_SHAPES, dim, padding_idx=0)
        self.conv1 = nn.Conv1d(dim, hidden, 3, padding=1)
        self.conv2 = nn.Conv1d(hidden, hidden, 3, padding=1)
        self.out = nn.Linear(hidden, hidden)
        self.char_dropout = char_dropout

    def forward(self, ids, shapes):
        mask = ids > 0
        if self.training and self.char_dropout:
            ids = torch.where(mask & (torch.rand(ids.shape, device=ids.device) < self.char_dropout),
                              torch.full_like(ids, N_CHARS - 1), ids)
        e = (self.ch(ids) + self.sh(shapes)).transpose(1, 2)              # (words, dim, chars)
        h = F.relu(self.conv2(F.relu(self.conv1(e))))
        h = h.masked_fill(~mask.unsqueeze(1), -1e4).max(2).values
        h = torch.where(mask.any(1, keepdim=True), h, torch.zeros_like(h))
        return self.out(h)


class BeamGNN(nn.Module):
    def __init__(self, n_feat, n_classes, n_edge_types, edge_dim, pair_dim, hidden=64, layers=3, dropout=0.1,
                 read_text=True, pattern_cols=(), hint=False, n_pair_kinds=None):
        super().__init__()
        self.config = dict(n_feat=n_feat, n_classes=n_classes, n_edge_types=n_edge_types, edge_dim=edge_dim,
                           pair_dim=pair_dim, hidden=hidden, layers=layers, dropout=dropout,
                           read_text=read_text, pattern_cols=list(pattern_cols), hint=hint, n_pair_kinds=n_pair_kinds)
        # the rule's same-bars hint (last pair feature) also goes straight to the pair score, with its own
        # weight: inside the pair head it is 1 input of ~200 and, with few true pairs, the network learned
        # the label texts instead. Starts trusting the hint; training turns the weight down if it misleads.
        # One weight per pair kind: the same-bars hint and the link-shape hint are not equally reliable, and a
        # shared weight raised for one also raised the other. Models saved before this have one shared weight.
        self.hint_w = (nn.Parameter(torch.full((n_pair_kinds,), 4.0)) if n_pair_kinds else nn.Parameter(torch.tensor(4.0))) if hint else None
        keep = torch.ones(n_feat)
        keep[list(pattern_cols)] = 0                 # hand-made pattern features switched off
        self.register_buffer("keep", keep)
        self.text = TextEncoder(hidden) if read_text else None
        self.inp = nn.Sequential(nn.Linear(n_feat, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.convs = nn.ModuleList([EdgeTypeConv(hidden, n_edge_types, edge_dim) for _ in range(layers)])
        self.drop = nn.Dropout(dropout)
        self.node_head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, n_classes))
        self.pair_head = nn.Sequential(nn.Linear(3 * hidden + pair_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
                                       nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1))

    def forward(self, g):
        h = self.inp(g["x"] * self.keep)
        if self.text is not None and len(g["text_idx"]):
            h = h.index_add(0, g["text_idx"], self.text(g["chars"], g["shapes"]))
        for conv in self.convs:
            h = self.drop(conv(h, g["edge_index"], g["edge_type"], g["edge_attr"]))
        i, j = g["pair_index"]
        pair_in = torch.cat([h[i], h[j], h[i] * h[j], g["pair_x"]], 1)
        pair = self.pair_head(pair_in).squeeze(1)
        if self.hint_w is not None:
            w = self.hint_w[g["pair_kind"]] if self.hint_w.dim() else self.hint_w
            pair = pair + w * g["pair_x"][:, -1]
        return self.node_head(h), pair


def new_model(example, hidden=64, layers=3, dropout=0.1, read_text=True, use_patterns=False):
    """read_text: the model reads each word itself (character encoder). use_patterns: also give it the
    hand-made text patterns ('looks like 6T20'); off by default, since a consultant's own style breaks them."""
    cols = [] if use_patterns else [example["feature_names"].index(f) for f in PATTERN_FEATURES if f in example["feature_names"]]
    return BeamGNN(example["x"].shape[1], len(example["class_names"]), len(example["edge_types"]),
                   example["edge_attr"].shape[1], example["pair_x"].shape[1], hidden, layers, dropout, read_text, cols,
                   hint=bool(example.get("has_hint")), n_pair_kinds=len(example["pair_kinds"]))


# ------------------------------------------------------------------------------------------- training
def weights(graphs):
    """Class weights for the node loss and, per pair kind, the weight of a true pair (true pairs are rare)."""
    n_cls, n_kinds = len(graphs[0]["class_names"]), len(graphs[0]["pair_kinds"])
    cnt = torch.zeros(n_cls)
    pos, tot = torch.zeros(n_kinds), torch.zeros(n_kinds)
    for g in graphs:
        cnt += torch.bincount(g["y"][g["y"] >= 0], minlength=n_cls).float()
        pos += torch.bincount(g["pair_kind"], weights=g["pair_label"], minlength=n_kinds)
        tot += torch.bincount(g["pair_kind"], minlength=n_kinds).float()
    cw = (cnt.sum() / (n_cls * cnt.clamp(min=1))).sqrt().clamp(0.2, 10.0)
    cw[cnt == 0] = 0.0
    pw = ((tot - pos) / pos.clamp(min=1)).clamp(1.0, 30.0)
    return cw, pw


def loss_fn(model, g, cw, pw):
    node_logits, pair_logits = model(g)
    node_loss = F.cross_entropy(node_logits, g["y"], weight=cw, ignore_index=IGNORE)
    w = torch.where(g["pair_label"] > 0, pw[g["pair_kind"]], torch.ones_like(g["pair_label"]))
    bce = F.binary_cross_entropy_with_logits(pair_logits, g["pair_label"], weight=w, reduction="none")
    kinds = torch.unique(g["pair_kind"])
    pair_loss = torch.stack([bce[g["pair_kind"] == k].mean() for k in kinds]).mean()   # every pair kind counts
    return node_loss + pair_loss, node_loss.item(), pair_loss.item()


@torch.no_grad()
def predict(model, g):
    """Node class probabilities (N, classes) and relation probabilities of the candidate pairs (P,), on the CPU."""
    model.eval()
    node_logits, pair_logits = model(on(g, next(model.parameters()).device))
    return node_logits.softmax(1).cpu(), pair_logits.sigmoid().cpu()


def _prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return dict(precision=round(p, 3), recall=round(r, 3), f1=round(2 * p * r / (p + r), 3) if p + r else 0.0)


def _relation_sets(g, prob, kind, thr):
    """Relations as Label Studio has them (between regions, not node pieces): the true ones of this pair
    kind, and the predicted ones (any node pair of the two regions above the threshold)."""
    ent = g["_ent"] if "_ent" in g else g.setdefault("_ent", [n.get("entity") for n in g["meta"]["nodes"]])
    pi = g["pair_index"].numpy(); lab = g["pair_label"].numpy() > 0; m = g["pair_kind"].numpy() == kind
    key = lambda a, b: frozenset((ent[a] or f"n{a}", ent[b] or f"n{b}"))
    true = {key(a, b) for a, b in zip(pi[0][m & lab], pi[1][m & lab])}
    sel = m & (prob > thr)
    pred = {key(a, b) for a, b in zip(pi[0][sel], pi[1][sel]) if (ent[a] is None or ent[a] != ent[b])}
    return true, pred


def _avg_precision(score, label):
    if label.sum() == 0:
        return None
    o = np.argsort(-score); l = label[o]
    hits = np.cumsum(l)
    return float((hits[l > 0] / (np.nonzero(l)[0] + 1)).mean())


@torch.no_grad()
def evaluate(model, graphs, thresholds=None):
    """Node classes and relations against your labels. Relations are counted as in Label Studio (region to
    region): found if any node pair of the two regions scores above the threshold of its pair kind."""
    classes, kinds = graphs[0]["class_names"], graphs[0]["pair_kinds"]
    thresholds = thresholds or getattr(model, "thresholds", None) or {}
    n_cls, n_k = len(classes), len(kinds)
    ct = np.zeros((n_cls, 3)); kt = np.zeros((n_k, 3)); correct = total = 0
    lt = np.zeros((n_cls, 3))                             # the same, counted in line length (pt): what a takeoff measures
    scores = [[] for _ in range(n_k)]; labels = [[] for _ in range(n_k)]
    for g in graphs:
        nprob, pprob = predict(model, g)
        yp, y = nprob.argmax(1).numpy(), g["y"].numpy()
        ln = np.array([n.get("length", 0.0) if n["kind"] == "line" else 0.0 for n in g["meta"]["nodes"]])
        yp, y, ln = yp[y != IGNORE], y[y != IGNORE], ln[y != IGNORE]   # ignored labels (Dim without its DimLine) not scored
        correct += int((yp == y).sum()); total += len(y)
        for c in range(1, n_cls):
            ct[c] += [((yp == c) & (y == c)).sum(), ((yp == c) & (y != c)).sum(), ((yp != c) & (y == c)).sum()]
            lt[c] += [ln[(yp == c) & (y == c)].sum(), ln[(yp == c) & (y != c)].sum(), ln[(yp != c) & (y == c)].sum()]
        pp, pk, pl = pprob.numpy(), g["pair_kind"].numpy(), g["pair_label"].numpy()
        for k in range(n_k):
            true, pred = _relation_sets(g, pp, k, thresholds.get(kinds[k], THRESHOLD))
            kt[k] += [len(true & pred), len(pred - true), len(true - pred)]
            scores[k].append(pp[pk == k]); labels[k].append(pl[pk == k])
    per_class = {classes[c]: dict(**_prf(*ct[c]), true=int(ct[c, 0] + ct[c, 2])) for c in range(1, n_cls) if ct[c, 0] + ct[c, 2] > 0}
    for c in range(1, n_cls):                             # line classes: also by length (tiny stubs barely count)
        if classes[c] in per_class and lt[c, 0] + lt[c, 2] > 0:
            per_class[classes[c]]["by_length"] = _prf(*lt[c])
    per_kind = {}
    for k in range(n_k):
        if kt[k, 0] + kt[k, 2] > 0:
            ap = _avg_precision(np.concatenate(scores[k]), np.concatenate(labels[k]))
            per_kind[kinds[k]] = dict(**_prf(*kt[k]), true=int(kt[k, 0] + kt[k, 2]), threshold=thresholds.get(kinds[k], THRESHOLD),
                                      pair_ap=round(ap, 3) if ap is not None else None)
    mean = lambda d, f: round(float(np.mean([v[f] for v in d.values() if v.get(f) is not None])), 3) if d else 0.0
    return dict(node_accuracy=round(correct / max(total, 1), 4), node_f1=mean(per_class, "f1"), per_class=per_class,
                relation_f1=mean(per_kind, "f1"), relation_ap=mean(per_kind, "pair_ap"), per_kind=per_kind,
                pages=[g["name"] for g in graphs])


@torch.no_grad()
def fit_thresholds(model, graphs):
    """Per pair kind, the threshold that finds the relations of these pages best (F1)."""
    kinds = graphs[0]["pair_kinds"]
    probs = [predict(model, g)[1].numpy() for g in graphs]
    out = {}
    for k, name in enumerate(kinds):
        best = (-1.0, THRESHOLD)
        for t in np.r_[np.linspace(0.05, 0.95, 19), 0.97, 0.98, 0.99, 0.995]:   # above 0.95 too: the model is often that sure
            tp = fp = fn = 0
            for g, pp in zip(graphs, probs):
                true, pred = _relation_sets(g, pp, k, t)
                tp += len(true & pred); fp += len(pred - true); fn += len(true - pred)
            if tp + fn == 0:
                break
            f1 = _prf(tp, fp, fn)["f1"]
            if f1 > best[0]:
                best = (f1, round(float(t), 3))
        out[name] = best[1]
    return out


def train(train_graphs, val_graphs=None, epochs=150, lr=3e-3, hidden=64, layers=3, dropout=0.1, seed=0,
          eval_every=10, progress=None, should_stop=None, verbose=True, device=None, read_text=True, use_patterns=False,
          augment_pages=False):
    """Train on train_graphs; keep the model that scores best on val_graphs (or on the training pages when
    there are none). progress(dict) is called after every epoch; should_stop() can end training early.
    augment_pages: every step trains on a changed copy of the page (augment(); text swaps only without
    use_patterns, whose hand-made features would no longer match the words).
    Returns (model, history, metrics of the kept model)."""
    if not train_graphs:
        raise ValueError("no training graphs")
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    dev = torch.device(device) if device else default_device()
    if dev.type == "cpu":
        torch.set_num_threads(max(1, os.cpu_count() or 1))
    if verbose:
        print(f"training on {dev}" + (f" ({torch.cuda.get_device_name(dev)})" if dev.type == "cuda" else ""))
    model = new_model(train_graphs[0], hidden, layers, dropout, read_text, use_patterns).to(dev)
    gpu_graphs = [on(g, dev) for g in train_graphs]
    rng = np.random.default_rng(seed)
    step_graph = (lambda k: on(augment(train_graphs[k], rng, text=0 if use_patterns else 0.5), dev)) if augment_pages \
        else (lambda k: gpu_graphs[k])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * len(train_graphs), pct_start=0.15)
    cw, pw = (w.to(dev) for w in weights(train_graphs))
    check = val_graphs or train_graphs
    best, best_state, best_epoch, history, t0 = -1.0, None, epochs, [], time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        order = list(range(len(train_graphs))); random.shuffle(order)
        tot = nl = pl = 0.0
        for k in order:
            loss, a, b = loss_fn(model, step_graph(k), cw, pw)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tot += loss.item(); nl += a; pl += b
        n = len(order)
        row = dict(epoch=epoch, epochs=epochs, loss=round(tot / n, 4), node_loss=round(nl / n, 4), pair_loss=round(pl / n, 4),
                   seconds=round(time.time() - t0, 1))
        if epoch % eval_every == 0 or epoch == epochs:
            m = evaluate(model, check)
            score = (m["node_f1"] + m["relation_ap"]) / 2
            row.update(node_f1=m["node_f1"], relation_ap=m["relation_ap"])
            if score > best:
                best, best_epoch = score, epoch
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if verbose:
                print(f"epoch {epoch:4d}  loss {row['loss']:.4f}  node F1 {m['node_f1']:.3f}  relation AP {m['relation_ap']:.3f}"
                      f"  ({'validation' if val_graphs else 'training'} pages, {row['seconds']:.0f} s)")
        history.append(row)
        if progress:
            progress(row)
        if should_stop and should_stop():
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.thresholds = fit_thresholds(model, train_graphs)        # chosen on the training pages only
    best_metrics = evaluate(model, check, model.thresholds)
    best_metrics["epoch"] = best_epoch
    best_metrics["checked_on"] = "validation pages" if val_graphs else "training pages (no validation page yet)"
    return model, history, best_metrics


# ------------------------------------------------------------------------------------------- save / show
def save_model(model, path, metrics=None, trained_on=None, example=None):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state_dict=model.state_dict(), config=model.config, thresholds=getattr(model, "thresholds", None),
                    metrics=metrics, trained_on=trained_on,
                    feature_names=example["feature_names"] if example else None,
                    class_names=example["class_names"] if example else None,
                    pair_kinds=example["pair_kinds"] if example else None,
                    saved=time.strftime("%Y-%m-%d %H:%M")), path)
    if metrics:
        info = dict(metrics, trained_on=trained_on, saved=time.strftime("%Y-%m-%d %H:%M"),
                    thresholds=getattr(model, "thresholds", None))
        path.with_suffix(".metrics.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    return path


def load_model(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = BeamGNN(**ck["config"])
    model.load_state_dict(ck["state_dict"])
    model.thresholds = ck.get("thresholds") or {}
    model.eval()
    return model, ck


def metrics_text(m):
    """The metrics as a few readable lines."""
    lines = [f"checked on {m.get('checked_on', '?')}: {', '.join(m.get('pages', []))}",
             f"node classes: accuracy {m['node_accuracy']:.1%}, mean F1 {m['node_f1']:.3f}"]
    for c, v in m["per_class"].items():
        lines.append(f"   {c:<13} P {v['precision']:.2f}  R {v['recall']:.2f}  F1 {v['f1']:.2f}  ({v['true']} nodes)")
        if "by_length" in v:
            b = v["by_length"]
            lines.append(f"   {'':<13} by length: P {b['precision']:.2f}  R {b['recall']:.2f}  F1 {b['f1']:.2f}")
    lines.append(f"relations (region to region, as in Label Studio): mean F1 {m['relation_f1']:.3f}")
    for k, v in m["per_kind"].items():
        lines.append(f"   {k:<38} P {v['precision']:.2f}  R {v['recall']:.2f}  F1 {v['f1']:.2f}  ({v['true']} relations, threshold {v['threshold']})")
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------- AI answer as a page
LS_LABEL = {"BeamName": "BeamID", "Dim": "Dimension", "DimLine": "DimensionLine"}     # class -> Label Studio label


def _paths(segs, tol=0.05):
    """A line node's segments [[x0, y0], [x1, y1]] -> point chains, joined where ends meet."""
    chains = []
    for a, b in (map(tuple, s) for s in segs):
        for c in chains:
            if math.dist(c[-1], a) < tol: c.append(b); break
            if math.dist(c[-1], b) < tol: c.append(a); break
            if math.dist(c[0], b) < tol: c.insert(0, a); break
            if math.dist(c[0], a) < tol: c.insert(0, b); break
        else:
            chains.append([a, b])
    return chains


@torch.no_grad()
def prediction_task(model, g, task, page_w, page_h, img_w=4967, img_h=3509):
    """What the model reads on a page, as a Label Studio task the review page opens: a region for each item it
    labels (nodes of one of your regions with the same class are joined) and the links above its thresholds.
    A region whose class differs from your label says so in its note ('AI: RebarSize · yours: BeamSize')."""
    nprob, pprob = predict(model, g)
    yp, y = nprob.argmax(1).numpy(), g["y"].numpy()
    classes, kinds = g["class_names"], g["pair_kinds"]
    thr = getattr(model, "thresholds", None) or {}
    nodes = g["meta"]["nodes"]
    groups = defaultdict(list)                        # (your region or the node itself, predicted class) -> nodes
    for i, n in enumerate(nodes):
        if yp[i] > 0:
            groups[(n.get("entity") or f"n{i}", int(yp[i]))].append(i)
    rid, result = {}, []
    px, py = lambda v: v / page_w * 100, lambda v: v / page_h * 100
    base = dict(to_name="image", original_width=img_w, original_height=img_h, image_rotation=0)
    for k, ((key, c), idx) in enumerate(groups.items()):
        lab = LS_LABEL.get(classes[c], classes[c])
        mine = {classes[y[i]] for i in idx if y[i] > 0}
        note = [] if mine == {classes[c]} else [f"AI: {lab} · yours: {', '.join(LS_LABEL.get(m, m) for m in sorted(mine)) or 'nothing'}"]
        prob = float(nprob[idx, c].mean())
        new = []
        segs = [s for i in idx for s in nodes[i].get("segs", [])]
        if classes[c] in ("RebarLine", "DimLine") and segs:
            for j, ch in enumerate(_paths(segs)):
                new.append(dict(base, id=f"ai{k}_{j}", type="vectorlabels", from_name="vec", score=prob,
                                   value=dict(closed=False, vectorlabels=[lab], vertices=[
                                       dict(id=f"ai{k}_{j}_{q}", x=px(x), y=py(yy), prevPointId=f"ai{k}_{j}_{q - 1}" if q else None,
                                            isBezier=False) for q, (x, yy) in enumerate(ch)])))
        else:
            bb = np.array([nodes[i]["bbox"] for i in idx])
            x0, y0, x1, y1 = bb[:, 0].min(), bb[:, 1].min(), bb[:, 2].max(), bb[:, 3].max()
            new.append(dict(base, id=f"ai{k}", type="rectanglelabels", from_name="label", score=prob,
                               value=dict(x=px(x0), y=py(y0), width=px(x1 - x0), height=py(y1 - y0), rotation=0,
                                          rectanglelabels=[lab])))
        for r in new:
            if note:
                r["meta"] = {"text": note}
        result += new
        for i in idx:
            rid[i] = [r["id"] for r in new]
    pi, pk, pp = g["pair_index"].numpy(), g["pair_kind"].numpy(), pprob.numpy()
    seen = set()
    for a, b, k, p in zip(pi[0], pi[1], pk, pp):
        if p <= thr.get(kinds[k], THRESHOLD) or a not in rid or b not in rid:
            continue
        for fa in rid[a]:
            for tb in rid[b]:
                if fa != tb and (fa, tb) not in seen and (tb, fa) not in seen:
                    seen.add((fa, tb))
                    result.append(dict(from_id=fa, to_id=tb, type="relation", direction="right"))
    meta = dict(task.get("meta") or {}, edited_in="AI test (model's own answer)")
    return dict(id=task.get("id"), data=task.get("data"), meta=meta, predictions=[],
                annotations=[dict(id=1, result=result, was_cancelled=False, ground_truth=False)])
