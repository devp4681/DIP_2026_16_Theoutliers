"""
week2_visualize.py

Top-5 retrieval strips for validation queries (Engine v1), saved to ./figures/.
Query types: easy, difficult, colour-dominated, spatial.

    python week2_visualize.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

import retrieval_engine_v1 as eng
from week1_evaluate import load_labels
from week2_experiments import make_split, jaccard_matrix

K = 5
OUT = Path("figures")


def topk(S, k=K):
    return np.argsort(-S, axis=1, kind="stable")[:, :k]


def per_query_p(order, q_cls, g_cls):
    return (g_cls[order] == q_cls[:, None]).mean(1)


def img_path(p):
    return Path(p.replace("\\", "/"))


def draw(qi, hits, title, tr_paths, tr_cls, ref, val, ref_J_row, fname):
    fig, ax = plt.subplots(1, K + 1, figsize=(3 * (K + 1), 3.6))
    q = val[qi]
    ax[0].imshow(Image.open(img_path(tr_paths[q])))
    ax[0].set_title(f"QUERY\n{tr_cls[q]}", fontsize=10)
    for r, g in enumerate(hits):
        gi = ref[g]
        ok = tr_cls[gi] == tr_cls[q]
        ax[r + 1].imshow(Image.open(img_path(tr_paths[gi])))
        ax[r + 1].set_title(f"Rank {r+1}: {tr_cls[gi]}\nJaccard {ref_J_row[g]:.2f}", fontsize=9,
                            color="green" if ok else "red")
    for a in ax:
        a.axis("off")
    fig.suptitle(title, fontsize=12, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / fname, dpi=110)
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    cfg = json.loads(Path(eng.CONFIG_FILE).read_text())
    tr_f, tr_names, tr_cls, tr_paths = eng.load_split_features("train")
    ref, val = make_split(tr_cls, tr_names)
    J = jaccard_matrix(*(lambda L: (L[val], L[ref]))(load_labels(tr_names)))
    sims = eng.component_sims(eng.subset(tr_f, val), eng.subset(tr_f, ref))
    q_cls, g_cls = tr_cls[val], tr_cls[ref]

    final = topk(eng.fuse(sims, cfg["weights"], cfg["znorm"]))
    hsv_only = topk(sims["hsv"])
    glob_cfg = {c: w for c, w in cfg["weights"].items() if c in ("hsv", "lbp", "hog")}
    global_only = topk(eng.fuse(sims, glob_cfg or {"hsv": 1}, cfg["znorm"]))

    p_final = per_query_p(final, q_cls, g_cls)
    p_hsv = per_query_p(hsv_only, q_cls, g_cls)
    p_glob = per_query_p(global_only, q_cls, g_cls)
    p_hog = per_query_p(topk(sims["hog"]), q_cls, g_cls)
    rng = np.random.default_rng(0)
    jitter = rng.random(len(val)) * 1e-6          # deterministic tie-break

    # best configuration that actually USES spatial features (from validation_results.csv),
    # so the "spatial" figure is meaningful even when Engine v1 itself has no spatial weight
    v = pd.read_csv("validation_results.csv")
    r = v[(v.w_shsv > 0) | (v.w_slbp > 0)].sort_values("selection_score", ascending=False).iloc[0]
    sp_w = {c: float(r[f"w_{c}"]) for c in eng.COMPONENTS if r[f"w_{c}"] > 0}
    sp_top = topk(eng.fuse(sims, sp_w, bool(r["znorm"])))
    p_sp = per_query_p(sp_top, q_cls, g_cls)
    sp_name = "+".join(f"{c.upper()}{w:.1f}" for c, w in sp_w.items())

    picks = {
        "easy": np.argmax(p_final + jitter),
        "difficult": np.argmin(p_final + jitter),      # a query whose top-5 are mostly wrong-class
        "color_dominated": np.argmax(p_hsv - p_hog + jitter),
        "spatial": np.argmax(p_sp - p_final + jitter),
    }
    titles = {"easy": "Easy query", "difficult": "Difficult query (wrong-class neighbours)",
              "color_dominated": "Colour-dominated query", "spatial": "Spatial query (spatial config beats Engine v1)"}
    for key, qi in picks.items():
        hits = sp_top[qi] if key == "spatial" else final[qi]
        who = f"best spatial config ({sp_name})" if key == "spatial" else "Engine v1"
        draw(qi, hits, f"{titles[key]}  --  {who} top-{K}", tr_paths, tr_cls, ref, val,
             J[qi], f"top{K}_{key}.png")
        print(f"saved figures/top{K}_{key}.png  (query {tr_names[val[qi]]}, class P@5 = {p_final[qi]:.2f})")


if __name__ == "__main__":
    main()