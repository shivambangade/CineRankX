"""Draw the diagrams and result charts used in the study guide (data charts read real saved artifacts)."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path("docs/_build/figs")
OUT.mkdir(parents=True, exist_ok=True)
P = Path("data/processed")
ORANGE, BLUE, TEAL, SLATE, GREY = "#F97316", "#2563EB", "#0D9488", "#0F172A", "#E2E8F0"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})


def box(ax, x, y, w, h, text, fc="#FFFFFF", ec=SLATE, size=8.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=SLATE,
            fontweight="bold" if bold else "normal", wrap=True)


def arrow(ax, p, q, color=SLATE, style="-|>", ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=11, color=color, lw=1.2, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}"))


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def architecture():
    fig, ax = canvas(10, 6.6)
    layers = [("Presentation  (Streamlit)", 5.6, "#FFEDD5"), ("Evaluation & benchmarking", 4.4, "#DBEAFE"),
              ("Recommendation engines", 2.7, "#CCFBF1"), ("Data engineering & storage", 0.9, "#F1F5F9")]
    for name, y, c in layers:
        ax.add_patch(FancyBboxPatch((0.2, y), 9.6, 1.0 if y != 2.7 else 1.6, boxstyle="round,pad=0.02,rounding_size=0.1", fc=c, ec="none"))
        ax.text(9.7, y + (0.82 if y != 2.7 else 1.42), name, fontsize=8.5, fontweight="bold", color="#475569", ha="right")
    for i, t in enumerate(["app.py\n(router, cache)", "Overview", "Engines", "Run Engine\n(runner.py)", "User Profile\n(profile*.py)", "KPI Benchmarks\n(kpi_view)"]):
        box(ax, 0.4 + i * 1.58, 5.68, 1.45, 0.62, t, "#FFFFFF", size=7.2)
    for i, t in enumerate(["src/evaluation\n(23 KPIs)", "src/benchmark\n(frozen split + baseline)", "scripts/\n(run_benchmark, KPI report)", "reports/*.json\n(saved results)"]):
        box(ax, 0.5 + i * 2.38, 4.5, 2.2, 0.62, t, "#FFFFFF", size=7.5)
    box(ax, 0.4, 2.85, 2.9, 1.0, "Module 2  IR engine\nTF-IDF + cosine + Trie\n(src/ir_engine)", "#FFFFFF", size=7.8)
    box(ax, 3.55, 2.85, 2.9, 1.0, "Module 3  Strategy selector\nfeatures -> LightGBM -> strategy\n(src/strategy_selector)", "#FFFFFF", size=7.8)
    box(ax, 6.7, 2.85, 3.0, 1.0, "Module 4  Ranking engine\ncandidates + 6 objectives\n(src/ranking_engine)", "#FFFFFF", size=7.8)
    for i, t in enumerate(["MovieLens 20M CSV\n+ TMDB 2023 CSV", "src/ingestion\nload - join - clean", "data/processed\nmovies_merged.csv, TF-IDF,\nmodels (.pkl)", "SQLite\ncinerankx.db\n(ratings, runs)"]):
        box(ax, 0.5 + i * 2.38, 1.0, 2.2, 0.78, t, "#FFFFFF", size=7.3)
    for x in (1.6, 3.98, 6.36):
        arrow(ax, (x, 1.8), (x + 0.2, 2.82), "#64748B")
    arrow(ax, (3.35, 3.35), (3.55, 3.35)); arrow(ax, (6.45, 3.35), (6.7, 3.35))
    arrow(ax, (5.0, 3.88), (5.0, 4.48), "#64748B"); arrow(ax, (5.0, 5.14), (5.0, 5.66), "#64748B")
    ax.set_title("CineRankX layered architecture", fontsize=11, fontweight="bold", color=SLATE)
    fig.savefig(OUT / "architecture.png", dpi=170, bbox_inches="tight"); plt.close(fig)


def dataflow():
    fig, ax = canvas(10, 5.6)
    spec = [(0.2, 4.4, 2.0, 0.8, "rating.csv\nmovie.csv, tag.csv\nlink.csv"), (0.2, 3.2, 2.0, 0.8, "TMDB 2023 CSV\n(overview, keywords)"),
            (2.7, 3.8, 2.0, 1.0, "Ingestion\njoin on tmdbId,\nclean text (NLTK)"), (5.2, 3.8, 2.0, 1.0, "movies_merged.csv\n+ TF-IDF matrix"),
            (7.7, 4.4, 2.1, 0.8, "SQLite DB\n20M ratings, users"), (7.7, 3.2, 2.1, 0.8, "user ratings\n(profile)"),
            (0.2, 1.9, 2.2, 0.8, "Profile features\n(taste, activity,\ninner backtests)"), (2.9, 1.9, 2.0, 0.8, "Strategy classifier\n(LightGBM)"),
            (5.4, 1.9, 2.0, 0.8, "strategy +\nconfidence"), (7.9, 1.9, 1.9, 0.8, "source weights\n(CB / CF / Pop)"),
            (0.2, 0.3, 2.3, 0.9, "Candidate pool\nIR + user-kNN +\npopularity (200)"), (3.0, 0.3, 2.6, 0.9, "Multi-objective ranker\n(6 objectives, greedy)"),
            (6.1, 0.3, 1.7, 0.9, "Top-K list\n+ explanations"), (8.2, 0.3, 1.6, 0.9, "Streamlit UI\nCSV export,\nrun log")]
    for x, y, w, h, t in spec:
        box(ax, x, y, w, h, t, "#FFFFFF", size=7.4)
    A = [((2.2, 4.8), (2.7, 4.4)), ((2.2, 3.6), (2.7, 4.1)), ((4.7, 4.3), (5.2, 4.3)), ((7.2, 4.3), (7.7, 4.8)), ((8.75, 4.4), (8.75, 4.0)),
         ((7.7, 3.4), (2.3, 2.7)), ((2.4, 2.3), (2.9, 2.3)), ((4.9, 2.3), (5.4, 2.3)), ((7.4, 2.3), (7.9, 2.3)),
         ((8.85, 1.9), (2.4, 1.2)), ((2.5, 0.75), (3.0, 0.75)), ((5.6, 0.75), (6.1, 0.75)), ((7.8, 0.75), (8.2, 0.75)), ((6.2, 3.8), (1.6, 1.2))]
    for p, q in A:
        arrow(ax, p, q, "#64748B", rad=0.0)
    ax.set_title("End-to-end data flow (live recommendation path)", fontsize=11, fontweight="bold", color=SLATE)
    fig.savefig(OUT / "dataflow.png", dpi=170, bbox_inches="tight"); plt.close(fig)


def sequence():
    fig, ax = canvas(10, 6.4)
    names = ["User", "app.py /\nrunner.py", "Engine cache\n(load_engine)", "Strategy\nselector", "Candidate\ngenerator", "Multi-objective\nranker", "SQLite\n(run log)"]
    xs = np.linspace(0.8, 9.2, len(names))
    for x, n in zip(xs, names):
        box(ax, x - 0.55, 5.55, 1.1, 0.6, n, "#FFEDD5", size=7)
        ax.plot([x, x], [0.2, 5.5], color="#94A3B8", lw=0.8, ls="--")
    msgs = [(0, 1, "1 enter user ID, K, weights; click Generate", 5.2), (1, 2, "2 get ratings / catalog / ranker (cached)", 4.8),
            (1, 3, "3 build_features(user ratings)", 4.4), (3, 1, "4 strategy + confidence", 4.0), (1, 4, "5 source weights from strategy", 3.6),
            (4, 5, "6 pool: IR + CF + popularity (200)", 3.2), (5, 5, "7 greedy pick: 6 objectives", 2.8), (5, 1, "8 ranked top-K + score breakdown", 2.4),
            (1, 6, "9 record_run(weights, strategy, results)", 2.0), (1, 0, "10 movie cards, export CSV", 1.6)]
    for a, b, text, y in msgs:
        if a == b:
            ax.annotate("", xy=(xs[a] + 0.02, y - 0.12), xytext=(xs[a] + 0.02, y + 0.12), arrowprops=dict(arrowstyle="-|>", color=SLATE, connectionstyle="arc3,rad=-1.8"))
            ax.text(xs[a] + 0.35, y, text, fontsize=6.8, va="center")
        else:
            arrow(ax, (xs[a], y), (xs[b], y), SLATE)
            ax.text((xs[a] + xs[b]) / 2, y + 0.07, text, fontsize=6.8, ha="center", va="bottom")
    ax.set_title("Sequence of one live recommendation", fontsize=11, fontweight="bold", color=SLATE)
    fig.savefig(OUT / "sequence.png", dpi=170, bbox_inches="tight"); plt.close(fig)


def training_pipeline():
    fig, ax = canvas(10, 3.4)
    steps = ["Sample 5,000 users\n(30-1000 ratings)", "Per-user split\nprofile 70% / holdout 30%", "Features from\nprofile only",
             "Backtest 3 strategies\non the holdout", "Label = best\nhit-rate (tie -> popularity)", "80/20 split,\n5-fold CV on 80%", "Fit LightGBM,\nscore holdout once"]
    for i, t in enumerate(steps):
        x = 0.1 + i * 1.4
        box(ax, x, 1.2, 1.25, 1.0, t, "#FFFFFF" if i not in (2, 4) else "#FFEDD5", size=6.8)
        if i:
            arrow(ax, (x - 0.15, 1.7), (x, 1.7))
    ax.text(5.0, 0.6, "Features never see the holdout ratings; labels come only from the holdout.", ha="center", fontsize=8, color="#475569")
    ax.set_title("Strategy-selector training pipeline", fontsize=11, fontweight="bold", color=SLATE)
    fig.savefig(OUT / "training_pipeline.png", dpi=170, bbox_inches="tight"); plt.close(fig)


def charts():
    a = json.loads((P / "selector_analysis.json").read_text())
    bins = a["accuracy_by_label_margin"]
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    labels = ["most ambiguous\n(lowest quarter)", "second", "third", "clearest\n(top quarter)"]
    vals = [v["accuracy"] for v in bins.values()]
    ax.bar(labels, vals, color=[ORANGE, "#FDBA74", "#93C5FD", BLUE])
    ax.axhline(a["current_model_full_population"]["majority_baseline"], color="#64748B", ls="--", lw=1)
    ax.text(3.45, a["current_model_full_population"]["majority_baseline"] + 0.01, "majority class 0.644", ha="right", fontsize=7.5, color="#475569")
    ax.set_ylabel("held-out accuracy"); ax.set_ylim(0.4, 0.8)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.008, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_title("Accuracy rises with label clarity (999 held-out users)", fontsize=9.5)
    fig.savefig(OUT / "label_margin.png", dpi=170, bbox_inches="tight"); plt.close(fig)

    c = json.loads(Path("reports/strategy_comparison.json").read_text())
    base, new, old = c["baseline"]["results"], c["cinerankx"]["after"], c["cinerankx"]["before"]
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    x = np.arange(2); w = 0.26
    for j, (lab, src, col) in enumerate([("Frozen baseline (most popular)", base, "#94A3B8"), ("CineRankX before (old pipeline)", old, BLUE), ("CineRankX now", new, ORANGE)]):
        v = [src["overall"]["hit_rate@10"], src["cold"]["hit_rate@10"]]
        ax.bar(x + (j - 1) * w, v, w, label=lab, color=col)
        for i, t in enumerate(v):
            ax.text(x[i] + (j - 1) * w, t + 0.01, f"{t:.3f}", ha="center", fontsize=7.5)
    ax.set_xticks(x, ["Overall users (2,000)", "Cold-start users (1,000)"]); ax.set_ylim(0, 1.0); ax.set_ylabel("Hit Rate@10")
    ax.legend(fontsize=7, loc="upper right"); ax.set_title("Recommendation accuracy on the frozen test split", fontsize=9.5)
    fig.savefig(OUT / "strategy_comparison.png", dpi=170, bbox_inches="tight"); plt.close(fig)

    d = json.loads((P / "strategy_eval.json").read_text())["machine_learning"]
    cm = np.array(d["confusion_matrix"]); names = d["confusion_matrix_labels"]
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    ax.imshow(cm, cmap="Oranges")
    ax.set_xticks(range(3), [n.replace("_", "\n") for n in names]); ax.set_yticks(range(3), names)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else SLATE, fontsize=10)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title("Deployed selector confusion matrix", fontsize=9.5)
    fig.savefig(OUT / "confusion.png", dpi=170, bbox_inches="tight"); plt.close(fig)

    z = json.loads((P / "selector_experiments_inner3.json").read_text())["table"]
    items = sorted(z.items(), key=lambda kv: kv[1]["cv"]["macro_f1"])
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    ax.barh([k for k, _ in items], [v["cv"]["macro_f1"] for _, v in items], color=[ORANGE if k == "multiclass/lightgbm" else "#93C5FD" for k, _ in items])
    ax.set_xlim(0.38, 0.5); ax.set_xlabel("5-fold CV macro-F1 (development users)")
    for i, (_, v) in enumerate(items):
        ax.text(v["cv"]["macro_f1"] + 0.002, i, f"{v['cv']['macro_f1']:.3f}", va="center", fontsize=7)
    ax.tick_params(axis="y", labelsize=7); ax.set_title("14 candidates, one protocol: all within a narrow band", fontsize=9.5)
    fig.savefig(OUT / "model_zoo.png", dpi=170, bbox_inches="tight"); plt.close(fig)

    ce = json.loads((P / "content_experiments.json").read_text())["variants"]
    pick = ["base (original)", "rich TF-IDF (bigrams, 20k)", "quality prior a=2.0", "rich + quality a=2.0", "popularity (reference)", "collaborative (reference)"]
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    m = [ce[k]["confirm_mean"] for k in pick]; lo = [ce[k]["ci95"][0] for k in pick]; hi = [ce[k]["ci95"][1] for k in pick]
    ax.bar(range(len(pick)), m, yerr=[np.array(m) - lo, np.array(hi) - m], color=[BLUE if "reference" not in k else "#94A3B8" for k in pick], capsize=3)
    ax.set_xticks(range(len(pick)), [k.replace(" (", "\n(").replace("rich + ", "rich +\n") for k in pick], fontsize=6.3)
    ax.set_ylabel("hit-rate on holdout (30 recs)"); ax.set_title("Content-based variants on 1,000 unseen users (95% CI)", fontsize=9.5)
    fig.savefig(OUT / "content_variants.png", dpi=170, bbox_inches="tight"); plt.close(fig)

    sc = json.loads((P / "selector_phase2_base.json").read_text())["final"]["decision"]
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    labs = ["always\npopularity", "always\ncollaborative", "CineRankX\nselector", "oracle\n(upper bound)"]
    v = [sc["always_popularity"], sc["always_collaborative"], sc["realised_hit_rate"], sc["oracle_hit_rate"]]
    ax.bar(labs, v, color=["#94A3B8", "#94A3B8", ORANGE, TEAL])
    for i, t in enumerate(v):
        ax.text(i, t + 0.002, f"{t:.4f}", ha="center", fontsize=8)
    ax.set_ylim(0.09, 0.16); ax.set_ylabel("mean hit-rate (999 held-out users)"); ax.set_title("What strategy choice is worth", fontsize=9.5)
    fig.savefig(OUT / "decision_value.png", dpi=170, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    architecture(); dataflow(); sequence(); training_pipeline(); charts()
    print(sorted(p.name for p in OUT.iterdir()))
