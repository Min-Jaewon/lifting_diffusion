import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPORT_DIR = "weight_analysis"
TENSOR_CSV = os.path.join(REPORT_DIR, "tensor_level.csv")
GROUP_CSV  = os.path.join(REPORT_DIR, "group_summary.csv")
OUT_DIR    = os.path.join(REPORT_DIR, "plots")
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(TENSOR_CSV)
gs = pd.read_csv(GROUP_CSV)

# ---------- 0) Safety check ----------
req_df = {"key","group","numel","A_l2_ratio","B_l2_ratio","A_l2_delta","B_l2_delta","B_over_A_ratio"}
req_gs = {"group","A_l2_delta_sum","B_l2_delta_sum","B_over_A_delta"}

if not req_df.issubset(df.columns):
    raise ValueError(f"tensor_level.csv missing: {req_df - set(df.columns)}")
if not req_gs.issubset(gs.columns):
    raise ValueError(f"group_summary.csv missing: {req_gs - set(gs.columns)}")

# ============================================================
# 1) Group delta sum bar (A vs B)
# ============================================================
topk = 25
gs_top = gs.sort_values("B_l2_delta_sum", ascending=False).head(topk)

x = np.arange(len(gs_top))
w = 0.4

plt.figure(figsize=(14, 6))
plt.bar(x - w/2, gs_top["A_l2_delta_sum"], width=w, label="A (partial FT)")
plt.bar(x + w/2, gs_top["B_l2_delta_sum"], width=w, label="B (full FT)")
plt.xticks(x, gs_top["group"], rotation=75, ha="right")
plt.ylabel("Sum of L2 delta")
plt.title(f"Top-{topk} groups changed vs original")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "group_delta_sum_topk.png"), dpi=200)
plt.close()

# ============================================================
# 2) Transformer block heatmap (A/B ratio)
# ============================================================
block_pat = re.compile(r"(?:^|\.)(?:transformer\.)?(?:blocks|block)\.(\d+)(?:\.|$)")

def extract_block_id(key):
    m = block_pat.search(str(key))
    return int(m.group(1)) if m else None

mask = df["key"].astype(str).str.contains(
    r"(?:^|\.)(?:transformer\.)?(?:blocks|block)\.\d+(?:\.|$)", regex=True
)
t = df[mask].copy()
t["block"] = t["key"].apply(extract_block_id)
t = t.dropna(subset=["block"])
t["block"] = t["block"].astype(int)

def wmean(v, w):
    v = np.asarray(v, float)
    w = np.asarray(w, float)
    return (v * w).sum() / (w.sum() + 1e-12)

if len(t) > 0:
    blk = (
        t.groupby("block", as_index=False)
         .apply(lambda g: pd.Series({
             "A": wmean(g["A_l2_ratio"], g["numel"]),
             "B": wmean(g["B_l2_ratio"], g["numel"]),
         }))
         .sort_values("block")
    )

    M = np.vstack([blk["A"].to_numpy(), blk["B"].to_numpy()])
    blocks = blk["block"].to_numpy()

    plt.figure(figsize=(14, 3))
    plt.imshow(M, aspect="auto")
    plt.yticks([0, 1], ["A", "B"])
    plt.xticks(np.arange(len(blocks)), blocks, rotation=90)
    plt.colorbar(label="Weighted mean L2 ratio")
    plt.title("Transformer blocks: change vs original")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "transformer_block_heatmap_ratio.png"), dpi=200)
    plt.close()

# ============================================================
# 3) Tensor-level scatter (A vs B)
# ============================================================
plot_df = df.dropna(subset=["A_l2_ratio","B_l2_ratio"])
N = 30000
if len(plot_df) > N:
    plot_df = plot_df.sort_values("B_l2_ratio", ascending=False).head(N)

x = plot_df["A_l2_ratio"].to_numpy()
y = plot_df["B_l2_ratio"].to_numpy()

plt.figure(figsize=(7, 7))
plt.scatter(x, y, s=6, alpha=0.3)
m = max(x.max(), y.max())
plt.plot([0, m], [0, m])
plt.xlabel("A: L2 ratio")
plt.ylabel("B: L2 ratio")
plt.title("Tensor-level change (above line = B changed more)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "tensor_scatter_A_vs_B.png"), dpi=200)
plt.close()

# ============================================================
# 4) NEW — B_over_A_ratio histogram (tensor-level)
# ============================================================
h = df["B_over_A_ratio"].replace([np.inf, -np.inf], np.nan).dropna()

plt.figure(figsize=(7, 4))
plt.hist(h, bins=80)
plt.axvline(1.0)
plt.xlabel("B_over_A_ratio")
plt.ylabel("Tensor count")
plt.title("How much more tensors changed in B vs A")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "tensor_B_over_A_ratio_hist.png"), dpi=200)
plt.close()

# ============================================================
# 5) NEW — Group bar: B_over_A_delta
# ============================================================
gs_bar = gs.sort_values("B_over_A_delta", ascending=False).head(topk)

x = np.arange(len(gs_bar))

plt.figure(figsize=(14, 6))
plt.bar(x, gs_bar["B_over_A_delta"])
plt.xticks(x, gs_bar["group"], rotation=75, ha="right")
plt.axhline(1.0)
plt.ylabel("B_over_A_delta (group)")
plt.title(f"Top-{topk} groups where full FT changed more than partial FT")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "group_B_over_A_delta_topk.png"), dpi=200)
plt.close()

print(f"[DONE] Plots saved to {OUT_DIR}")
print(" - group_delta_sum_topk.png")
print(" - transformer_block_heatmap_ratio.png")
print(" - tensor_scatter_A_vs_B.png")
print(" - tensor_B_over_A_ratio_hist.png")
print(" - group_B_over_A_delta_topk.png")
