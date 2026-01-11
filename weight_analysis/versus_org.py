import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SRC_DIR = "weight_analysis"
SRC_TENSOR = os.path.join(SRC_DIR, "tensor_level.csv")
SRC_GROUP  = os.path.join(SRC_DIR, "group_summary.csv")

OUT_DIR = "weight_analysis"
PLOT_DIR = os.path.join(OUT_DIR, "plots_vs_org")
os.makedirs(PLOT_DIR, exist_ok=True)

df = pd.read_csv(SRC_TENSOR)
gs = pd.read_csv(SRC_GROUP)

# -----------------------------
# (A) B 전용 테이블로 정리해서 저장
# -----------------------------
dfB = df[["key", "group", "numel", "B_l2_delta", "B_l2_ratio", "B_mad", "B_cos"]].copy()
dfB.to_csv(os.path.join(OUT_DIR, "tensor_level_B_only.csv"), index=False)

gsB = gs[["group", "tensors", "numel_sum", "B_l2_delta_sum", "B_l2_ratio_mean"]].copy()
gsB.to_csv(os.path.join(OUT_DIR, "group_summary_B_only.csv"), index=False)

# -----------------------------
# (1) Group bar: B_l2_delta_sum (TopK)
# -----------------------------
topk = 25
gs_top = gsB.sort_values("B_l2_delta_sum", ascending=False).head(topk)

x = np.arange(len(gs_top))

plt.figure(figsize=(14, 6))
plt.bar(x, gs_top["B_l2_delta_sum"].to_numpy())
plt.xticks(x, gs_top["group"].astype(str).to_list(), rotation=75, ha="right")
plt.ylabel("Sum of L2 delta (B vs original)")
plt.title(f"Top-{topk} groups changed in B vs original")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "B_group_delta_sum_topk.png"), dpi=200)
plt.close()

# -----------------------------
# (2) Transformer block heatmap: B_l2_ratio (weighted mean)
# -----------------------------
block_pat = re.compile(r"(?:^|\.)(?:transformer\.)?(?:blocks|block)\.(\d+)(?:\.|$)")

def extract_block_id(key):
    m = block_pat.search(str(key))
    return int(m.group(1)) if m else None

mask = dfB["key"].astype(str).str.contains(
    r"(?:^|\.)(?:transformer\.)?(?:blocks|block)\.\d+(?:\.|$)", regex=True
)
t = dfB[mask].copy()
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
             "B_ratio_wmean": wmean(g["B_l2_ratio"], g["numel"]),
         }))
         .sort_values("block")
    )
    blocks = blk["block"].to_numpy()
    M = blk["B_ratio_wmean"].to_numpy()[None, :]  # (1, n)

    plt.figure(figsize=(14, 2.6))
    plt.imshow(M, aspect="auto")
    plt.yticks([0], ["B"])
    plt.xticks(np.arange(len(blocks)), blocks, rotation=90)
    plt.colorbar(label="Weighted mean L2 ratio (B vs original)")
    plt.title("Transformer blocks: B change vs original")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "B_transformer_block_heatmap_ratio.png"), dpi=200)
    plt.close()
else:
    print("[WARN] No transformer blocks detected from keys. Skip block heatmap.")

# -----------------------------
# (3) Scatter: numel vs B_l2_ratio (log-numel)
#     -> '큰 텐서가 얼마나 변했는지' 보기 좋음
# -----------------------------
plot_df = dfB.dropna(subset=["B_l2_ratio", "numel"]).copy()
N = 30000
if len(plot_df) > N:
    plot_df = plot_df.sort_values("B_l2_ratio", ascending=False).head(N)

x = np.log10(plot_df["numel"].to_numpy() + 1.0)
y = plot_df["B_l2_ratio"].to_numpy()

plt.figure(figsize=(7, 5))
plt.scatter(x, y, s=6, alpha=0.3)
plt.xlabel("log10(numel)")
plt.ylabel("B: L2 ratio vs original")
plt.title("Tensor change in B vs original (size vs relative change)")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "B_scatter_lognumel_vs_ratio.png"), dpi=200)
plt.close()

# -----------------------------
# (4) Top tensors in B: bar plot (TopK)
# -----------------------------
topk_t = 30
top_tensors = dfB.sort_values("B_l2_ratio", ascending=False).head(topk_t)

x = np.arange(len(top_tensors))
plt.figure(figsize=(14, 6))
plt.bar(x, top_tensors["B_l2_ratio"].to_numpy())
plt.xticks(x, top_tensors["key"].astype(str).to_list(), rotation=80, ha="right")
plt.ylabel("B: L2 ratio vs original")
plt.title(f"Top-{topk_t} tensors changed in B vs original")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "B_top_tensors_ratio.png"), dpi=200)
plt.close()

print(f"[DONE] Saved B-only report to: {OUT_DIR}")
print("CSV:")
print(" - tensor_level_B_only.csv")
print(" - group_summary_B_only.csv")
print("Plots:")
print(" - B_group_delta_sum_topk.png")
print(" - B_transformer_block_heatmap_ratio.png")
print(" - B_scatter_lognumel_vs_ratio.png")
print(" - B_top_tensors_ratio.png")
