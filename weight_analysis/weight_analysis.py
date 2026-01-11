import os
import re
import json
import torch
import pandas as pd

EPS = 1e-12

def load_state_dict(path, map_location="cpu"):
    import os
    path = os.path.expanduser(path)

    # Directory: sharded safetensors (diffusers)
    if os.path.isdir(path):
        index_files = [f for f in os.listdir(path) if f.endswith(".safetensors.index.json")]
        if not index_files:
            raise FileNotFoundError(f"No *.safetensors.index.json found in directory: {path}")

        preferred = "diffusion_pytorch_model.safetensors.index.json"
        index_name = preferred if preferred in index_files else sorted(index_files)[0]
        index_path = os.path.join(path, index_name)

        from safetensors.torch import load_file
        with open(index_path, "r") as f:
            index = json.load(f)

        weight_map = index.get("weight_map", {})
        if not weight_map:
            raise ValueError(f"Invalid index file (no weight_map): {index_path}")

        state_dict = {}
        shard_cache = {}
        for key, shard_name in weight_map.items():
            shard_path = os.path.join(path, shard_name)
            if shard_path not in shard_cache:
                shard_cache[shard_path] = load_file(shard_path, device=map_location)
            state_dict[key] = shard_cache[shard_path][key]
        return state_dict

    # File: .safetensors
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(path, device=map_location)

    # torch serialized
    ckpt = torch.load(path, map_location=map_location)
    if isinstance(ckpt, dict):
        for k in ["state_dict", "model", "ema", "module"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                if k == "ema" and "state_dict" in ckpt[k] and isinstance(ckpt[k]["state_dict"], dict):
                    return ckpt[k]["state_dict"]
                return ckpt[k]
    return ckpt


def default_group(key, depth=3):
    parts = key.split(".")
    return ".".join(parts[:depth])


@torch.inference_mode()
def tensor_metrics_gpu(w0, w1, device="cuda:0", dtype=torch.float32):
    """
    Compute metrics on GPU for a single tensor pair.
    Moves only this tensor pair to GPU (safe for memory).
    Returns python floats.
    """
    # move to GPU + dtype
    x0 = w0.detach().to(device=device, dtype=dtype, non_blocking=True).reshape(-1)
    x1 = w1.detach().to(device=device, dtype=dtype, non_blocking=True).reshape(-1)

    d = x1 - x0

    # L2 norms
    l2_d = torch.linalg.vector_norm(d)
    l2_0 = torch.linalg.vector_norm(x0)
    ratio = l2_d / (l2_0 + EPS)

    mad = d.abs().mean()

    # cosine similarity
    denom = (torch.linalg.vector_norm(x0) * torch.linalg.vector_norm(x1) + EPS)
    cos = (x0 @ x1) / denom

    # sync minimal (item() triggers sync)
    return l2_d.item(), ratio.item(), mad.item(), cos.item()


def compare(sd0, sdX, tagX, group_depth=3, key_filter=None,
            device="cuda:0", dtype=torch.float32, verbose_every=2000):
    rows = []
    common = sorted(set(sd0.keys()) & set(sdX.keys()))

    # 속도: 파이썬 루프 자체도 크니까, print는 가끔만
    for i, k in enumerate(common):
        v0, v1 = sd0[k], sdX[k]
        if not (torch.is_tensor(v0) and torch.is_tensor(v1)):
            continue
        if v0.shape != v1.shape:
            continue
        if key_filter and not key_filter(k):
            continue

        l2_d, ratio, mad, cos = tensor_metrics_gpu(v0, v1, device=device, dtype=dtype)

        rows.append({
            "key": k,
            "group": default_group(k, depth=group_depth),
            f"{tagX}_l2_delta": l2_d,
            f"{tagX}_l2_ratio": ratio,
            f"{tagX}_mad": mad,
            f"{tagX}_cos": cos,
            "numel": v0.numel(),
        })

        if verbose_every and (i + 1) % verbose_every == 0:
            print(f"[{tagX}] processed {i+1}/{len(common)} keys...")

    return pd.DataFrame(rows)


def merge_A_B(dfA, dfB):
    return pd.merge(dfA, dfB, on=["key", "group", "numel"], how="outer")


def group_summary(df, tagA="A", tagB="B"):
    g = df.groupby("group", dropna=False)
    out = pd.DataFrame({
        "tensors": g.size(),
        "numel_sum": g["numel"].sum(),
        f"{tagA}_l2_delta_sum": g[f"{tagA}_l2_delta"].sum(),
        f"{tagB}_l2_delta_sum": g[f"{tagB}_l2_delta"].sum(),
        f"{tagA}_l2_ratio_mean": g[f"{tagA}_l2_ratio"].mean(),
        f"{tagB}_l2_ratio_mean": g[f"{tagB}_l2_ratio"].mean(),
    }).reset_index()

    out["B_over_A_delta"] = out[f"{tagB}_l2_delta_sum"] / (out[f"{tagA}_l2_delta_sum"] + EPS)
    return out.sort_values(f"{tagB}_l2_delta_sum", ascending=False)


if __name__ == "__main__":
    path0 = "/mnt/dataset1/m_jaewon/icml26/DiT4SR/preset/models/dit4sr_q/transformer/diffusion_pytorch_model.safetensors"
    pathA = "/mnt/dataset3/jaewon/_hanwha_weights/[Later]_Baseline/checkpoint-400000/transformer/diffusion_pytorch_model.safetensors"
    pathB = "/mnt/dataset3/jaewon/_hanwha_weights/[Later]_Baseline_FullFT/checkpoint-400000/transformer"

    sd0 = load_state_dict(path0, map_location="cpu")
    sdA = load_state_dict(pathA, map_location="cpu")
    sdB = load_state_dict(pathB, map_location="cpu")

    # 필요하면 transformer만 필터
    key_filter = None  # lambda k: k.startswith("transformer.")

    # GPU 설정
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32  # 안전/정확/속도 밸런스

    print("Device:", device)

    dfA = compare(sd0, sdA, "A", group_depth=3, key_filter=key_filter,
                  device=device, dtype=dtype, verbose_every=2000)
    # GPU 메모리 캐시 정리(안정)
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    dfB = compare(sd0, sdB, "B", group_depth=3, key_filter=key_filter,
                  device=device, dtype=dtype, verbose_every=2000)
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    df = merge_A_B(dfA, dfB)
    df["B_over_A_ratio"] = df["B_l2_ratio"] / (df["A_l2_ratio"] + EPS)

    save_dir = "weight_analysis"
    os.makedirs(save_dir, exist_ok=True)
    df.to_csv(f"{save_dir}/tensor_level.csv", index=False)

    summ = group_summary(df, "A", "B")
    summ.to_csv(f"{save_dir}/group_summary.csv", index=False)

    print("Top changed tensors (A):")
    print(df.sort_values("A_l2_ratio", ascending=False).head(20)[["key","A_l2_ratio","B_l2_ratio","B_over_A_ratio"]])

    print("\nTop changed tensors (B):")
    print(df.sort_values("B_l2_ratio", ascending=False).head(20)[["key","A_l2_ratio","B_l2_ratio","B_over_A_ratio"]])

    print(f"\nSaved to {save_dir}/")
