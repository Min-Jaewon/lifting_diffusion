import os
import numpy as np
from glob import glob

# === 설정 ===
stat_dir = "/mnt/dataset1/m_jaewon/cvpr26/DiT4SR/results/layer_abl_stat"   # metric npz들이 저장된 폴더
metric_files = glob(os.path.join(stat_dir, "*_metrics.npz"))

# metric별 누적 저장용 dict
accum_metrics = {'EPE': {}, '1px': {}, '3px': {}, '5px': {}}
num_samples = 0


# Load timestep list from .txt
with open('./timesteps_list.txt', 'r') as f:
    # 소수점 3자리까지 표기
    timesteps = [round(float(line.strip()), 3) for line in f.readlines()]

# === 각 sample의 metric 로드 ===
for path in metric_files:
    data = np.load(path, allow_pickle=True)
    epe = data['epe'].item()      # dict(layer_idx → list of values per timestep)
    _1px = data['_1px'].item()
    _3px = data['_3px'].item()
    _5px = data['_5px'].item()

    for metric_name, metric_dict in zip(['EPE', '1px', '3px', '5px'], [epe, _1px, _3px, _5px]):
        for layer_idx, values in metric_dict.items():
            values = np.array(values)
            if layer_idx not in accum_metrics[metric_name]:
                accum_metrics[metric_name][layer_idx] = values.copy()
            else:
                accum_metrics[metric_name][layer_idx] += values
    num_samples += 1
    
# Average 1px metric over all samples (Keep layer & timestep dimensions)
average_1px_metrics = {}

for layer_idx, values in accum_metrics['1px'].items():
    average_1px_metrics[layer_idx] = values / num_samples
    
if len(average_1px_metrics) == 0:
    raise ValueError("No metrics found to plot.")
import matplotlib.pyplot as plt

# Plot the surface of average 1px metric
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

layer_indices = sorted(average_1px_metrics.keys())
timestep_indices = timesteps  # 실제 timestep 값 사용

L, T = np.meshgrid(layer_indices, timestep_indices)
Z = np.array([average_1px_metrics[layer_idx] for layer_idx in layer_indices]).T  # (len(timesteps), len(layers))

ax.plot_surface(L, T, Z, cmap='viridis')
ax.set_xlabel('Layer Index', fontsize=12)
ax.set_ylabel('Timestep', fontsize=12)   # Index 대신 실제 timestep
# ax.set_zlabel('1px Accuracy', fontsize=12)

plt.savefig(os.path.join(stat_dir, "average_1px_metric_surface.png"), dpi=200)



# Average EPE metric over all samples and timesteps (Keep layer dimension)
average_epe_metrics = {}
for layer_idx, values in accum_metrics['EPE'].items():
    average_epe_metrics[layer_idx] = np.mean(values) / num_samples

# Plot the average EPE metric per layer
plt.figure()
layer_indices = sorted(average_epe_metrics.keys())
epe_values = [average_epe_metrics[layer_idx] for layer_idx in layer_indices]
plt.plot(layer_indices, epe_values, marker='o')
plt.xlabel('Layer Index', fontsize=15)
plt.ylabel('AEPE', fontsize=15)
# plt.title('Average EPE Metric per Layer')
plt.grid()
plt.savefig(os.path.join(stat_dir, "average_epe_metric_per_layer.png"), dpi=200)
plt.show()



# Average EPE metric over all samples and layer (keep timestep dimension)
average_epe_timestep_metrics = {}
max_timesteps = max(len(v) for v in accum_metrics['EPE'].values())
for t in range(max_timesteps):
    timestep_values = []
    for layer_idx, values in accum_metrics['EPE'].items():
        if t < len(values):
            timestep_values.append(values[t])
    average_epe_timestep_metrics[t] = np.mean(timestep_values) / num_samples 

# 원래 timestep 값 (비균일)
x_raw = np.array(timesteps[:max_timesteps])
y = np.array([average_epe_timestep_metrics[t] for t in range(len(x_raw))])

# 플롯용 균일한 x축 (index)
x = np.arange(len(x_raw))

# 최소값 위치 (index 기준)
min_idx = int(np.argmin(y))
min_epe = y[min_idx]
min_timestep = x_raw[min_idx]

plt.figure()

# 라인 + 최소값 점
plt.plot(x, y, marker='o', markersize=5, color='#0055AA', linewidth=1.8)
plt.scatter([min_idx], [min_epe], color='red', s=60, zorder=3)

# 최소 timestep에 세로 점선 (index 기준)
plt.axvline(x=min_idx, color='black', linestyle='--', linewidth=1, alpha=0.7)

# x축을 index 범위로 두고,
plt.xlim(-1, len(x_raw))

# x축 눈금: 0~1000 + 최소 timestep (label만 실제 timestep)
ticks = list(range(0, 1001, 200))
# ticks.append(int(min_timestep))
print(f'Minimum EPE: {min_epe:.4f} at Timestep: {min_timestep}')
ticks = sorted(set(ticks))

tick_pos = [np.argmin(np.abs(x_raw - t)) for t in ticks]  # 눈금이 찍힐 index 위치
plt.xticks(tick_pos, ticks)

plt.grid(alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(stat_dir, "average_epe_metric_per_timestep.png"), dpi=200)
plt.show()


# 24개의 layer를 timestep에 따른 epe를 한 plot에 그리기
plt.figure()
layers = sorted(accum_metrics['EPE'].keys())
colors = plt.cm.plasma(np.linspace(0, 1, len(layers)))
all_epe = []

x = timesteps  # [999, 973, ..., 0] 실제 timestep 값

for i, layer_idx in enumerate(layers):
    epe_values = accum_metrics['EPE'][layer_idx] / num_samples
    all_epe.append(epe_values)
    plt.plot(x, epe_values,
             color=colors[i], label=f'Layer {layer_idx}', linewidth=1.5)

# 너무 큰 값 잘라내기 + 여백 주기
all_epe = np.concatenate(all_epe)
y_min = all_epe.min()
y_max = np.percentile(all_epe, 95)
margin = (y_max - y_min) * 0.05

plt.ylim(y_min - margin, y_max + margin)
plt.xlim(0, 1000)  # x축 0~1000

plt.margins(x=0.02)

# plt.xlabel('Timestep', fontsize=15)
# plt.ylabel('EPE', fontsize=15)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
plt.grid()
plt.savefig(os.path.join(stat_dir, "epe_metric_per_layer_over_timesteps.png"),
            dpi=200, bbox_inches='tight')
plt.show()