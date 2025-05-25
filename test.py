import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 定義每層的節點數
layers = [7, 250, 100, 50, 3]

# 繪圖參數
layer_distance = 3
node_radius = 0.1

fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlim(0, layer_distance * (len(layers) - 1) + 1)
ax.set_ylim(-max(layers) * 0.03, max(layers) * 0.03)
ax.axis('off')

# 畫節點
positions = []  # 紀錄每層節點的位置以便畫線
for i, num_nodes in enumerate(layers):
    x = i * layer_distance
    y_start = -(num_nodes - 1) / 2
    layer_positions = []
    for j in range(num_nodes):
        y = y_start + j
        circle = patches.Circle((x, y), node_radius, edgecolor='black', facecolor='skyblue', lw=1)
        ax.add_patch(circle)
        layer_positions.append((x, y))
    positions.append(layer_positions)

# 畫線
for i in range(len(positions) - 1):
    for src in positions[i]:
        for dst in positions[i + 1]:
            ax.plot([src[0], dst[0]], [src[1], dst[1]], 'gray', linewidth=0.2)

plt.title("MLP Structure: (7×250) → (250×100) → (100×50) → (50×3)")
plt.show()
