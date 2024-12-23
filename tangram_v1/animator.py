import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import json
import numpy as np

with open("tangram_v1/logs/movement_logs.json", "r") as file:
    logs = json.load(file)

fig, ax = plt.subplots(figsize=(10, 8))
ax.set_xlim(0, 2000)
ax.set_ylim(0, 1200)
ax.set_aspect('equal')
ax.set_title("Tangram Animation")

def create_tangram_shapes():
    """Define vertices of Tangram shapes (relative to the centroid)."""
    shapes = {
        "Large Triangle 1": np.array([[0, 0], [100, 200], [200, 0]]),
        "Large Triangle 2": np.array([[0, 0], [100, 200], [-100, 200]]),
        "Medium Triangle": np.array([[0, 0], [50, 100], [100, 0]]),
        "Small Triangle 1": np.array([[0, 0], [25, 50], [50, 0]]),
        "Small Triangle 2": np.array([[0, 0], [25, 50], [-25, 50]]),
        "Square": np.array([[0, 0], [0, 50], [50, 50], [50, 0]]),
        "Parallelogram": np.array([[0, 0], [50, 50], [100, 50], [50, 0]])
    }
    return shapes

tangram_shapes = create_tangram_shapes()
pieces = {}

for log in logs:
    shape = log["shape"]
    color = log["color"]
    centroid = log["centroid"]

    if shape in tangram_shapes:
        polygon = patches.Polygon(
            tangram_shapes[shape] + centroid,
            closed=True,
            color=color,
            alpha=0.7,
            label=shape
        )
        ax.add_patch(polygon)
        pieces[shape] = polygon

def update(frame):
    ax.clear()
    ax.set_xlim(0, 2000)
    ax.set_ylim(0, 1200)
    ax.set_aspect('equal')
    ax.set_title(f"Tangram Animation - Frame {frame}")

    for log in logs[:frame + 1]:
        shape = log["shape"]
        if shape not in tangram_shapes:
            continue  # unknown shapes

        color = log["color"]
        centroid = log["centroid"]
        rotation_angle = log["rotation_angle"]

        polygon = tangram_shapes[shape]
        theta = np.radians(rotation_angle)
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        rotated_polygon = np.dot(polygon, rotation_matrix.T)

        transformed_polygon = rotated_polygon + centroid

        patch = patches.Polygon(
            transformed_polygon,
            closed=True,
            color=color,
            alpha=0.7,
            label=shape
        )
        ax.add_patch(patch)

ani = FuncAnimation(fig, update, frames=len(logs), interval=500)
plt.legend()
plt.show()
