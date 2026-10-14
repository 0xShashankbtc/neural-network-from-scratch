"""
Visualization utilities for datasets, loss curves, and 2D decision boundaries.
"""

from typing import List, Optional
import numpy as np


def plot_dataset(
    dataset: List[List[float]],
    query_point: Optional[List[float]] = None,
    save_path: Optional[str] = None,
) -> None:
    """Plot training samples and optional query sample in 2D feature space."""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 5))
    plt.grid(True, linestyle="--", alpha=0.5)

    for point in dataset:
        color = "red" if point[2] == 1 else "blue"
        plt.scatter(point[0], point[1], c=color, s=70, edgecolors="k")

    if query_point is not None:
        plt.scatter(
            query_point[0],
            query_point[1],
            c="gold",
            marker="*",
            s=160,
            edgecolors="k",
            label=f"Query Sample {query_point}",
        )

    plt.xlabel("Petal Length")
    plt.ylabel("Petal Width")
    plt.title("Training Observation Distribution in Euclidean Feature Space")
    plt.legend(loc="upper left")
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_loss_curve(costs: List[float], save_path: Optional[str] = None) -> None:
    """Plot cumulative squared error convergence across optimization iterations."""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 4))
    plt.plot(costs, color="teal", linewidth=1.8)
    plt.xlabel("Optimization Steps (x100)")
    plt.ylabel("Cumulative Squared Loss")
    plt.title("Convergence of Stochastic Gradient Descent")
    plt.grid(True, linestyle="--", alpha=0.5)
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_decision_boundary(
    model,
    dataset: List[List[float]],
    query_point: Optional[List[float]] = None,
    save_path: Optional[str] = None,
) -> None:
    """Plot learned dense 2D classification decision boundary."""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    grid_x = np.linspace(0, 6, 50)
    grid_y = np.linspace(0, 3, 50)

    for x in grid_x:
        for y in grid_y:
            pred = model.forward(x, y)
            c = "red" if pred >= 0.5 else "blue"
            plt.scatter(x, y, color=c, alpha=0.12, s=25)

    for point in dataset:
        c = "red" if point[2] == 1 else "blue"
        plt.scatter(point[0], point[1], color=c, s=80, edgecolors="k", linewidth=1.2)

    if query_point is not None:
        pred_q = model.forward(query_point[0], query_point[1])
        c_q = "red" if pred_q >= 0.5 else "blue"
        plt.scatter(
            query_point[0],
            query_point[1],
            color=c_q,
            edgecolors="black",
            s=180,
            marker="*",
            label=f"Query {query_point} (Pred: {pred_q:.3f})",
        )

    plt.xlabel("Petal Length")
    plt.ylabel("Petal Width")
    plt.title("Learned Decision Boundary in 2D Feature Space")
    plt.legend(loc="upper left")
    plt.grid(True, linestyle="--", alpha=0.5)
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
