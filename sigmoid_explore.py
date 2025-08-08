import os
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec

# Reproducibility
SEED = 42
rng = np.random.default_rng(SEED)

# Sigmoid

def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))

def dsigmoid_from_s(s: np.ndarray) -> np.ndarray:
    return s * (1.0 - s)

# Target polynomial: y = a*x^3 + b*x^2 + c*x + d
POLY_COEFFS = {"a": 0.4, "b": -0.8, "c": 1.5, "d": 0.5}

def generate_data(num_points: int = 200, x_min: float = -2.5, x_max: float = 2.5) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(x_min, x_max, num_points)
    a, b, c, d = POLY_COEFFS.values()
    y = a * x**3 + b * x**2 + c * x + d
    return x, y

class SingleParamSigmoid:
    """y_hat = sigmoid(w * x) with a single parameter w"""

    def __init__(self):
        self.w = float(rng.normal(scale=0.5))

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, dict]:
        z = self.w * x
        s = sigmoid(z)
        return s, {"x": x, "z": z, "s": s}

    @staticmethod
    def loss(y_hat: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean((y_hat - y) ** 2))

    def backward(self, cache: dict, y: np.ndarray) -> dict:
        x, s = cache["x"], cache["s"]
        n = y.shape[0]
        dL_dy = 2.0 * (s - y) / n
        dL_dw = np.sum(dL_dy * dsigmoid_from_s(s) * x)
        return {"w": dL_dw}

    def step(self, grads: dict, lr: float) -> None:
        self.w -= lr * grads["w"]


def train_sigmoid(x: np.ndarray, y: np.ndarray, lr: float = 5e-3, epochs: int = 800, snapshot_every: int = 10):
    model = SingleParamSigmoid()
    losses: list[float] = []
    best_loss = math.inf
    best_w = None
    snapshots_w: list[float] = []

    for epoch in range(epochs):
        y_hat, cache = model.forward(x)
        loss = model.loss(y_hat, y)
        losses.append(loss)
        if loss < best_loss:
            best_loss = loss
            best_w = model.w
        if epoch % snapshot_every == 0:
            snapshots_w.append(model.w)
        grads = self_grads = model.backward(cache, y)
        model.step(self_grads, lr)
        if epoch % 200 == 199 and len(losses) > 50:
            recent = float(np.mean(losses[-50:]))
            prev = float(np.mean(losses[-100:-50])) if len(losses) >= 100 else recent
            if recent > prev:
                lr = max(lr * 0.5, 1e-5)

    snapshots_w.append(model.w)
    if best_w is not None:
        model.w = best_w
    return model, best_loss, losses, snapshots_w


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def main() -> None:
    out_dir = os.path.join("outputs", "sigmoid")
    ensure_dir(out_dir)

    x, y = generate_data()
    x_dense = np.linspace(np.min(x), np.max(x), 600)
    a, b, c, d = POLY_COEFFS.values()
    y_true = a * x_dense**3 + b * x_dense**2 + c * x_dense + d

    model, best_loss, losses, snapshots_w = train_sigmoid(x, y, lr=5e-3, epochs=800, snapshot_every=10)

    # Static final fit
    yhat_best, _ = model.forward(x_dense)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(x, y, s=12, color="black", alpha=0.5, label="data (polynomial)")
    ax.plot(x_dense, y_true, "k--", linewidth=2, label="true polynomial")
    ax.plot(x_dense, yhat_best, color="tab:purple", linewidth=2.5, alpha=0.95, label=f"sigmoid(w x) (MSE={best_loss:.3f})")
    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min, y_max = float(np.min(yhat_best)), float(np.max(yhat_best))
    text = (
        f"x∈[{x_min:.2f},{x_max:.2f}]\n"
        f"w={model.w:.4f}\n"
        f"ŷ∈[{y_min:.2f},{y_max:.2f}]"
    )
    ax.text(0.02, 0.98, text, transform=ax.transAxes, va='top', ha='left', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='tab:purple', linewidth=1.5))
    fig.text(0.5, 0.995, r"$\hat{y}(x) = \sigma(w x)$", ha='center', va='top', fontsize=11)
    ax.set_title("Single-Parameter Fit: y=σ(w x)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(out_dir, "sigmoid_fit.png"), dpi=150)
    plt.close(fig)

    # GIF with two panels: left fit, right trajectory of w
    fig2 = plt.figure(figsize=(10, 6))
    gs = GridSpec(1, 2, width_ratios=[3, 2], figure=fig2)
    ax_fit = fig2.add_subplot(gs[0, 0])
    ax_w = fig2.add_subplot(gs[0, 1])

    ax_fit.scatter(x, y, s=12, color="black", alpha=0.4, label="data (polynomial)")
    ax_fit.plot(x_dense, y_true, "k--", linewidth=1.8, alpha=0.9, label="true polynomial")
    (line_pred,) = ax_fit.plot([], [], color="tab:purple", linewidth=2.0, label="σ(w x)")

    # W trajectory axis
    steps = np.arange(len(snapshots_w))
    ax_w.plot(steps, snapshots_w, color="tab:purple", linewidth=1.8)
    (pt_w,) = ax_w.plot([], [], marker='o', color="tab:red", markersize=6)
    ax_w.set_title("w trajectory (snapshots)")
    ax_w.set_xlabel("snapshot idx")
    ax_w.set_ylabel("w")
    ax_w.grid(True, alpha=0.25)

    # Dynamic annotation
    ann = ax_fit.text(0.02, 0.98, "", transform=ax_fit.transAxes, va='top', ha='left', fontsize=9,
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='tab:purple', linewidth=1.5))

    fig2.text(0.5, 0.995, r"$\hat{y}(x) = \sigma(w x)$", ha='center', va='top', fontsize=11)
    ax_fit.set_title("Convergence (σ(w x))")
    ax_fit.set_xlabel("x")
    ax_fit.set_ylabel("y")
    ax_fit.legend()
    ax_fit.grid(True, alpha=0.25)

    y_all_min = float(np.min(y_true))
    y_all_max = float(np.max(y_true))

    def init_anim():
        line_pred.set_data([], [])
        pt_w.set_data([], [])
        ann.set_text("")
        return (line_pred, pt_w, ann)

    def update_anim(idx: int):
        w = float(snapshots_w[idx])
        yhat = sigmoid(w * x_dense)
        line_pred.set_data(x_dense, yhat)
        pt_w.set_data([steps[idx]], [w])
        ann.set_text("\n".join([
            f"x∈[{float(x_dense[0]):.2f},{float(x_dense[-1]):.2f}]",
            f"w={w:.4f}",
            f"ŷ∈[{float(np.min(yhat)):.2f},{float(np.max(yhat)):.2f}]",
        ]))
        ax_fit.set_xlim(float(x_dense[0]), float(x_dense[-1]))
        ax_fit.set_ylim(min(y_all_min, float(np.min(yhat))) - 0.5, max(y_all_max, float(np.max(yhat))) + 0.5)
        return (line_pred, pt_w, ann)

    anim = animation.FuncAnimation(fig2, update_anim, init_func=init_anim, frames=len(snapshots_w), interval=120, blit=True, repeat=False)
    fig2.tight_layout(rect=[0, 0, 1, 0.95])
    anim.save(os.path.join(out_dir, "sigmoid_convergence.gif"), writer="pillow", fps=12)
    plt.close(fig2)


if __name__ == "__main__":
    main() 