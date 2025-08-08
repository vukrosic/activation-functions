import os
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Reproducibility
SEED = 42
rng = np.random.default_rng(SEED)

# Activations and derivatives

def tanh(z: np.ndarray) -> np.ndarray:
    return np.tanh(z)

def dtanh(z: np.ndarray) -> np.ndarray:
    t = np.tanh(z)
    return 1.0 - t * t

def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, z)

def drelu(z: np.ndarray) -> np.ndarray:
    return (z > 0.0).astype(float)

ACTIVATIONS = {
    "tanh": (tanh, dtanh, r"$\hat{y}(x) = v\,\tanh(w x + b) + c$"),
    "relu": (relu, drelu, r"$\hat{y}(x) = v\,\mathrm{ReLU}(w x + b) + c$"),
}

# Target polynomial: y = a*x^3 + b*x^2 + c*x + d
POLY_COEFFS = {"a": 0.4, "b": -0.8, "c": 1.5, "d": 0.5}

def generate_data(num_points: int = 200, x_min: float = -2.5, x_max: float = 2.5) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(x_min, x_max, num_points)
    a, b, c, d = POLY_COEFFS.values()
    y = a * x**3 + b * x**2 + c * x + d
    return x, y

class SingleHiddenNeuron:
    """y_hat = v * act(w * x + b) + c"""

    def __init__(self, activation_name: str):
        self.activation_name = activation_name
        self.act, self.dact, _ = ACTIVATIONS[activation_name]
        self.w = rng.normal(scale=0.5)
        self.b = rng.normal(scale=0.5)
        self.v = rng.normal(scale=0.5)
        self.c = rng.normal(scale=0.5)

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, dict]:
        z = self.w * x + self.b
        a = self.act(z)
        y_hat = self.v * a + self.c
        return y_hat, {"x": x, "z": z, "a": a, "y_hat": y_hat}

    @staticmethod
    def loss(y_hat: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean((y_hat - y) ** 2))

    def backward(self, cache: dict, y: np.ndarray) -> dict:
        x, z, a, y_hat = cache["x"], cache["z"], cache["a"], cache["y_hat"]
        n = y.shape[0]
        dL_dy = 2.0 * (y_hat - y) / n
        dL_dv = np.sum(dL_dy * a)
        dL_dc = np.sum(dL_dy)
        dL_da = dL_dy * self.v
        dL_dz = dL_da * self.dact(z)
        dL_dw = np.sum(dL_dz * x)
        dL_db = np.sum(dL_dz)
        return {"w": dL_dw, "b": dL_db, "v": dL_dv, "c": dL_dc}

    def step(self, grads: dict, lr: float) -> None:
        self.w -= lr * grads["w"]
        self.b -= lr * grads["b"]
        self.v -= lr * grads["v"]
        self.c -= lr * grads["c"]


def train_single(
    activation_name: str,
    x: np.ndarray,
    y: np.ndarray,
    lr: float = 2e-3,
    epochs: int = 1200,
    snapshot_every: int = 20,
) -> tuple[SingleHiddenNeuron, float, list[float], list[tuple[float, float, float, float]]]:
    model = SingleHiddenNeuron(activation_name)
    losses: list[float] = []
    best_loss = math.inf
    best_params: tuple[float, float, float, float] | None = None
    snapshots: list[tuple[float, float, float, float]] = []

    for epoch in range(epochs):
        y_hat, cache = model.forward(x)
        loss = model.loss(y_hat, y)
        losses.append(loss)
        if loss < best_loss:
            best_loss = loss
            best_params = (model.w, model.b, model.v, model.c)
        if epoch % snapshot_every == 0:
            snapshots.append((model.w, model.b, model.v, model.c))
        grads = model.backward(cache, y)
        model.step(grads, lr)
        if epoch % 400 == 399 and len(losses) > 50:
            recent = np.mean(losses[-50:])
            prev = np.mean(losses[-100:-50]) if len(losses) >= 100 else recent
            if recent > prev:
                lr = max(lr * 0.5, 1e-5)

    snapshots.append((model.w, model.b, model.v, model.c))
    if best_params is not None:
        model.w, model.b, model.v, model.c = best_params
    return model, best_loss, losses, snapshots


def forward_with_params_single(x_dense: np.ndarray, params: tuple[float, float, float, float], act) -> np.ndarray:
    w, b, v, c = params
    z = w * x_dense + b
    a = act(z)
    yhat = v * a + c
    return yhat


def annotate_box(ax, x_range: tuple[float, float], y_range: tuple[float, float], params: tuple[float, float, float, float], edge_color: str) -> None:
    x_min, x_max = x_range
    y_min, y_max = y_range
    w, b, v, c = params
    text = (
        f"x∈[{x_min:.2f},{x_max:.2f}]\n"
        f"w={w:.4f}\n"
        f"b={b:.4f}\n"
        f"v={v:.4f}\n"
        f"c={c:.4f}\n"
        f"ŷ∈[{y_min:.2f},{y_max:.2f}]"
    )
    ax.text(
        0.02, 0.98, text,
        transform=ax.transAxes,
        va='top', ha='left', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor=edge_color, linewidth=1.5)
    )


def run_case(name: str, color: str, x: np.ndarray, y: np.ndarray, x_dense: np.ndarray) -> None:
    act, _, fn_text = ACTIVATIONS[name]
    model, best_loss, losses, snapshots = train_single(name, x, y, lr=2e-3, epochs=1200, snapshot_every=20)
    print(f"{name:>5s} | best MSE: {best_loss:.6f} | final MSE: {losses[-1]:.6f}")

    # Static plot
    yhat_best, _ = model.forward(x_dense)
    fig, ax = plt.subplots(figsize=(9, 6))
    a, b, c, d = POLY_COEFFS.values()
    y_true_dense = a * x_dense**3 + b * x_dense**2 + c * x_dense + d
    ax.scatter(x, y, s=12, color="black", alpha=0.5, label="data (polynomial)")
    ax.plot(x_dense, y_true_dense, "k--", linewidth=2, label="true polynomial")
    ax.plot(x_dense, yhat_best, color=color, linewidth=2.5, alpha=0.95, label=f"{name} (MSE={best_loss:.3f})")

    annotate_box(
        ax,
        (float(np.min(x)), float(np.max(x))),
        (float(np.min(yhat_best)), float(np.max(yhat_best))),
        (model.w, model.b, model.v, model.c),
        edge_color=color,
    )

    fig.text(0.5, 0.995, fn_text, ha='center', va='top', fontsize=11, color='black')
    ax.set_title(f"Single-Neuron Fit: {name}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    ax.grid(True, alpha=0.25)

    os.makedirs("outputs", exist_ok=True)
    out_img = os.path.join("outputs", f"{name}_single_fit.png")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_img, dpi=150)
    plt.close(fig)
    print(f"Saved plot to: {out_img}")

    # Animated convergence
    fig2, ax2 = plt.subplots(figsize=(9, 6))
    ax2.scatter(x, y, s=12, color="black", alpha=0.4, label="data (polynomial)")
    ax2.plot(x_dense, y_true_dense, "k--", linewidth=1.8, alpha=0.9, label="true polynomial")
    (line_pred,) = ax2.plot([], [], color=color, linewidth=2.0, label=f"{name} (training)")
    ann = ax2.text(
        0.02, 0.98, "",
        transform=ax2.transAxes,
        va='top', ha='left', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor=color, linewidth=1.5)
    )
    fig2.text(0.5, 0.995, fn_text, ha='center', va='top', fontsize=11, color='black')
    ax2.set_title(f"Convergence ({name})")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.legend()
    ax2.grid(True, alpha=0.25)

    y_all_min = float(np.min(y_true_dense))
    y_all_max = float(np.max(y_true_dense))

    def init_anim():
        line_pred.set_data([], [])
        ann.set_text("")
        return (line_pred, ann)

    def update_anim(frame_idx: int):
        params = snapshots[frame_idx]
        yhat = forward_with_params_single(x_dense, params, act)
        line_pred.set_data(x_dense, yhat)
        ann.set_text(
            "\n".join(
                [
                    f"x∈[{float(x_dense[0]):.2f},{float(x_dense[-1]):.2f}]",
                    f"w={params[0]:.4f}",
                    f"b={params[1]:.4f}",
                    f"v={params[2]:.4f}",
                    f"c={params[3]:.4f}",
                    f"ŷ∈[{float(np.min(yhat)):.2f},{float(np.max(yhat)):.2f}]",
                ]
            )
        )
        ax2.set_xlim(float(x_dense[0]), float(x_dense[-1]))
        ax2.set_ylim(min(y_all_min, float(np.min(yhat))) - 0.5, max(y_all_max, float(np.max(yhat))) + 0.5)
        return (line_pred, ann)

    anim = animation.FuncAnimation(
        fig2,
        update_anim,
        init_func=init_anim,
        frames=len(snapshots),
        interval=120,
        blit=True,
        repeat=False,
    )

    out_gif = os.path.join("outputs", f"{name}_single_convergence.gif")
    fig2.tight_layout(rect=[0, 0, 1, 0.95])
    anim.save(out_gif, writer="pillow", fps=12)
    plt.close(fig2)
    print(f"Saved animation to: {out_gif}")


def main() -> None:
    x, y = generate_data()
    x_dense = np.linspace(np.min(x), np.max(x), 600)
    run_case("tanh", color="tab:blue", x=x, y=y, x_dense=x_dense)
    run_case("relu", color="tab:orange", x=x, y=y, x_dense=x_dense)


if __name__ == "__main__":
    main() 