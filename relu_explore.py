import os
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Reproducibility
SEED = 42
rng = np.random.default_rng(SEED)

# ReLU activation and derivative

def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, z)

def drelu(z: np.ndarray) -> np.ndarray:
    return (z > 0.0).astype(float)

# Target polynomial: y = a*x^3 + b*x^2 + c*x + d
POLY_COEFFS = {"a": 0.4, "b": -0.8, "c": 1.5, "d": 0.5}

def generate_data(num_points: int = 200, x_min: float = -2.5, x_max: float = 2.5) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(x_min, x_max, num_points)
    a, b, c, d = POLY_COEFFS.values()
    y = a * x**3 + b * x**2 + c * x + d
    return x, y

class SingleHiddenNeuronReLU:
    """y_hat = v * ReLU(w * x + b) + c"""

    def __init__(self):
        self.w = rng.normal(scale=0.5)
        self.b = rng.normal(scale=0.5)
        self.v = rng.normal(scale=0.5)
        self.c = rng.normal(scale=0.5)

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, dict]:
        z = self.w * x + self.b
        a = relu(z)
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
        dL_dz = dL_da * drelu(z)
        dL_dw = np.sum(dL_dz * x)
        dL_db = np.sum(dL_dz)
        return {"w": dL_dw, "b": dL_db, "v": dL_dv, "c": dL_dc}

    def step(self, grads: dict, lr: float) -> None:
        self.w -= lr * grads["w"]
        self.b -= lr * grads["b"]
        self.v -= lr * grads["v"]
        self.c -= lr * grads["c"]


def train_relu(x: np.ndarray, y: np.ndarray, lr: float = 2e-3, epochs: int = 1200, snapshot_every: int = 20):
    model = SingleHiddenNeuronReLU()
    losses: list[float] = []
    best_loss = math.inf
    best_params = None
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


def fit_vc_least_squares(a_feat: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    # Solve min_{v,c} || v*a + c - y ||^2
    F = np.stack([a_feat, np.ones_like(a_feat)], axis=1)
    theta, *_ = np.linalg.lstsq(F, y, rcond=None)
    v_opt, c_opt = float(theta[0]), float(theta[1])
    return v_opt, c_opt


def loss_for_wb(x: np.ndarray, y: np.ndarray, w: float, b: float) -> float:
    a_feat = relu(w * x + b)
    v_opt, c_opt = fit_vc_least_squares(a_feat, y)
    y_pred = v_opt * a_feat + c_opt
    return float(np.mean((y_pred - y) ** 2))


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def plot_activation_and_derivative(out_dir: str) -> None:
    z = np.linspace(-5, 5, 1000)
    f = relu(z)
    df = drelu(z)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(z, f, label="ReLU(z)", color="tab:blue")
    ax.plot(z, df, label="ReLU'(z)", color="tab:orange")
    ax.set_title("ReLU and derivative")
    ax.set_xlabel("z")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "relu_activation_derivative.png"), dpi=150)
    plt.close(fig)


def plot_static_and_gif(x: np.ndarray, y: np.ndarray, out_dir: str) -> tuple[list[tuple[float,float,float,float]], np.ndarray, np.ndarray]:
    model, best_loss, losses, snapshots = train_relu(x, y, lr=2e-3, epochs=1200, snapshot_every=20)
    x_dense = np.linspace(np.min(x), np.max(x), 600)
    a, b, c, d = POLY_COEFFS.values()
    y_true = a * x_dense**3 + b * x_dense**2 + c * x_dense + d
    yhat, _ = model.forward(x_dense)

    # Static plot
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(x, y, s=12, color="black", alpha=0.5, label="data (polynomial)")
    ax.plot(x_dense, y_true, "k--", linewidth=2, label="true polynomial")
    ax.plot(x_dense, yhat, color="tab:orange", linewidth=2.5, label=f"ReLU (MSE={best_loss:.3f})")

    # Annotation box
    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min, y_max = float(np.min(yhat)), float(np.max(yhat))
    text = (
        f"x∈[{x_min:.2f},{x_max:.2f}]\n"
        f"w={model.w:.4f}\n"
        f"b={model.b:.4f}\n"
        f"v={model.v:.4f}\n"
        f"c={model.c:.4f}\n"
        f"ŷ∈[{y_min:.2f},{y_max:.2f}]"
    )
    ax.text(0.02, 0.98, text, transform=ax.transAxes, va='top', ha='left', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='tab:orange', linewidth=1.5))

    fig.text(0.5, 0.995, r"$\hat{y}(x) = v\,\mathrm{ReLU}(w x + b) + c$", ha='center', va='top', fontsize=11)
    ax.set_title("Single-Neuron ReLU Fit")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(out_dir, "relu_single_fit.png"), dpi=150)
    plt.close(fig)

    # GIF
    fig2, ax2 = plt.subplots(figsize=(9, 6))
    ax2.scatter(x, y, s=12, color="black", alpha=0.4, label="data (polynomial)")
    ax2.plot(x_dense, y_true, "k--", linewidth=1.8, alpha=0.9, label="true polynomial")
    (line_pred,) = ax2.plot([], [], color="tab:orange", linewidth=2.0, label="ReLU (training)")
    ann = ax2.text(0.02, 0.98, "", transform=ax2.transAxes, va='top', ha='left', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='tab:orange', linewidth=1.5))
    fig2.text(0.5, 0.995, r"$\hat{y}(x) = v\,\mathrm{ReLU}(w x + b) + c$", ha='center', va='top', fontsize=11)
    ax2.set_title("Convergence (ReLU)")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.legend()
    ax2.grid(True, alpha=0.25)

    y_all_min = float(np.min(y_true))
    y_all_max = float(np.max(y_true))

    def init_anim():
        line_pred.set_data([], [])
        ann.set_text("")
        return (line_pred, ann)

    def update_anim(idx: int):
        w, b, v, c = snapshots[idx]
        yhat_s = v * relu(w * x_dense + b) + c
        line_pred.set_data(x_dense, yhat_s)
        ann.set_text(
            "\n".join([
                f"x∈[{float(x_dense[0]):.2f},{float(x_dense[-1]):.2f}]",
                f"w={w:.4f}", f"b={b:.4f}", f"v={v:.4f}", f"c={c:.4f}",
                f"ŷ∈[{float(np.min(yhat_s)):.2f},{float(np.max(yhat_s)):.2f}]",
            ])
        )
        ax2.set_xlim(float(x_dense[0]), float(x_dense[-1]))
        ax2.set_ylim(min(y_all_min, float(np.min(yhat_s))) - 0.5, max(y_all_max, float(np.max(yhat_s))) + 0.5)
        return (line_pred, ann)

    anim = animation.FuncAnimation(fig2, update_anim, init_func=init_anim, frames=len(snapshots), interval=120, blit=True, repeat=False)
    fig2.tight_layout(rect=[0, 0, 1, 0.95])
    anim.save(os.path.join(out_dir, "relu_single_convergence.gif"), writer="pillow", fps=12)
    plt.close(fig2)

    return snapshots, x_dense, y_true


def plot_param_trajectories(snapshots: list[tuple[float,float,float,float]], out_dir: str) -> None:
    arr = np.array(snapshots)
    w, b, v, c = arr[:,0], arr[:,1], arr[:,2], arr[:,3]
    steps = np.arange(len(snapshots))
    fig, ax = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
    ax[0,0].plot(steps, w, label='w', color='tab:blue'); ax[0,0].set_title('w'); ax[0,0].grid(True, alpha=0.25)
    ax[0,1].plot(steps, b, label='b', color='tab:orange'); ax[0,1].set_title('b'); ax[0,1].grid(True, alpha=0.25)
    ax[1,0].plot(steps, v, label='v', color='tab:green'); ax[1,0].set_title('v'); ax[1,0].grid(True, alpha=0.25)
    ax[1,1].plot(steps, c, label='c', color='tab:red'); ax[1,1].set_title('c'); ax[1,1].grid(True, alpha=0.25)
    for axi in ax.flat:
        axi.set_xlabel('snapshot idx')
    fig.suptitle('Parameter trajectories (snapshots)')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(out_dir, "relu_param_trajectories.png"), dpi=150)
    plt.close(fig)


def plot_wb_loss_heatmap(x: np.ndarray, y: np.ndarray, snapshots: list[tuple[float,float,float,float]], out_dir: str) -> None:
    w_vals = np.linspace(-4, 4, 151)
    b_vals = np.linspace(-4, 4, 151)
    heat = np.empty((len(w_vals), len(b_vals)))
    for i, w in enumerate(w_vals):
        for j, b in enumerate(b_vals):
            heat[i, j] = loss_for_wb(x, y, float(w), float(b))

    # Prepare trajectory for overlay
    traj_w = [p[0] for p in snapshots]
    traj_b = [p[1] for p in snapshots]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(
        heat.T, origin='lower', aspect='auto',
        extent=[w_vals[0], w_vals[-1], b_vals[0], b_vals[-1]], cmap='viridis'
    )
    cbar = fig.colorbar(im, ax=ax, label='MSE (v,c optimized)')
    ax.contour(w_vals, b_vals, heat.T, colors='white', alpha=0.3, linewidths=0.5)
    ax.plot(traj_w, traj_b, color='tab:orange', linewidth=2.0, marker='o', markersize=2, alpha=0.9, label='training trajectory')
    ax.set_xlabel('w')
    ax.set_ylabel('b')
    ax.set_title('Loss landscape over (w,b) with optimal (v,c)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "relu_wb_loss_heatmap.png"), dpi=150)
    plt.close(fig)


def main() -> None:
    out_dir = os.path.join("outputs", "relu")
    ensure_dir(out_dir)

    # Activation shape and derivative
    plot_activation_and_derivative(out_dir)

    # Data and core visuals
    x, y = generate_data()
    snapshots, x_dense, y_true = plot_static_and_gif(x, y, out_dir)

    # Parameter trajectories
    plot_param_trajectories(snapshots, out_dir)

    # Loss heatmap (w,b)
    plot_wb_loss_heatmap(x, y, snapshots, out_dir)


if __name__ == "__main__":
    main() 