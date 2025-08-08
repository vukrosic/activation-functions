import os
import math
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import Callable

# Reproducibility
SEED = 42
rng = np.random.default_rng(SEED)

# Activations and derivatives

def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))

def dsigmoid(z: np.ndarray) -> np.ndarray:
    s = sigmoid(z)
    return s * (1.0 - s)

def tanh(z: np.ndarray) -> np.ndarray:
    return np.tanh(z)

def dtanh(z: np.ndarray) -> np.ndarray:
    t = np.tanh(z)
    return 1.0 - t * t

def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, z)

def drelu(z: np.ndarray) -> np.ndarray:
    return (z > 0.0).astype(float)

def leaky_relu(z: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    return np.where(z > 0.0, z, alpha * z)

def dleaky_relu(z: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    dz = np.ones_like(z)
    dz[z < 0.0] = alpha
    return dz

def sine(z: np.ndarray) -> np.ndarray:
    return np.sin(z)

def dsine(z: np.ndarray) -> np.ndarray:
    return np.cos(z)

def identity(z: np.ndarray) -> np.ndarray:
    return z

def didentity(z: np.ndarray) -> np.ndarray:
    return np.ones_like(z)

ACTIVATIONS: dict[str, tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray], str]] = {
    "tanh": (tanh, dtanh, r"$\hat{y}(x) = v\,\tanh(w x + b) + c$"),
    "relu": (relu, drelu, r"$\hat{y}(x) = v\,\mathrm{ReLU}(w x + b) + c$"),
    "sigmoid": (sigmoid, dsigmoid, r"$\hat{y}(x) = v\,\sigma(w x + b) + c$"),
    "leaky_relu": (leaky_relu, dleaky_relu, r"$\hat{y}(x) = v\,\mathrm{LReLU}_{\alpha}(w x + b) + c$"),
    "sine": (sine, dsine, r"$\hat{y}(x) = v\,\sin(w x + b) + c$"),
    "identity": (identity, didentity, r"$\hat{y}(x) = v\,(w x + b) + c$"),
}

# Polynomials to fit
POLYS: list[dict] = [
    {"id": "linear_pos",  "coeffs": {"a": 0.0, "b": 0.0, "c": 1.2, "d": 0.0}},
    {"id": "quadratic_u", "coeffs": {"a": 0.0, "b": 0.6, "c": 0.0, "d": 0.2}},
    {"id": "cubic_s",    "coeffs": {"a": 0.4, "b": -0.8, "c": 1.5, "d": 0.5}},
    {"id": "cubic_neg",  "coeffs": {"a": -0.5, "b": 0.3, "c": -0.8, "d": 0.0}},
    {"id": "quartic_w",  "coeffs": {"a": 0.0, "b": 0.2, "c": 0.0, "d": 0.0}},  # actually y = 0.2 x^2 + 0; we'll square later
]


def poly_eval(x: np.ndarray, coeffs: dict) -> np.ndarray:
    a, b, c, d = coeffs.get("a", 0.0), coeffs.get("b", 0.0), coeffs.get("c", 0.0), coeffs.get("d", 0.0)
    # For quartic_w, interpret as y = (b * x^2 + d)^2 to create W shape
    if coeffs is POLYS[4]["coeffs"]:
        return (b * x**2 + d) ** 2
    return a * x**3 + b * x**2 + c * x + d


def generate_data(coeffs: dict, num_points: int = 200, x_min: float = -2.5, x_max: float = 2.5) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(x_min, x_max, num_points)
    y = poly_eval(x, coeffs)
    return x, y


class SingleHiddenNeuron:
    def __init__(self, act_name: str):
        self.act_name = act_name
        self.act, self.dact, self.fn_text = ACTIVATIONS[act_name]
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


def train_single(act_name: str, x: np.ndarray, y: np.ndarray, lr: float = 2e-3, epochs: int = 1000, snapshot_every: int = 25):
    model = SingleHiddenNeuron(act_name)
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
        if epoch % 250 == 249 and len(losses) > 50:
            recent = float(np.mean(losses[-50:]))
            prev = float(np.mean(losses[-100:-50])) if len(losses) >= 100 else recent
            if recent > prev:
                lr = max(lr * 0.5, 1e-5)
    snapshots.append((model.w, model.b, model.v, model.c))
    if best_params is not None:
        model.w, model.b, model.v, model.c = best_params
    return model, best_loss, losses, snapshots


def annotate_box(ax, x_range, y_range, params, edge_color: str) -> None:
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
    ax.text(0.02, 0.98, text, transform=ax.transAxes, va='top', ha='left', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor=edge_color, linewidth=1.5))


def save_static_and_gif(poly_id: str, coeffs: dict, act_name: str, out_root: str) -> dict:
    out_dir = os.path.join(out_root, poly_id, act_name)
    os.makedirs(out_dir, exist_ok=True)

    x, y = generate_data(coeffs)
    x_dense = np.linspace(np.min(x), np.max(x), 600)
    y_true = poly_eval(x_dense, coeffs)

    model, best_loss, losses, snapshots = train_single(act_name, x, y)

    # Static
    yhat_best, _ = model.forward(x_dense)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(x, y, s=12, color="black", alpha=0.5, label="data (polynomial)")
    ax.plot(x_dense, y_true, "k--", linewidth=2, label="true polynomial")
    color = {
        "tanh": "tab:blue",
        "relu": "tab:orange",
        "sigmoid": "tab:green",
        "leaky_relu": "tab:red",
        "sine": "tab:purple",
        "identity": "tab:brown",
    }.get(act_name, "tab:gray")
    ax.plot(x_dense, yhat_best, color=color, linewidth=2.5, alpha=0.95, label=f"{act_name} (MSE={best_loss:.3f})")

    annotate_box(ax, (float(np.min(x)), float(np.max(x))), (float(np.min(yhat_best)), float(np.max(yhat_best))), (model.w, model.b, model.v, model.c), color)
    _, _, fn_text = ACTIVATIONS[act_name]
    fig.text(0.5, 0.995, fn_text, ha='center', va='top', fontsize=11)
    ax.set_title(f"Single-Neuron Fit: {act_name} on {poly_id}")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.legend(); ax.grid(True, alpha=0.25)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    static_path = os.path.join(out_dir, f"{poly_id}_{act_name}_fit.png")
    fig.savefig(static_path, dpi=150)
    plt.close(fig)

    # GIF
    fig2, ax2 = plt.subplots(figsize=(9, 6))
    ax2.scatter(x, y, s=12, color="black", alpha=0.4, label="data (polynomial)")
    ax2.plot(x_dense, y_true, "k--", linewidth=1.8, alpha=0.9, label="true polynomial")
    (line_pred,) = ax2.plot([], [], color=color, linewidth=2.0, label=f"{act_name} (training)")
    ann = ax2.text(0.02, 0.98, "", transform=ax2.transAxes, va='top', ha='left', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor=color, linewidth=1.5))
    fig2.text(0.5, 0.995, fn_text, ha='center', va='top', fontsize=11)
    ax2.set_title(f"Convergence: {act_name} on {poly_id}")
    ax2.set_xlabel("x"); ax2.set_ylabel("y"); ax2.legend(); ax2.grid(True, alpha=0.25)

    y_all_min = float(np.min(y_true)); y_all_max = float(np.max(y_true))

    def init_anim():
        line_pred.set_data([], []); ann.set_text(""); return (line_pred, ann)

    def update_anim(idx: int):
        w, b, v, c = snapshots[idx]
        yhat_s = v * ACTIVATIONS[act_name][0](w * x_dense + b) + c
        line_pred.set_data(x_dense, yhat_s)
        ann.set_text("\n".join([
            f"x∈[{float(x_dense[0]):.2f},{float(x_dense[-1]):.2f}]",
            f"w={w:.4f}", f"b={b:.4f}", f"v={v:.4f}", f"c={c:.4f}",
            f"ŷ∈[{float(np.min(yhat_s)):.2f},{float(np.max(yhat_s)):.2f}]",
        ]))
        ax2.set_xlim(float(x_dense[0]), float(x_dense[-1]))
        ax2.set_ylim(min(y_all_min, float(np.min(yhat_s))) - 0.5, max(y_all_max, float(np.max(yhat_s))) + 0.5)
        return (line_pred, ann)

    anim = animation.FuncAnimation(fig2, update_anim, init_func=init_anim, frames=len(snapshots), interval=120, blit=True, repeat=False)
    gif_path = os.path.join(out_dir, f"{poly_id}_{act_name}_convergence.gif")
    fig2.tight_layout(rect=[0, 0, 1, 0.95])
    anim.save(gif_path, writer="pillow", fps=12)
    plt.close(fig2)

    return {
        "poly_id": poly_id,
        "activation": act_name,
        "best_mse": float(best_loss),
        "final_mse": float(losses[-1]),
        "params": {"w": model.w, "b": model.b, "v": model.v, "c": model.c},
        "static": os.path.relpath(static_path),
        "gif": os.path.relpath(gif_path),
    }


def run_benchmark(out_root: str = os.path.join("outputs", "benchmark")) -> list[dict]:
    os.makedirs(out_root, exist_ok=True)
    results: list[dict] = []
    for poly in POLYS:
        poly_id = poly["id"]
        coeffs = poly["coeffs"]
        print(f"Polynomial: {poly_id}")
        for act_name in ACTIVATIONS.keys():
            print(f"  - Training activation: {act_name}")
            res = save_static_and_gif(poly_id, coeffs, act_name, out_root)
            results.append(res)
    # Save json and csv
    with open(os.path.join(out_root, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    # CSV
    import csv
    with open(os.path.join(out_root, "results.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["poly_id", "activation", "best_mse", "final_mse", "static", "gif"])
        for r in results:
            writer.writerow([r["poly_id"], r["activation"], r["best_mse"], r["final_mse"], r["static"], r["gif"]])
    return results


def write_report(results: list[dict], out_root: str = os.path.join("outputs", "benchmark")) -> None:
    # Group by polynomial
    by_poly: dict[str, list[dict]] = {}
    for r in results:
        by_poly.setdefault(r["poly_id"], []).append(r)

    lines: list[str] = []
    lines.append("# Single-Neuron Activation Benchmark")
    lines.append("")
    lines.append("This report compares 6 activations on multiple target polynomials using a single hidden neuron and a linear output.")
    lines.append("")
    for poly_id, group in by_poly.items():
        lines.append(f"## {poly_id}")
        lines.append("")
        # Table header
        lines.append("| activation | best MSE | preview |")
        lines.append("|--|--:|--|")
        group_sorted = sorted(group, key=lambda r: r["best_mse"])  # best first
        for r in group_sorted:
            img = r["static"].replace(" ", "%20")
            lines.append(f"| {r['activation']} | {r['best_mse']:.4f} | ![]({img}) |")
        best = group_sorted[0]
        lines.append("")
        lines.append(f"Best: **{best['activation']}** with MSE {best['best_mse']:.4f}")
        lines.append("")
    report_path = os.path.join(out_root, "REPORT.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote report to {report_path}")


def main() -> None:
    results = run_benchmark()
    write_report(results)


if __name__ == "__main__":
    main() 