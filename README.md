# activation-functions

A minimal playground to study activation functions with single-neuron and tiny-MLP models. Includes static plots, convergence GIFs, and a benchmark across activations and target polynomials.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Scripts

- `fit_single_neuron.py`: Train a single hidden neuron (choose activation) to fit a polynomial. Produces static plots and optional GIFs.
- `relu_explore.py`: ReLU-only exploration (activation/derivative, static fit, convergence GIF, parameter trajectories, (w,b) loss heatmap).
- `sigmoid_explore.py`: Single-parameter model `y = σ(w x)` with a fit figure and a two-panel convergence GIF (fit + `w` trajectory).
- `benchmark_activations.py`: Compare 6 activations (`tanh`, `relu`, `sigmoid`, `leaky_relu`, `sine`, `identity`) over multiple polynomials. Produces per-case figures/GIFs and a Markdown report.

## Quick start

```bash
# Single-neuron, tanh vs relu examples
python fit_single_neuron.py

# ReLU-only visuals
python relu_explore.py

# Sigmoid-only (y = σ(w x))
python sigmoid_explore.py

# Full benchmark across activations and polynomials
python benchmark_activations.py
```

Outputs are written under `outputs/`:
- ReLU visuals: `outputs/relu/`
- Sigmoid visuals: `outputs/sigmoid/`
- Benchmark: `outputs/benchmark/` (per-polynomial/per-activation folders plus `REPORT.md`)

## Benchmark summary

We train a single hidden neuron with linear output for each activation and target polynomial. Loss: mean squared error (MSE). The report with inline previews is at `outputs/benchmark/REPORT.md`.

Example winners on the generated run (your results may vary with randomness):

| polynomial   | best activation | best MSE |
|--------------|-----------------|---------:|
| linear_pos   | identity        | 0.0000   |
| quadratic_u  | relu            | ~0.96    |
| cubic_s      | leaky_relu      | ~1.42    |
| cubic_neg    | identity        | ~1.76    |
| quartic_w    | leaky_relu      | ~0.15    |

Why these patterns?
- Identity excels on linear-like targets (no nonlinearity needed).
- ReLU often approximates convex shapes (e.g., upward quadratics) via a hinge + scaling.
- Leaky ReLU and sine can better capture asymmetric or oscillatory features; leaky ReLU avoids dead zones of pure ReLU.
- Tanh and sigmoid are bounded and can be strong for saturating shapes; with a single neuron, they cannot represent polynomials exactly.

## Tutorial: Visual gallery (all images with explanations)

Each fit figure has: a figure-level header with the model, an on-plot box listing x-range (top), parameters (`w, b, v, c`) in order, and predicted ŷ-range (bottom). GIFs animate training snapshots.

### Benchmark: single-neuron across activations and polynomials
- Dataset note: PNGs are shown inline. GIFs are now shown inline as well.

#### linear_pos
| Activation | Fit (PNG) | GIF |
|--|--|--|
| tanh | ![](outputs/benchmark/linear_pos/tanh/linear_pos_tanh_fit.png) | ![](outputs/benchmark/linear_pos/tanh/linear_pos_tanh_convergence.gif) |
| relu | ![](outputs/benchmark/linear_pos/relu/linear_pos_relu_fit.png) | ![](outputs/benchmark/linear_pos/relu/linear_pos_relu_convergence.gif) |
| sigmoid | ![](outputs/benchmark/linear_pos/sigmoid/linear_pos_sigmoid_fit.png) | ![](outputs/benchmark/linear_pos/sigmoid/linear_pos_sigmoid_convergence.gif) |
| leaky_relu | ![](outputs/benchmark/linear_pos/leaky_relu/linear_pos_leaky_relu_fit.png) | ![](outputs/benchmark/linear_pos/leaky_relu/linear_pos_leaky_relu_convergence.gif) |
| sine | ![](outputs/benchmark/linear_pos/sine/linear_pos_sine_fit.png) | ![](outputs/benchmark/linear_pos/sine/linear_pos_sine_convergence.gif) |
| identity | ![](outputs/benchmark/linear_pos/identity/linear_pos_identity_fit.png) | ![](outputs/benchmark/linear_pos/identity/linear_pos_identity_convergence.gif) |

- Explanation: Linear target; `identity` matches exactly; other activations approximate via their nonlinearity.

#### quadratic_u
| Activation | Fit (PNG) | GIF |
|--|--|--|
| tanh | ![](outputs/benchmark/quadratic_u/tanh/quadratic_u_tanh_fit.png) | ![](outputs/benchmark/quadratic_u/tanh/quadratic_u_tanh_convergence.gif) |
| relu | ![](outputs/benchmark/quadratic_u/relu/quadratic_u_relu_fit.png) | ![](outputs/benchmark/quadratic_u/relu/quadratic_u_relu_convergence.gif) |
| sigmoid | ![](outputs/benchmark/quadratic_u/sigmoid/quadratic_u_sigmoid_fit.png) | ![](outputs/benchmark/quadratic_u/sigmoid/quadratic_u_sigmoid_convergence.gif) |
| leaky_relu | ![](outputs/benchmark/quadratic_u/leaky_relu/quadratic_u_leaky_relu_fit.png) | ![](outputs/benchmark/quadratic_u/leaky_relu/quadratic_u_leaky_relu_convergence.gif) |
| sine | ![](outputs/benchmark/quadratic_u/sine/quadratic_u_sine_fit.png) | ![](outputs/benchmark/quadratic_u/sine/quadratic_u_sine_convergence.gif) |
| identity | ![](outputs/benchmark/quadratic_u/identity/quadratic_u_identity_fit.png) | ![](outputs/benchmark/quadratic_u/identity/quadratic_u_identity_convergence.gif) |

- Explanation: Upward-convex; `relu`/`leaky_relu` emulate convexity via a hinge; bounded activations may underfit with one neuron.

#### cubic_s
| Activation | Fit (PNG) | GIF |
|--|--|--|
| tanh | ![](outputs/benchmark/cubic_s/tanh/cubic_s_tanh_fit.png) | ![](outputs/benchmark/cubic_s/tanh/cubic_s_tanh_convergence.gif) |
| relu | ![](outputs/benchmark/cubic_s/relu/cubic_s_relu_fit.png) | ![](outputs/benchmark/cubic_s/relu/cubic_s_relu_convergence.gif) |
| sigmoid | ![](outputs/benchmark/cubic_s/sigmoid/cubic_s_sigmoid_fit.png) | ![](outputs/benchmark/cubic_s/sigmoid/cubic_s_sigmoid_convergence.gif) |
| leaky_relu | ![](outputs/benchmark/cubic_s/leaky_relu/cubic_s_leaky_relu_fit.png) | ![](outputs/benchmark/cubic_s/leaky_relu/cubic_s_leaky_relu_convergence.gif) |
| sine | ![](outputs/benchmark/cubic_s/sine/cubic_s_sine_fit.png) | ![](outputs/benchmark/cubic_s/sine/cubic_s_sine_convergence.gif) |
| identity | ![](outputs/benchmark/cubic_s/identity/cubic_s_identity_fit.png) | ![](outputs/benchmark/cubic_s/identity/cubic_s_identity_convergence.gif) |

- Explanation: S-shaped cubic; `leaky_relu` captures asymmetry; `tanh`/`sigmoid` are bounded, so they match locally.

#### cubic_neg
| Activation | Fit (PNG) | GIF |
|--|--|--|
| tanh | ![](outputs/benchmark/cubic_neg/tanh/cubic_neg_tanh_fit.png) | ![](outputs/benchmark/cubic_neg/tanh/cubic_neg_tanh_convergence.gif) |
| relu | ![](outputs/benchmark/cubic_neg/relu/cubic_neg_relu_fit.png) | ![](outputs/benchmark/cubic_neg/relu/cubic_neg_relu_convergence.gif) |
| sigmoid | ![](outputs/benchmark/cubic_neg/sigmoid/cubic_neg_sigmoid_fit.png) | ![](outputs/benchmark/cubic_neg/sigmoid/cubic_neg_sigmoid_convergence.gif) |
| leaky_relu | ![](outputs/benchmark/cubic_neg/leaky_relu/cubic_neg_leaky_relu_fit.png) | ![](outputs/benchmark/cubic_neg/leaky_relu/cubic_neg_leaky_relu_convergence.gif) |
| sine | ![](outputs/benchmark/cubic_neg/sine/cubic_neg_sine_fit.png) | ![](outputs/benchmark/cubic_neg/sine/cubic_neg_sine_convergence.gif) |
| identity | ![](outputs/benchmark/cubic_neg/identity/cubic_neg_identity_fit.png) | ![](outputs/benchmark/cubic_neg/identity/cubic_neg_identity_convergence.gif) |

- Explanation: Negative-slope cubic region; `identity` follows the broad linear trend best with a single unit.

#### quartic_w
| Activation | Fit (PNG) | GIF |
|--|--|--|
| tanh | ![](outputs/benchmark/quartic_w/tanh/quartic_w_tanh_fit.png) | ![](outputs/benchmark/quartic_w/tanh/quartic_w_tanh_convergence.gif) |
| relu | ![](outputs/benchmark/quartic_w/relu/quartic_w_relu_fit.png) | ![](outputs/benchmark/quartic_w/relu/quartic_w_relu_convergence.gif) |
| sigmoid | ![](outputs/benchmark/quartic_w/sigmoid/quartic_w_sigmoid_fit.png) | ![](outputs/benchmark/quartic_w/sigmoid/quartic_w_sigmoid_convergence.gif) |
| leaky_relu | ![](outputs/benchmark/quartic_w/leaky_relu/quartic_w_leaky_relu_fit.png) | ![](outputs/benchmark/quartic_w/leaky_relu/quartic_w_leaky_relu_convergence.gif) |
| sine | ![](outputs/benchmark/quartic_w/sine/quartic_w_sine_fit.png) | ![](outputs/benchmark/quartic_w/sine/quartic_w_sine_convergence.gif) |
| identity | ![](outputs/benchmark/quartic_w/identity/quartic_w_identity_fit.png) | ![](outputs/benchmark/quartic_w/identity/quartic_w_identity_convergence.gif) |

- Explanation: “W” shape; `leaky_relu` fits best thanks to non-zero negative slope.

### ReLU exploration (dedicated)
- ![](outputs/relu/relu_activation_derivative.png) — ReLU(z) and derivative ReLU′(z) vs z.
- ![](outputs/relu/relu_single_fit.png) — Single-neuron ReLU best fit with ordered annotation.
- ![](outputs/relu/relu_single_convergence.gif) — ReLU training over snapshots; annotation updates each frame.
- ![](outputs/relu/relu_param_trajectories.png) — Snapshot trajectories for w, b, v, c.
- ![](outputs/relu/relu_wb_loss_heatmap.png) — MSE over (w,b) with optimal (v,c), training trajectory overlaid.

### Sigmoid exploration (y = σ(w x))
- ![](outputs/sigmoid/sigmoid_fit.png) — Best fit for `σ(w x)`; annotation shows x-range, w, ŷ-range.
- ![](outputs/sigmoid/sigmoid_convergence.gif) — Two-panel: Left fit; Right `w` trajectory with a moving marker.

### Extras from earlier runs
- ![](outputs/single_neuron_fits.png) — Single-neuron comparison across activations on a fixed polynomial.
- ![](outputs/tanh_single_fit.png) — ![](outputs/tanh_single_convergence.gif) — Tanh single-neuron.
- ![](outputs/relu_single_fit.png) — ![](outputs/relu_single_convergence.gif) — ReLU single-neuron.
- ![](outputs/tanh_mlp_fit.png) — ![](outputs/tanh_mlp_convergence.gif) — Tiny tanh-MLP.
- ![](outputs/tanh_fit.png) — ![](outputs/tanh_convergence.gif) — Original tanh-only.

## Notes

- All models are intentionally tiny to stay interpretable. Use `hidden_sizes` or `epochs`/`lr` in scripts to adjust capacity or training length.
- Plots include figure-level function forms and annotation boxes in the order: x-range (top) → parameters → predicted y-range (bottom).
- GIFs record snapshots every N steps for a short, readable animation.

## License

MIT
