# Single-Neuron Activation Benchmark

This report compares 6 activations on multiple target polynomials using a single hidden neuron and a linear output.

## linear_pos

| activation | best MSE | preview |
|--|--:|--|
| identity | 0.0000 | ![](outputs/benchmark/linear_pos/identity/linear_pos_identity_fit.png) |
| sine | 0.1060 | ![](outputs/benchmark/linear_pos/sine/linear_pos_sine_fit.png) |
| tanh | 0.1862 | ![](outputs/benchmark/linear_pos/tanh/linear_pos_tanh_fit.png) |
| leaky_relu | 0.2797 | ![](outputs/benchmark/linear_pos/leaky_relu/linear_pos_leaky_relu_fit.png) |
| relu | 0.6642 | ![](outputs/benchmark/linear_pos/relu/linear_pos_relu_fit.png) |
| sigmoid | 2.1210 | ![](outputs/benchmark/linear_pos/sigmoid/linear_pos_sigmoid_fit.png) |

Best: **identity** with MSE 0.0000

## quadratic_u

| activation | best MSE | preview |
|--|--:|--|
| relu | 0.9606 | ![](outputs/benchmark/quadratic_u/relu/quadratic_u_relu_fit.png) |
| sine | 1.2737 | ![](outputs/benchmark/quadratic_u/sine/quadratic_u_sine_fit.png) |
| sigmoid | 1.2744 | ![](outputs/benchmark/quadratic_u/sigmoid/quadratic_u_sigmoid_fit.png) |
| identity | 1.2752 | ![](outputs/benchmark/quadratic_u/identity/quadratic_u_identity_fit.png) |
| leaky_relu | 1.2756 | ![](outputs/benchmark/quadratic_u/leaky_relu/quadratic_u_leaky_relu_fit.png) |
| tanh | 1.2770 | ![](outputs/benchmark/quadratic_u/tanh/quadratic_u_tanh_fit.png) |

Best: **relu** with MSE 0.9606

## cubic_s

| activation | best MSE | preview |
|--|--:|--|
| leaky_relu | 1.4163 | ![](outputs/benchmark/cubic_s/leaky_relu/cubic_s_leaky_relu_fit.png) |
| relu | 1.4974 | ![](outputs/benchmark/cubic_s/relu/cubic_s_relu_fit.png) |
| identity | 3.1867 | ![](outputs/benchmark/cubic_s/identity/cubic_s_identity_fit.png) |
| tanh | 3.5832 | ![](outputs/benchmark/cubic_s/tanh/cubic_s_tanh_fit.png) |
| sine | 3.8689 | ![](outputs/benchmark/cubic_s/sine/cubic_s_sine_fit.png) |
| sigmoid | 10.9220 | ![](outputs/benchmark/cubic_s/sigmoid/cubic_s_sigmoid_fit.png) |

Best: **leaky_relu** with MSE 1.4163

## cubic_neg

| activation | best MSE | preview |
|--|--:|--|
| identity | 1.7560 | ![](outputs/benchmark/cubic_neg/identity/cubic_neg_identity_fit.png) |
| leaky_relu | 3.0051 | ![](outputs/benchmark/cubic_neg/leaky_relu/cubic_neg_leaky_relu_fit.png) |
| relu | 3.0506 | ![](outputs/benchmark/cubic_neg/relu/cubic_neg_relu_fit.png) |
| sine | 3.2844 | ![](outputs/benchmark/cubic_neg/sine/cubic_neg_sine_fit.png) |
| tanh | 3.7740 | ![](outputs/benchmark/cubic_neg/tanh/cubic_neg_tanh_fit.png) |
| sigmoid | 10.9407 | ![](outputs/benchmark/cubic_neg/sigmoid/cubic_neg_sigmoid_fit.png) |

Best: **identity** with MSE 1.7560

## quartic_w

| activation | best MSE | preview |
|--|--:|--|
| leaky_relu | 0.1542 | ![](outputs/benchmark/quartic_w/leaky_relu/quartic_w_leaky_relu_fit.png) |
| tanh | 0.1765 | ![](outputs/benchmark/quartic_w/tanh/quartic_w_tanh_fit.png) |
| identity | 0.1806 | ![](outputs/benchmark/quartic_w/identity/quartic_w_identity_fit.png) |
| relu | 0.1815 | ![](outputs/benchmark/quartic_w/relu/quartic_w_relu_fit.png) |
| sigmoid | 0.1818 | ![](outputs/benchmark/quartic_w/sigmoid/quartic_w_sigmoid_fit.png) |
| sine | 0.1848 | ![](outputs/benchmark/quartic_w/sine/quartic_w_sine_fit.png) |

Best: **leaky_relu** with MSE 0.1542
