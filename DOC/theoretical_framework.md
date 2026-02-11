# Theoretical Framework

## 1. Problem Formulation

Given a text sequence, the task is to determine whether its local lexical dynamics are better explained by a human-derived or synthetic-derived reference distribution.

The framework is comparative by construction: classification does not depend on absolute entropy magnitude, but on differential compatibility across two calibrated references.

## 2. Dual-Reference Hypothesis

Let:

- \(\mathcal{R}_H\): human reference distribution (PAISA-derived)
- \(\mathcal{R}_S\): synthetic reference distribution (AI-derived)

Both references are estimated with identical preprocessing and smoothing to preserve comparability.

## 3. Tournament Mechanics

For each sliding window \(W\), the pipeline computes surprisal-based statistics under both references. Operationally, this induces four comparisons over labeled corpora:

1. human samples under \(\mathcal{R}_H\)
2. human samples under \(\mathcal{R}_S\)
3. synthetic samples under \(\mathcal{R}_H\)
4. synthetic samples under \(\mathcal{R}_S\)

The decisive statistic is the within-window entropy differential between references.

## 4. Differential Entropy Statistic

For text \(T\) and window \(W\):

$$
H_H(T,W) = \frac{1}{|W|}\sum_{w\in W} -\log P_{\mathcal{R}_H}(w)
$$

$$
H_S(T,W) = \frac{1}{|W|}\sum_{w\in W} -\log P_{\mathcal{R}_S}(w)
$$

Define:

$$
\Delta H(T,W) = H_H(T,W) - H_S(T,W)
$$

In exported outputs, \(\Delta H\) is represented by `delta_h`.

## 5. Interpretation Regime

- \(\Delta H > 0\): the window is relatively more probable under the human reference.
- \(\Delta H \approx 0\) or \(\Delta H < 0\): the window is relatively more compatible with synthetic-reference regularities.

Robust inference should rely on empirical distributions of \(\Delta H\), not isolated windows.

## 6. Burstiness as Orthogonal Descriptor

A second descriptor quantifies local fluctuation of surprisal under the human reference:

$$
B_H(T,W)=\operatorname{Var}\left(-\log P_{\mathcal{R}_H}(w)\right)_{w\in W}
$$

This quantity is exposed as `burstiness_paisa`.

## 7. Phase-Space Geometry

Operational visualization is conducted in the two-dimensional space:

- x-axis: `delta_h`
- y-axis: `burstiness_paisa`

This embedding supports geometric separation between regions with dominant human-reference affinity and regions with dominant synthetic-reference affinity.

## 8. Validity Constraint

The method is formally defined only if both calibrated references are available:

- `paisa_ref_dict.json`
- `synthetic_ref_dict.json`

If either reference is absent, dual-reference discrimination is underdetermined.
