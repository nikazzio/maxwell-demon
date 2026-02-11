# Theoretical Framework

This section formalizes the mathematical basis of Maxwell-Demon, linking Shannon entropy, compression, and burstiness for distinguishing human-written text from LLM-generated text.

## 1) The Entropic Signature Hypothesis

Human language can be modeled as a non‑stationary stochastic process shaped by biological and energetic constraints: people optimize communicative efficiency but introduce structured noise (creativity, emphasis, error, stylistic deviation).

LLMs, by contrast, optimize a Maximum Likelihood Estimation (MLE) objective, minimizing average loss over training data. This tends to produce a statistically smoother texture: local deviations are reduced to maximize global likelihood.

## 2) Compression as Surprise (The Dictionary Concept)

Compressing a text with a reference dictionary $D$ is equivalent to estimating the **local cross‑entropy** against a baseline language model. The informational surprise (surprisal) of a word $w_i$ is:

$$S(w_i) = -\log_2 P_{ref}(w_i)$$

where $P_{ref}$ is the probability of the word in a “human ground‑truth” dictionary (e.g., a 2018 Italian frequency list pre‑LLM). In coding‑theoretic terms, $S(w_i)$ is the **optimal code length** (in bits) required to compress $w_i$ if the standard language distribution were perfectly known.

## 3) Divergence and Residuals

We measure the divergence between the generated text $T$ and the standard language model $M$. A token‑level approximation of Kullback–Leibler divergence is:

$$D_{KL}(T || M) \approx \frac{1}{N} \sum_{i=1}^{N} \left(-\log P_M(w_i)\right) - H(T)$$

where $H(T)$ is the entropy of the text itself. The tool isolates **residuals**: when an author uses unexpected words, $S(w_i)$ increases and the local compression cost rises, signaling deviations from standard language.

## 4) Burstiness: Variance as a Discriminator

This is the core signal. LLMs tend to keep $S(w_i)$ relatively constant (low variance) to avoid destabilizing generation, while humans alternate common words ($S \approx 0$) with rare or highly contextual terms ($S \gg 0$).

We define **burstiness** on a sliding window $W$ as the standard deviation of surprisal:

$$\mathcal{B}_W = \sigma(S_W) = \sqrt{\frac{1}{|W|} \sum_{w \in W} (S(w) - \mu_W)^2}$$

High $\mathcal{B}_W$ indicates a dynamic information profile typical of human writing.

## 5) Visual Conclusion

Projecting texts into a 2D space:

- $X = \text{Mean Surprisal}$ (local lexical richness)
- $Y = \text{Burstiness}$ (attention dynamics)

reveals a **phase separation**: human texts tend to occupy high‑variance regions (criticality), while LLM texts cluster in low‑variance regions (thermodynamic equilibrium).
