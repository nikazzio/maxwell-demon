# Fondamenti Teorici (Theoretical Framework)

Questa sezione formalizza la base matematica di Maxwell-Demon, collegando Entropia di Shannon, Compressione e Burstiness per la distinzione tra testi umani e testi generati da LLM.

## 1) L'Ipotesi della Firma Entropica

Il linguaggio umano puo essere modellato come un processo stocastico non stazionario, plasmato da vincoli biologici ed energetici: gli esseri umani cercano efficienza comunicativa ma introducono rumore strutturato (creativita, enfasi, errore, deviazione stilistica).

Gli LLM, al contrario, ottimizzano una funzione obiettivo di Maximum Likelihood Estimation (MLE), minimizzando la perdita media sui dati di training. Questo induce una “texture” statistica piu levigata: il modello riduce le deviazioni locali per massimizzare la verosimiglianza globale.

## 2) La Compressione come Misura di Sorpresa (Il concetto del Dizionario)

Comprimere un testo con un dizionario di riferimento $D$ equivale a stimare la **cross‑entropy locale** rispetto a un modello linguistico di base. La **sorpresa informativa** (surprisal) di una parola $w_i$ e definita come:

$$S(w_i) = -\log_2 P_{ref}(w_i)$$

dove $P_{ref}$ e la probabilita della parola nel dizionario “Human Ground Truth” (es. lista frequenze italiane 2018 pre‑LLM). In termini di teoria dei codici, $S(w_i)$ rappresenta la **lunghezza ottimale del codice** (in bit) necessaria per comprimere $w_i$ se la lingua standard fosse perfettamente nota.

## 3) La Divergenza e i Residui

Stiamo misurando la divergenza tra il testo generato $T$ e il modello linguistico standard $M$. Una forma approssimata della divergenza di Kullback‑Leibler sui token e:

$$D_{KL}(T || M) \approx \frac{1}{N} \sum_{i=1}^{N} \left(-\log P_M(w_i)\right) - H(T)$$

dove $H(T)$ e l'entropia del testo stesso. Il tool isola i **residui**: quando l'autore usa parole inattese, $S(w_i)$ cresce e il costo di compressione locale aumenta, segnalando deviazioni dalla lingua standard.

## 4) Burstiness: La Varianza come Discriminatore

Questo e il cuore del metodo. Gli LLM tendono a mantenere $S(w_i)$ relativamente costante (bassa varianza) per non destabilizzare la generazione, mentre gli umani alternano parole banali ($S \approx 0$) a parole rare o fortemente contestuali ($S \gg 0$).

Definiamo la **Burstiness** su una finestra mobile $W$ come la deviazione standard della surprisal:

$$\mathcal{B}_W = \sigma(S_W) = \sqrt{\frac{1}{|W|} \sum_{w \in W} (S(w) - \mu_W)^2}$$

Valori alti di $\mathcal{B}_W$ indicano un dinamismo informativo tipico della produzione umana.

## 5) Conclusione Visiva

Proiettando i testi in uno spazio bidimensionale:

- $X = \text{Mean Surprisal}$ (ricchezza lessicale locale)
- $Y = \text{Burstiness}$ (dinamica dell'attenzione)

emerge una **separazione di fase**: i testi umani tendono a occupare regioni ad alta varianza (criticita), mentre i testi LLM si addensano in regioni a bassa varianza (equilibrio termodinamico).
