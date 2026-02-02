# CommonGen: Metrics and Validator Outputs

This repo implements a CommonGen-style keyword-to-sentence task. Given a concept set (keywords) and (optionally) multiple reference sentences, we evaluate generated sentences using constraint satisfaction (coverage) and standard text generation metrics.

Implementation reference: `chemgfn/models/validators.py` (`CommonGenValidator`).

## Notation

- Concept set: \(\mathcal{C} = \{c_1, \dots, c_{K}\}\), where each \(c_k\) is a (possibly multi-word) concept.
- Candidate (generated) sentence: \(y\).
- References: \(\mathcal{R} = \{r_1, \dots, r_M\}\) (may be empty for test-only splits).
- Let \(\mathrm{norm}(\cdot)\) denote the text normalization used for concept matching (lowercasing, removing non-alphanumerics, and collapsing whitespace).
- Let \(\mathrm{lemma}(\cdot)\) denote lemmatization using spaCy (`en_core_web_sm`).

The task treats `.` as an end-of-sentence token (EOS) during generation; decoded candidates are truncated to EOS (or to max length if EOS is absent).

## Coverage (keyword satisfaction)

We compute keyword coverage as the fraction of concepts that appear in the generated sentence. We implement two matching modes:

1) Lemma-based coverage (used for reporting):

- Lemmatize the candidate sentence and each concept phrase, normalize them, and check whole-word matches.
- This is robust to inflections (e.g., "run" vs "running").

2) Surface-form coverage (used for per-prefix shaping inside the validator):

- Normalize the candidate prefix without lemmatization, then check whole-word matches.

Reported metric:

- `coverage` (percentage): \(100 \times \mathrm{Cov}(y, \mathcal{C})\).
- Internally we also track `cov_ratio` in \([0,1]\).

### LaTeX (paper-ready)

```tex
\paragraph{Concept coverage.}
Given a concept set $\mathcal{C}=\{c_1,\ldots,c_K\}$ and a generated sentence $y$, we compute concept coverage as
\begin{equation}
\mathrm{Cov}(y,\mathcal{C}) = \frac{1}{K}\sum_{k=1}^{K} \mathbf{1}\big[\mathrm{match}(c_k, y)\big],
\end{equation}
where $\mathrm{match}(c_k,y)$ is a whole-word match between the lemmatized concept phrase and the lemmatized sentence (spaCy lemmatizer). We report coverage as a percentage: $100\times\mathrm{Cov}(y,\mathcal{C})$.
```

## Hard coverage accuracy (all keywords present)

We define a hard constraint satisfaction indicator:

- `acc`: \(\mathbf{1}[\mathrm{Cov}(y,\mathcal{C}) = 1]\).

This matches the “all concepts covered” notion commonly used in CommonGen evaluations.

### LaTeX

```tex
\paragraph{Hard concept satisfaction.}
We define hard satisfaction as $\mathrm{Acc}(y,\mathcal{C})=\mathbf{1}[\mathrm{Cov}(y,\mathcal{C})=1]$.
```

## Coverage@valid (coverage_filter)

To separate semantic/fluency effects from constraint satisfaction, we also report coverage conditioned on hard-valid generations:

- `coverage_filter`: average \(\mathrm{Cov}(y,\mathcal{C})\) over examples with \(\mathrm{Acc}(y,\mathcal{C})=1\) (0 if none are valid).

### LaTeX

```tex
\paragraph{Coverage conditioned on validity.}
Let $\mathcal{I}=\{i: \mathrm{Acc}(y_i,\mathcal{C}_i)=1\}$. We report
\begin{equation}
\mathrm{CovFilt} = \frac{1}{|\mathcal{I}|}\sum_{i\in\mathcal{I}} \mathrm{Cov}(y_i,\mathcal{C}_i),
\end{equation}
with $\mathrm{CovFilt}=0$ when $|\mathcal{I}|=0$.
```

## BLEU-3 / BLEU-4

We report corpus-level BLEU-3 and BLEU-4 against the available reference set \(\mathcal{R}\) (when present).

Implementation details:

- Primary implementation uses COCO caption evaluation code (`pycocoevalcap`) with PTB tokenization (`PTBTokenizer`).
- We report only BLEU-3 and BLEU-4.
- Values are scaled to \([0,100]\).
- If `pycocoevalcap` is unavailable, we fall back to NLTK `corpus_bleu` with a simple smoothing.

### LaTeX

```tex
\paragraph{BLEU.}
We report corpus BLEU-3 and BLEU-4 (scaled to $[0,100]$) with Penn Treebank tokenization, computed against the set of references for each prompt using the standard BLEU definition~\citep{papineni2002bleu}.
```

## N-gram F1 (auxiliary; shaping)

For lightweight quality shaping during training, we compute an n-gram F1 score between the candidate and a reference.

- Tokenization: simple word tokens (lowercased alphanumerics + apostrophes).
- n-grams: contiguous n-grams, treated as multisets.
- Over multiple references, we take the maximum score.

Metric name:

- `ngram_f1` in \([0,1]\).

### LaTeX

```tex
\paragraph{N-gram F1 (auxiliary).}
Let $G_n(x)$ be the multiset of word $n$-grams in text $x$. For a candidate $y$ and reference $r$, we define
\begin{equation}
\mathrm{F1}_n(y,r)=\frac{2\,|G_n(y)\cap G_n(r)|}{|G_n(y)|+|G_n(r)|},
\end{equation}
and use $\max_{r\in\mathcal{R}} \mathrm{F1}_n(y,r)$ when multiple references are available.
```

## BERTScore (optional)

An optional semantic similarity metric:

- `bertscore_f1`: mean BERTScore F1 over the batch.
- Only computed in evaluation (not during training) due to cost.
- Default model (if enabled): `microsoft/deberta-xlarge-mnli`.

### LaTeX

```tex
\paragraph{BERTScore (optional).}
When enabled, we report BERTScore F1~\citep{zhang2019bertscore} between candidates and references using a fixed pretrained encoder.
```

## Validator outputs used by the GFlowNet training

The validator returns per-sample tensors used by reward/loss code:

- `invalid` (shape \(B\times(T+1)\)): invalid-mask over prefix states; `invalid[:,0]=1` for the empty prefix.
- `local_score` (shape \(B\times(T+1)\)): per-prefix shaping score.
- `global_score` (shape \(B\)): final lemma coverage ratio \(\mathrm{Cov}(y,\mathcal{C})\) in \([0,1]\).
- `full_tokens`: decoded final text (up to EOS).

In this task, the terminal state is treated as valid (`invalid=0`) iff all concepts are covered (hard satisfaction). The final state `invalid[:,-1]` mirrors the terminal validity.

## Metric configuration knobs

Key config fields (see `configs/model/common_gen.yaml`):

- `model.reward.sentence_validator.coverage_weight`, `quality_weight`, `hard_coverage_bonus`, `ngram_n`
- `model.reward.sentence_validator.compute_bleu`, `compute_bertscore`
- `model.constraint_config.end_of_sentence_token` (set to `.`)
