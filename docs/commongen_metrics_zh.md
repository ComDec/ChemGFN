# CommonGen: 指标与验证器输出（中文）

本仓库实现了 CommonGen 风格的“关键词（concepts）到句子”的生成任务：给定一个概念集合（关键词集合）以及（可选的）多条参考句，评估模型生成句子对关键词约束的满足情况，以及常用的文本生成质量指标。

实现位置：`chemgfn/models/validators.py` 中的 `CommonGenValidator`。

## 记号说明

- 概念集合：\(\mathcal{C}=\{c_1,\dots,c_K\}\)，其中每个 \(c_k\) 是一个概念（可能是多词短语）。
- 模型生成句子：\(y\)。
- 参考句集合：\(\mathcal{R}=\{r_1,\dots,r_M\}\)（某些 split 可能没有参考句）。
- \(\mathrm{norm}(\cdot)\)：用于概念匹配的规范化（小写化、去除非字母数字字符、合并空白）。
- \(\mathrm{lemma}(\cdot)\)：spaCy (`en_core_web_sm`) 的词形还原。

本任务在生成时将 `.` 视为句末符（EOS）；解码时候会截断到 EOS（若未出现则截断到最大长度）。

## Coverage（关键词覆盖率）

Coverage 衡量生成句子中“覆盖了多少比例的概念”。我们实现了两种匹配方式：

1) Lemma-based coverage（用于最终报告）：

- 对候选句和每个概念短语做词形还原（lemma），再做规范化，并进行 whole-word 匹配。
- 能更稳健地处理词形变化（例如 "run" vs "running"）。

2) Surface-form coverage（用于训练中 prefix 级 shaping）：

- 不做 lemma，仅做规范化后匹配，用于快速给 prefix 打分。

报告指标：

- `coverage`（百分比）：\(100\times\mathrm{Cov}(y,\mathcal{C})\)。
- 内部也会保留 `cov_ratio`（\([0,1]\)）。

### LaTeX（可直接写入论文 Method）

```tex
\paragraph{Concept coverage.}
Given a concept set $\mathcal{C}=\{c_1,\ldots,c_K\}$ and a generated sentence $y$, we compute concept coverage as
\begin{equation}
\mathrm{Cov}(y,\mathcal{C}) = \frac{1}{K}\sum_{k=1}^{K} \mathbf{1}\big[\mathrm{match}(c_k, y)\big],
\end{equation}
where $\mathrm{match}(c_k,y)$ is a whole-word match between the lemmatized concept phrase and the lemmatized sentence (spaCy lemmatizer). We report coverage as a percentage: $100\times\mathrm{Cov}(y,\mathcal{C})$.
```

## Hard coverage accuracy（全覆盖准确率）

我们定义硬约束满足指标：

- `acc`：若所有概念均被覆盖则为 1，否则为 0，即 \(\mathbf{1}[\mathrm{Cov}(y,\mathcal{C})=1]\)。

这对应 CommonGen 常用的“是否覆盖所有 concepts”评估口径。

### LaTeX

```tex
\paragraph{Hard concept satisfaction.}
We define hard satisfaction as $\mathrm{Acc}(y,\mathcal{C})=\mathbf{1}[\mathrm{Cov}(y,\mathcal{C})=1]$.
```

## coverage_filter（只在 hard-valid 上的覆盖率）

为了区分“是否满足硬约束”和“软覆盖程度”，我们还报告在 hard-valid 样本上的平均覆盖率：

- `coverage_filter`：只对 \(\mathrm{Acc}(y,\mathcal{C})=1\) 的样本求 \(\mathrm{Cov}(y,\mathcal{C})\) 平均；若没有 hard-valid 样本则为 0。

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

当参考句 \(\mathcal{R}\) 存在时，我们报告 corpus-level BLEU-3 与 BLEU-4：

实现细节：

- 优先使用 COCO caption 的 BLEU 实现（`pycocoevalcap`）与 PTBTokenizer（Penn Treebank tokenization）。
- 仅报告 BLEU-3 与 BLEU-4。
- 指标统一缩放到 \([0,100]\)。
- 若 `pycocoevalcap` 不可用，则 fallback 到 NLTK `corpus_bleu` 并使用简单平滑。

### LaTeX

```tex
\paragraph{BLEU.}
We report corpus BLEU-3 and BLEU-4 (scaled to $[0,100]$) with Penn Treebank tokenization, computed against the set of references for each prompt using the standard BLEU definition~\citep{papineni2002bleu}.
```

## N-gram F1（辅助指标；用于 shaping）

为了在训练阶段提供轻量级质量 shaping，我们实现了一个基于 n-gram 的 F1：

- Tokenization：简单的 word tokens（小写的字母数字 + apostrophe）。
- n-grams：连续 n-gram，并作为 multiset 计数。
- 若有多参考句，则取最大值（max-over-refs）。

指标名：

- `ngram_f1`，取值范围 \([0,1]\)。

### LaTeX

```tex
\paragraph{N-gram F1 (auxiliary).}
Let $G_n(x)$ be the multiset of word $n$-grams in text $x$. For a candidate $y$ and reference $r$, we define
\begin{equation}
\mathrm{F1}_n(y,r)=\frac{2\,|G_n(y)\cap G_n(r)|}{|G_n(y)|+|G_n(r)|},
\end{equation}
and use $\max_{r\in\mathcal{R}} \mathrm{F1}_n(y,r)$ when multiple references are available.
```

## BERTScore（可选）

可选的语义相似度指标：

- `bertscore_f1`：batch 平均 BERTScore F1。
- 仅在 eval 时计算（训练时不算）以避免额外开销。
- 默认模型（开启时）：`microsoft/deberta-xlarge-mnli`。

### LaTeX

```tex
\paragraph{BERTScore (optional).}
When enabled, we report BERTScore F1~\citep{zhang2019bertscore} between candidates and references using a fixed pretrained encoder.
```

## GFlowNet 训练中使用的验证器输出

验证器会返回以下张量，供 reward/loss 计算使用：

- `invalid`（形状 \(B\times(T+1)\)）：对 prefix 状态的无效掩码；空 prefix 处 `invalid[:,0]=1`。
- `local_score`（形状 \(B\times(T+1)\)）：prefix 级 shaping 分数（软 coverage + 可选质量 shaping）。
- `global_score`（形状 \(B\)）：最终 lemma coverage ratio \(\mathrm{Cov}(y,\mathcal{C})\)，取值 \([0,1]\)。
- `full_tokens`：解码后的最终文本（截断到 EOS）。

在该任务中，终止状态（terminal state）仅当“覆盖所有概念”（hard satisfaction）时被视为有效（`invalid=0`）；最终状态 `invalid[:,-1]` 会镜像终止状态的有效性。

## 配置项（可调参数）

关键配置字段（参考 `configs/model/common_gen.yaml`）：

- `model.reward.sentence_validator.coverage_weight`, `quality_weight`, `hard_coverage_bonus`, `ngram_n`
- `model.reward.sentence_validator.compute_bleu`, `compute_bertscore`
- `model.constraint_config.end_of_sentence_token`（设置为 `.`）
