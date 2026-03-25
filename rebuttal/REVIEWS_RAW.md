# Raw Reviews — Submission 13383

Verbatim copy from OpenReview. See `/data2/xw3763/gflow/ChemGFN/refs/reviews.md` for the source file.

## Reviewer Pd1v — Score 3 (Weak Reject), Confidence 2

**Summary**: Paper aims to fix (1) weak credit assignment to early prefixes and (2) biased replay in LLM-GFlowNet. RapTB as new objective + SubM for replay. Experiments on SMILES.

**Strengths**: RapTB vs TB/SubTB comparison; SubM for diversity; SMILES generation interesting; decent results.

**Weaknesses**:
- W1: Experimental validation not strong enough — small LLM, narrow benchmarks (molecule, synthetic arithmetic, CommonGen)
- W2: Baseline methods not sufficient — only TB and SubTB compared
- W3: No theoretical guarantees for RapTB preserving reward-proportional terminal distribution

**Questions**: See weaknesses.

---

## Reviewer cA3o — Score 4 (Weak Accept), Confidence 3

**Summary**: RapTB proposes rooted prefix constraints + absorbed suffix backups + pterm stop-grad. SubM replaces RP with submodular objective. Evaluated on SMILES (QED), Expr24, CommonGen diagnostic subset.

**Strengths**:
- S1: Mechanistic diagnosis of SubTB failure on terminable prefix trees (Appendix C.6)
- S2: Absorbed suffix backup is clean variance-reduction (Appendix C.5)
- S3: Figure 1 and Figure 3 clear justifications

**Weaknesses**:
- W1: Empirical support rests almost entirely on SMILES/QED — needs biological sequences, code gen, or math reasoning
- W2: TB+SubM outperforms RapTB alone on coverage (Table 3) — when does RapTB provide additive benefit?
- W3: Seven task-specific hyperparameters (eta, gamma, alpha, beta, rho, k_min, K) — sensitivity unknown

**Questions**:
- Q1: When does RapTB provide meaningful additive benefit over SubM alone?
- Q2: How sensitive are results to beta and rho?
- Q3: Do gains hold on longer sequences, larger vocabularies, non-SMILES domains?
- Q4: Is RapTB equivalent to a particular GAE estimator in the GFlowNet framework?

---

## Reviewer JxzD — Score 4 (Weak Accept), Confidence 4

**Summary**: RapTB with rooted subtrajectories + absorbed suffix reward target + SubM replay buffer. Finetuning Llama-3.2-1B on SMILES, arithmetic, sentence generation.

**Strengths**:
- S1: Novel solutions for GFN training issues
- S2: Large-scale model on variety of tasks, promising results
- S3: Good ablation studies
- S4: Submodular replay is clever and generally applicable

**Weaknesses**:
- W1: Lack of convergence/global optimal analysis
- W2: Absorbed suffix backups need clearer motivation
- W3: Many design choices, multiple hyperparameters, robustness unknown

**Questions**:
- Q1: Why do SubTB constraints include termination probabilities? Have you tested learning state flow values?
- Q2: What is the prefix survival metric?
- Q3: Have you tested longer sequence tasks (AMP/GFP)?
- Q4: Why finetune LLM rather than train small model from scratch?

---

## Reviewer QHmk — Score 2 (Reject), Confidence 4

**Summary**: RapTB as GFlowNet-based LLM fine-tuning method for reward-proportional posteriors. Modified TB objective + submodular replay. Evaluation shows improvement over TB/SubTB.

**Strengths**:
- S1: Investigating GFlowNet-based LLM fine-tuning is highly interesting
- S2: In-depth experimental evaluation and ablation studies, well-crafted and insightful

**Weaknesses**:
- W1: Fails to contextualize within RL literature — GFlowNets reformulable as entropy-regularized RL
- W2: Missing PPO/GRPO baselines
- W3: Missing TBA baseline (most direct competitor)
- W4: RapTB loss section needs expanded mathematical explanation
- W5: Terminology used before definition (e.g., termination drift)

**Questions**:
- Q1: Mathematical explanation of RapTB loss — are credit bounds (Eqs 6-8) to avoid learnable Z? Why chosen this way? Does global optimum guarantee target distribution?
- Q2: Simpler baseline — averaging usual TB loss over all prefixes. Is this viable? Will RapTB produce meaningfully different results?

**References cited**: [1] GFlowNets as Entropy-Regularized RL (Tiapkin+, AISTATS 2024), [2] Discrete Probabilistic Inference as Control (Deleu+, UAI 2024), [3] TBA (Bartoldson+, NeurIPS 2025)
