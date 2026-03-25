Official Review of Submission13383 by Reviewer Pd1v
Official Reviewby Reviewer Pd1v17 Mar 2026, 17:57 (modified: 24 Mar 2026, 06:21)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer Pd1vRevisions
Summary:
The paper aims to fix (1) weak credit assignment to early prefixes and (2) biased replay when using GFlowNet to finetune LLM to approximate reward-proportional posterios. Specifically to address (1), the paper introduces Rooted absorbed prefix Trajectory Balance as a new objective. To mitigate the replay bias problem (2), the paper proposes a submodular replay refresh strategy that promotes both high reward and diversity. Experiments are conducted on SMILE-based molucule generation with LLM.

Strengths And Weaknesses:
Strengths

The paper compares the proposed RapTB with standard tajectory balance and subtrajectory balance. RapTB additionally considers rooted prefix supervision and absorbed suffix rewards.

Submodular replay as an additional improvement to enforce diversity in addition to the exploration of GFlowNet.

SMILES-based molecule generation with LLMs are interesting applications. Experiments show decent effectiveness.

Weaknesses

The experimental validation is not strong enough. Experiments mostly use a small LLM and the evaluated benchmarks are a bit narrow, as it only covers molecule generation, synthetic arithmetic and CommonGen. To support the claims made in the paper, it will be better to include LLMs with larger size and more benchmarks.

The baseline methods are not sufficient, as it only compares TB and SubTB. I think there are definitely more competitive baseline methods available.

No theoretical guarantees are provided for RapTB to show that it can indeed preserve the desired reward-proportional terminal distribution. It will be nice to have a deeper analysis regarding the effectiveness of RapTB.

Soundness: 2: fair
Presentation: 2: fair
Significance: 2: fair
Originality: 2: fair
Key Questions For Authors:
See the weakness part above.

Limitations:
N/A

Overall Recommendation: 3: Weak reject: A paper with clear merits, but also some weaknesses, which overall outweigh the merits. Papers in this category require revisions before they can be meaningfully built upon by others. Please use sparingly.
Confidence: 2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.
Compliance With LLM Reviewing Policy: Affirmed.
Code Of Conduct Acknowledgement: Affirmed.
Add:
Official Review of Submission13383 by Reviewer cA3o
Official Reviewby Reviewer cA3o15 Mar 2026, 07:02 (modified: 24 Mar 2026, 06:21)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer cA3oRevisions
Summary:
Here, the authors a method, RapTB, that proposes two complementary components for training LLM-GFlowNets on terminable prefix trees. First, RapTB augments the standard terminal Trajectory Balance (TB) objective with O(N) rooted prefix constraints that propagate terminal reward to intermediate prefixes via suffix-absorbed backups (a convex combination of max and soft log-sum-exp aggregations over the suffix), while stopping gradients through the termination head in the auxiliary branch to prevent the termination drift observed in SubTB. Second, Submodular Replay (SubM) replaces reward-prioritized replay with greedy maximization of a facility-location + length-bin coverage submodular objective over the union of the current buffer and a new batch. The method is evaluated primarily on scaffold-conditioned SMILES generation (QED), with supporting experiments on Expr24 arithmetic generation and a small CommonGen diagnostic subset (10 concept-sets, N=320 samples each). Their main claim is that RapTB+SubM simultaneously improves reward quality, molecular diversity, and long-horizon coverage relative to TB and SubTB, while maintaining high chemical validty.

Strengths And Weaknesses:
Strengths
Soundness. For me, the mechanistic diagnosis of SubTB's failure on terminable prefix trees is defintiely their strongest technical contribution. The analysis in Appendix C.6 looks correct: in the SubTB window residual (Eq. 33), when rewards are sparse or weakly prefix-dependent, the difference term carries the dominant gradient signal, and because each termination logit participates in O(N²) windows simultaneously, the optimizer can reduce the aggregate squared loss by globally shifting the termination head rather than improving token-level transitions. This is a reproducible failure mode; they nicely demonstrated this in Table 4 (SubTB log p_term = −79.638 under RP, −86.415 under Oracle) and Table 5 (SubTB delta log p_term = −28.32, saturating length at 20.00 on CommonGen). The ablation in Table 4 confirms this, where restricting SubTB to rooted windows and reintroducing Z_theta recovers accuracy to ~100%, directly validating the diagnosis.

Originality. The absorbed suffix backup (Eqs. 22–24) is a super clean variance-reduction method. Replacing the terminal stop-reward with a surrogate prefix credit is well-motivated by the variance decomposition in Appendix C.5 (Eq. 32), and the exponential distance discounting in Eq. (25) is a nice way to downweight distant suffix evidence. The combination of this with gradient stopping on the termination head in the auxiliary branch is a minima fix that directly addresses the identified failure channel.

Presentation. Figure 1 and Figure 3 are definitely the paper's clearest justifications. Figure 1 makes the O(1)/O(N^2)/O(N) complexity comparison between TB/SubTB/RapTB immediately legible alongside qualitative SMILES samples. Figure 3's prefix-collapse diagnostics (survival, entropy, top-1 mass by prefix depth) are (and underused) evaluation methodsthat the paper uses well to distinguish between terminal diversity and genuine prefix-level branching.

Weaknesses
The paper's primary experimental setting is scaffold-conditioned SMILES generation with a fixed vocabulary (the EBNF grammar in Figure 4) and a single scalar reward (QED). The Expr24 task uses a minimal vocabulary with a binary sparse reward. CommonGen uses 10 concept-sets with N=320 samples, which is explicitly described as a "diagnostic subset" rather than a benchmark. This means the paper's empirical support for RapTB as a general LLM-GFlowNet training objective rests almost entirely on one domain (molecular SMILES) with one reward (QED) and one model (Llama-3.2-1B with LoRA rank 16). The paper's claims, however, are stated at the level of terminable LLM-GFlowNets generally. The failure modes identified (prefix collapse and termination drift) arise from the structure of the prefix tree and the SubTB objective, not from anything specific to SMILES. Domains that would test these claims more directly include: biological sequence generation with unique vocabularies (amino acids, nucleotides) and non-differentiable reward functions; code generation with structural validity constraints and functional correctness rewards; or mathematical reasoning with verifiable sparse rewards analogous to Expr24 but at longer horizons and larger output spaces. None of these are evaluated. The practical significance of the method is therefore difficult for me to assess beyond the molecular generation setting.

Table 3 (Expr24) shows that SubM applied to TB alone (TB+SubM) achieves NormCov=0.100 and Unique=642.0, while RapTB without SubM achieves NormCov of 0.039 and Unique=246.7 under RP. So in other words, SubM on top of vanilla TB outperforms RapTB alone on coverage by a large margin. The paper acknowledges this in Appendix A.1 for SMILES ("applying SubM to TB also yields substantial improvements"), but unfortunatley does not discuss it in the main text as a challenge to the narrative that RapTB is the primary driver of improvement. The strongest result in the paper, that RapTB+SubM reaching NormCov=0.209, doubling TB+SubM, is really strong, but the paper does not clearly characterize when RapTB provides additive benefit over SubM alone vs. when SubM dominates.

RapTB introduces seven task-specific hyperparameters beyond the base TB objective: aux weight (eta), distance discount (gamma), mix weight between max and soft backup (alpha), soft backup temperature (beta), distance penalty (rho), minimum prefix depth (k_min, with a linear schedule), and auxiliary horizon cap (K). From Table 23, these differ meaningfully between tasks (for example, alpha is 0.5 for SMILES and 0.8 for Expr24, rho is 0.1 for SMILES and 0.5 for Expr24) yet the Table 6 results only show ablations for reward absorption on/off and max-only vs. soft-only backups, leaving alpha, beta, rho, eta, and gamma entirely unablated. As such, the sensitivity of the main results to these choices is not known. The k_min schedule, which linearly anneals the minimum eligible prefix depth from 5 to 2 over 5000 training steps for SMILES, is a non-trivial design choice that determines which prefixes receive auxiliary supervision during the critical early phase of training, and the authors never study its effect.

Soundness: 2: fair
Presentation: 3: good
Significance: 4: excellent
Originality: 4: excellent
Key Questions For Authors:
Table 3 shows TB+SubM achieves NormCov of 0.100 on Expr24 while RapTB without SubM reaches only 0.039 under RP -- SubM alone outperforms RapTB alone on coverage by a large margin. Under what conditions does RapTB provide meaningful additive benefit over SubM alone, and is there a task or regime where RapTB without SubM is the dominant contributor?

RapTB introduces 7 hyperparameters (eta, gamma, alpha, beta, rho, k_min schedule, K) that differ between SMILES and Expr24 (e.g., alpha=0.5 vs. 0.8, rho=0.1 vs. 0.5). How sensitive are the main results in Tables 1 and 3 to these choices, particularly beta and rho, which directly control the variance-reduction behavior central to the method's theoretical motivation?

All primary results use a single model (Llama-3.2-1B, LoRA rank 16), a single molecular domain (QED), and sequence lengths of at most 15 tokens. Can the authors clarify for me whether RapTB's gains over TB hold on longer sequences, larger vocabularies, or non-SMILES domains such as amino acid or nucleotide generation with non-differentiable reward functions?

The rooted residual and absorbed suffix backup are structurally analogous to advantage estimation with multi-step returns (GAE) in actor-critic methods. Can the authors let me know if RapTB equivalent to a particular GAE estimator instantiated within the GFlowNet framework, or does the GFlowNet partition function constraint introduce a substantive difference?

Limitations:
Yes.

Overall Recommendation: 4: Weak accept: Technically solid paper that advances at least one sub-area of AI, with a contribution that others are likely to build on, but with some weaknesses that limit its impact (e.g., limited evaluation). Please use sparingly.
Confidence: 3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.
Compliance With LLM Reviewing Policy: Affirmed.
Code Of Conduct Acknowledgement: Affirmed.
Add:
Official Review of Submission13383 by Reviewer JxzD
Official Reviewby Reviewer JxzD11 Mar 2026, 18:39 (modified: 24 Mar 2026, 06:21)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer JxzDRevisions
Summary:
The authors propose a new objective and replay buffer to GFN training to improve credit assignment and address mode collapse. Instead of considering constraints on all subtrajectories, RapTB only considers trajectories rooted at $s_0. In addition, it provides an additional absorbed suffix reward target consisting of an interpolation of a hard/soft operator applied to downstream rewards. Finally, to address the mode collapse issue, they propose selecting a buffer that maximizes a submodular function which uses a mix of a quality term, a diversity term and a length diversity term.

Experimentally, the authors finetune Llama-3.2-1B for a variety of sequence generation tasks (SMILES, arithmetic and sentence generation). They find that RapTB achieves better diversity while maintaining similar quality in most task. Particularly, the length diversity and prefix diversity of RapTB is higher.

Strengths And Weaknesses:
Strengths

The paper proposes a series of novel solutions to address issues with GFN training
The author train a large scale model on a variety of tasks. The experimental results are promising and indicate that their method does provide a meaningful improvement to diversity while maintaining or improving quality.
The authors ablate the different additional components of their method, identifying the impact of the changes
The submodular replay is clever and generally applicable (e.g. by varying the objective)
Weaknesses

There is a lack of analysis of the new objective and whether it will converge to the global optimal.
The absorbed suffix backups should be explained and motivated more clearly in the main text. It is unclear to me why it is a good idea to mix the intermediate task rewards with some sort of interpolation between a hard and soft operator.
The methods makes many design choices and requires multiple new hyperparameters. It would interesting to examine robustness to these hyperparameters or have a good way to set them.
I am willing to increase my score if my questions are answered.

Soundness: 2: fair
Presentation: 3: good
Significance: 3: good
Originality: 2: fair
Key Questions For Authors:
I am a bit confused about the issue of overlapping constraints in SubTB. My understanding was that the SubTB constraint related flow at a state
 to flow at a state
 through the intermediate probabilities. Why are the termination probabilities included for each subtrajectory? (I see this is explained in the appendix, have you tested with learning the state flow values and if this is still an issue?)
Could you explain the prefix survival metric in Fig.3.?
Have you looked at longer sequence generation tasks (AMP/GFP) which are typically tested by GFNs?
The finetuning of a LLM for these tasks is interesting, what is the reason for this and have you also tested training a smaller model from scratch?
Limitations:
yes

Overall Recommendation: 4: Weak accept: Technically solid paper that advances at least one sub-area of AI, with a contribution that others are likely to build on, but with some weaknesses that limit its impact (e.g., limited evaluation). Please use sparingly.
Confidence: 4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.
Compliance With LLM Reviewing Policy: Affirmed.
Code Of Conduct Acknowledgement: Affirmed.
Add:
Official Review of Submission13383 by Reviewer QHmk
Official Reviewby Reviewer QHmk10 Mar 2026, 06:16 (modified: 24 Mar 2026, 06:21)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer QHmkRevisions
Summary:
The paper presents RapTB — a GFlowNet-based method for fine-tuning language models to approximate reward-proportional posteriors. It combines a modification of trajectory balance objective that provides supervision over all intermediate prefixes of a sequence, as well as a submodular strategy for picking samples from the replay buffer for optimization that promotes both high reward and diversity. Experimental evaluation shows problems of previous GFlowNet-based approaches, such as prefix collapse and length bias, and demonstrates the improved performance of the proposed method in comparison to them.

Strengths And Weaknesses:
Strengths:

Investigating GFlowNet-based approaches for fine-tuning LLMs is a highly interesting and relevant topic to the research community.
The paper presents an in-depth experimental evaluation and ablation studies across a number of tasks, demonstrating failure modes of previous approaches based on TB and SubTB objectives, as well as showing empirical improvements of the proposed approach across various metrics. I found it well-crafted and insightful.
Weaknesses:

The paper fails to put itself in the broader context of RL literature related to GFlowNets and LLM fine-tuning. The authors state: "In contrast to reward-maximizing reinforcement learning, the objective of GFlowNets is distributional: spread probability mass across many high-reward modes in proportion to reward, rather than concentrating on a single optimum." While this view was shared by some of the earlier works on GFlowNets, it was shown that GFlowNet training can be equivalently reformulated as an RL problem [1, 2]. Moreover, it is well-known that sampling from the reward distribution defined as a product of pre-trained model and and a reward function (which is the case studied in the paper) is equivalent to fine-tuning a pre-trained model to solve a KL-regularized RL problem, see e.g. Equation 1 and Equation 2 in [3]. While I am not an expert in LLMs, up to my knowledge this is the standard way to do RL fine-tuning in modern LLM literature. I believe that it is crucial that the problem being solved and the contributions of the paper are properly put in this context.
Following the previous point, I believe that it will be great to add some standard RL approaches like PPO and GRPO as baselines to the experimental evaluation in the paper.
An important baseline from GFlowNet literature is missing: TBA [3]. This paper also uses trajectory balance objective to fine-tune language models, and also utilizes a replay buffer with a specific sampling strategy. This is the most direct competitor from previous literature, which follows a very similar training pipeline to the approach proposed by the authors, this I believe it should also be added to experimental evaluation and discussed in the paper.
I believe that the section presenting the proposed RapTB loss should be expanded. When reading the text, I struggled to understand the design choices behind the proposed training objective. A detailed mathematical derivation would be very helpful in my opinion (see Questions).
A more minor point, but the paper uses some terminology across the text without timely explaining its meaning, or referencing some previous works where it is defined. For example, termination drift is mentioned across different parts of the paper, but its meaning is only explained in the experiments section, which complicates readability.
The paper presents interesting contributions, but proper contextualization and framing within previous literature, as well as experimental comparison to important missing baselines, is necessary to recommend acceptance in my opinion. I was thinking between reject and weak reject for the score, but decided to settle on reject for the time being. I am willing to increase my score if my concerns are addressed.

References:
[1] Generative Flow Networks as Entropy-Regularized RL. Daniil Tiapkin, Nikita Morozov, Alexey Naumov, Dmitry Vetrov. AISTATS 2024.
[2] Discrete Probabilistic Inference as Control in Multi-path Environments. Tristan Deleu, Padideh Nouri, Nikolay Malkin, Doina Precup, Yoshua Bengio. UAI 2024.
[3] Trajectory Balance with Asynchrony: Decoupling Exploration and Learning for Fast, Scalable LLM Post-Training. Brian Bartoldson, Siddarth Venkatraman, James Diffenderfer, Moksh Jain, Tal Ben-Nun, Seanie Lee, Minsu Kim, Johan Obando-Ceron, Yoshua Bengio, Bhavya Kailkhura. NeurIPS 2025.

Soundness: 3: good
Presentation: 2: fair
Significance: 2: fair
Originality: 3: good
Key Questions For Authors:
Could you please provide a more in-depth mathematical explanation of the proposed RapTB loss? Are the credit bounds (Equations 6, 7, 8) introduced as a means to avoid having a learnable
 scalar in TB loss? Why are they chosen in such a way? Does reaching the global optimum of the loss guarantee that the model samples from the distribution of interest (line 102)?
In my opinion, a much simpler loss function that can be used as a baseline for ablation is just averaging the usual TB loss over all prefixes of the sequence (since every prefix can be considered a complete trajectory with an addition of a stop token). Is this design choice viable for the provided experiments? Will the proposed approach produce meaningfully different results from it?
Limitations:
No concern.

Overall Recommendation: 2: Reject: For instance, a paper with technical flaws, weak evaluation, inadequate reproducibility, incompletely addressed ethical considerations, or writing so poor that it is not possible to understand its key claims.
Confidence: 4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.
Compliance With LLM Reviewing Policy: Affirmed.
Code Of Conduct Acknowledgement: Affirmed.
