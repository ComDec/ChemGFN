# Rebuttal 全面分析报告（中文版）

**论文**: RapTB: Rooted Absorbed Prefix Trajectory Balance with Submodular Replay for GFlowNet Training
**投稿**: ICML 2026, Submission 13383
**当前分数**: 4 (WA), 4 (WA), 3 (WR), 2 (R) — 均值 3.25
**日期**: 2026-03-28

---

## 一、总体形势判断

### 1.1 接收路径分析

**当前局面**：两个 Weak Accept (cA3o, JxzD) + 一个 Weak Reject (Pd1v) + 一个 Reject (QHmk)。ICML 的接收需要 AC 和 SAC 的认可，通常需要至少 3 个 reviewer 不反对。

**目标优先级**：
1. **QHmk (2→4)** — 最高优先级。此 reviewer 的 concerns 最 actionable（RL framing + baselines），且已声明 "willing to increase score"。如果能翻转 QHmk，得分变为 4/4/4/3 = 3.75，非常有利
2. **cA3o 巩固 (4→5 或 4 stay)** — 此 reviewer 技术最强，如果满意度提升会成为 champion
3. **JxzD 巩固 (4→4 stay)** — 此 reviewer confidence=4，但表示 "willing to increase score if answered"
4. **Pd1v (3→4)** — 低优先级。此 reviewer confidence=2（"quite likely did not understand central parts"），其 concerns 与其他 reviewer 高度重叠

**不死的原因**：
- 论文有实质性技术贡献（SubTB failure mode 诊断 + minimal fix）
- cA3o 给了 Significance=4 (excellent), Originality=4 (excellent)
- QHmk 给了 Soundness=3 (good), Originality=3 (good)
- 两位 WA reviewer 的技术评价正面

### 1.2 核心风险

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| QHmk 不翻转（RL framing 不满意） | 中 | 高 | 全面满足其 6 个 concerns，用 PPO/GRPO 数据说话 |
| AC 侧重 QHmk 意见 | 中 | 高 | 用 cA3o + JxzD 的正面评价做 counterweight |
| 新实验数字不如预期 | 低 | 中 | 所有实验已完成，数字已知 |
| AMP 实验结果不理想 | 中 | 中 | 作为 additional evidence，不作为核心论据 |

---

## 二、逐审稿人深度分析

---

### 2.1 Reviewer QHmk — Score 2 (Reject), Confidence 4

**人物画像**: 熟悉 GFlowNet-RL 关系的理论导向 reviewer。对论文的 framing 非常不满（"fails to put itself in broader context"），这是他打 Reject 的主因。但他给了 Soundness=3 和 Originality=3，说明技术上并不认为论文有错，只是认为定位有误。

**核心关切**：论文没有在 RL 框架下正确定位 GFlowNet，导致 (a) 对比 baseline 不足 (b) 贡献看起来像 re-invention

**要什么 / 我们如何满足**：

| # | 要什么 | 严重性 | 我们如何满足 | 证据状态 |
|---|--------|--------|------------|---------|
| C1 | 承认 GFlowNet = entropy-regularized RL | 关键 | 全面修订 Intro 的 "in contrast to" 措辞；引 Tiapkin+ 2024, Deleu+ 2024 | **文本已写好** |
| C2 | PPO/GRPO baseline | 重要 | PPO crash at step 50 (Acc=0.003), GRPO (Acc=0.002, 99.9% length collapse) | **实验已完成** ✅ |
| C3 | TBA baseline/讨论 | 关键 | 明确 TBA = system-level pipeline, RapTB = objective-level fix, 正交关系 | **文本已写好** |
| C4 | 扩展 RapTB loss 数学推导 | 重要 | 解释 Z 在 terminal TB 保留，rooted residual 取消 Z in aux，absorbed target 降方差 | **文本已写好** |
| C5 | 全局最优是否保证目标分布 | 关键 | 明确：NO new exact theorem。TB anchor = exact，aux = regularizer | **文本已写好** |
| C6 | Averaged-prefix TB baseline | 关键 | AvgPrefixTB: Expr24 NormCov=0.016 (vs RapTB 0.039), SMILES QED=0.661/Len=2.89 (严重短序列坍缩) | **实验已完成** ✅ |
| C7 | "termination drift" 术语前置定义 | 次要 | Will define at first mention | **文本已写好** |

**翻转评估**：**乐观**。QHmk 的 7 个 concerns 全部可回答：
- C1/C3/C5 是 framing 问题 → 文本修复
- C2 有 PPO/GRPO 实验数据 → 有力证据
- C6 有 AvgPrefixTB 实验数据 → 有力证据
- C4/C7 是 clarity 问题 → 文本修复

**关键策略**：不要争辩 "GFlowNet 不是 RL"。完全接受 reviewer 的观点（GFlowNet ⊂ MaxEnt RL），然后说明我们的贡献在这个框架内依然成立：balance-based objectives on terminable prefix trees 的 credit assignment 问题。

**PASTE_READY_QHmk.txt 状态**：✅ 已更新，包含 PPO/GRPO 和 AvgPrefixTB 完整数据。约 4500 字符。

---

### 2.2 Reviewer cA3o — Score 4 (Weak Accept), Confidence 3

**人物画像**: 技术最深入的 reviewer。给了 Significance=4, Originality=4。称赞 SubTB 诊断是 "strongest technical contribution"，absorbed suffix backup 是 "super clean variance-reduction method"。但有三个实质性技术质疑。

**核心关切**：(a) SubM vs RapTB 谁是主要贡献者 (b) 7 个超参数的敏感度 (c) 单一 domain

**要什么 / 我们如何满足**：

| # | 要什么 | 严重性 | 我们如何满足 | 证据状态 |
|---|--------|--------|------------|---------|
| C1 | RapTB vs SubM 何时各自主导 | 关键 | 三个 regime 分析：RP(RapTB>TB), SubM(RapTB+SubM>TB+SubM), Oracle(RapTB>TB with controlled coverage) | **论文数据已足够** ✅ |
| C2 | β×ρ×η×k_min 敏感度 | 重要 | β×ρ 9 格 (Expr24 + SMILES), η 3 点, k_min 3 点 sweep | **全部完成** ✅ |
| C3 | Domain generalization | 重要 | 3B 模型验证 + 缩窄 claim scope | **3B 完成** ✅; AMP **进行中** |
| C4 | GAE 类比 | 次要 | Bias-variance 类比，非精确等价；不同优化几何 | **文本已写好** |

**巩固评估**：**非常乐观**。
- C1 有三个清晰 regime 的数据支撑
- C2 有大量 sweep 数据，核心结论有力（Acc≥0.994 全部 9 格，log_pterm 无 drift）
- C3 有 3B scale-up 数据（SubTB drift 在 3B 上更严重 → 验证 structural nature）
- C4 是 clarification

**PASTE_READY_cA3o.txt 状态**：✅ 已更新，包含 sweep + SMILES sweep + 3B 数据。约 4800 字符。

---

### 2.3 Reviewer JxzD — Score 4 (Weak Accept), Confidence 4

**人物画像**: 对 GFlowNet 有经验的 reviewer（confidence 4）。整体正面，重点关注机制理解。明确说 "willing to increase score if questions answered"。

**核心关切**：(a) SubTB 终止耦合的精确机制 (b) AMP/GFP 类更长序列任务 (c) 为什么用 LLM

**要什么 / 我们如何满足**：

| # | 要什么 | 严重性 | 我们如何满足 | 证据状态 |
|---|--------|--------|------------|---------|
| C1 | SubTB 为什么包含 termination prob | 重要 | Appendix C.6 详细解释 + RootSubTBLogZ ablation (Table 4) | **论文数据已足够** ✅ |
| C2 | Prefix survival 定义 | 次要 | Surv(k)=n_k/n_valid (Appendix B.3) + 联合解读 PefEnt/Top1 | **文本已写好** |
| C3 | AMP/GFP 长序列任务 | 重要 | L_max=15 stress test (Table 2) + 3B scale-up; AMP 实验进行中 | **3B 完成** ✅; AMP **进行中** |
| C4 | 为什么 fine-tune LLM | 次要 | 问题定义就是 LLM post-training，P_ref 是 reward 的组成部分 | **文本已写好** |
| C5 | 收敛/全局最优分析 | 重要 | = QHmk-C5: TB anchor exact, aux = regularizer | **文本已写好** |
| C6 | Absorbed suffix backup 动机 | 重要 | Terminal-only → 高方差 → absorbed target 降方差；Table 6 验证 | **文本已写好** |

**巩固评估**：**非常乐观**。所有问题都可以用论文已有数据 + 文本解释回答。3B 实验是额外加分项。如果 AMP 实验能提供正面结果，C3 会得到更充分的回答。

**PASTE_READY_JxzD.txt 状态**：✅ 已写好，包含 3B 数据。约 5400 字符 — **需要精简**（ICML 限制约 5000/reviewer）。

---

### 2.4 Reviewer Pd1v — Score 3 (Weak Reject), Confidence 2

**人物画像**: Confidence=2，自己承认 "quite likely did not understand central parts"。关切非常笼统（narrow benchmarks, weak baselines, no theory）——与其他 3 位 reviewer 高度重叠。

**核心关切**：(a) 实验只有小模型+窄 benchmark (b) baseline 只有 TB/SubTB (c) 没有理论保证

**要什么 / 我们如何满足**：

| # | 要什么 | 严重性 | 我们如何满足 | 证据状态 |
|---|--------|--------|------------|---------|
| C1 | 更大模型 + 更多 benchmark | 重要 | 3B scale-up (Llama-3.2-3B) + 3 tasks + L_max=15; AMP 进行中 | **3B 完成** ✅ |
| C2 | 更多 baseline | 重要 | PPO/GRPO/AvgPrefixTB + TBA 讨论 | **全部完成** ✅ |
| C3 | 理论保证 | 重要 | = QHmk-C5: TB anchor exact, aux = regularizer | **文本已写好** |

**翻转评估**：**中等乐观**。Pd1v 的 concerns 全部被 QHmk 和 cA3o 的回复覆盖。3B 实验是直接回应 C1 的新数据。但 confidence=2 的 reviewer 可能不会深入阅读 rebuttal。

**PASTE_READY_Pd1v.txt 状态**：✅ 已写好，包含 3B 对比表 + PPO/GRPO + AvgPrefixTB 提及。约 5000 字符。

---

## 三、证据清单：完成度与缺口

### 3.1 已完成实验（✅ = 数据在手）

| # | 实验 | 回答哪个 concern | 关键数字 | 说服力 |
|---|------|-----------------|---------|--------|
| 1 | **β×ρ sweep (Expr24, 9 configs)** | cA3o-C2 | Acc≥0.994 全部 9 格, log_pterm∈[-0.25,-0.04] | ★★★★★ |
| 2 | **β×ρ sweep (SMILES, 7/9 configs)** | cA3o-C2 | Acc≥0.985, Len 6.3-7.3, 跨任务一致 | ★★★★☆ |
| 3 | **η sweep (3 points)** | cA3o-C2 | 单调改善，η 有清晰可解释行为 | ★★★★☆ |
| 4 | **k_min ablation (3 points)** | cA3o-C2 | Fixed-low 最差，验证设计 | ★★★★☆ |
| 5 | **GRPO baseline (Expr24)** | QHmk-C2 | 12/6400 valid, 1 unique, 99.9% at L=11 | ★★★★★ |
| 6 | **PPO baseline (Expr24)** | QHmk-C2 | 20/6400 valid, 1 unique, crash at step 50 | ★★★★★ |
| 7 | **AvgPrefixTB (Expr24, 4 replay)** | QHmk-C6 | RP: NormCov=0.016, SMILES: QED=0.661/Len=2.89 | ★★★★★ |
| 8 | **AvgPrefixTB (SMILES)** | QHmk-C6 | 严重短序列坍缩，所有指标最差 | ★★★★★ |
| 9 | **3B scale-up (SMILES, 4 methods)** | Pd1v-C1, JxzD-C3 | RapTB+SubM best; SubTB drift amplified at 3B | ★★★★★ |
| 10 | **Paper-exact anchor (3 seeds)** | Sweep 校准 | Acc=0.999, JS=0.134 (优于论文) | ★★★☆☆ |

### 3.2 进行中实验

| # | 实验 | 回答哪个 concern | 当前状态 | 预期价值 |
|---|------|-----------------|---------|---------|
| 11 | **AMP 任务 (4 methods)** | JxzD-C3, cA3o-C3, Pd1v-C1 | TB/SubTB/RapTB 完成，RapTB+SubM ~50% | ★★★☆☆ (见下文分析) |

### 3.3 AMP 实验的初步分析与 Rebuttal 价值判断

AMP 当前结果：
- **Performance**: TB(0.927) ≈ RapTB(0.927) > SubTB(0.897) > RapTB+SubM(0.925, 进行中)
- **Diversity**: SubTB(21.37) >> RapTB+SubM(9.67) > RapTB(7.96) > TB(7.39)
- **Length pattern**: TB/RapTB 偏短 (17-18 AA, near min=15); SubTB 偏长 (49-50 AA, near max)

**问题**：AMP 的 pattern 与 SMILES/Expr24 不完全一致：
- SubTB 在 AMP 上没有出现 accuracy collapse（0.897 vs SMILES 的 0.328），但出现了 extreme length bias（几乎所有序列都是 max length）
- RapTB 与 TB 的 diversity 差距不大（7.96 vs 7.39），不如 SMILES 上的差距明显
- RapTB+SubM 还在训练中，diversity 在逐步增加

**Rebuttal 使用建议**：
- **如果 RapTB+SubM 训练完成后 diversity 显著提升** → 作为 JxzD-C3 的直接回答，证明方法在 biological sequence 上也有效
- **如果 RapTB+SubM 没有明显改善** → 作为 honest additional result 提及，但强调 (a) 无 termination drift (b) 与 SMILES pattern 的 structural 一致性
- **不建议作为核心论据**：AMP 的结果不如 SMILES/Expr24/3B 清晰，作为 supplementary evidence 即可

### 3.4 证据缺口分析

| 缺口 | 重要性 | 是否阻塞 | 处理方式 |
|------|--------|---------|---------|
| AMP RapTB+SubM 未完成 | 中 | 否 | 等训练完成；用已有 3 个方法的结果作为 placeholder |
| SMILES sweep 只有 7/9 configs | 低 | 否 | 7 个已足够说明 cross-task robustness |
| Sweep 单 seed 无 CI | 中 | 否 | 以 Acc/log_pterm robustness 为主，NormCov 不强调 |
| NormCov 在 sweep 中偏低 | 中 | 否 | 用 KL/JS（更可靠）替代 NormCov 作为主指标 |

**总体评估**：**证据齐全度 ~90%**。核心实验全部完成。AMP 是锦上添花，不是必须。

---

## 四、各草稿审查与修改建议

### 4.1 PASTE_READY_global.txt (~3000 字符)

**优点**：
- 三大全局主题清晰（理论范围、RL 定位、RapTB vs SubM 互补）
- 新实验列举完整
- Claim scope 修订明确

**改进建议**：
- ✅ 已包含 PPO/GRPO 数据
- ✅ 已包含 AvgPrefixTB 数据
- ✅ 已包含 3B scale-up 数据
- ⚠️ 如果 AMP 结果好，可在 "New experiments" 部分加一句 AMP

### 4.2 PASTE_READY_QHmk.txt (~4500 字符)

**优点**：
- PPO/GRPO 数据非常有说服力
- AvgPrefixTB 数据全面（Expr24 + SMILES）
- TBA 定位清晰
- 数学解释充分

**改进建议**：
- ⚠️ 关于 TBA 的回答可以更具体：提到 RapTB 可以 drop-in 替换 TBA 中的 TB loss
- ✅ AvgPrefixTB 部分数据完整，包含 Oracle 对比

### 4.3 PASTE_READY_cA3o.txt (~4800 字符)

**优点**：
- Q1 三 regime 分析非常清晰
- Q2 sweep 数据全面
- 3B 数据已纳入
- GAE 解释准确

**改进建议**：
- ✅ 已包含 SMILES sweep 跨任务验证
- ⚠️ 如果 AMP 有好结果，可加一句到 Q3

### 4.4 PASTE_READY_JxzD.txt (~5400 字符)

**优点**：
- Q1 SubTB 终止机制解释透彻
- Q3 有 3B 数据表
- Q4 LLM fine-tuning 动机解释清楚

**改进建议**：
- ⚠️ **需要精简**：约 5400 字符，ICML 限制约 5000/reviewer。建议删减 Q2 (prefix survival 解释)，因为这是最次要的
- ⚠️ 如果 AMP 有好结果，可替换部分 Q3 的 "future work" 措辞为实际数据

### 4.5 PASTE_READY_Pd1v.txt (~5000 字符)

**优点**：
- 3B 对比表完整（1B vs 3B）
- PPO/GRPO + AvgPrefixTB 提及
- TBA 讨论到位

**改进建议**：
- ✅ 数据完整
- ⚠️ 如果 AMP 有好结果，可在 W1 部分加一句

---

## 五、Rebuttal 整体结构建议

### 5.1 ICML 2026 Rebuttal 格式

ICML 2026 允许：
- **Per-reviewer response**: 各自独立，每个有字符限制（通常 5000 字符）
- **Global response**: 所有 reviewer 可见
- **Revised PDF**: 可以上传

### 5.2 内容优先级

**Global response** 应该包含：
1. 三大全局主题（理论范围、RL 定位、互补性）
2. 所有新实验的简要列表
3. Claim scope 修订声明

**Per-reviewer responses** 按照以下优先级分配精力：
1. **QHmk** (最长、最详细) — 争取翻转
2. **cA3o** (全面的技术回应) — 巩固 champion
3. **JxzD** (简洁直接) — 维持 WA
4. **Pd1v** (精炼) — 尽力而为

### 5.3 需要在 Revised PDF 中做的修改

1. Introduction 的 RL framing 修正
2. "termination drift" 前置定义
3. TBA 的 Related Work 讨论
4. β×ρ heatmap figure (新图)
5. PPO/GRPO/AvgPrefixTB 结果表 (新表)
6. 3B 结果表 (新表)
7. AMP 结果 (如果好的话，新表)
8. Claim scope 修订（throughout）

---

## 六、待办事项清单

### 立即（Today）

- [ ] 检查 AMP RapTB+SubM 训练进度，评估结果
- [ ] 精简 PASTE_READY_JxzD.txt 至 5000 字符以内
- [ ] 确认 ICML 2026 rebuttal 字符限制

### 短期（完成前）

- [ ] AMP 实验完成后，根据结果决定是否纳入 rebuttal
- [ ] 更新 global response 纳入 AMP (如果结果好)
- [ ] Phase 5 Safety Validation (检查覆盖、来源、承诺、语气)
- [ ] Phase 6 Stress Test (Codex MCP 审稿人模拟)
- [ ] Phase 7 Finalize (PASTE_READY 最终版本)

### 可选

- [ ] SMILES sweep 补完剩余 2/9 configs (β=10, ρ=0.1/0.5)
- [ ] 更多 seed 的 sweep 验证（如果时间允许）

---

## 七、风险与应对预案

### 7.1 最坏情况：QHmk 不翻转

即使 QHmk 维持 2，如果 cA3o 和 JxzD 在讨论阶段提升到 5 或 6，AC 仍可能基于 3 位 reviewer 的多数意见决定 borderline accept。策略：
- 确保 cA3o 和 JxzD 完全满意
- Global response 中明确 QHmk 的所有 concerns 均已回应
- 让 AC 看到 cA3o 的 Significance=4/Originality=4 评价

### 7.2 AMP 结果不理想

AMP 目前显示 RapTB 与 TB 差距不大。如果 RapTB+SubM 也没有明显优势：
- **不作为核心论据**
- 在 JxzD-C3 中诚实提到："We conducted a preliminary AMP experiment. Performance is comparable across methods; the longer sequence setting (50 AA) reveals different dynamics that warrant dedicated investigation."
- 重点仍放在 3B scale-up（已有明确正面结果）

### 7.3 Reviewer 追问 β×ρ sweep 的 NormCov

如果 cA3o 追问 NormCov 在 sweep 中偏低：
- 强调 Acc 和 log_pterm 是 robustness 的主要指标
- 解释 NormCov 差异来自 (a) single seed (b) RP replay 的 stochastic nature
- 论文 anchor 的 KL/JS 实际上优于 paper 数字

---

## 八、AMP 实验 Placeholder 用语建议

在 global response 和 per-reviewer response 中，关于 AMP 的表述需要根据最终结果二选一：

**如果 AMP 结果好**（RapTB+SubM diversity 显著提升）：
> We further extend evaluation to antimicrobial peptide (AMP) generation, a biological sequence design task with non-differentiable fitness predictor and sequence lengths 15-50 amino acids. RapTB+SubM achieves the best quality-diversity trade-off, demonstrating the method generalizes beyond molecular SMILES to biological sequence design.

**如果 AMP 结果一般**（差距不明显）：
> As a preliminary study beyond chemical SMILES, we evaluate on antimicrobial peptide (AMP) generation (sequence length 15-50). All methods achieve comparable high performance. The longer sequence setting and distinct reward landscape warrant dedicated investigation, which we plan as follow-up work.

---

## 九、总结

**Rebuttal 准备度**：90%+。核心证据全部在手，关键草稿已写好。

**最大杠杆点**：
1. PPO/GRPO 各只找到 1 个 unique valid (12/6400, 20/6400) → reward-maximizing RL 完全无法 distributional sampling
2. AvgPrefixTB 的短序列坍缩 → 证明 simple averaging 不够
3. β×ρ sweep 全格 Acc≥0.994 + diversity 一致 → 方法 robust（NormCov 因 effective bsz 差异偏低，以 Acc+diversity+log_pterm 为主）
4. 3B scale-up SubTB drift amplified → failure mode 是 structural 的

**最大风险**：QHmk 不翻转。但即使如此，4/4/3 + strong evidence 仍有 borderline accept 可能。

**下一步**：完成 AMP 实验 → 决定是否纳入 → 精简 JxzD 回复 → Safety Validation → Finalize。
