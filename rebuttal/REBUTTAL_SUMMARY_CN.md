# Rebuttal 总结报告（中文）— v2 文献加强版

**论文**: RapTB: Rooted Absorbed Prefix Trajectory Balance with Submodular Replay
**投稿**: ICML 2026, Submission 13383
**当前分数**: 4, 4, 3, 2 → 目标全部 ≥ 4

---

## 一、审稿人概况与翻转策略

| 审稿人 | 分数 | 核心关切 | 翻转难度 |
|--------|------|---------|---------|
| **QHmk** | 2 (Reject) | RL framing + PPO/GRPO/TBA baseline + AvgPrefixTB + 全局最优 | **高优先级** |
| **Pd1v** | 3 (Weak Reject) | 窄 benchmark + 弱 baseline + 无理论 | **中等** — confidence=2 |
| **cA3o** | 4 (Weak Accept) | SubM vs RapTB 角色 + 超参敏感度 + 域泛化 + GAE类比 | **巩固** |
| **JxzD** | 4 (Weak Accept) | SubTB 机制 + 长序列 + 为什么用 LLM + 全局最优 | **巩固** |

---

## 二、三个核心论点的文献支撑

### 2.1 RapTB 全局最优性（两个角度）

**角度一：结构分解 — RapTB = TB + 正则化**

TB 的不动点性质由 Malkin et al. (2022, NeurIPS, Theorem 1) 建立：TB loss = 0 当且仅当 $q_\theta^\top(x) \propto R(x)$。Tiapkin et al. (2024, AISTATS, Oral, Theorem 1) 进一步证明 GFlowNet 前向策略等价于 MaxEnt RL 最优策略。RapTB 保留 TB 作为唯一精确平衡条件，辅助项是有限权重的正则化器。

**文献类比**：
- **LED-GFN** (Jang et al. ICLR 2024 Oral)：对流重参数化 + 平滑正则化器。有限正则化权重引入偏差，权重→0 恢复精确最优。
- **损失形状修改** (arXiv 2410.02596)：改变回归损失形状改变最小化的散度，但 loss=0 处的全局最小值不变。
- **控制变量** (da Silva et al. NeurIPS 2024)：无偏方差缩减——理论上最干净的方法。
- **TBA** (Bartoldson et al. NeurIPS 2025)：通过增加采样和控制变量缩减方差，不移动最优解。

**角度二：零点一致性 — TB=0 时 L_aux 也=0**

在可终止前缀树（Deleu et al. UAI 2022; Hu et al. ICLR 2024）中：
1. 当 TB 条件对所有终止点成立：$\Delta_k^{TB}(\xi) = 0$ 对所有 $k$
2. 根子轨迹残差 $\bar{\Delta}_k = \Delta_k^{TB} - \Delta_0^{TB} = 0$
3. 吸收修正项消失
4. 因此 $\mathcal{L}_{aux} = 0$

**与 SubTB 的类比**：SubTB 通过伸缩论证共享 TB 的不动点（Madan et al. ICML 2023）——子轨迹残差伸缩到完整 TB 残差。RapTB 通过不同机制共享不动点：根残差和吸收修正在 TB 最优处独立消失。

**关键区别**：不像 LED-GFN 的平滑正则化器在 TB 最优处 L_reg > 0，RapTB 的辅助项在 TB 不动点处代数保证为零。

**诚实表述**：这是代数一致性，不是收敛速率定理。

### 2.2 SubTB + LLM 的终止机制

**推导链条**：

1. **原始 SubTB**（Madan et al. ICML 2023）：使用**显式学习的状态流** $F(s; \theta)$ 作为独立网络头。
2. **树结构 → P_B = 1**（Malkin et al. 2022, Eq. 16 后注释）："In the case of auto-regressive generation, G is a directed tree... P_B is trivially P_B = 1."
3. **可终止前缀树 → F(s) 替换**（Deleu et al. UAI 2022; Hu et al. ICLR 2024）：$F(s) = R(s^\top)/P_F(\top|s)$。Hu et al. 原文："since each state is a valid terminable state, we can incorporate the modification to account for this from Deleu et al. (2022)."
4. **SubTB = PCL 等价**（Deleu et al. UAI 2024, Proposition 3.2）：$L_{PCL} = \alpha^2 \cdot L_{SubTB}$——SubTB 等价于最大熵 RL 中的路径一致性学习。
5. **后果**：终止概率同时充当（a）停止决策和（b）缺失状态流的替代。$O(N^2)$ 窗口在同一终止头上施加梯度压力。
6. **终止漂移**：稀疏奖励下，优化器通过全局平移停止 logits 减少多个残差。证据：SubTB $\log p_{term} = -79.638$（Table 4），vs RapTB 的 $-0.065$。

### 2.3 Max/Soft 吸收目标

**u_max — 保守事后目标（非正式下界）**

u_max 不是真实流 $F(s_{0:k})$ 的正式下界（真实流对指数多条路径求和）。它是：沿一条采样后缀观察到的最佳任务专有信号。"保守"的含义：max 是保证匹配或超过任何单个后缀观察值的最紧算子，不进行外推。

**文献定位**：
- **FL-GFN** (Pan et al. ICML 2023)：通过中间能量 $E(s)$ 重参数化流。要求 $E(s)$ 在每个中间状态可评估（Assumption 4.1）——在稀疏奖励 LLM 任务中违反。
- **LED-GFN** (Jang et al. ICLR 2024)：学习加法势能分解，需要辅助网络。
- **TBA** (Bartoldson et al. NeurIPS 2025)：明确呼吁"learning partial energy functions"——RapTB 的吸收目标是其非参数实现。
- **GAE** (Schulman et al. ICLR 2016)：指数加权 TD 误差。u_soft 的距离衰减 ρ 与 GAE 的 γλ 结构类似，但操作对象不同（原始任务奖励 vs 使用学习 V(s) 的 TD 误差）。
- **RUDDER** (Arjona-Medina et al. NeurIPS 2019)：学习回报分解，保证守恒。RapTB 不保证守恒。
- **HCA** (Harutyunyan et al. NeurIPS 2019)：基于贝叶斯规则的事后信用。RapTB 使用 max/softmax 聚合。

**在 MaxEnt RL 视角下**（Tiapkin et al. 2024）：$\log F(s) = V^*(s)$（软价值函数）。FL-GFN 通过已知中间能量重参数化它。LED-GFN 用辅助势能学习它。**RapTB 完全绕过学习 $V^*(s)$**，使用非参数后缀聚合作为廉价、有偏但低方差的替代。

**u_soft — 平滑长程信号**

距离衰减 $\rho(j-k)$ 下权远处噪声观测，避免过于乐观的估计。Table 6 确认混合优于任一端点。

---

## 三、新实验证据

### 3.1 SMILES β×ρ Sweep（9 configs, bsz=128, 5000 steps）

| Config | Acc | QED | FPDiv |
|--------|-----|-----|-------|
| β=1,ρ=0 | 0.992 | 0.751 | 0.849 |
| **β=5,ρ=0.1（论文）** | **0.991** | **0.783** | **0.864** |
| β=10,ρ=0 | 0.968 | 0.727 | 0.883 |
| ... | 8/9 Acc≥0.991 | QED 0.727-0.804 | FPDiv 0.849-0.883 |

### 3.2 Expr24 β×ρ Sweep（9 configs）

全部 9 configs Acc≥0.983，$\log p_{term} \in [-0.25, -0.04]$。

### 3.3 3B Scale-Up

| 方法 | Acc | QED | FPDiv |
|------|-----|-----|-------|
| SubTB | **0.313** ❌ | 0.221 | 0.854 |
| **RapTB+SubM** | **0.996** ✅ | **0.856** | **0.937** |

### 3.4 AMP 生物序列

| 方法 | Perf | Div | Nov | Avg Len |
|------|------|-----|-----|---------|
| SubTB† | 0.897 | 21.37† | 28.68† | **49.3†** |
| **RapTB+SubM** | 0.916 | **16.92** | **15.77** | 25.6 |

### 3.5 PPO/GRPO

SMILES：GRPO QED=0.661, Entropy=0.98（GFlowNet 的 36%）；PPO Entropy=0
Expr24：GRPO Acc=0.002 (eval), 1 unique valid

### 3.6 AvgPrefixTB

SMILES：Diversity=0.665, Len=2.89（坍缩）
Expr24：NormCov=0.016, Len=5.74（坍缩）

---

## 四、完整文献引用表

| 文献 | 引用方式 | 用于回答 |
|------|---------|---------|
| Malkin et al. (2022) NeurIPS — TB Theorem 1 | TB 不动点的精确性 | QHmk-Q1, JxzD-W1, Pd1v-W3 |
| Tiapkin et al. (2024) AISTATS Oral — Theorem 1 | GFlowNet = MaxEnt RL 最优策略 | QHmk-W1, RL 框架 |
| Deleu et al. (2024) UAI — Prop 3.2 | SubTB = PCL 等价 | QHmk-W1, JxzD-Q1 |
| Deleu et al. (2022) UAI | F(s) = R(s^⊤)/P_F(⊤|s) 终止修改 | JxzD-Q1, SubTB 机制 |
| Hu et al. (2024) ICLR Oral | LLM-GFlowNet SubTB 公式 + 引用 | JxzD-Q1, SubTB 机制 |
| Madan et al. (2023) ICML | SubTB 伸缩论证，相同不动点 | 零点一致性论证 |
| Pan et al. (2023) ICML — FL-GFN | 中间能量重参数化, Assumption 4.1 | 吸收目标对比 |
| Jang et al. (2024) ICLR Oral — LED-GFN | 学习势能分解 + 正则化 | 吸收目标对比, 不动点讨论 |
| Bartoldson et al. (2025) NeurIPS — TBA | TB 方差限制 + "partial energy" 未来工作 | QHmk-W3, 吸收目标动机 |
| Schulman et al. (2015) ICLR 2016 — GAE | γλ 衰减结构类比 | cA3o-Q4, 吸收目标 |
| Arjona-Medina et al. (2019) NeurIPS — RUDDER | 回报分解对比 | JxzD-W2 |
| Harutyunyan et al. (2019) NeurIPS — HCA | 事后信用对比 | JxzD-W2 |
| Ng et al. (1999) ICML — PBRS | 势能整形不适用论证 | cA3o-Q4 |
| da Silva et al. (2024) NeurIPS | 无偏控制变量 | 零点一致性讨论 |
| arXiv 2410.02596 (2024) | 损失-散度对应 | 不动点保持论证 |

---

## 五、核心翻转杠杆（更新版）

1. **零点一致性论证**：TB=0 时 L_aux=0 → RapTB 不移动不动点（类比 SubTB 伸缩论证）
2. **SubTB 终止漂移机制**：完整文献链（Deleu 2022 → Hu 2024 → 我们的诊断）
3. **吸收目标 = TBA 呼吁的 "partial energy functions" 的非参数实现**
4. **PPO/GRPO 坍缩** → reward-max RL 不适合分布采样
5. **AvgPrefixTB 坍缩** → 简单前缀监督不够
6. **18 configs 稳健性** → 方法 robust
7. **3B SubTB 崩溃加剧** → 失败是结构性的
8. **AMP SubTB 长度坍缩** → 生物序列也有同样问题
9. **完全接受 RL 框架** → 不争辩，reframe

---

## 六、提交文件

| 文件 | 字符数 | 状态 |
|------|-------|------|
| `PASTE_READY_global.txt` | ~3,800 | ✅ 更新：文献加强 |
| `PASTE_READY_QHmk_v2.md` | ~4,900 | ✅ 更新：文献加强 |
| `PASTE_READY_cA3o_v3.txt` | ~3,900 | ✅ 更新：文献加强 |
| `PASTE_READY_JxzD_v3.txt` | ~4,900 | ✅ 更新：文献加强 |
| `PASTE_READY_Pd1v_v3.md` | ~4,700 | ✅ 更新：文献加强 |
| `REBUTTAL_DRAFT_QHmk_v2_rich.md` | — | ✅ 更新：完整文献引用 |
| `REBUTTAL_DRAFT_JxzD_v3_rich.md` | — | ✅ 更新：完整文献引用 |
| `REBUTTAL_DRAFT_cA3o_v2_rich.md` | — | ✅ 更新：完整文献引用 |

---

## 七、安全验证

| 检查项 | 结果 |
|--------|------|
| 覆盖率 | ✅ 18/18 issues |
| 来源 | ✅ 无虚构数据，所有文献引用经调研确认 |
| 承诺 | ✅ 无超范围承诺 |
| 语气 | ✅ 专业、委婉 |
| 一致性 | ✅ 理论/RL/互补性框架统一 |
| 字符限制 | ✅ 全部 < 5000 |
| 文献准确性 | ✅ 所有定理/命题编号已核实 |
