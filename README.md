# DensForge

[![CI](https://github.com/CJX0712/densforge/actions/workflows/ci.yml/badge.svg)](https://github.com/CJX0712/densforge/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Quality](https://img.shields.io/badge/quality-C%20(blocked)-red.svg)](BLOCKED.md)

**密度估计 + 流形学习，统一为一套估计器。**
一次 `fit` 同时输出归一化密度、几何保持嵌入与可机器验证的不变量。

作者：晨星 <CJX0712@users.noreply.github.com> · MIT © 2026

---

## ⚠️ 质量等级：**C 级（不合格，未交付）**

**本仓库未推送到 GitHub、未打 tag、未创建 Release。**

定稿前的性能自查在**两个子任务上均未达 DoD 门槛**：

| 子任务 | 主门 | 实测（仓库实际生成器与实现） | 结论 |
|---|---|---|---|
| 密度估计 | G-D1 中位 rel ≥ 0% | **−0.94%** | ❌ FAIL |
| 流形学习 | G-M1 TW ≥ 基线 −0.01 | **中位 −0.1065，5/5 全败** | ❌ FAIL |

完整卡点、80 组消融证据、已尝试方案与下一步建议见 **[`BLOCKED.md`](BLOCKED.md)**。

**代码本身完整可运行**：423 项测试、24/24 可验证不变量、ruff 全绿、纯 numpy 离线兜底齐全，
可作为代码资产直接复用。**C 级是对性能主张的判定，不是对代码质量的判定。**

> **安装说明**：`pip install densforge` **暂不可用**（未发布到 PyPI）。
> 请直接使用本仓库源码：`pip install -r requirements.txt && pip install -e .`

---

## 这是什么

| 维度 | 说明 |
|---|---|
| **密度估计** | 学 `p(x)`，主指标 **held-out test NLL**（越小越好） |
| **流形学习** | 恢复内在几何，主指标 **Trustworthiness**（越大越好） |
| **旗舰** | `DensFuse` —— 固定带宽 `KernelDensity` 的**零参数自适应推广** |
| **运行环境** | 纯 CPU · 无 GPU · 无 torch · 运行时零网络请求 |

### 旗舰的真实定位（不是"超越基线"）

`DensFuse` **不宣称性能超越**经典基线。实测证据：

> `DensFuse(η=0, β=0)` 与 `KernelDensity` 的 log-density **逐位相同**（max|diff| = **1.6e-14**）。

即：**η=0, β=0 时旗舰就是固定带宽 KDE**。η（自适应带宽）与 β（局部各向异性）是仅有的区分点，
而 **β 在全部 5 个数据集上为负贡献**。因此本项目诚实地把旗舰定位为：

1. **免调参** —— 基线的最优带宽跨数据集差 **79 倍**（实测 `h*` 从 1.0 到 79.4），`DensFuse` 用一组固定超参覆盖全部数据集。
2. **统一性** —— 一次拟合同时给出密度与嵌入，共享同一套局部几何。
3. **可验证性** —— 24 条可机器验证的不变量（归一性、尺度/平移/旋转等变性、留一法上界等）。

---

## 性能实测（全部来自真实运行输出）

### 密度估计：仓库口径下轻微劣化 ❌

基线为**公平搜索**：`KernelDensity` 带宽在 val 上做 **61 点 log 网格**（跨 4 个数量级）搜索，3 seeds。

| 数据集 | 旗舰 vs 公平基线（test NLL 相对变化） | 门槛 | 结论 |
|---|---|---|---|
| `aniso_gmm` | **−0.69%** | ≥ 0% | ❌ |
| `t_mixture` | **−1.29%** | ≥ 0% | ❌ |
| `sparse_dim` | **+0.06%** | ≥ 0% | ✅ |
| `hetero_density` | **−1.85%** | ≥ 0% | ❌ |
| `heavy_tail` | **−1.05%** | ≥ 0% | ❌ |
| **真实数据集中位数** | **−1.05%** | ≥ 0% | ❌ **FAIL** |
| 最差单数据集 | −1.85% | ≥ −5% | ✅ |
| `iso_gauss`（反作弊对照） | **−2.25%** | ∈ [−5%, +5%] | ✅ 通过 |
| `uniform_hypercube`（负对照） | **+0.04%** | ∈ [−5%, +5%] | ✅ 通过 |

> ⚠️ **口径更正**：早期校准脚本（使用不同的私有生成器）曾给出 **+1.01%**。用
> **仓库实际生成器 + 实际实现**复算后为 **−1.05%**（中位）。**以仓库口径为准**，+1.01% 已作废。

**反作弊对照门通过**（−2.25% / +0.04% 均在容差内），说明实现无数据泄漏、无虚高 ——
这是**诚实的轻微劣化**，不是错误。

> **γ 网格下界（诊断，非改动）**：`sparse_dim`、`uniform_hypercube` 在 val 选到 shipped 网格下端点 γ=0.0625。放宽下界对照显示仅这 2 个触底数据集受影响，二者均非 5 个真实数据集的中位数，**中位数 −1.05% 维持不变**，G-D1 仍 FAIL。

### 流形学习：劣于基线 ❌（C 级主卡点）

**协议**：val 选超参 / test 报告 / **3 seeds**。基线为各数据集最强者。

| 数据集 | 最强基线 | 基线 TW | DensFuse TW | 实测 Δ |
|---|---|---|---|---|
| moons | **Isomap** | **1.0000** | 0.9723 | **−0.0277** |
| swiss_roll | **Isomap** | 0.9853 | 0.8697 | **−0.1156** |
| circles | **Isomap** | 0.9730 | 0.8665 | **−0.1065** |
| spiral_double | **Isomap** | 0.9996 | 0.8738 | **−0.1258** |
| hd40_imbalanced | **Isomap** | 0.7407 | 0.6833 | **−0.0574** |

**中位 Δ = −0.1065，5/5 全败。**

**80 组全因子消融**（α(3) × MST(2) × β(3) 配置 × 5 数据集 × 3 seeds）中，
**连最优配置（α=0, MST=on, β=0）仍是 −0.0761，0/5 通过**。

> **根因**：旗舰嵌入路径是 **Diffusion Maps（谱方法）**，而非 LLE 的**局部线性重构**
> 或 Isomap 的**测地距离**。这是**机制层面的能力缺口**，非组件可修。

### 消融：被否决的组件（负结果保留）

| 组件 | 贡献 | 判定 |
|---|---|---|
| 自适应带宽轴（η>0, β=0） | +0.2% ~ +1.0% | ✅ 微弱正贡献 |
| 局部各向异性白化轴（β=1） | **−0.1% ~ −5.65%** | ❌ **全部为负，未生效** |
| 密度平衡扩散嵌入（α） | 流形侧 TW 全部落后 | ❌ 劣于基线 |

### α 的作用范围（实测确认）

> **α 只作用于嵌入路径，从不进入 `log_density`。**
> 实测 α ∈ {0, 0.25, 0.5, 1} 的密度输出**逐位相同**（max|diff| = **0.000e+00**）。

故密度侧超参只有 η / β / γ / k 四轴，**对 α 做密度消融是数学空洞的**（结果必然相同）。
原设计文档中「两范式互喂 / 三轴闭环」的表述**不成立**，已更正。

---

## 三轮自证失败（本次交付最重要的产出）

定稿前对性能主张做了三轮独立自查，**三次自证失败**。全部如实保留：

| 轮次 | 曾声称 | 证伪方式 | 修正后 |
|---|---|---|---|
| v1 | +34.84% | 实现 bug：`logdetH` 符号反 + 核未逐项归一化 | — |
| v2 | +33.71% | 公平基线（带宽网格 8 点窄 → 61 点 log 网格） | +1.01% → **−1.05%**（仓库口径） |
| v3 | M4 全局白化 **+52.84%** | 等预算审计 + 基线补预处理维度 | **+0.00%** |

**共同根因**：性能"提升"全部来自**基线调参不足**，而非结构优势。
M4 尤其典型 —— 其全部增益等价于"按各维标准差做一次全局标准化"，基线被允许搜索该强度后
两者找到**完全相同**的解（λ\*=1, h\*=0.7943）。**M4 不是新机制，只是基线欠了一个预处理步骤。**

### 核心科学结论：指标与机制不对齐

> held-out test NLL 是**全局平均**指标，被**样本密集区主导**；而局部自适应机制恰恰在
> **稀疏区最激进**，其贡献被平均抹掉；且稀疏区放大的核更容易"漏掉"测试点，反过来**推高**平均 NLL。
>
> **推论**：一个在局部几何上明确有效的机制，在全局平均指标上可能完全不可见，甚至反向。
> **评估指标的选择可能比算法改进更决定结论。**

---

## 架构

调用单向无环：`cli → pipeline → {data, density, manifold, hpo, training, eval} → core`

```
densforge/
  core/       types · errors · config · interfaces(Protocol) · seed(全局确定性)
  data/       synth(6 族 + 2 对照) · datasets(registry, train/val/test 独立采样)
  density/    baselines(Tier-0) · kdet(变带宽) · flagship(DensFuse) · tier1(纯 numpy)
  manifold/   baselines · diffusionmaps · flagship · tier1
  hpo/        bandwidth · alpha · coordinate(坐标下降)
  training/   fitter
  eval/       metrics(NLL/TW) · leakage(防泄漏三重断言)
  pipeline/   densforge_pipeline
  cli.py      run / bench / demo / info
```

`tests/test_architecture.py` 会**解析两个旗舰模块的 AST**，若出现
`KernelDensity` / `GaussianMixture` / `Isomap` / `SpectralEmbedding` 等基类引用即构建失败 ——
确保旗舰是独立核估计器而非基线包装器。唯一的 sklearn 依赖是 `NearestNeighbors`。

## 复现

```bash
pip install -r requirements.txt
pip install -e ".[dev]"

densforge info                       # 列出可用后端（Tier-0 sklearn / Tier-1 纯 numpy）
densforge demo --quick               # 端到端冒烟
densforge bench --out benchmark.json --seeds 3
python examples/run_demo.py --quick  # 带全部章节说明的同一 demo
```

`Makefile` 提供 `make install` / `make test` / `make cov` / `make demo` / `make verify`。

## 工程化

| 项 | 状态 |
|---|---|
| 测试 | 423 项 |
| 可验证不变量 | **24/24 通过**（候选 32 条，实测启用 24） |
| 确定性 | 全部生成器用 legacy `np.random.RandomState`（NEP 19 跨 numpy 版本冻结流），同 seed 逐位一致 |
| 离线兜底 | Tier-1 纯 numpy：NumpyKDE / NumpyIsomap / NumpySpectral / NumpyDiffusionMaps |
| 防泄漏 | 三重断言：种子隔离 / 指标来源隔离 / 模型指纹隔离 + 采样重叠检查 |
| 依赖锁定 | `requirements.lock.txt`（numpy 2.5.3 / scipy 1.18.1 / sklearn 1.9.1） |
| CI | ruff lint + format · pytest 矩阵 3.12/3.13 · demo 冒烟 |

## 文档

- [`BLOCKED.md`](BLOCKED.md) —— C 级卡点、80 组消融证据、下一步建议
- [`docs/architecture.md`](docs/architecture.md) —— 架构、数学推导、门槛表、方法论讨论
- [`docs/model_card.md`](docs/model_card.md) —— 用途 / 数据 / 指标 / 局限 / 伦理风险
- [`docs/math_verification.md`](docs/math_verification.md) —— 数学核验记录

## 伦理与风险

密度估计与嵌入可被用于 **membership inference**：若攻击者已知候选点属于训练集，
可通过其密度异常高来推断成员身份。发布任何基于本仓库训练的密度模型前，请评估该风险。

## 许可

MIT © 2026 [晨星](https://github.com/CJX0712)