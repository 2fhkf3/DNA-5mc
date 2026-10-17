# Test2：基于天然 chr16 测试数据的 DNA 5mC 假阴性评测方案

方案与实现日期：2026-10-18。当前可用数据为已有天然样本 chr16 test，其参考标签由原 BED 筛选规则产生。评测脚本已在 [workflow](workflow/README.md) 实现，支持复用预测与阳性缓存重新推理；尚未执行正式全量评测。

**主分析从原 test 中选择 `label=1`，即原构建流程中 BED 修饰百分比为 100% 的阳性窗口，计算 read/window 与 site 两个层面的假阴性率 FNR，并同时报告 Recall/TPR。** 这是原测试集阳性子集的补充分析，不是新制备的全甲基化对照，也不构成新的独立实验重复。

## 1. 评测目标与解释范围

本次回答三个问题：

1. 在被原参考标签标记为阳性的 read–位点窗口中，有多少被模型判为阴性？
2. 将同一阳性位点的窗口概率汇总后，有多少位点被漏检？
3. 漏检是否集中在特定序列上下文、覆盖度或基因组区段？降低 Test1 的误报时，是否同时损失了这些阳性的检出能力？

| 数据分支 | 参考阳性/阴性的来源 | 主要结果 |
| --- | --- | --- |
| Test1：WGA UNMOD | WGA 阴性文库身份 | WGA 上的 FPR / specificity |
| Test1：天然 BED=0 | 原 chr16 test 的 `label=0` | 天然参考阴性子集上的 FPR |
| **Test2：天然 BED=100** | **原 chr16 test 的 `label=1`** | **天然参考阳性子集上的 FNR / Recall** |

当前结果应表述为“相对于原 BED 阳性参考标签的观察漏检率”。天然 BED=100 的可靠性依赖 BED 的生成方法、有效覆盖度和调用阈值：有限次观测中的 100% 并不证明每条分子必然修饰；若 BED 来自另一个甲基化 caller，结果也包含对该 caller 标签的一致性。

应补充记录 BED 的产生工具、版本、参数、参考版本、链规则、覆盖度和可获得的独立实验依据。暂时无法核实时，保留 `reference_validation_status: unverified`，不将参考标签描述为已实验确认的逐 read 真值。Modkit 的 bedMethyl 百分比来自过滤后的修饰调用计数，定义见[官方说明](https://nanoporetech.github.io/modkit/intro_pileup.html)。

## 2. 已有数据与可复用结果

### 2.1 阳性特征来源

各目录均包含 `test.pt`、特征二进制、标签二进制、`metadata_test.tsv` 和 `feature_summary.json`。当前特征为 `[N, 5, 768]`，来自原 15 nt 输入上下文的中心裁剪。

| 模型 | 特征目录 | 全 test 窗口数 | 阳性窗口数 | 阳性 site 数 |
| --- | --- | ---: | ---: | ---: |
| V011 / Diffusion | `train/data/V011-V2` | 2,590,329 | **1,295,163** | **51,314** |
| V543 / BERT | `train/data/V543-V2` | 2,590,340 | **1,295,170** | **51,314** |
| V554 / BERT | `train/data/V554-V2` | 2,590,340 | **1,295,170** | **51,314** |
| V610 / GPT | `train/data/V610-V2` | 2,590,329 | **1,295,163** | **51,314** |

这些计数已根据现有训练 `metrics.json` 核对；是各主干自己的可评测集合，不代表四模型已匹配的共同分母。未来加入 V548 等模型时，读取实际清单和预测结果重新计数，不沿用上表数值。

默认 V011 输入绝对路径：

```text
/home/shj/ZJ-lab/Modification/DNA-5mC/train/data/V011-V2
```

### 2.2 现成预测文件

当前四组各有 15 个 run 保存了以下文件：

```text
/home/shj/ZJ-lab/Modification/DNA-5mC/train/output/<experiment>/runs/<strategy>_seed<N>/
├── metrics.json
├── test_predictions.tsv
├── test_site_predictions.tsv
└── best.pt
```

- `test_predictions.tsv` 含 `read_id/chrom/center_genome_pos0/strand/source/label/prob/seq/ref`。
- `test_site_predictions.tsv` 含 `label/prob_mean/n_reads` 等位点字段。
- `metrics.json` 已记录原 test 的 `tp`、`fn`、`recall` 等，因此能直接得到基线 FNR。

**优先复用已有预测完成 Test2。** 只有预测缺失、检查点变化、特征协议变化或需要验证一致性时，才重新推理。

## 3. 数据构造规则

1. 只从既有 `test` 选择 `label=1`，不读取 train/valid 作为测试样本。
2. 依据原参考标签选择，不依据当前模型概率、预测正负、已有 FN 清单挑选样本。
3. 保持标签为 1，不将阴性改成阳性，也不把阳性翻成 0 以复用阴性评测器。
4. 保留原始 read ID、染色体、0-based 中心坐标、链、序列、原输入行号和原裁剪上下文。
5. 同一原始 read 的同一位点只保留一个观测；发现重复或冲突标签时中止审计，不静默增加样本数。
6. 不重新平衡类别，不额外重采样阳性窗口作为主结果。统计所有符合既有条件且成功预测的阳性。
7. 记录原质量筛选条件；缺失的 MAPQ、BED 覆盖度等保留为空，不虚构数值。

推荐身份字段：

```yaml
dataset_kind: native_bed100
truth_source: native_test_reference_label_one
reference_label_rule: original_test_label_equals_1
reference_validation_status: unverified
library_id: NATIVE_250F500490011
source_split: test
partial_test: false
```

`library_id` 应对应真实天然文库，同文库在 Test1 天然阴性与 Test2 阳性分支中保持一致；切换模型也不改变它。`source=mod` 是原天然文库来源，应保留，它本身不能替代 `label`。

本方案继承原 test 的既有筛选、平衡和缓存成功条件，因而结果只代表这部分阳性。尤其 BED=100 是极端参考位点，不能外推为所有部分甲基化位点的逐分子漏检率。

## 4. 推荐执行路线

### 路线 A：直接汇总已保存的预测，作为首选

```text
冻结 run 与阈值
  → 审计既有 test_predictions.tsv / metrics.json / 检查点来源
  → 选择 label=1，保留每条阳性的原始概率
  → 计算 read FNR / Recall，重新汇总 site 概率与覆盖度
  → 按上下文、覆盖度、基因组区块分层
  → 匹配各模型共同观测并汇总 seed
```

此路线不需要主干提取或 GPU 推理，也不必复制约 10 GB 的单模型阳性特征。

要求核对预测文件所属 run、模型、seed、检查点、特征签名和行数。不能把后来重新训练的 `best.pt` 与旧预测混用；旧 TSV 未嵌入检查点哈希，当前仅能作来源一致性核对，无法证明历史权重绑定。无法确认来源时改用路线 B 重新推理。源码中的既有实现可参考[预测导出](../../train/train.py)、[分类头](../../train/model)。

主结果建议从阳性 read 重新聚合 site，以便统一共同 read 和最终覆盖度。原 `test_site_predictions.tsv` 的阳性行可作为一致性校验；只有观测集合完全一致时，两者才应一致。

下面命令仅只读展示一个现有 run 的基线，不生成 Test2 结果：

```bash
/mnt/zzbnew/rnamodel/zhoukexuan/miniconda3/envs/PoreDLM/bin/python - <<'PY'
import json
from pathlib import Path
p = Path('/home/shj/ZJ-lab/Modification/DNA-5mC/train/output/V011-v2/runs/center_5mer_mlp_raw768_seed1/metrics.json')
m = json.loads(p.read_text())
for level in ('read', 'site'):
    r = m[f'test_{level}_level']
    n = r['tp'] + r['fn']
    if not n:
        raise ValueError(f'No reference-positive samples at {level} level')
    print(level, {'N_positive': n, 'FN': r['fn'], 'TP': r['tp'],
                  'FNR': r['fn']/n, 'Recall': r['tp']/n})
PY
```

该命令沿用原结果的阈值；如需确认数值，检查 run 的实际配置。当前四组的原测试工作点均为 0.5。

### 路线 B：构造全阳性缓存并重新推理

已新增 [Step1](workflow/step1_select_native_positives.py)，参照 Test1 的 Step1.1 分块同步筛选，选择 `label == 1`，输出身份为天然 BED=100；[Step2](workflow/step2_evaluate.py) 的 features 模式加载对应分类头重新推理。

- 同步筛选特征、标签和元数据，保留原顺序、dtype 和全部特征值。
- 保存 `source_sample_index`，重新生成连续的 `sample_index`；复制缓存后使用相对 `.bin` 路径。
- 当前输入已裁剪到中心 5 个位置，保持原 15 nt 主干上下文，不重新编码短信号。
- 保留提取协议、主干/encoder/tokenizer 身份、裁剪记录及输入签名。
- 默认只输出 test，不生成训练或验证数据；测试前缀/限量输出必须标记 `partial_test=true`。
- 全量 V011 阳性数组约 9.95 GB，另需元数据与审计空间；非空输出目录拒绝覆盖。
- 各主干必须使用自己的特征。V011 隐藏状态不能交给 BERT/GPT 上训练的分类头。

推理复用对应 run 的 `best.pt` 和模型定义，使用 `model.eval()` 与无梯度模式，不训练、不更新权重。输出正类概率 `P(label=1)`，不对概率取反。

**Test1 的 Step1.1 固定选择 label=0，Step4 也严格要求全阴性并计算 FP/TN，不能直接承担 Test2。** 请使用本目录专用全阳性入口，操作与参数见 [workflow/README.md](workflow/README.md)。

路线 B 的抽样结果必须与路线 A 的相同检查点预测一致；先检查特征字节及行号，再检查概率误差和 0.5 阈值调用，记录数值容差。

## 5. 核心指标与阈值

### 5.1 Read/window 级：主要指标

一条样本是一条原始 read 上的一个目标窗口，不等于一条独立完整 read。对参考标签为 1 的窗口，沿用 `p_i >= t_read` 判阳性：

$$
TP_{read}=\sum_i\mathbf{1}(p_i\geq t_{read}),\qquad
FN_{read}=\sum_i\mathbf{1}(p_i<t_{read}).
$$

$$
FNR_{read}=\frac{FN_{read}}{TP_{read}+FN_{read}},\qquad
Recall_{read}=TPR_{read}=1-FNR_{read}.
$$

主工作点为 `t_read=0.5`，恰好 0.5 判阳性。分母为零时记为 NA，不填 0。指标定义对应[混淆矩阵约定](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html)。

必须同时报告 `N_positive`、TP、FN、FNR、Recall 和区间；FNR 以比例存储，表格展示时乘 100%。`FN/(FN+TN)` 是另一种条件比例，不是本方案的 FNR。

### 5.2 Site 级：聚合后的漏检

推荐物理位点键为 `(library_id, chrom, center_genome_pos0, strand)`；每个位点按原始 read 去重。现有训练汇总还包含 `source`，当前单一天然来源 `mod` 下等价。扩展来源时应明确处理，避免人为重复计数同一物理位点。

对最终成功预测的窗口：

$$
s_j=\frac{1}{n_j}\sum_{i=1}^{n_j}p_{ij},\qquad
FNR_{site}=\frac{\sum_j\mathbf{1}(s_j<t_{site})}{N_{positive\ sites}}.
$$

主工作点为 `t_site=0.5`，Recall_site 为其补数。先匹配共同窗口，再重新计算 `n_j`、`s_j` 和覆盖度筛选，不能直接沿用输入中的旧覆盖度。

位点平均分可能掩盖部分 read 漏检。例如五条 read 中一条低分，均值仍可超过 0.5。因此增加：

$$
r_j=\frac{1}{n_j}\sum_i\mathbf{1}(p_{ij}<t_{read}),
$$

报告位点等权的 `r_j` 均值、P50/P95/P99，以及存在漏检 read 的位点比例。`r_j` 与 `s_j` 的阈值调用是不同指标；平均概率未经校准不能直接作为真实甲基化比例。

### 5.3 分数及单类别指标的边界

| 指标 | 用途 |
| --- | --- |
| FNR / Recall | 主指标，分别展示 read 与 site |
| 阳性分数 P05/P50/P95、阈值附近窗口比例 | 观察低分尾部及阈值敏感性 |
| `mean((1-p)^2)`，Brier | 阳性分数偏离 1 的程度，辅助描述 |
| `mean(-log(clip(p, eps, 1)))`，BCE | 阳性低分惩罚；需记录 eps，不修改用于阈值判断的原概率 |
| Accuracy | 全阳性时等于 Recall，无需重复作为主成绩 |
| Precision / F1 | 缺少阴性约束，不能评价模型的误报控制；F1 只是 Recall 的变换 |
| AUROC / specificity / FPR / MCC | 全阳性子集不能给出有效的完整二分类评价 |
| AUPRC / AP | 全阳性时部分实现可能返回 1，不反映区分能力，不作为 Test2 成绩 |

读取概率时检查有限性及 `[0,1]` 范围。若需要 AUROC、AP 或完整 F1，应另使用原天然正负完整 test，而不是将单类别结果补成一张虚构的混淆矩阵。

## 6. 分层分析与覆盖度

主结果保留原 test 中所有符合条件且成功预测的阳性；另提供最终 coverage ≥ 5 的统一子集，与 Test1 的高覆盖分析相对应。

| 分层 | 要求 |
| --- | --- |
| 中心 5-mer | 输出 N、FN、TP、FNR、Recall，覆盖全部有观测的上下文 |
| 其他上下文长度 | 只有原 `crop_input_ref/seq` 或可信参考支持时使用，不能从 5-mer 猜测外侧序列 |
| 实际 site 覆盖度 | 记录 1、2、3、4、5–9、10–19、≥20；空层记 NA |
| 基因组区块 | 按预先定义的 chr16 区块汇总，不先挑选误差最高区段作主结果 |
| 信号窗口长度 | 有真实 `signal_len` 时分层，注明其对应原 15 nt 信号而非裁剪后 5 个位置 |
| 质量 | 仅在保留 MAPQ 等真实字段时分层，缺失时明确不可分析 |

Test2 继承了原天然 test 的覆盖度和筛选，不能称为“5× 阳性评测”，也不能假设其含有足够低覆盖位点。自然出现的覆盖度分层还混合了不同位点集合，不足以证明覆盖度本身的因果影响。

若要研究低深度，单独固定具有足够 coverage 的共同阳性位点，再按原始 read ID 下采样到 1、2、3、5、10 条；各模型使用相同位点和相同 read 子集，并报告预先约定的重复次数与随机种子。这是高覆盖阳性子集的下采样实验，不能替代独立低深度文库。

## 7. 公平比较与未成功调用的处理

### 7.1 两种分母分别报告

1. **各模型原始可评测集合**：描述该模型成功预测的全部原 test 阳性。
2. **跨模型共同集合**：按观测键 `(library_id, read_id, chrom, position, strand)` 取交集，检查标签、seq/ref 一致，再计算所有模型的指标和跨 seed 汇总。

不能仅因表中阳性 site 数相同，就认定 read 子集相同。主干跳过少量窗口后，即便 site 仍存在，均值和 coverage 也可能改变。共同集合的 aggregate 必须由共同集合重新计算；原始集合结果另存，不混用。

原始 BED/JSONL 候选阳性窗口清单若可恢复，应另报告：

```text
N_reference_positive → N_features_available → N_predictions → TP / FN / no-call
call_rate = N_predictions / N_reference_positive
```

无特征、提取失败、非有限分数或拒绝判断均单列为 no-call，不默认计作 TP 或已调用 FN。可另定义 `TP/N_reference_positive` 作为包含 no-call 的端到端检出率，明确分母及含义。

当前只有成功缓存时，不能将缓存层的 100% 推理完成率描述为完整上游 call rate；应说明原始候选分母暂不可核实。

### 7.2 检查点、阈值与 seeds

- 使用原 valid 选择的 `best.pt`；不根据 Test2 FNR 再选最佳 epoch、seed 或隐藏层。
- 固定 0.5 为主工作点，可附预先约定的 0.1/0.3/0.5/0.7/0.9 曲线。FNR 随阈值升高应不下降，作为检查项。
- 如需其他阈值，在独立 valid/校准数据上确定后冻结；不在全阳性 Test2 上把阈值降到最低来优化成绩。
- 列出每组实际训练预算、早停、seed 和协议。现有 V554 的部分轮数/早停与其他组不同，不能直接将差异全部归因于主干架构。
- 逐 seed 展示 FN/TP/FNR，再报告均值与样本标准差 `ddof=1`；不将三次预测合并为三倍独立测试样本。
- 相同 seed 的组可做预先约定的配对分析。seed 不齐时仍可描述各组均值，但不能只挑共同的低误差 seed 或假装所有策略都有四模型配对结果。

当前主要策略的 seed 情况：

| 策略 | V011 | V543 | V554 | V610 |
| --- | --- | --- | --- | --- |
| 两个 Linear、MLP | 1/2/3 | 1/2/3 | 1/2/3 | 1/2/3 |
| Residual DWConv | 4/5/6 | 4/5/6 | 1/2/3 | 4/5/6 |
| 融合头 `emb8_none` | 2/3/4 | 2/3/4 | 1/2/3 | 2/3/4 |

这些是当前已完成结果的状态，不是未来必须采用的 seed 规则。共同观测可独立于 seed 预先固定，避免把“缺少相同 seed”误当作“没有共同数据”。

## 8. 不确定性与统计解释

默认采用预先固定的 1 Mb chr16 基因组区块、1000 次配对 bootstrap，并记录实际区块数和重采样随机种子。每次对相同区块抽样，重新计算全局 FNR 或模型间 FNR 差值；不简单平均不同大小区块的 FNR。

窗口共享 read 和位点，不能按完全独立观测解释数百万窗口的区间。区块 bootstrap 仅描述当前文库内部的空间抽样变化；必要时检查区块尺度敏感性，不能替代独立文库重复。

Wilson 区间可作为明确标注的独立观测近似。零 FN 时，普通百分位 bootstrap 的退化 `[0,0]` 不证明真实漏检率必为零；少区块、零事件或退化情形应注明区间不可估计，另报带独立性前提的上界参考。seed 标准差反映训练随机性，不是生物或制备重复。

上下文表以探索性描述为主；若挑选高漏检 motif 并据此改模型，应在未参与优化的数据上复核，不能把同一 Test2 的回测改善视为独立验证。

## 9. 与 Test1 联合报告

建议分别展示以下三个工作点结果：

| 模型 / 分类头 / seed | 天然 chr16 BED=0 FPR | 天然 chr16 BED=100 FNR / Recall | WGA FPR |
| --- | --- | --- | --- |
| 每个已冻结 run | 来自原天然阴性子集 | 来自本 Test2 | 标明 WGA 文库、染色体和筛选条件 |

天然 chr16 的正负两类来自同一完整 test，可恢复该 test 的混淆矩阵并检查一致性。WGA 与天然样本属于不同条件，不能直接拼接它们的 FP/FN 后宣称得到同一总体上的 Accuracy、F1、PPV 或 FDR。

如果关注相同 sensitivity 下的 WGA FPR，应先在独立 valid/校准集确定工作点，再分别应用于 Test1/Test2。仅展示低 FPR 或低 FNR 都不足以证明综合性能最好。

## 10. 已实现目录和输出合同

脚本已经实现，正式执行后按下述合同生成结果。完整命令见 [workflow/README.md](workflow/README.md)。默认推荐 predictions 路线：

```bash
bash /home/shj/ZJ-lab/Modification/DNA-5mC/eval/Test2/workflow/run_all.sh --test
bash /home/shj/ZJ-lab/Modification/DNA-5mC/eval/Test2/workflow/run_all.sh --dry-run
# 配置核对完成后，正式运行：
bash /home/shj/ZJ-lab/Modification/DNA-5mC/eval/Test2/workflow/run_all.sh --run
```

```text
eval/Test2/
├── README.md
├── workflow/                       # config.yaml、Step1/2/3 Python/Shell、测试及一键入口
├── data/
│   └── native_bed100/<version>/    # 仅路线 B 需要：test.pt / bin / metadata / 来源清单
└── output/
    ├── results/<version>/<run>/
    │   ├── metrics.json
    │   ├── read_predictions.tsv.gz
    │   ├── site_predictions.tsv.gz
    │   ├── false_negative_windows.tsv.gz
    │   ├── false_negative_sites.tsv.gz
    │   ├── context_metrics.tsv
    │   ├── coverage_metrics.tsv
    │   ├── threshold_metrics.tsv
    │   └── run_manifest.json
    └── comparison/
        ├── raw_summary.tsv
        ├── common_summary.tsv
        ├── common_aggregate.tsv
        ├── paired_differences.tsv
        └── comparison_manifest.json
```

`metrics.json` 至少记录：

- 数据身份：`dataset_kind`、`truth_source`、`library_id`、`reference_validation_status`、`partial_test`。
- 模型身份：版本、架构、策略、seed、最佳 epoch、检查点 SHA256。
- 阈值：read/site 的数值、来源及冻结规则。
- 每层指标：N_positive、TP、FN、FNR、Recall、区间及区间方法。
- 位点覆盖度、上下文和失败/no-call 计数；不可获得的上游分母显式为 null。
- 原始/共同集合标识、共同样本键、筛选条件、观测清单哈希。

`run_manifest.json` 保存原预测/特征/配置签名、代码版本、源行号或观测键映射、概率生成协议、主干/encoder/tokenizer 身份。推理时还需记录设备、精度和批次；文件大小与 mtime 的签名不能替代模型权重身份核对。

建议结果表：

| 模型 | 分类头 | Seed | 集合 | read N+ | read FN | read FNR% [CI] | site N+ | site FN | site FNR% [CI] | coverage / 阈值 |
| --- | --- | --- | --- | ---: | ---: | --- | ---: | ---: | --- | --- |
| 待评测 | 待评测 | 待评测 | 原始 / 共同 | — | — | — | — | — | — | 明确记录 |

## 11. 实施顺序与验收

1. **冻结实验清单**：确定参与比较的 run、原测试阈值、BED 来源状态和主次指标。
2. **先完成路线 A**：审计已有预测，筛选阳性，输出整体、上下文及 coverage 结果，不重新提取特征。
3. **统一比较分母**：匹配共同观测，重新聚合位点及 coverage，再生成 common aggregate 和差值区间。
4. **补齐质量与来源记录**：记录候选/成功/无调用数量，无法追溯的项目明确标注。
5. **有需要再运行路线 B**：已实现分块阳性缓存与分类头推理；CPU 小样本核对后再决定是否全量执行。
6. **联合检查 Test1 与 Test2**：并列天然 FPR、天然 FNR/Recall、WGA FPR；保持各自数据身份。

验收至少包括：

- 只使用 test，所有纳入标签为 1；没有依据预测值选择阳性。
- 标签二进制、metadata 和预测严格对齐；无重复原始 read–位点，位点标签无冲突。
- `TP+FN=N_positive`、`FNR+Recall=1`，且与同集合、同阈值原 `metrics.json` 的 FN/Recall 一致。
- 路线 B 的特征与输入阳性行逐字节一致；小样本概率和分类调用与路线 A 对应行一致。
- 原/共同集合分开命名；所有模型共享共同列表后再计算覆盖度和最终指标。
- 无阳性、空覆盖层、非有限概率、文件截断、阈值非法和输入变化均能显式处理。
- smoke 输出标记 partial_test，不与正式结果合并；写入临时目录，成功后发布，拒绝覆盖已有正式数据。

当前数据足以完成一次有意义的参考阳性漏检分析。由于它已经属于反复查看过的原 chr16 test，结论应保留“既有测试集的补充分析”这一范围；后续若根据 Test2 优化模型，需另留独立阳性文库或独立样本检验改进。
