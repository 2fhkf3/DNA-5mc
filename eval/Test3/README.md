# Test3：天然 chr16 测试集的 Coverage 鲁棒性评测方案

方案与实现日期：2026-10-18。本目录用于 **5 / 10 / 20 reads per site** 的 site-level 稳定性评测。已实现统一配置、五步 Python/Shell 脚本、一键入口及临时数据测试，尚未执行正式全量评测。具体操作见 [workflow/README.md](workflow/README.md)，配置见 [workflow/config.yaml](workflow/config.yaml)。

**主实验：先固定同一批具有至少 20 条共同可用 read 的 chr16 测试位点，再在每个位点无放回抽取 5、10、20 条 read；所有模型、分类头和训练 seed 使用相同抽样清单，比较 site-level 性能、性能下降及抽样波动。**

## 1. 这个实验回答什么问题

本实验回答：在同一批既有测试位点上，将用于位点汇总的 read 数从 20 减少到 10 或 5 时，模型的位点分类性能下降多少，结果是否容易受抽到哪些 read 影响。

这里的 Coverage 必须明确为：

| 本实验设置 | 实际含义 |
| --- | --- |
| 5× | 每个入选位点恰好使用 5 条不同原始 read 的窗口概率 |
| 10× | 每个入选位点恰好使用 10 条不同原始 read 的窗口概率 |
| 20× | 每个入选位点恰好使用 20 条不同原始 read 的窗口概率 |
| Full-common | 同一批入选位点，使用全部共同可用 read，作为辅助参照 |

更准确的图表标题为 **Site-level performance at 5, 10 and 20 reads per site**。这不是将整条染色体或整份文库制备为平均 5×/10×/20× 测序深度：同一条长 read 在不同位点可被分别选中或未选中，不能据此计算真实低深度文库的位点检出率或未覆盖率。

现有分类头分别对各 read–位点窗口输出概率，再对同一位点求均值；read 数不作为分类头输入。因此本实验主要评估 **有限 read 汇总造成的位点判断变化**，不评估低深度重新测序对 basecalling、信号预处理、比对和覆盖缺失的影响。

## 2. 数据来源与当前可行性

### 2.1 使用完整天然 test 的正负两类

使用此前天然样本 chr16 test：

- `label=0`：原构建流程保留的 BED 修饰比例为 0% 的参考阴性。
- `label=1`：原构建流程保留的 BED 修饰比例为 100% 的参考阳性。
- 文库 ID：`NATIVE_250F500490011`，跨模型和 Test1/Test2 保持一致。
- 原 `source=mod` 表示天然来源；两类都保留该 source，不能将它当成阳性标签。
- 当前沿用正链、原测试划分及既有特征协议，不额外加入 train/valid。

**本实验同时需要 label=0 和 label=1。** 不能直接使用 Test1 的全阴性子集或 Test2 的全阳性子集来计算完整 AUROC、F1、Precision 等指标。

这些 BED 标签是既有参考标签，BED 的实验来源和逐分子准确性尚未在本流程中核实，统一记录 `reference_validation_status: unverified`。结果描述为“相对于既有参考标签的位点分类性能”；不把同一位点所有 read 的真实修饰状态视为已逐分子实验确认。

### 2.2 可直接复用的输入路径

| 模型 | 架构 | 原 test 特征目录 | 已完成分类头预测目录 |
| --- | --- | --- | --- |
| V011 | Diffusion | `/home/shj/ZJ-lab/Modification/DNA-5mC/train/data/V011-V2` | `/home/shj/ZJ-lab/Modification/DNA-5mC/train/output/V011-v2/runs` |
| V543 | BERT | `/home/shj/ZJ-lab/Modification/DNA-5mC/train/data/V543-V2` | `/home/shj/ZJ-lab/Modification/DNA-5mC/train/output/V543-v2/runs` |
| V554 | BERT | `/home/shj/ZJ-lab/Modification/DNA-5mC/train/data/V554-V2` | `/home/shj/ZJ-lab/Modification/DNA-5mC/train/output/V554-v2/runs` |
| V610 | GPT | `/home/shj/ZJ-lab/Modification/DNA-5mC/train/data/V610-V2` | `/home/shj/ZJ-lab/Modification/DNA-5mC/train/output/V610-v2/runs` |

优先读取各 run 中：

```text
<strategy>_seed<N>/
├── config.json
├── metrics.json
├── best.pt
├── test_predictions.tsv         # 原始逐窗口正类概率，主输入
└── test_site_predictions.tsv    # 原始全部 read 汇总结果，仅用于审计/基线核对
```

`test_predictions.tsv` 的实际列为：

```text
read_id  chrom  center_genome_pos0  strand  source  label  prob  split  seq  ref
```

必须从逐窗口预测抽样后重新聚合，不能对已有 `test_site_predictions.tsv` 的一条平均分“抽样”，也不能复制同一个全覆盖位点概率充当三种深度的结果。

各模型对应的 `test.pt`、标签数组、`metadata_test.tsv` 和 `feature_summary.json` 用于校验来源及行对应关系。若缺少旧预测，则对该主干已有的 test 特征完整推理一次，保存逐窗口概率；不需要为每个深度、每个重复重新提取隐藏状态。

### 2.3 本次只读核对的覆盖度

以下来自四组 MLP seed1 的实际 `test_site_predictions.tsv`，同时核对了 `metrics.json` 的窗口数：

| 模型 | 原 test 窗口数 | 原 site 数 | 阴性 site 数 | 阳性 site 数 | 阴性 coverage 范围 | 阳性 coverage 范围 | 自身 coverage≥20 的 site 数 |
| --- | ---: | ---: | ---: | ---: | --- | --- | ---: |
| V011 | 2,590,329 | 104,616 | 53,302 | 51,314 | 20–51 | 19–58 | 104,615 |
| V543 | 2,590,340 | 104,616 | 53,302 | 51,314 | 20–51 | 20–58 | 104,616 |
| V554 | 2,590,340 | 104,616 | 53,302 | 51,314 | 20–51 | 20–58 | 104,616 |
| V610 | 2,590,329 | 104,616 | 53,302 | 51,314 | 20–51 | 19–58 | 104,615 |

其中 `chr16:73837520:+` 的参考标签为 1，在 V011/V610 中只有 19 条可评测 read，在 V543/V554 中为 20 条。它必须退出主实验的 **全部三种深度**，不能只退出 20×。

上述数字尚不是最终共同位点数。四模型窗口数存在 11 条差异；即使某个位点在各模型中各有 20 条，也不能仅凭计数断定这 20 条 read 相同。正式流程仍需匹配 read ID，取交集后重算 coverage。最终位点数及正负比例以该审计为准，不在方案中假定为 104,615。

## 3. 建议先冻结的实验协议

| 项目 | 主实验建议 |
| --- | --- |
| 测试染色体与链 | 原 chr16 test，正链 |
| 数据类别 | 同时使用原 label=0/1，不额外重平衡 |
| 模型 | 首批 V011、V543、V554、V610；V548 等后续单独冻结新版本协议 |
| 分类头 | 既有五种策略，见下表 |
| 检查点 | 原 valid 选择的 best.pt，不根据 Test3 重新挑 epoch |
| 位点集合 | 所有纳入模型/策略/run 的共同观测中，coverage≥20 的位点 |
| 深度 | 5、10、20，不根据评测结果临时改变 |
| 抽样 | 每个位点等概率、无放回、深度之间嵌套 |
| 抽样重复次数 | 50 次，`repeat_id=0..49`；这是计划参数，不是已完成重复 |
| 抽样主 seed | 20261018，与训练 seed 和 bootstrap seed 分开 |
| site 汇总 | 抽中窗口的正类概率算术平均 |
| site 阈值 | 0.5，恰等于阈值判阳性 |
| 主观察指标 | Site AUROC、Balanced Accuracy，以及从 20 降至 5 的性能损失 |
| 次要指标 | AP/AUPRC、Acc、F1、MCC、FPR、FNR、Recall、Brier、抽样稳定性 |
| 区间 | 抽样分布范围与基因组区块 bootstrap 区间分别命名 |
| 结果状态 | 测试前缀为 partial，不能进入正式比较 |

五种分类头：

| 策略名 | 含义 |
| --- | --- |
| `center_base_linear_raw768` | 中心碱基线性探针 |
| `center_5mer_linear_raw768` | 中心 5-mer 线性探针 |
| `center_5mer_mlp_raw768` | 中心 5-mer MLP |
| `center_5mer_residual_dwconv_raw768` | 中心 5-mer Residual DWConv |
| `center_5mer_emb8_none` | 隐藏状态与碱基嵌入融合头 |

训练 seed 不固定要求为 1/2/3。当前部分策略使用 4/5/6 或 2/3/4；冻结清单时记录实际已完成的 runs。不得根据低深度成绩只保留表现好的 seed。

## 4. 观测键、固定测试位点与公平性

### 4.1 两个必须区分的键

物理位点键：

```text
(library_id, chrom, center_genome_pos0, strand)
```

逐窗口观测键：

```text
(library_id, read_id, chrom, center_genome_pos0, strand)
```

同一原始 read 在同一位点只能贡献一个窗口；若有重复、标签冲突、ref 冲突或来源冲突，应中止审计并定位原因，不能将重复窗口当成多条 read。`source` 需要保留并核对，本实验单一天然来源不把它作为人为拆分物理位点的依据。

### 4.2 先固定共同 read 池，再固定 coverage≥20 的位点

令 `a` 表示冻结清单中的一个“主干—分类头—训练 seed”运行，`R_a(j)` 是其在位点 j 成功预测的原始 read 集合。定义：

$$
R_{common}(j)=\bigcap_a R_a(j),\qquad
n_j=|R_{common}(j)|,\qquad
J=\{j:n_j\geq20\}.
$$

主分析在整个协议中只使用这一套 J，而非每个深度或每个分类头分别取一套位点。步骤顺序必须为：

```text
冻结所有待比较 run
  → 核对原标签和完整预测
  → 以观测键取所有 run 的交集
  → 核对相同观测的 label/seq/ref
  → 按物理位点统计共同 read 数
  → 保留 n_j≥20 的固定 J
  → 再构造 5/10/20 抽样
```

若同一主干的所有分类头/seed 确实共享完全相同的观测清单，可通过清单哈希证明后复用一次索引，减少重复计算；不能未经核对直接假定相同。

筛选仅依赖测试划分、已有质量规则、数据完整性和 coverage，不依赖概率高低、预测是否正确或模型间差异。主分析不按 label 再筛成等量位点；三种深度自然继承同一个 J 的类别比例。应分别记录 N0、N1 和阳性比例，避免把 read 数平衡误称为 site 数平衡。

如果增加新主干后共同集合变化，建立新协议 ID、重建全部比较结果；不能将新集合成绩直接拼入旧集合的表格。不同集合的补充分析应另列。

## 5. 分步评测过程

### Step0：冻结配置与来源清单

1. 列出参与的模型、策略、全部训练 seed、run 路径与 checkpoint SHA256。
2. 保存原 `config.json`、最佳 epoch、验证集选择指标、训练轮数/早停、特征提取协议和特征签名。
3. 固定文库、染色体、链、coverage、重复次数、抽样 seed、site 汇总规则及阈值。
4. 确认每个 run 是已完成的 test 预测，拒绝半截文件和开发联调数据。
5. 保存软件实际版本、代码版本、配置哈希及 protocol ID，例如 `chr16_common_cov20_v1`。

旧 TSV 没有内嵌生成它的检查点哈希。可核对 run 配置、epoch、特征签名及原 metrics 中的混淆矩阵，但本次读取 best.pt 的哈希不能单独证明旧 TSV 的历史来源。记录 `historical_prediction_binding: consistency_checks_only`；来源无法确认时，用冻结权重完整推理一次。

**正式输出**：根目录 `protocol.yaml`、`manifests/run_inventory.tsv`、`manifests/audit_manifest.json`。当前仅在临时目录测试，未生成正式清单。

### Step1：导入完整正负预测并审计

1. 逐行对应 `test_predictions.tsv`、`metadata_test.tsv` 与二进制标签。
2. 检查原始 read ID、chr16、坐标、正链、source=mod、split=test、label 及 seq/ref。
3. 检查概率有限且位于 `[0,1]`，保留原始 `P(label=1)`，不反转概率。
4. 拒绝重复观测、标签冲突、错位、缺行、额外行、输入在处理中变化等情况。
5. 对全部原始预测按原阈值重算 read/site 混淆矩阵，与原 metrics 对照；同时检查原位点 read 数。
6. 导出输入观测索引，保留源文件与源行号，便于追查抽样中的任一窗口。

不得只导入已预测正确的窗口，也不得从 Test1/Test2 的 FN/FP 清单反向挑选测试数据。

如果少数 run 缺少预测，先补齐或在实验开始前明确调整冻结清单。协议冻结后出现缺失不能按深度临时删除位点；记录为失败/no-call，修复后重跑，或建立新的明确版本。

**预期输出**：完整正负预测缓存、观测索引、重复/标签审计、各模型原始 coverage 表。

### Step2：构建固定共同 read 池和位点清单

按第4节求交集，重算每个位点的真实可用 read 数 n_j，再保留 n_j≥20。

`fixed_sites.tsv` 至少包含：

```text
site_index library_id chrom position strand label ref5
n_common_reads original_coverage_by_model retained exclusion_reason
```

建议另存 `excluded_sites.tsv`，逐位点记录共同 coverage 不足、共同观测缺失等原因。将固定 J、正负标签、共同 read 池及索引顺序分别计算 SHA256。所有后续结果都引用这些哈希。

验收时明确报告：原始位点数 → 各模型共同位点数 → 共同 read coverage≥20 的位点数，及每一步的 label=0/1 数量。不能把原 test 的 104,616 直接当作最终 J 的大小。

### Step3：生成跨模型通用的嵌套抽样清单

对每个位点 j、每次重复 r：

1. 从固定共同 read 池 R_common(j) 生成一个与模型分数无关的均匀随机顺序。
2. 取前 20 条不同 read 作为本次 20×。
3. 取其中前 10 条作为 10×，前 5 条作为 5×。
4. 所有主干、分类头、训练 seed 读取这份清单，禁止各自重新抽样。

因此同一次重复满足：

$$
S_{j,r,5}\subset S_{j,r,10}\subset S_{j,r,20}\subseteq R_{common}(j),
\qquad |S_{j,r,d}|=d.
$$

可采用如下稳定、与输入行顺序无关的确定性实现：对每条候选 read 计算规范化 JSON 编码的

```text
SHA256([sampling_seed, repeat_id, library_id, chrom, position, strand, read_id])
```

以摘要排序，read_id 作为极小概率哈希相同情况下的次级排序。不同 repeat_id 生成新的伪随机排序；**排序键不包含模型名、分类头、训练 seed、depth、label 或 prob**。该方案的排序及“随机性”规则应写入清单；固定 seed 并不代替保存抽中 read 的审计记录。

若使用 NumPy 实现等概率无放回抽样，显式设 `replace=False`，不要沿用其默认可重复抽样。对同一随机顺序取前缀即可得到嵌套深度；需同时固定输入 ID 顺序和 RNG 版本。定义见 [NumPy 官方 choice 文档](https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.choice.html)。

当 n_j=20 时，20× 在 50 次重复中总是同一组 20 条 read；只有其 5×/10× 子集变化。这是有限 read 池的正常结果，不应该通过有放回复制来强行制造 20× 波动。不同重复允许抽到相同组合，但同一次同一位点内不得重复同一 read。

**预期输出**：每个 repeat 的排名前20观测索引、统一 read ID 查找表、抽样算法说明及抽样清单哈希。无需导出三份隐藏状态或三份原始信号。

### Step4：按清单重新聚合 site 概率

对主干 m、分类头 h、训练 seed s、重复 r、深度 d，计算：

$$
q_{m,h,s,r,d}(j)=\frac{1}{d}\sum_{i\in S_{j,r,d}}p_{m,h,s}(i,j),
\qquad
\hat y_{m,h,s,r,d}(j)=\mathbf1[q_{m,h,s,r,d}(j)\geq0.5].
$$

沿用当前训练评测的 **平均概率** 聚合，不临时替换为多数投票、平均 logit、最高概率或其他能使某个模型更好的规则。使用 float64 做求和和除法；用于判定的原始概率不能为方便储存而随意舍入。

同一位点在三种深度都使用同一个参考 label，不重新按抽中的 read 生成 BED 标签。每个位点在 site-level 指标中只贡献一次，权重相同，不按原始 coverage 加权。

同时生成 Full-common：在固定 J 上使用各位点全部共同 read 的平均概率。它是同集合辅助基线，不能拿不同位点集合、不同 read 池的原 test site 指标直接替代。

主流程只需对既有概率做索引、求和和排序计算，通常无需 GPU。现有模型对各窗口独立推理，因此复用冻结的逐窗口概率与对同一批选中窗口重新推理具有相同的汇总定义；数值实现可能因设备/算子产生微小差异，若更换推理来源应另记协议并核对阈值附近调用。

### Step5：逐 run、逐深度、逐重复计算性能与稳定性

每个 `(version, strategy, training_seed, repeat_id, depth)` 生成一行 site-level 指标。先保留这些原始结果，再做训练 seed 和抽样重复两个维度的汇总，不能只保存一条平均曲线。

此外，按同一位点、同一 repeat 配对比较 5 vs20、10 vs20；按同一抽样清单比较不同模型/分类头。具体指标见第6节。

### Step6：汇总、区间与可视化

输出按训练 seed 汇总、按模型/策略汇总、深度差值、模型差值、上下文分层及来源清单。抽样波动与空间抽样不确定性分开报告，见第7节。

展示 5/10/20 曲线及相同测试位点上的性能下降，不根据当前结果挑选最有利的一次随机重复。所有图都标明固定 N_sites、正负数、训练 seed 数和抽样次数。

## 6. 评测指标：性能和稳定性都要看

### 6.1 每次重复的 site-level 分类指标

以固定 J 上的参考标签 y 和平均概率 q 计算：

| 指标 | 作用与报告要求 |
| --- | --- |
| AUROC | 主指标之一，衡量位点正负排序能力；各深度使用相同 J |
| Balanced Accuracy | 主指标之一，`(Recall + Specificity)/2`，同时关注两类 |
| AUPRC / AP | 沿用项目 average precision 定义；明确当前阳性比例，不与不同阳性率/类别比例的数据直接比较 |
| Accuracy | 位点总体正确率 |
| F1、Precision、MCC | 补充阈值表现；F1 的正类为 label=1 |
| FPR、Specificity | 阴性位点在低 read 数下是否更容易误报 |
| FNR、Recall | 阳性位点在低 read 数下是否更容易漏检 |
| Site Brier / BCE | 参考二分类标签下的分数误差；BCE 记录 clip epsilon |
| N、N0、N1、TP、TN、FP、FN | 每一行均保存，避免缺失分母或混淆矩阵含义 |

定义：

$$
FPR=\frac{FP}{FP+TN},\quad FNR=\frac{FN}{FN+TP},\quad
Recall=\frac{TP}{TP+FN},\quad BA=\frac{Recall+Specificity}{2}.
$$

Balanced Accuracy 的类别平均召回定义见 [scikit-learn 官方文档](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html)。AP 使用 recall 增量加权的 precision，而非直接把 PR 曲线按梯形积分；见 [average_precision_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html)。正式实现需记录实际软件版本。

分母为零时留 NA 并说明；单类别联调子集不能产出有效 AUROC。不要将 NA 填成 0 后参与组均值。

平均概率 q 不是经验证的真实甲基化百分比。本数据只含原标签0/1的极端位点，因此不把它作为部分甲基化频率估计基准，也不据此声称证明频率 MAE 或相关性性能。

### 6.2 相对 20× 的性能损失

对 AUROC、BA、F1 等越高越好的指标 Z，定义：

$$
L_Z(5)=Z(20)-Z(5),\qquad L_Z(10)=Z(20)-Z(10).
$$

正值代表低深度下降，负值表示该次抽样低深度分数更高，保留符号，不截断为0。对于 FPR/FNR/Brier 等越低越好的指标，使用：

$$
L_E(d)=E(d)-E(20).
$$

差值先在相同模型、训练 seed、repeat 上配对计算，再汇总。AUROC/BA/F1 的绝对差可展示为百分点，例如差值0.01等于1个百分点；不把相对百分比和百分点混写。

不要只看“曲线很平”：一个模型可能在 20× 和 5× 都很差却非常稳定。至少同时展示 `5×绝对性能 + 20→5损失 + 抽样波动 + FPR/FNR`。

### 6.3 对抽到哪些 read 的敏感性

每个固定训练 seed、深度下，计算 50 次抽样指标的均值、样本标准差 `ddof=1` 和经验分位数。波动越大，说明结果更依赖被抽到的 read。

对每个位点，还可保存：

1. `score_sd`：该深度下 q 在50次抽样中的标准差。
2. `call_instability`：`min(π,1-π)`，π 为50次中判阳性的比例；范围0–0.5，0代表调用不变。
3. `error_frequency`：50次中与参考标签不一致的比例；稳定但一直错误的位点在这里仍会暴露。
4. `flip_5_vs20`：同一次抽样中5×与20×调用不同的位点比例；10×同理。
5. 与 Full-common 分数的平均绝对差，作为汇总稳定性描述，不称为对真实甲基化比例的误差。

对这些位点统计分别按原 label、5-mer ref 上下文及原共同 coverage 分层。上下文属于探索性描述；小样本 motif 标注 N，不依据其误差再选择主测试集合。

### 6.4 预期变化与有限 read 池

对固定模型、位点，等概率无放回抽样的均分满足：

$$
\mathbb E[q_d\mid R_{common}]=q_{Full},\qquad
\operatorname{Var}(q_d\mid R_{common})=\left(1-\frac{d}{n_j}\right)\frac{S_j^2}{d},
$$

其中 S_j² 为该模型在完整共同 read 池上概率的样本方差，分母为 n_j−1。这是本方案有限总体抽样设计下的数学性质，不是当前已观测的模型结果。

较少 read 通常会增加均分波动，但分类阈值、模型偏差和排序会共同影响 AUROC/F1，因此不能要求每个 seed、每次重复的所有性能都随深度严格单调提高。验证单调性只能作为结果观察，不能作为删除“不好看”结果的规则。

## 7. 重复、训练 seed 和不确定性如何报告

### 7.1 三种数量分别记录

| 数量 | 本方案中的含义 | 不能代替什么 |
| --- | --- | --- |
| N_sites | 同一冻结测试集中的可评测位点数 | 不是独立文库数 |
| R=50 | 同一位点和有限 read 池上的随机下采样重复 | 不是50份新生物样本 |
| S | 各模型/策略已冻结的训练 seed 数 | 不是独立测序实验重复 |

目前是既有单一天然文库的补充分析，不能写成 `n=50 biological replicates`，也不能把 `S×50` 当成独立实验样本量后做普通独立样本 t 检验。

### 7.2 建议的汇总顺序

记一个指标为 Z_s,r,d：

1. **逐训练 seed**：对 r 求均值、SD及2.5%/97.5%经验分位数，保留完整分布。
2. **组均值**：先得到每个 seed 的50次均值，再在所有冻结 seed 间等权平均。
3. **训练随机性**：报告各 seed 均值之间的样本 SD，`ddof=1`；只有1个 seed 时为 NA。
4. **下采样随机性**：单列逐 seed 的抽样 SD，或报告这些 SD 的均方根；不和训练 SD 混为一个不明来源的“±”。
5. **抽样带**：也可对每个 repeat 先求所有 seed 的指标平均，再取其经验分位数，命名为“固定已训练模型下的下采样分布范围”。这不是95%生物学置信区间，也不是均值标准误。

分类头比较仍使用每个分类头独立的原始概率；在不同训练 seed 的“指标”上求均值，不先把 seed 概率做 ensemble 后当成单模型成绩。

### 7.3 深度与模型差异的区间

主分析以效应差值和波动为重点。预先定义空间区间分析：chr16 每1Mb为一个基因组区块，1000次配对 block bootstrap，bootstrap seed为42。

- 相同重采样区块权重用于所有深度、模型和抽样 repeat，保持比较配对。
- 阈值指标可先按 `(block,run,repeat,depth)` 保存 TP/TN/FP/FN，每次重采样合并计数，重新计算全局指标及差值，再对50次/训练seed汇总。
- 不平均“各区块的 BA/F1”来代替全局 BA/F1；区块类别组成和大小不同。
- AUROC/AP 不是简单的区块计数函数。若需要其 block CI，必须在每次空间重采样后的位点分数及权重上重算全局 AUROC/AP，不能平均区块 AUC。
- 为控制计算量，默认区块区间可先用于预先指定的 BA、FPR/FNR 及其深度差值；AUROC/AP 仍报告全部50次的抽样分布，不能将该分布伪装成空间区间。若扩展其空间区间，另记录运行范围和计算预算。
- 少区块、零事件或退化区间明确标记，不将 `[0,0]` 解释为总体绝无误差。空间区间仍仅描述当前文库内部的变动，不能代替独立文库验证。

不同模型训练 seed 不一致时，仍可比较各自全部冻结 seed 的指标均值；因测试位点、抽样清单一致，空间/抽样比较仍然有共同观测。若要对“训练 seed”逐一配对，只匹配实际共有的 seed，另报有效配对数，不凭序号将 seed4 与 seed1 强行配对，也不因此删除整个分类头。

不默认生成大量显著性星号。若后续做正式假设检验，先冻结比较家族、方向和多重比较方法，不能看完结果后挑显著深度或 motif。

## 8. 推荐图表与结果表

### 8.1 图形

1. **性能曲线**：横轴5/10/20 reads per site；纵轴AUROC或BA。每个分类头独立面板，颜色区分主干。可显示各训练seed的细线和组均值粗线；抽样带注明定义。
2. **低深度损失图**：20→5的AUROC/BA下降，辅以训练seed散点；同时保留5×绝对性能。
3. **FPR/FNR曲线**：检查低深度变化是否对正负两类不对称。
4. **稳定性分布**：5/10/20的逐位点score_sd、error_frequency或与20×调用翻转率；按label分开。
5. **探索性上下文表/热图**：固定5-mer上展示深度变化与N，避免只展示最高错误motif。

Step5 已支持 AUROC、BA、FPR/FNR 曲线 PNG/PDF；当前只用临时样本验证图形导出。逐位点稳定性和上下文表也会导出，可用于进一步绘制分布图或探索性热图。

### 8.2 主结果表建议

| Backbone | Head | Depth | N sites / N0 / N1 | Training seeds | AUROC | AP | BA | F1 | FPR | FNR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 待评测 | 待评测 | 5 | 固定J | 实际seed列表 | 待计算 | 待计算 | 待计算 | 待计算 | 待计算 | 待计算 |
| 同上 | 同上 | 10 | 与5×完全相同 | 同上 | — | — | — | — | — | — |
| 同上 | 同上 | 20 | 与5×完全相同 | 同上 | — | — | — | — | — | — |

表下注明：均值先跨50次下采样再跨训练seed；括号或±具体表示训练SD还是抽样SD。另用稳定性表报告20→5损失、抽样SD、调用翻转率，不把不同来源的不确定性合成一个无法解释的数。

## 9. 已实现代码、配置与输出

五个步骤均已有 Python 和带中文参数注释的 Shell 入口，详见 [workflow/README.md](workflow/README.md)。默认入口只运行临时测试，正式运行须显式选择 `--run`：

```bash
bash /home/shj/ZJ-lab/Modification/DNA-5mC/eval/Test3/workflow/run_all.sh --test
bash /home/shj/ZJ-lab/Modification/DNA-5mC/eval/Test3/workflow/run_all.sh --dry-run
# 核对 config.yaml 后执行正式流程：
bash /home/shj/ZJ-lab/Modification/DNA-5mC/eval/Test3/workflow/run_all.sh --run
```

主要配置为 `enabled_versions/strategies/training_seeds`、`depths`、`minimum_common_reads`、`repeats`、`sampling_seed`、`site_balance`、`site_threshold` 和 `bootstrap`。所有路径采用绝对路径，配置与 run 清单在 Step1 冻结。

默认 `site_balance: none` 保留所有合格位点。严格1:1的补充实验使用 `equal` 并指定新的协议目录：按 `balance_seed` 只选一次等量正负位点，随后所有模型/深度/抽样重复共用，不复制 read、不逐深度重新平衡。

```text
<output_dir>/
├── protocol.yaml
├── manifests/                 # 冻结run、审计与环境版本
├── audit/versions/<version>/  # 每主干一份原始观测索引
├── audit/runs/<version>/<run>/ # 每run的完整正负概率向量与来源
├── common/                    # 固定位点、排除表、共同观测数据库及各主干行号映射
├── sampling/                  # [repeat,site,max_depth] .npy索引及抽样清单
├── results/<version>/<run>/   # 逐repeat指标、Full-common、区块计数、上下文及逐site稳定性
└── summary/                   # seed/模型汇总、深度/模型差值、区间、图形及清单
```

为减少磁盘开销，已实现版本用 NumPy `.npy/.npz` 保存抽样索引和区块计数、SQLite 保存共同观测，替代重复导出大规模逐read文本。主要性能表仍为 TSV；实际文件名和列含义见操作文档。

`metrics_by_repeat.tsv` 包含协议、主干、策略、训练seed、repeat、depth、位点数/正负数、TP/TN/FP/FN以及全部核心指标。每阶段 `manifest.json` 保存配置与文件SHA256，结果绑定同一共同集合和抽样；源标签状态保留 `unverified`。

## 10. 计算与存储安排

默认流程无需复制特征或原始信号，也不对50次重复分别调用主干。一次导入逐窗口概率后，按共享索引计算三种位点均分即可。

以 **约104,615个位点作为规模估算，不作为最终纳入数**，4主干×5分类头×3训练seed=60个runs、50次抽样时：

- 主指标表约为60×3×50=9,000行，适合集中保存。
- 同一重复只需保存每个位点20个观测索引，5/10由前缀恢复；50×104,615×20个int32约418MB，另需ID查找表和清单。
- 若保存所有run、所有重复和深度的逐位点分数，共约9.42亿个分数；float64纯分数约7.53GB，尚未计ID、表格和压缩开销。默认不导出逐重复的大型TSV。
- 阈值区块bootstrap优先使用区块混淆计数；位点稳定性可流式累积sum/sum²/阳性调用次数，不必把全部重复读入内存。
- 全局AUROC/AP仍需每次重复对全体site分数计算；不能用区块AUROC平均值替代以节省时间。

真实所需资源由最终J、run数量、输出选项和软件实现决定；执行前做磁盘空间检查。不要为每个depth/repeat重复保存约10GB级的筛选隐藏状态。

## 11. 实现后的验收清单

1. 数据只来自原chr16 test，正负标签不变，参考来源与文库身份明确。
2. 原始预测与元数据、标签严格对齐，原混淆矩阵可复核，重复观测被拒绝。
3. 共同read池在所有选定run之间一致；最终固定J的每个位点均有至少20条不同read。
4. 三种深度的位点ID、标签、N0/N1完全相同，每次位点贡献read数严格为5/10/20。
5. 同一次重复满足5⊂10⊂20；抽样不依赖label、prob、模型和训练seed。
6. 模型间选中read ID完全相同，改变输入行顺序、工作进程数或批次大小不改变抽样结果。
7. 单次抽样无重复read，跨重复可出现相同组合；n_j=20时20×不变。
8. 手算样本能复核平均概率和TP/TN/FP/FN；q=0.5判阳性；全0/全1概率、并列分数、单类别子集和空集合显式处理。
9. 使用某个位点全部read时，均分能复现对应共同read池的Full-common结果，而非不相同原始池的结果。
10. 指标按site等权，训练seed、抽样重复、位点数和文库数在表格中分别记录。
11. 经验抽样范围不冒称生物学CI，区块bootstrap保持配对，不将区块AUC平均成全局AUC。
12. 小样本测试只写临时目录；正式数据有配置/清单/检查点/代码哈希，失败不发布部分结果，不覆盖旧协议。

## 12. 合理性、结论范围与后续补充

这个设计适合当前目标：已有test大多具有20条以上可用read，能够在不更换位点和参考标签的条件下，受控地减少用于位点汇总的证据数量，并比较各主干/分类头对read抽样的敏感性。

解释结果时保留以下边界：

- 固定J来自原本高覆盖、通过既有质量与缓存条件的位点，因此不能外推为全基因组低覆盖位点的表现。
- 原test经过既有筛选与类别处理；本实验评价该缓存分布，不还原未经处理的天然覆盖度分布和人群阳性率。
- BED=0/100参考的可靠性、天然分子异质性及潜在caller标签偏差并未被下采样消除。
- 50次重复复用同一有限read池，20×附近的波动受原pool大小限制；低波动不等于独立新文库也低波动。
- 同一完整read可覆盖多个位点，邻近site仍有关联；区块区间只部分处理这种相关性，不提供独立生物重复。
- 这是已多次查看过的既有chr16 test补充分析。若据此选择模型或修改聚合规则，后续需用未参与优化的样本/文库验证改善。

若后续目标变为“真实平均5×/10×/20×测序的端到端表现”，另建立按整条原始read统一保留或基于BAM/POD5的全局下采样实验，让同一read在其覆盖的所有位点同时保留/删除，重新统计实际coverage、未覆盖位点及call rate。该实验与本次固定site、固定read数的汇总鲁棒性分析分别报告。
