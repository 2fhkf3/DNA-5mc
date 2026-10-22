# Test4：HG002 chr16 甲基化频率与独立 WGBS 的相关性评测方案

方案与实现日期：2026-10-20。状态：**已实现六步工作流与测试；尚未扫描全量信号或计算正式相关性结果。** 操作入口见 [工作流说明](workflow/README.md) 和 [实际配置](workflow/config.yaml)。

目标：使用现有冻结 5mC 模型，对 HG002 chr16 的天然测序信号做逐 read–CpG 推理，聚合为连续的位点甲基化频率，与独立 WGBS 的位点甲基化比例配对，报告 Pearson 相关系数及绝对误差。与 Test1/2/3 不同，**主实验保留 0%～100% 的全部有效参考比例，不只选择 0%/100%，不做正负平衡。**

用户已确认给定 BED 是独立 WGBS 结果；参考基因组版本、WGBS 原始样本/文库信息、BED 转换流程和信号归一化历史仍需在正式运行前记录。

## 1. 参考实验与本实验能回答的问题

[华大智造发布的 CycloneSEQ 功能介绍](https://www.instrument.com.cn/netshow/SH104318/news_797100.htm) 报告：NA12878 的甲基化检测结果与 WGBS 在覆盖 ≥5× 的 CpG 位点上，Pearson `r=0.9267`，发布时间为 2025-07-10。

本实验借鉴“连续位点甲基化频率与 WGBS 做相关分析”的设计，但当前是 **HG002、chr16、正链 CpG 子集**。网页未充分说明该图的双平台覆盖门槛、逐 read 判定规则、链合并规则、过滤及完整实现，不能将其数值当作当前数据必须达到的验收线，也不能声称完全复现了原实验。以下参数是本方案预先提出的实验选择。

主问题为：

> 在参考基因组、坐标、链和有效覆盖度一致的 chr16 CpG 位点上，现有模型估计的甲基化频率，能否准确跟随独立 WGBS 的连续甲基化比例？

Pearson 高并不等价于绝对频率准确。例如，存在系统性偏高或偏低时，仍可能高度相关。因此必须同时报告误差、偏差及回归斜率，而非只给一个相关系数。

## 2. 已有输入与本次只读检查

### 2.1 两份输入

| 输入 | 实际路径 | 文件逻辑大小 | 作用 |
| --- | --- | ---: | --- |
| 独立 WGBS BED | `/home/shj/ZJ-lab/Modification/DNA-5mc/output/step2_input/bed/HG002_250F500490011.bed.gz` | 724,425,729 bytes，约 0.675 GiB | 位点甲基化百分比和参考覆盖度 |
| chr16 全 read 信号 JSONL | `/mnt/zzbnew/rnamodel/jinyue/data/processed_data/chr16/mod/s2_all_reads.genome_signal.preprocessed.jsonl` | 2,165,335,685,171 bytes，约 **1.969 TiB** | 逐 read 的信号、基因组映射及碱基信号区间 |

注意输入目录是 `DNA-5mc`，本实验输出项目是 `DNA-5mC`，大小写不同。输入保持只读，不覆盖、不搬迁。

### 2.2 BED 的实际字段

已读取前 20,000 条记录：全部为 11 列，修饰代码均为 `m`，包含 `+/-` 链，百分比范围为 0～100；其中 15,067 条为 `0 < percent < 100`。**这个前缀全部位于 chr10，不是 chr16 统计，也不是全文件统计。**

首条示例：

```text
chr10  2124  2125  m  4  +  2124  2125  255,0,0  4  100.00
```

拟按现有数据约定解析的关键字段如下，正式运行前仍需对照 WGBS 导出脚本确认：

| 列号，1-based | 含义 | 使用方式 |
| --- | --- | --- |
| 1 | 染色体 | 选择 `chr16`，记录命名映射 |
| 2–3 | BED 左闭右开区间 | 暂按 0-based `[start,end)`，核对 `end=start+1` |
| 4 | 修饰类型代码 | 选择 `m`，再用参考序列确认 CpG 上下文 |
| 6 | 链 | 主实验选 `+` |
| 10 | 参考有效覆盖度 | 主实验要求 ≥5；不使用第 5 列 score 替代覆盖度 |
| 11 | 甲基化百分比 | `wgbs_beta=percent/100`，保留所有 0～100 的有效值 |

这是 bedMethyl 风格的布局，**布局本身不能证明测序平台**；WGBS 来源在本方案中依据用户确认，后续需补充来源记录。标准 Bismark coverage 文件通常是另一种六列格式，坐标约定也不同，不能把不同格式直接套用同一解析器。[Bismark 官方格式说明](https://felixkrueger.github.io/Bismark/options/methylation-extraction/)、[modkit 官方字段说明](https://github.com/nanoporetech/modkit/blob/master/README.md?plain=1)

当前只有 11 列，没有直接提供 WGBS 的甲基化/未甲基化整数计数。**不把经过两位小数舍入的百分比乘以 coverage 后四舍五入，伪造原始计数。** 主实验可用现有 beta 与覆盖度开展；若需精确计数加权合并、二项模型或转换率校正，应取得原始计数表或明确可验证的转换记录。

### 2.3 信号记录与归一化疑点

已读取信号文件前 12 条完整记录，约 78 MB；未扫描全量 1.969 TiB 文件。样例字段包括：

```text
read_id, source_part, source_bam, fast5, chrom,
genome_start0, genome_end0, strand, mapq, cigar, seq,
signal, moves, base_sample_span_seq, per_base,
input_granularity, signal_base_shift, normal_mode, signal_calibration
```

`per_base` 中有 `genome_pos0`、`query_pos_signal`、`query_base`、`ref_base`、`signal_span`；可用于构建与训练一致的 15nt 窗口。样例为完整 read，包含正负链，`signal_base_shift=-4` 已编码到映射说明中，**不能在下游重复平移 -4**。

需要优先解决的疑点：样例 `normal_mode=fast5_calibrated`，校准描述仍写着 `no apple normalization`；但前三条信号的中位数约为 0、标准差约 0.88～0.89，提示数值已发生标准化。仅凭分布不能确认具体算法，文件名 `preprocessed` 也不能证明已执行正确的 Apple 流程。

正式提取前应找到上游预处理脚本/日志，核对实际 extractor、配置与代码版本，并与原训练输入中相同 read、相同位点的信号窗口做逐点抽样比较。确认一致才采用主干提取时 `preprocess=none`；若证实是原始电流，则按训练协议先处理整条 read，再裁窗口。不能对已标准化信号重复执行物理电流修复/Apple，也不能用逐窗口重新归一化替代原整条 read 处理。

**目前不能据此给出 chr16 的完整 CpG 数、有效 read 数或可配对位点数；由 Step1/2 实际审计后填写。**

## 3. 建议冻结的主实验协议

| 项目 | 主实验建议 |
| --- | --- |
| 协议 ID | `hg002_chr16_wgbs_correlation_v1`，确定全部输入与预处理后冻结 |
| 参考 | 用户确认的独立 HG002 WGBS；样本/细胞系批次、文库、处理软件和基因组版本需记录 |
| 区域与链 | chr16、正链 CpG；与现有训练协议一致 |
| 候选集合 | WGBS 中覆盖度 ≥5 的全部有效正链 CpG；不按甲基化比例挑选 |
| 信号质量 | MAPQ≥60、合法 15nt 信号窗口、中心参考 `CG`、沿用原训练的窗口序列一致性规则 |
| 主干与检查点 | 冻结既有 checkpoint，沿用 chr19 valid 选出的 `best.pt`，不在 chr16 上重选 |
| 两平台覆盖度 | 每个位点 WGBS coverage≥5，模型共同可评测的不同 read 数≥5 |
| 主频率估计 | 逐 read 正类概率 `p≥0.5` 计为甲基化；计数比例作为位点频率 |
| 辅频率估计 | 同一批 read 的正类概率均值；预先报告，不能只保留相关性更高的估计器 |
| 主指标 | Pearson `r`，每个位点等权 |
| 辅指标 | Spearman `rho`、MAE、RMSE、平均偏差、OLS 斜率/截距、配对位点数与覆盖情况 |
| 公平比较集合 | 所有纳入模型/分类头/seed 的共同有效 read–site 观测，重算共同覆盖度再定站点 |
| 跨 seed | 每个 seed 单独评测，再报告三个 seed 的均值与样本方差；不默认做概率集成 |
| 新训练/校准 | 主实验不新增训练，不用 chr16 WGBS 调阈值或校准概率 |

`coverage≥5` 的“双平台”要求是本方案的明确定义，网页并未充分交代其 ≥5× 具体指哪一侧。

### 3.1 为什么不能直接复用 Test3 的共同测试集

Test3 的 104,615 个位点来自原 BED 的 0%/100% 子集，还做过窗口类别下采样与共同覆盖度≥20 筛选。它没有覆盖完整的连续甲基化分布，而且被下采样后的 read 集合不一定保留原位点的真实观测比例。

因此，Test3 只能作为“与旧实验的极端位点一致性检查”，不能直接用作本实验主相关图的输入。Test4 要从当前完整 chr16 信号重新构建候选窗口，保留所有符合质量条件的不同 read，不做正负类别平衡，也不按预测分数筛 read。

当前 `workflow/step2_build_chrom19_dataset.py` 会排除中间甲基化比例，并要求二分类标签，不能原样调用生成 Test4。应增加 Test4 专用的**无逐 read 真值标签**窗口构建和推理适配层：`wgbs_beta` 只供最终位点配对，不伪装成每条 read 的 0/1 真值，也不输入模型。

### 3.2 样本与数据泄漏边界

核对训练为 chr19、Test4 为 chr16，没有训练/验证位点重叠；原始 read 跨染色体情况也应检查。WGBS 是独立检测来源，不代表其标签从未被研究流程使用：现有 chr19 训练标签和原 chr16 评测参考也来自此 BED，应如实记录。

已有 chr16 Test1/2/3 曾被用于性能分析，不能将本实验描述为从未查看过的全新盲测。此次应在得到连续位点相关性前冻结模型、阈值和过滤规则，所有预定组合都报告，不能按 chr16 结果挑最好模型再声称独立验证。

普通 WGBS 不能区分 5mC 和 5hmC，需在结果解释中保留这一检测对象差异；不能将所有不一致都归因于模型错误。[原始方法研究](https://pmc.ncbi.nlm.nih.gov/articles/PMC2811190/)

## 4. 坐标、CpG 和链的处理

1. **确认参考基因组。** 从原 BAM header、WGBS 流程记录取得参考 FASTA 及版本/校验值；当前不猜测为 hg19、hg38 或 T2T。先核对 chr16 长度和选定位置的上下文，不以哪个坐标偏移产生更高相关性来选择转换规则。
2. **统一为 0-based 正链 C 坐标。** JSONL 的 `per_base.genome_pos0` 为已有基因组位置；BED 暂按第 2 列 0-based，仍需确认转换历史。以 FASTA 验证位置 `p` 是 C，`p+1` 是 G；11 列 BED 的 `m` 不足以证明每行都是 CpG。
3. **主实验只用 `+`。** 正链 read、正链 CpG、链特异 WGBS 正链记录配对，保持与既有分类头的训练方向一致。若 WGBS 实际已合并双链，必须单独记录并说明与模型正链频率的可比性，不能把它冒充正链单独结果。
4. **双链 CpG dyad 为后续扩展。** 反链 C 在参考正链的 G 位置，需经上下文验证后映射到前一位的正链 C。只有格式确认为“反链 C 坐标”时才减 1；不能对所有负链数据机械减 1。合并参考时优先使用真实甲基化/未甲基化计数，不能简单平均两条百分比。当前正链训练模型也不能未经验证直接用于负链。
5. **重复和冲突。** 以 `(library_id, original_read_id, chrom, CpG_C_pos0, strand)` 去重；完全相同记录可保留一份，映射/信号冲突需报错或按预冻结规则排除。secondary/supplementary、低 MAPQ、重复原始 read 的处理要依原 BAM 标记核对，不能仅因 JSONL 没有相应字段就声称已剔除。

完整窗口缺失、变异导致 `seq!=ref`、信号区间非法、过长等原因分别统计。严格匹配筛选会限制可评测区域，尤其在 HG002 变异位点附近；这是适用范围的一部分，不宜直接外推到所有 chr16 CpG。

## 5. 位点频率与指标公式

对同一位点 `i` 的共同有效 read 集合 `R_i`，令 `n_i=|R_i|`，第 `j` 条 read 的模型正类概率为 `p_ij`。

### 5.1 两种频率估计

主口径为甲基化 read 的比例：

```text
beta_call(i) = sum_j I[p_ij >= 0.5] / n_i
```

辅口径为模型概率均值：

```text
beta_prob(i) = sum_j p_ij / n_i
```

二者都是位点频率的估计方式，但一般不相等。`beta_prob` 的频率解释依赖概率校准质量；`beta_call` 受阈值影响。两者均与相同 `wgbs_beta(i)` 配对并分别报告，不能把 Test3 用于位点二分类的“均值≥0.5”混同为甲基化频率。

如果将来采用正负置信阈值丢弃低置信 read，必须在训练/验证集固定阈值，报告保留率，并重新定义有效 coverage 及共同观测；本方案主口径不增加该过滤。

### 5.2 主指标与辅助误差

令 `x_i=wgbs_beta(i)`，`y_i=beta_call(i)` 或 `beta_prob(i)`，在同一个共同位点集合上计算：

```text
Pearson r = sum[(x_i-x_mean)(y_i-y_mean)]
            / sqrt(sum[(x_i-x_mean)^2] * sum[(y_i-y_mean)^2])

MAE       = mean(abs(y_i-x_i))
RMSE      = sqrt(mean((y_i-x_i)^2))
Bias      = mean(y_i-x_i)
```

Spearman 使用并列值的平均秩。OLS 拟合 `y=a+b*x`，报告截距 `a` 和斜率 `b`；这里仅用于描述偏差，不将拟合后的预测重新代入主相关性/误差。Pearson `r²` 与预测意义的 `R²=1-SSE/SST` 不混用，主表不强行增加 R²。

所有 beta 使用 0～1；MAE/RMSE/Bias 可附百分点评价，展示时注明乘以 100。位点等权是主口径，coverage 加权相关性如要做，只能另列补充，不能悄悄让高覆盖位点主导主指标。配对数不足或任一序列无方差时相关系数返回 NA 及原因，不填 0 或 1。

### 5.3 分层与不确定性

- 主报告为全部合格位点；同时按 WGBS beta 展示低、中、高区间，尤其单列 `0 < beta < 1` 和 `0.2 < beta < 0.8`，避免极端位点的双峰分布掩盖中间比例预测不准。各区间报告位点数，区间内相关性可能因动态范围缩小而下降，结合 MAE 解释。
- 分别按 WGBS coverage 和共同 read coverage 的 `5–9 / 10–19 / 20–49 / ≥50` 分层，保留分母、缺失率及误差；不同层不作为不同生物重复。
- 主实验使用各合格位点全部共同 read。若加做 5/10/20 reads/site，下采样应像 Test3 一样先固定共同 coverage≥20 的**新连续比例位点集**，再共享嵌套无放回抽样；不能复用 Test3 的极端标签集合。
- 建议以 chr16 上预定义 **1 Mb 区块**做 1,000 次配对 bootstrap，seed=42；每次抽区块后用对应原始位点重算 Pearson、MAE 等，不能平均区块 Pearson。模型间使用同一组区块抽样，直接报告配对指标差值的区间。有效区块不足或退化时保留状态，不伪造区间。
- 三个训练 seed 的波动与空间区块区间分开；一个 HG002 来源样本不因多个 seed 或 bootstrap 获得独立生物重复。主表均值/方差跨 seed；区块区间对应其预定义汇总统计量，不把 3×1000 次当作独立样本。

## 6. 模型范围与特征协议

### 6.1 推荐分两阶段推进

先完成 **V554 的三个主要分类头 × 三个训练 seed**：MLP、残差 DWConv、BiLSTM 融合，共 9 个运行，用于验证数据接口和频率聚合。该选择只是实施顺序，不能按 pilot 的 chr16 WGBS 相关性决定是否隐去其他预定模型。

正式扩展为已有六主干 V011/V013/V543/V548/V554/V610 × 五分类头 × 三 seed，共 90 个运行；两种线性探针作为容量基线。模型路径以当前 `train/backbone` 软链接及原 run 配置为准，分类头从 `train/output/<版本>-v2/runs/<策略>_seed<N>/best.pt` 加载。V557/V558 等新模型待完成对应训练后另立协议加入。

### 6.2 必须按 checkpoint 的原特征协议提取

一个 read 的 15nt 输入上下文先由冻结主干编码，再取中心 5 个碱基特征供分类头使用；不能把 15nt 输入直接改成 5nt 再编码，也不替换为 release 的 7nt/CpG4 池化向量。

需要冻结 encoder/decoder 权重及代码指纹、预处理、Codec 层、token 映射、特殊 token、隐藏层、final norm 的取值位置、逐碱基池化、裁切和精度。相同主干提取一次，供其全部分类头/seed 使用。

特别注意当前已有缓存中的实际协议：

| 模型 | 已有训练缓存记录，供新推理对齐 |
| --- | --- |
| V543/V548/V554 | BERT 最后层 `hidden_layer=-1`，Codec layer=0；需沿用原实现的特殊 token、隐藏状态和池化路径 |
| V011 | `ode_steps=1`、`ode_start_t=0.97`、CFG=0.5，不能随当前默认参数变化 |
| V013 | 现有分类头来源缓存同样记录 `steps=1 / start_t=0.97 / CFG=0.5`；统一 YAML 的新提取默认曾为 `steps=2 / start_t=0.98`，不能直接套用 |
| V610 | 现有分类头训练缓存记录为 **仅 mapped codec tokens、无 BOS/EOS**，最后层、Codec layer=1；近期 GPT 独立脚本默认已改为 `bos_eos`，对旧分类头仍须明确使用原 `none` 协议 |

上述记录来自当前迁移后缓存的 `feature_summary.json`；正式工作流应逐 run 追溯嵌套来源记录和训练配置。若希望采用 GPT 的新 BOS/EOS 协议，应另行提取训练/验证特征并重新训练分类头，作为独立方法比较，不能只在 Test4 改输入。

旧 Test1/2/3 范围内重叠窗口可用于回归验证：逐点检查预处理/窗口，核对 Codec token、中心特征和预测概率。若使用相同硬件/批次仍不能复现，应先定位协议差异，再开展连续位点推理，不能把历史保存概率和新协议概率混合平均。

## 7. 已实现的工作流及目录

代码和 Shell 入口位于 [workflow](workflow/README.md)，沿用 Test3 的配置、分步运行和协议冻结方式。配置的实际结构以 [workflow/config.yaml](workflow/config.yaml) 为准；参考基因组、FASTA、坐标和信号预处理依据仍需在正式运行前补齐。

| 阶段 | 已实现入口（均有 Python 和 Shell） | 主要工作及产物 |
| --- | --- | --- |
| Step0 | `step0_audit_inputs` | 解析 chr16 连续 beta，参考 CpG 验证，冻结模型/分类头/seed、输入与特征协议 |
| Step1 | `step1_build_windows` | 单次流式扫描原 JSONL，输出无标签 15nt 窗口；原始 read–位点去重、完整 read 边界续跑 |
| Step2 | `step2_predict_reads` | 每主干分片提取一次特征，复用各分类头/seed；保存索引、概率及有效掩码 |
| Step3 | `step3_aggregate_sites` | 求共同 read–site 集合后筛选共同 coverage≥5；输出连续 beta、固定配对位点及共享区块抽样权重 |
| Step4 | `step4_evaluate_correlation` | 逐 seed 相关性、误差、预设分层及配对区块 bootstrap |
| Step5 | `step5_compare` | 逐 seed 明细、均值/样本方差、配对模型差值、HTML/CSV 和 PNG/PDF |

实际产物放在同一个可迁移的协议根目录，默认 `output/hg002_chr16_wgbs_correlation_v1/`：

```text
<protocol_dir>/
├── protocol.yaml
├── audit/                         # reference.sqlite、运行清单、输入/模型/代码校验
├── windows/                       # part_*.jsonl.gz、observations.sqlite、扫描断点与摘要
├── predictions/<版本>/part_*/     # probabilities.npz 与分片校验
├── common/
│   ├── fixed_sites.tsv / fixed_sites.npz
│   ├── scores/<版本>/<运行>.npz   # 甲基化read整数计数、概率均值；与fixed_sites同序
│   ├── common_indices_*.npz       # 共同有效窗口；还需关联fixed_sites取得最终coverage合格集合
│   └── bootstrap_weights.npz
├── metrics/<版本>/<运行>/         # 完整指标、分层及bootstrap抽样统计
└── summary/
    ├── correlation_by_seed.tsv
    ├── correlation_mean_variance.tsv
    ├── model_differences.tsv
    ├── data_funnel.json
    ├── compare/correlation_comparison.csv / .html
    └── figures/<版本>/<运行>.png / .pdf
```

`common/fixed_sites` 和每个运行的 `common/scores` 按行关联，可独立重算逐位点频率；模型版本、seed、权重校验和特征协议由冻结清单追溯。原 BED 行号保存在参考 SQLite，原信号行号保存在窗口分片。`wgbs_beta` 不进入主干或分类头输入。

```bash
# 临时合成数据和CPU分类头测试，不扫描真实大文件。
bash eval/Test4/workflow/run_all.sh --test

# 只读计划：发现模型、显示缺项，不启动正式处理。
bash eval/Test4/workflow/run_all.sh --dry-run

# 补齐实际配置后，正式运行。
bash eval/Test4/workflow/run_all.sh --run
```

默认无参数等同 `--test`。改变配置后使用新的协议目录；完整命令、分主干运行和中断恢复见 [工作流说明](workflow/README.md)。

## 8. 工程成本与执行顺序

**最昂贵的是完整信号扫描与主干提取，方案必须避免为每个模型、seed 或阈值重读 1.969 TiB JSONL。**

- Step0 先顺序解析一次 BED，导出 chr16 候选位点的小索引；不假设 gzip 可随机按染色体定位。
- Step1 大 JSONL 只顺序扫描一次，按候选位点导出压缩分片和窗口身份，不复制完整 read 信号。保持有界内存，批量写 SQLite 和 gzip JSONL，避免每个窗口单独提交数据库事务。
- 按分片断点续跑；完成标记绑定输入大小/mtime、内容校验策略、记录范围、配置和代码指纹。大文件完整 SHA256 可在必需的首次顺序读取时计算，避免为了 hash 单独多扫 2 TB。前缀测试必须标记 partial。
- 相同主干的特征供五分类头、三个 seed 共用；默认逐 Codec 路径及批次遵循旧协议。更改 Codec batch 可能改变离散 token，属于需要验证的提取变体，不能只为速度直接切换。
- 不默认保存全量 15nt embedding：每 100 万窗口的 `[N,15,768]` float16 就约 **21.46 GiB/主干**；中心 5nt 仍约 **7.15 GiB/主干**。优先分片提取后立即分类，持久化紧凑概率、身份和必要的审计样本。
- 可以反复读取较小的窗口分片，但不要反复解析原 2 TB read JSONL。共同有效窗口掩码明确后再聚合，避免不同模型各用自己的 read 集合制造不公平相关性。
- GPU 使用前检查其他任务的实际算力占用，不以空余显存判断空闲。当前项目曾出现 A800 被其他训练占用后，特征提取显著变慢的情况。

仅按顺序读取原文件计算，持续 100 MB/s 约需 **6.0 小时**，200 MB/s 约 **3.0 小时**；这是纯字节读取下限，不含 JSON 解码、窗口构建、网络抖动和模型推理，**不是本实验实测完成时间**。准确 ETA 要在前缀性能测试后，按实际输入扫描速率、有效窗口数和各主干吞吐估计。

建议先做 100～1,000 条完整 read 的接口测试，检查坐标/归一化/窗口与旧协议；该规模是接口前缀，不保证自然达到每个位点 5 条 read，不能替代正式相关性实验。随后以预先固定的较小 chr16 区域构建足够覆盖的连续 beta 位点测试，不能根据其相关性调整主协议。

## 9. 最终应交付什么

1. **数据漏斗表**：chr16 WGBS 总 CpG → 参考 coverage≥5 → 有合法信号窗口 → 所有模型共同观测 → 共同 coverage≥5 → 最终配对位点；逐步列出排除数及原因。
2. **主对比表**：主干、分类头、配对位点数、共同 read 数、Pearson r、Spearman rho、MAE、RMSE、Bias；三个 seed 的均值与样本方差，并附单 seed 明细。
3. **两种频率估计表**：beta_call 为主、beta_prob 为辅；不按 WGBS 结果临时决定哪一列作为主结果。
4. **相关图**：横轴 WGBS beta、纵轴模型 beta，坐标均固定 0～1，标注 N、r、MAE，显示 y=x；点多时用 hexbin/密度，所有模型相同色标和共同位点。若仅显示抽样点，指标仍在完整共同集合计算并注明。
5. **误差与分层结果**：中间比例、覆盖度、上下文或预定义基因组区域；模型间差值使用同位点配对，不把不同集合的 r 直接相减解释。
6. **可复现清单**：输入来源、基因组/坐标/链、真实预处理、完整模型/检查点和特征协议、排除规则、有效集合及版本、seed、阈值和 bootstrap 规则。

验收重点是数据与协议可追溯、不同模型使用同一集合、无标签泄漏、保留中间甲基化比例，以及结果能由逐位点表独立复算；不将“达到网页中的 0.9267”作为通过标准。

## 10. 检查和实现范围

已做：阅读用户指定网页及官方格式说明；核对两份输入的路径与大小；抽查 BED 前 20,000 行、信号前 12 条；核对现有窗口构建逻辑与六主干缓存中的关键特征协议；依据用户确认，将参考来源记为独立 WGBS。

新增实现：六步 Python/Shell、六模型配置、一键入口、分片续跑、共同集合聚合、相关性与比较导出，以及临时合成数据测试。另用首条真实 read 的两个窗口进行少量 CPU 主干/分类头接口检查；这不等价于历史缓存逐字节复现，也不是正式相关性结果。验证明细见 [工作流说明](workflow/README.md)。

未做：全量 chr16 WGBS 统计、全量 read 扫描、参考基因组验证、信号归一化的最终判定，以及全量窗口提取、模型预测、正式相关性和真实数据图表。本次没有启动大规模任务或改写现有训练/评测配置。

相关本地实现：[原窗口构建脚本](../../workflow/step2_build_chrom19_dataset.py)、[统一特征入口](../../workflow/step4_extract_features.py)、[分类模型](../../train/model/)、[Test3 工作流说明](../Test3/workflow/README.md)。
