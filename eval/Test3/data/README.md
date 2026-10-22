# Test3 评测数据清单与规模说明

核对日期：2026-10-20。本文参考 [Test2 数据说明](../../Test2/data/README.md) 的结构，记录 Test3 的来源数据、共同位点、下采样规模、实际存储位置及已完成结果。数量以正式协议的冻结清单、位点表、数组头和结果文件为依据；不沿用 Test2 README 中较早的生成状态。

## 1. 当前状态与数据总览

**本次编写前，`eval/Test3/data/` 为空；写入后本目录仅有这份 README。Test3 正式评测已完成，数据索引、抽样清单和结果保存在 `../output/chr16_common_cov20_6models_v1/`，不在本目录中。**

Test3 复用原天然文库 chr16 test 的正负两类及已保存预测，研究同一批位点在每个位点恰好使用 5、10、20 条不同 read 时的检测性能。它不是 Test2 的全阳性假阴性实验，也不是新增的测序文库。

| 项目 | 已完成协议的实际规模或设置 |
| --- | --- |
| 协议 ID | `chr16_common_cov20_6models_v1` |
| 文库 | `NATIVE_250F500490011`，一个天然文库 |
| 染色体、链、来源集合 | chr16、正链 `+`、原 `test`、`source=mod` |
| 标签 | 原参考 BED=0/100 对应的 `label=0/1`，两类均保留 |
| 主干 | V011、V013、V543、V548、V554、V610，共 6 个 |
| 分类头与训练 seed | 每主干 5 种分类头，每种 3 个 seed；共 **90 个运行、30 个主干–分类头组合** |
| 原始候选位点 | **104,616** |
| 最终固定共同位点 | **104,615**，约 10.46 万 |
| 固定阴性 / 阳性位点 | **53,302 / 51,313** |
| 固定集合中的共同窗口 | **2,590,310**，约 259.03 万 |
| 共同窗口的阴性 / 阳性数量 | **1,295,166 / 1,295,144** |
| 保留位点的共同覆盖度 | **20–58 reads/site**，平均 **24.76041** |
| 位点类别平衡 | `site_balance=none`，没有再次抽成等量正负位点 |
| 深度 / 抽样重复 | **5、10、20 reads/site；每个深度 50 次** |
| 概率来源 | 90 个运行全部为 `saved`，复用已有预测 |
| 是否限量测试 | 否；正式清单 `partial_test=false`、`max_records=null` |

“窗口”指“一条原始 read 在一个目标位点上的一次观测”，不是独立完整 read。一个 read 可贡献多个位点的窗口；模型、分类头、seed 和 50 次下采样复用这些数据，不能将其数量相加作为新增生物样本量。

## 2. 来源缓存与预测结果

### 2.1 各主干原始 test 的数量

下表依据本次已完成评测的 [审计清单](../output/chr16_common_cov20_6models_v1/manifests/audit_manifest.json)，描述共同集合筛选前的来源数量。

| 版本 | 架构 | 原 test 窗口数 | 阴性窗口 | 阳性窗口 | 原 test 位点数 | 纳入运行数 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| V011 | Diffusion | 2,590,329 | 1,295,166 | 1,295,163 | 104,616 | 15 |
| V013 | Flow Matching | 2,590,329 | 1,295,166 | 1,295,163 | 104,616 | 15 |
| V543 | BERT | 2,590,340 | 1,295,170 | 1,295,170 | 104,616 | 15 |
| V548 | BERT | 2,590,340 | 1,295,170 | 1,295,170 | 104,616 | 15 |
| V554 | BERT | 2,590,340 | 1,295,170 | 1,295,170 | 104,616 | 15 |
| V610 | GPT | 2,590,329 | 1,295,166 | 1,295,163 | 104,616 | 15 |

每个主干的原始缓存特征形状为 `[N, 5, 768]`，特征落盘为 float16，标签为 float32；上游主干输入上下文为 15 nt。Test3 默认不重新提取这些 embedding，也不把缓存复制进本目录。

纳入的五种分类头为：

| 策略名 | 含义 |
| --- | --- |
| `center_base_linear_raw768` | 中心碱基线性探针 |
| `center_5mer_linear_raw768` | 中心 5mer 线性探针 |
| `center_5mer_mlp_raw768` | 中心 5mer MLP |
| `center_5mer_residual_dwconv_raw768` | 中心 5mer 残差深度卷积 |
| `center_5mer_emb8_none` | 碱基 8 维 embedding 与信号特征的 BiLSTM 融合 |

所有已冻结的三个训练 seed 都纳入统计，不按测试成绩选择 seed；不同分类头的 seed 编号不一定相同。具体身份见 [运行清单](../output/chr16_common_cov20_6models_v1/manifests/run_inventory.tsv)。

### 2.2 缓存路径的当前状态

**需区分历史审计位置与目前实际目录。** 当前 [工作流配置](../workflow/config.yaml)、已完成协议的 `protocol.yaml` 及旧迁移记录中的缓存根路径为：

```text
/mnt/nas_data/shenhaojie/step5-output/<版本>-V2/
```

本次检查时，六个版本的上述路径均已不存在；在以下新路径找到了六份缓存，并核对了下列五个文件均存在及其大小：

```text
/mnt/nas_data/shenhaojie/Modfication/train/step5-output/<版本>-V2/
├── test.pt
├── test.features.f16.bin
├── test.labels.f32.bin
├── metadata_test.tsv
└── feature_summary.json
```

版本目录分别为 `V011-V2`、`V013-V2`、`V543-V2`、`V548-V2`、`V554-V2`、`V610-V2`。注意路径中的 `Modfication` 是现有目录的实际拼写。

**本次仅核对新路径中文件的存在性与大小，没有重新计算大文件哈希，也没有更新配置或迁移绑定。** 因而不能把“新目录存在”视为旧审计绑定已自动迁移。已有正式 Test3 结果仍可直接读取；如需从 Step1 重新审计或推理，应先核对新路径、迁移记录和配置，使用独立输出协议，不能直接改写已完成协议的冻结文件。

### 2.3 原训练预测的位置与用途

```text
DNA-5mC/train/output/<版本>-v2/runs/<策略>_seed<N>/
├── best.pt
├── config.json
├── metrics.json
├── test_predictions.tsv
└── test_site_predictions.tsv
```

Test3 Step1 导入 `test_predictions.tsv` 的连续正类概率，逐行与完整 test 的标签、元数据和原行号对齐，并核对原 read/site 混淆计数。之后每个运行保存自己的 float64 概率向量，各主干只保存一份观测身份索引。

旧预测 TSV 未内嵌 checkpoint 哈希，因此本次正式清单中的 `historical_prediction_binding=consistency_checks_only` 表示来源一致性检查，不等价于已经证明历史预测由某个特定 checkpoint 字节生成。

## 3. 固定共同位点与覆盖度

Test3 在所有纳入运行之间，按文库、原始 read ID、染色体、位点和链匹配同一个物理观测，并检查标签及序列一致性。**不能仅凭各主干位点数相同，直接按行号匹配预测。**

在共同 read 池上重新计算每个位点的不同 read 数，要求至少 20 条，固定后由所有模型、分类头、seed、深度和抽样重复共享。当前未额外做位点类别平衡。

| 筛选阶段 | 位点数 |
| --- | ---: |
| 原始候选位点 | 104,616 |
| 有共同 read 的位点 | 104,616 |
| 共同覆盖度 ≥20 的最终位点 | **104,615** |
| 因共同覆盖度不足排除 | **1** |

被排除的是 **chr16:73837520，0-based、正链、阳性标签**，共同覆盖度为 19；该位点的全部 19 条共同观测退出三个深度的评测。故最终阳性位点从 51,314 变为 51,313，阳性共同窗口从 1,295,163 变为 1,295,144。

最终共同覆盖度分布为：

| 共同覆盖度 | 位点数 |
| --- | ---: |
| 20–29 | 92,959 |
| 30–39 | 11,298 |
| 40–49 | 337 |
| 50–58 | 21 |
| 合计 | **104,615** |

这里是成功进入所有冻结运行共同集合后的实际 read 数，既不是原 BED coverage，也不是整条染色体的平均测序深度。明细见 [固定共同位点](../output/chr16_common_cov20_6models_v1/common/fixed_sites.tsv)、[排除位点](../output/chr16_common_cov20_6models_v1/common/excluded_sites.tsv) 和 [共同集合清单](../output/chr16_common_cov20_6models_v1/common/manifest.json)。

## 4. 5 / 10 / 20 reads/site 数据如何生成

每个位点、每次重复都对共同 read 做确定性随机排名，无放回取前 20 条；5 条、10 条分别取这同一份 20 条清单的前缀。模型名、分类头、训练 seed、标签和预测概率不参与 read 排名，确保各模型比较的是同一次抽样。

| 每个位点使用 read 数 | 每次抽样的位点数 | 每次使用的 read–site 观测数 | 50 次累计使用次数 |
| --- | ---: | ---: | ---: |
| 5 | 104,615 | 523,075 | 26,153,750 |
| 10 | 104,615 | 1,046,150 | 52,307,500 |
| 20 | 104,615 | 2,092,300 | 104,615,000 |

表中累计使用次数包含跨重复、跨深度的复用，不代表这些数量的独立 read 或新增样本。

抽样索引实际保存在 `sampling/selected_indices.npy`：

```text
shape: [50, 104615, 20]
dtype: int32
文件大小: 418,460,128 bytes，约 399.07 MiB
```

它保存的是共同观测索引，而不是电信号或 embedding；数组只保存最大深度 20 的清单，5/10 条无需另存一份。抽样 seed 为 `20261018`，配置与算法见 [抽样清单](../output/chr16_common_cov20_6models_v1/sampling/manifest.json)。

## 5. 实际文件位置与磁盘量级

当前正式数据和结果的根目录为：

```text
eval/Test3/
├── data/
│   └── README.md                       # 本说明，不存大特征数组
└── output/chr16_common_cov20_6models_v1/
    ├── protocol.yaml                   # 已完成运行的冻结配置
    ├── manifests/                      # 90 个运行清单、来源审计、环境
    ├── audit/
    │   ├── versions/<版本>/             # 各主干完整观测索引与标签
    │   └── runs/<版本>/<运行名>/         # 每个运行的概率向量及审计清单
    ├── common/                         # 固定位点、排除位点、共同观测与映射
    ├── sampling/                       # 全模型共享的无放回抽样索引
    ├── results/
    │   ├── <版本>/<运行名>/              # 单运行逐重复指标与位点稳定性
    │   └── compare/                    # AUROC/AUPRC 对比表及明细
    └── summary/                        # 跨 seed、深度与模型汇总、已有图形
```

本次按文件逻辑大小求和，协议输出共 **887 个文件、7,617,217,950 bytes，约 7.09 GiB**。它不是磁盘实际分配块数，且后续增删输出会改变该统计。

| 存储部分 | 当前文件逻辑大小 |
| --- | ---: |
| `audit/`：6 份观测索引与 90 份概率向量等 | 约 **4.753 GiB** |
| `common/`：共同观测 SQLite、6 份行号映射等 | 约 **0.626 GiB** |
| `sampling/` | 约 **0.390 GiB** |
| `results/`，含 `compare/` | 约 **1.315 GiB** |
| `summary/` | 约 **0.010 GiB** |
| `manifests/` 与 `protocol.yaml` | 约 **0.001 GiB** |
| `data/` 中的特征数组副本 | **0 份** |

外部单模型完整 test 的 `[约259万,5,768]` float16 特征约 **18.53 GiB**，另有标签与元数据；这些外部文件不计入上述 7.09 GiB。默认复用预测没有复制六份大特征，但观测索引、概率和抽样清单仍会占用本地输出空间。

## 6. 已完成结果与指标含义

每个运行保存 50 次重复 × 3 个深度的 `metrics_by_repeat.tsv`，全部 90 个运行合计 **13,500 行**，每行含同一组位点上的 13 项指标。`metrics_full_common.json` 则在同一批固定 104,615 个位点上使用全部共同 read，不能与恰好 20 reads/site 的结果混同。

| 正式汇总文件 | 当前行数或用途 |
| --- | --- |
| `summary/seed_summary.tsv` | **3,510 行**，90 个运行 × 3 深度 × 13 指标 |
| `summary/model_summary.tsv` | **1,170 行**，30 个组合 × 3 深度 × 13 指标 |
| `summary/full_common_summary.tsv` | **390 行**，30 个组合 × 13 指标 |
| `summary/depth_differences.tsv` | **780 行**，低深度相对最大深度的差异 |
| `results/compare/coverage_comparison.html/.tex/.csv` | 主表 **30 行**，5/10/20 reads/site 各展示 AUROC、AUPRC |
| `results/compare/coverage_results_by_seed.csv` | **270 行**，逐训练 seed 与深度明细 |
| `results/compare/coverage_all_metrics.csv` | **1,170 行**，完整 13 指标及不同不确定性信息 |

位点分数是本次选中 read 的正类概率均值；AUROC 按连续位点分数计算，AUPRC 指 Average Precision（AP）。Accuracy、F1、FPR、FNR 等指标使用协议固定阈值 `p >= 0.5` 判阳性，不在 Test3 中按深度重新选择阈值。

对比表的统计顺序是：**每个训练 seed 内先对 50 次抽样指标取均值，再对三个训练 seed 的均值计算均值和样本方差（ddof=1）。** 不先平均三个 seed 的概率形成集成，也不把 150 个结果当作独立生物重复。训练 seed 波动、read 抽样波动及基因组区块区间分别报告；AUROC/AP 不从区块混淆计数伪造置信区间。

结果入口：[AUROC/AUPRC 表格](../output/chr16_common_cov20_6models_v1/results/compare/coverage_comparison.html)、[表格说明](../output/chr16_common_cov20_6models_v1/results/compare/README.md)、[官方逐深度汇总](../output/chr16_common_cov20_6models_v1/summary/model_summary.tsv)。

## 7. 标签来源、实验边界与本次核对范围

Test3 的正负标签继承同一天然文库 chr16 test 的 BED=0/100 参考，`reference_validation_status=unverified`。这不是新制备的全甲基化/未甲基化成对对照，也不是逐分子独立验证的修饰真值。`source=mod` 说明文库来源，不代表所有窗口都是阳性。

Test1 的天然阴性分支只取 `label=0`，Test2 只取 `label=1`；Test3 保留两类，并重新固定共同覆盖度 ≥20 的位点。三个评测复用同一来源测试集的观测，不是三套独立文库。Test3 也不使用 release 流程的 chr16 一半 read 阴性校准划分，不能直接混用其样本量或结果。

本次核对了：本地 `data/` 的实际内容；当前配置与正式清单；六主干来源数量、90 个运行身份和 `saved` 状态；固定/排除位点表、共同覆盖度及两类共同窗口数；主要数组头、形状、精度和文件大小；汇总表行数与输出目录逻辑大小；旧 NAS 路径失效及新路径五类源文件的存在性、大小。

**本次未重新扫描原始信号、重算新 NAS 文件完整哈希、重新核对全部历史预测或重新抽样/推理，也未修改流程配置与迁移记录。** 已完成正式审计的校验状态以冻结 manifest 为准，本 README 的文件盘点不替代内容一致性审计。

相关文档：[Test3 方案](../README.md)、[工作流说明](../workflow/README.md)、[当前配置](../workflow/config.yaml)、[冻结协议](../output/chr16_common_cov20_6models_v1/protocol.yaml)、[Test2 数据说明](../../Test2/data/README.md)。
