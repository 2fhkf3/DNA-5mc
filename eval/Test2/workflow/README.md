# Test2 workflow：天然 chr16 阳性漏检评测

本目录实现 [Test2 方案](../README.md)。仅使用原 test 的 `label=1`，保留正类概率，计算 read-window 与 site 的 FN/TP/FNR/Recall。当前原始标签来自 BED=100% 筛选，标签来源状态记录为 `unverified`，不能当作逐分子实验真值。

## 1. 文件及执行顺序

| 文件 | 作用 | 是否必须 |
| --- | --- | --- |
| `config.yaml` | 六主干（V011/V013/V543/V548/V554/V610）路径、文库、策略/seed、阈值、覆盖度、bootstrap 及设备 | 是 |
| `step1_select_native_positives.py/.sh` | 从本主干 test 缓存同步选择 label=1，保留原特征和源行号 | 仅重新推理路线 |
| `step2_evaluate.py/.sh` | 审计旧预测或调用分类头重新推理；输出全阳性指标与 FN 清单 | 是 |
| `step3_compare.py/.sh` | 原始集合与共同集合比较、seed 汇总和同 seed 配对差值 | 跨模型比较时 |
| `common.py/.sh` | 通用 I/O、配置、统计及脚本环境 | 被各步骤调用 |
| `cache_migration.py` / `cache_migrations/*.json` | 缓存迁移审计、原路径与 NAS 路径映射及冻结签名 | 缓存迁移时 |
| `test_cache_migration.py` | 临时目录中的迁移回归检查，不读取真实大缓存 | 验证迁移逻辑 |
| `run_all.sh` | 测试、只读计划、正式运行入口 | 可选 |
| `test_workflow.py` | `/tmp` 下的 CPU 集成测试，无正式导出 | 验证实现 |

本流程无需重跑 Test1 的窗口提取、主干隐藏状态提取或中心裁剪。原 test 已有 `[N,5,768]` 的缓存，原始主干输入为 15 nt。

## 2. 首先检查配置与测试

```bash
# 默认就是 --test；使用临时合成数据，结束后自动清理。
bash /home/shj/ZJ-lab/Modification/DNA-5mC/eval/Test2/workflow/run_all.sh --test

# 只读列出将处理的 run 和缺少的结果，不创建正式文件。
bash /home/shj/ZJ-lab/Modification/DNA-5mC/eval/Test2/workflow/run_all.sh --dry-run
```

解释器优先使用已激活的非 base 环境，其次是本机 PoreDLM；也可设置 `PYTHON_BIN`。依赖为 Python、PyTorch、NumPy、PyYAML 和标准库 SQLite。所有 Bash 入口可在任意工作目录调用。

### 已迁移到 NAS 的原训练缓存

六个模型的 `models.<版本>.feature_dir` 现指向 `/mnt/nas_data/shenhaojie/step5-output/<版本>-V2`。训练 run/config.json 与 best.pt 中的历史路径、签名保持原样。每个模型用 `cache_migration_manifest` 指向独立迁移记录，Step2（包括 `--dry-run`）会校验这些记录。

迁移记录核对全部训练 run 的原始 test 签名、Test1 chr16-native 缓存中归档的原始来源签名及完整 feature_summary，确认当前 NAS 协议与该归档一致，并检查文件大小、缓存形状及分类头维度。记录保存迁移前签名与迁移后路径/大小/mtime，并对当前 test.pt、metadata_test.tsv、标签数组及 feature_summary.json 固定 SHA256。评测时按当前签名缓存哈希验证结果，避免为 15 个分类头重复散列同一份元数据。

历史训练记录没有原文件内容哈希；这些新哈希固定的是迁移审计时的当前文件，不能追溯证明历史字节完全一致。约 20GB/模型的特征数组维持冻结的大小/mtime 校验，不声称已验证其历史字节。复用预测路线仍逐行核对当前元数据、标签数组与原预测，且全量运行核对原 TP/FN 计数。任何未记录的换路径、改协议、缺文件或当前签名变化均报错，不自动放行。

本次六份迁移记录已经生成，正常运行直接执行 Step2。再次移动缓存时，先更新 YAML 中的实际 `feature_dir`，再显式重新审计；无需重新提取主干特征：

```bash
# 首次为新迁移生成记录；已有记录默认只核验，不覆盖。
/mnt/zzbnew/rnamodel/zhoukexuan/miniconda3/envs/PoreDLM/bin/python \
  /home/shj/ZJ-lab/Modification/DNA-5mC/eval/Test2/workflow/cache_migration.py

# 再次迁移后重建指定模型记录；不会修改 NAS 或训练结果。
/mnt/zzbnew/rnamodel/zhoukexuan/miniconda3/envs/PoreDLM/bin/python \
  /home/shj/ZJ-lab/Modification/DNA-5mC/eval/Test2/workflow/cache_migration.py --version V543 --refresh

# 迁移逻辑的临时目录回归检查。
python /home/shj/ZJ-lab/Modification/DNA-5mC/eval/Test2/workflow/test_cache_migration.py
```

2026-10-20 迁移验证：10 项迁移回归检查全部通过；真实配置的只读计划通过六模型、90 个已完成 run 的迁移绑定校验。另在 `/tmp` 中对每个模型扫描前 128 个源窗口、各筛选出 55 个阳性，核对筛选特征与 NAS 源数组逐值一致；六模型 MLP 及 V011 融合头、Residual DWConv 共八组进行旧预测/CPU 重推理对照，阈值 0.5 的判定完全一致，最大概率绝对差约 2.53×10⁻⁴。临时测试自动清理；这些是小样本联调，没有生成正式全量 Test2 结果。

## 3. 路线 A：复用预测，推荐

默认 `evaluation.source: predictions`。先 Step2，再 Step3；无需 Step1，也无需 GPU：

```bash
# 执行 YAML 中 enabled_versions 的所有已完成 run，随后比较。
bash /home/shj/ZJ-lab/Modification/DNA-5mC/eval/Test2/workflow/run_all.sh --run

# 或分步执行。
bash /home/shj/ZJ-lab/Modification/DNA-5mC/eval/Test2/workflow/step2_evaluate.sh
bash /home/shj/ZJ-lab/Modification/DNA-5mC/eval/Test2/workflow/step3_compare.sh
```

单模型、单 run 示例：

```bash
VERSION=V011 \
RUN_DIR=/home/shj/ZJ-lab/Modification/DNA-5mC/train/output/V011-v2/runs/center_5mer_mlp_raw768_seed1 \
bash /home/shj/ZJ-lab/Modification/DNA-5mC/eval/Test2/workflow/step2_evaluate.sh
```

Step3 读取配置选定的完整 run 列表。只评测一个 run 后若想比较该范围，应同时在 YAML 中限制 `enabled_versions/strategies/seeds`；不能假装其他 run 已经完成。

Step2 同步读取旧 `test_predictions.tsv`、原 `metadata_test.tsv` 和标签数组；逐行校验 ID、坐标、链、source、标签及 seq/ref。所有概率必须有限且位于 `[0,1]`，只将原 label=1 写入 Test2。全量导入时还核对原 `metrics.json` 的阳性数及同阈值 TP/FN。发现缺行、额外行或不匹配时中止。

原预测没有内嵌 checkpoint SHA256，所以旧文件与权重的历史绑定只能做配置、epoch、特征签名与结果计数的一致性检查。`run_manifest.json` 明确记录 `historical_prediction_binding=consistency_checks_only...`。需要确认当前权重概率时，用路线 B；记录本次 `best.pt` 哈希本身不能证明历史 TSV 由它产生。

## 4. 路线 B：选择阳性缓存并重新推理

修改 YAML 为 `evaluation.source: features`，然后运行一键入口；或逐步执行：

```bash
# Step1 从各主干自身的原 test 选择 label=1，约需单模型 10GB 特征空间。
bash /home/shj/ZJ-lab/Modification/DNA-5mC/eval/Test2/workflow/step1_select_native_positives.sh

# 只加载分类头到设备，分批读取缓存，不重新运行 backbone。
SOURCE=features DEVICE=cpu \
bash /home/shj/ZJ-lab/Modification/DNA-5mC/eval/Test2/workflow/step2_evaluate.sh

bash /home/shj/ZJ-lab/Modification/DNA-5mC/eval/Test2/workflow/step3_compare.sh
```

`DEVICE=cpu` 可换成 `cuda:0` 或 `auto`。`batch_size` 控制分类头批次；CPU 内存与显存占用按批次增长。推理固定 `eval()`、`inference_mode()`、float32；融合头采用 `softmax[:,1]`，其余头采用 sigmoid。核对各主干提取协议、特征维度、原训练路径和保存配置，严格加载权重。

Step1 同步复制特征/标签/元数据，重新生成 `sample_index` 并保留 `source_sample_index`。不改变 dtype、数值、source=mod、文库身份或原 15 nt 信号信息。生成 test 缓存、来源清单、位点表及重复审计数据库；不生成 train/valid。

已创建过阳性缓存时，请直接运行 Step2；一键 features 路线会再运行 Step1，因此会拒绝已有非空缓存。两条路线若都要保留结果，分别配置不同 `results_root/comparison_dir`。

## 5. 参数覆盖

参数均在 YAML 或脚本内有中文注释。配置优先级：显式命令行 > 脚本环境变量 > YAML；`--config` 可以指定另一份绝对路径配置。

| 参数 / 环境变量 | 含义 |
| --- | --- |
| `enabled_versions` / `VERSION` / `--version` | 全部启用模型或单模型 |
| `evaluation.strategies/seeds` | 空列表表示全部已有已完成 run；不限定 seeds 为 1/2/3 |
| `evaluation.read_threshold` | null 沿用各 run 的训练阈值；比较要求工作点一致 |
| `evaluation.site_threshold` | site 平均概率阈值，默认 0.5 |
| `statistics.coverage_minima` | 默认 `[1,5]`，都在最终观测集合上重算 |
| `statistics.bootstrap/block_size/bootstrap_seed` | 默认 1000 次、1Mb、42 |
| `DRY_RUN=1` / `--dry-run` | 只读计划；Step1 还核对数组完整性及标签数 |
| `MAX_RECORDS` / `--max-records` | 扫描输入前 N 行；非“取前 N 个阳性”，输出标记 partial |
| `OUT_DIR` | Step1 为单模型缓存目录，Step2 为结果根目录，Step3 为比较目录 |
| `RESULTS_ROOT` / `--results-root` | Step3 读取的 Step2 结果根目录 |

只做小样本联调时，务必输出至独立目录。例如：

```bash
VERSION=V011 MAX_RECORDS=128 \
OUT_DIR=/tmp/dna5mc-test2-manual-results \
RUN_DIR=/home/shj/ZJ-lab/Modification/DNA-5mC/train/output/V011-v2/runs/center_5mer_mlp_raw768_seed1 \
bash /home/shj/ZJ-lab/Modification/DNA-5mC/eval/Test2/workflow/step2_evaluate.sh
```

正式比较默认拒绝 partial。`comparison.allow_partial: true` 仅供隔离联调；不得拿部分结果替代完整 test 结果。输出采用临时目录完成后发布，不覆盖旧结果；失败时删除未完成目录。再次实验应换输出根目录。脚本不会训练模型，也不会改动原特征或训练结果。

## 6. 指标与输出

主指标为 `FNR=FN/(TP+FN)`、`Recall=1-FNR`。`p >= threshold` 判阳性，恰等于阈值为 TP。read 指 read–位点窗口；site 按 `(library_id,chrom,position,strand)` 汇总平均概率。`source` 保留为 mod，label 始终为 1。

每个结果目录 `output/results/<version>/<run>/` 包含：

| 文件 | 内容 |
| --- | --- |
| `metrics.json` | 全部与 coverage≥5 的 FN/TP/FNR/Recall，区块区间、Wilson 近似区间、阳性 BCE/Brier、位点内漏检 read 比例等 |
| `predictions.sqlite` | 全阳性概率和观测主键，拒绝重复；供共同集合匹配 |
| `read_predictions.tsv.gz` | 全阳性逐窗口概率，配置可关闭 |
| `site_predictions.tsv.gz` | 平均概率 score、coverage、read_fn |
| `false_negative_windows.tsv.gz` | 全部被漏检的阳性窗口，按 read 阈值 |
| `false_negative_sites.tsv.gz` | 被漏检的阳性位点，按 site 阈值 |
| `context_metrics.tsv` | 每个保存的 ref 上下文在 read/site 两层的指标及 coverage 子集；当前为中心 5-mer |
| `coverage_metrics.tsv` | 1/2/3/4/5–9/10–19/≥20 分层，空层 FNR 留空 |
| `threshold_metrics.tsv` | 预先定义阈值下的描述性 FNR，不选择最优阈值 |
| `run_manifest.json` | 权重 SHA256、原配置、数据协议/签名、代码哈希、设备和来源限制 |

位置字段统一命名为 `position`，与输入 `center_genome_pos0` 相同，都是 0-based。原始候选 BED/JSONL 分母无法恢复时，`upstream_reference_positive_count` 与 `upstream_call_rate` 为 null；缓存行必须全部成功调用，不静默忽略 no-call。

全阳性数据无法评价 FPR、AUROC 或完整区分能力，脚本不输出这些单类别伪成绩。bootstrap 为同一文库内基因组区块的抽样变化，不代表独立生物重复；零事件、少区块、禁用或退化区间记 null。概率分位数使用单个 float64 向量，当前阳性规模约占 10MB，主干特征始终分块读取。

## 7. 共同集合比较

Step3 针对每个策略，将所有选定模型及其全部选定 seed 的观测键取交集，验证 seq/ref 一致。随后重新聚合 site，并在共同集合上做 coverage≥5 筛选。某 site 从原始 5 条降至共同 4 条时，整个 site 及其 read 都退出 ≥5 分析。

共同 mask 独立于 seed 配对，因此 V554 seeds1/2/3 与其他模型 seeds4/5/6 仍可进行共同集合汇总；不会因没有相同 seed 而省略整种策略。仅配对差值表需要相同 seed，缺少时明确写 `no_matching_training_seed`。

比较目录包含 `raw_summary.tsv`、`common_summary.tsv`、`common_aggregate.tsv`、`paired_differences.tsv`、`common_samples_<strategy>.tsv.gz` 和 `comparison_manifest.json`。均值和 `ddof=1` 样本标准差只跨训练 seed，不把多个 seed 视为新增生物重复。差值方向为 `FNR_B-FNR_A`。

共同集合按策略固定；若不同策略的可用观测未来不相同，跨策略比较也要额外统一分母，不将不同 mask 的均值直接解释为同数据比较。原始结果单列，不能用 raw 的均值冒充 common 的均值。

## 8. 测试范围与后续扩展

`run_all.sh --test` 在 `/tmp` 使用可手算的合成样本和真实项目分类头，覆盖标签筛选/字节保真、旧预测与 CPU 推理一致、阈值等号、空覆盖层、跨模型缺失窗口、seed 不同、同 seed 配对、重复观测、截断/错位/非法概率、协议/阈值不匹配及脚本入口。

2026-10-18 已通过 Python/Bash 语法检查、默认四模型的 60 个 run 只读计划检查和 **15 项 CPU 自动测试**。另外对四个真实缓存各扫描前 128 行，均选择到 55 条阳性，核对了输出特征逐字节不变及原始观测键；重新推理结果与各自旧预测比较如下。全部输出均在临时目录中，未生成正式评测结果。

| 真实前缀验证 | 阳性窗口数 | CPU 与原保存概率的最大绝对差 | 0.5 分类调用 |
| --- | ---: | ---: | --- |
| V011 MLP seed1 | 55 | 1.13×10⁻⁶ | 完全一致 |
| V543 MLP seed1 | 55 | 7.15×10⁻⁷ | 完全一致 |
| V554 MLP seed1 | 55 | 4.77×10⁻⁷ | 完全一致 |
| V610 MLP seed1 | 55 | 3.58×10⁻⁷ | 完全一致 |
| V011 融合头 seed2 | 55 | 1.92×10⁻⁴ | 完全一致 |
| V011 Residual DWConv seed4 | 55 | 2.52×10⁻⁴ | 完全一致 |

验证采用 MLP 概率误差 `<2×10⁻⁵`、涉及卷积/LSTM 的头 `<10⁻³`，并单独要求上述前缀的逐窗口调用一致。跨 CPU/GPU 算子和批次的计算不能要求概率逐位相同；接近阈值的其他样本仍可能改变调用。表格只用于验证实现，不能作为模型正式性能结论；复用预测路线可完整保留原 test 的工作点结果。

此版本实现总体、上下文、覆盖度、阈值及区块区间；低深度 read 下采样、额外 MAPQ/信号长度分层、BED 来源自动核实、独立阳性文库评测属于后续扩展，不在当前主流程内自动执行。
