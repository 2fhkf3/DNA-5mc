# Test1 可执行工作流

使用已经预处理的 WGA UNMOD 全 read JSONL，构建正链 CpG 阴性窗口，再用已有 V011 / V543 / V610 主干和分类头进行只读推理。不会调用训练启动器或更新模型。

默认输入：

```text
/mnt/zzbnew/rnamodel/jinyue/data/processed_data/chr19/unmod/s2_all_reads.genome_signal.preprocessed.jsonl
```

输入约 207 GB。约 5× 是总体深度，窗口构建默认保留实际覆盖度至少 1 条 read 的位点。阴性标签来自 WGA 文库身份，所有窗口标为 0，不读取其他模型的甲基化预测来筛选标签。

## 文件与步骤

| 顺序 | Python | Bash | 功能 |
| --- | --- | --- | --- |
| 1 | [step1_build_unmod_windows.py](step1_build_unmod_windows.py) | [step1_build_unmod_windows.sh](step1_build_unmod_windows.sh) | 提取 15 nt 正链窗口、按原始 read 去重、计算覆盖度和保存审计索引 |
| 2 | [step2_extract_features.py](step2_extract_features.py) | [step2_extract_features.sh](step2_extract_features.sh) | 复用原三种主干的提取实现，仅处理 test |
| 3 | [step3_center_crop.py](step3_center_crop.py) | [step3_center_crop.sh](step3_center_crop.sh) | 将 test 表征从 15 个位置裁剪到中心 5 个，保留原上下文 |
| 4 | [step4_evaluate.py](step4_evaluate.py) | [step4_evaluate.sh](step4_evaluate.sh) | 加载完成训练的 best.pt，输出 read/site FPR、覆盖度分层和置信区间 |
| 5 | [step5_compare.py](step5_compare.py) | [step5_compare.sh](step5_compare.sh) | 汇总各 seed，并匹配相同 read–位点比较不同主干 |
| 共用 | [common.py](common.py) | [common.sh](common.sh) | 原子发布、路径、SQLite 匹配和阴性统计 |
| 测试 | [test_workflow.py](test_workflow.py) | [run_all.sh](run_all.sh) | 临时目录内的统计、真实格式、分类器及可选真实主干测试 |

每个 Python 入口支持 `--help`。每个 Bash 脚本中已注释参数含义，支持环境变量覆盖和追加 Python 命令行参数。所有入口都可以从任意目录调用。

## 只做测试

默认一键入口只测试，测试数据、日志和缓存位于 `/tmp/dna5mc-test1-*`；成功后自动清理，不生成正式 `output/` 结果：

```bash
bash /home/shj/ZJ-lab/Modification/DNA-5mC/eval/Test1/workflow/run_all.sh --test
```

该测试包含少量真实 WGA 窗口和三个真实预训练主干，默认使用 CPU。需要调试日志时：

```bash
bash eval/Test1/workflow/run_all.sh --test --keep-temp
```

不加载真实主干的快速联调，可在已经激活项目环境后执行：

```bash
python eval/Test1/workflow/test_workflow.py
```

此模式仍会检查已有训练分类头的实际权重，不只是做语法检查。模型定义、原提取脚本、权重目录和原训练特征清单需要保留。

## 正式运行命令：本次未执行

正式输出默认写入 `eval/Test1/output/`。以下命令会读取完整输入，只有用户准备启动正式评测时才运行：

```bash
DEVICE=cuda:0 bash eval/Test1/workflow/run_all.sh --run
```

运行前确认 MAPQ 与原训练数据一致。如果原实验最后筛选到 MAPQ 60，可设置 `MIN_MAPQ=60`。原语料已经预处理，不应再次运行原始信号归一化脚本。

单步运行如下。每一行都会执行对应步骤，不属于 dry-run：

```bash
bash eval/Test1/workflow/step1_build_unmod_windows.sh
VERSION=V610 DEVICE=cuda:0 bash eval/Test1/workflow/step2_extract_features.sh
VERSION=V610 bash eval/Test1/workflow/step3_center_crop.sh
VERSION=V610 DEVICE=cuda:0 bash eval/Test1/workflow/step4_evaluate.sh
bash eval/Test1/workflow/step5_compare.sh
```

默认依次比较 V011、V543、V610 的所有已完成分类头、共同 seed1/2/3。`STRATEGIES` 和 `SEEDS` 可以限制范围，`RUN_DIR` 可以指定单个 run。不会根据 WGA 结果自动选择最优 seed、epoch 或阈值。

例如只预览一组分类头评测命令：

```bash
DRY_RUN=1 VERSION=V610 SEEDS="1 2 3" \
STRATEGIES="center_base_linear_raw768" \
  bash eval/Test1/workflow/step4_evaluate.sh
```

`DRY_RUN=1` 不创建输出，不检查完整数据和权重是否能执行。

## 常用参数

| 环境变量 | 默认值 | 含义 |
| --- | --- | --- |
| `PYTHON_BIN` | 原项目环境选择规则 | Python 解释器，需要 torch / numpy / transformers / PyYAML 等原有依赖 |
| `INPUT_JSONL` | 上述 WGA 文件 | Step1 输入，只读 |
| `TEST1_OUTPUT_ROOT` | `eval/Test1/output` | 所有正式输出的根目录 |
| `CHROM` / `MOTIF` | `chr19` / `CG` | 目标染色体和 CpG / 全 C 选择 |
| `MIN_MAPQ` / `MIN_COVERAGE` | `30` / `1` | read 质量阈值与去重后正链位点覆盖度 |
| `CANDIDATE_SITES` | 空 | 可选无表头 TSV，列为 chrom 和 0-based position，来自独立位点集合 |
| `LIBRARY_ID` | `WGA_UNMOD_chr19` | 文库标识，不同文库不能混用 |
| `MAX_RECORDS` / `MAX_WINDOWS` | 空 | 仅用于受限测试；有限输入结果带 `partial_test` 标记 |
| `VERSION` | `V610` | 单步运行的主干版本 |
| `DEVICE` / `CPU_THREADS` | `auto` / `4` | 设备和 CPU 算子线程 |
| `BATCH_SIZE` | `4` | 主干提取批次 |
| `FEATURE_DTYPE` | `float16` | 缓存存储精度 |
| `CROP_WINDOW` / `CHUNK_SIZE` | `5` / `4096` | 表征裁剪窗口和读写块大小 |
| `EVAL_BATCH_SIZE` | `2048` | 分类器推理批次 |
| `SEEDS` | `1 2 3` | 批量评测的训练 seed |
| `STRATEGIES` | 所有完成的分类头 | 空格分隔分类头名称 |
| `RUN_DIR` | 空 | 单个已完成 run；使用时 `RESULTS_DIR` 是该 run 的输出目录 |
| `READ_THRESHOLD` | checkpoint 中的阈值 | read 阳性判断；当前训练通常为 0.5 |
| `SITE_THRESHOLD` | `0.5` | site 平均概率的阳性阈值 |
| `BOOTSTRAP` / `BLOCK_SIZE` | `1000` / `1000000` | 重采样次数和区块长度 bp |
| `BOOTSTRAP_SEED` | `42` | 重采样随机种子 |
| `SAVE_READ_PREDICTIONS` | `1` | 是否额外输出逐窗口 TSV.gz；SQLite 索引始终保存 |
| `VERSIONS` | `V011 V543 V610` | 一键正式流程及汇总的版本列表 |

可单独覆盖 `DATA_DIR`、`FEATURES_DIR`、`CROPPED_DIR`、`RESULTS_DIR`、`RESULTS_ROOT` 和 `COMPARISON_DIR`。建议正式一键流程仅修改 `TEST1_OUTPUT_ROOT`，避免把不同阶段指向不一致目录。

## 输出和计算规则

```text
output/
├── data/
│   ├── test.jsonl.gz
│   ├── sample_audit.sqlite
│   ├── sites.tsv
│   └── dataset_manifest.json
├── features/<VERSION>/                # [N,15,768]
├── center5/<VERSION>/                 # [N,5,768]
├── results/<VERSION>/<run>/
│   ├── predictions.sqlite
│   ├── read_predictions.tsv.gz
│   ├── site_predictions.tsv.gz
│   ├── false_positive_sites.tsv.gz
│   ├── metrics.json
│   ├── coverage_metrics.tsv
│   ├── context_metrics.tsv
│   ├── threshold_metrics.tsv
│   └── run_manifest.json
└── comparison/
    ├── summary.tsv
    ├── aggregate.tsv
    ├── common_summary.tsv              # 有完整共同模型组时输出
    ├── common_<VERSION>_<head>_seed<N>.json
    └── comparison_manifest.json
```

`FPR_read = FP / 实际成功预测的阴性窗口数`。site 概率按相同文库、染色体、坐标、链对独立 read 的概率取均值，再判断是否达到 site 阈值。阈值比较统一使用 `>=`。

位点覆盖度按 `1 / 2 / 3 / 4 / 5–9 / >=10` 分层；另给出最小覆盖度 1、3、5 的累计结果。平均概率与阳性 read 比例分开输出。不同主干用文库、原始 read ID、染色体、坐标和链连接，匹配后重新计算覆盖度和 site FPR。

所有 FPR 数值存储为 0–1 的比例。没有样本的分层和没有阳性类的 AUROC 等记为 null。Wilson 区间和零误报时的二项上界依赖独立观测近似；区块 bootstrap 在区块过少、关闭重采样或结果全零/全一时不报告伪精确区间。

原始语料和模型权重只读，已有输出目录不覆盖；失败步骤的临时目录不发布。多个 run 的批量评测若中途失败，已经完成的 run 会保留，新目录可用于重试，或用 `RUN_DIR` 单独处理剩余模型。

## 测试范围与限制

测试工具检查已知概率样本的 FPR 分母、独立 read 去重、低覆盖位点、严格负标签、错误主干拒绝、融合头 softmax、中心裁剪等价性，以及共同样本比较。真实主干测试仅使用受限窗口，不用于判断模型真实假阳性水平。

2026-10-16 的验证已通过：12 个已有 seed1 分类器的实际权重加载与前向推理；三个真实预训练主干分别处理 4 个用户 WGA 窗口；完整的 test 裁剪、分类及共同样本汇总；已知概率的统计核对；缺失 read 后的交集和覆盖度重算；融合头分支；以及三个版本原 test 文件前 16 条保存概率的复现。真实模型测试使用 CPU；当前进程未检测到可用 CUDA，因此本次没有验证 GPU 路径。所有成功测试的中间结果已清理，没有启动全量评测。

训练协议核对依赖原 run 保存的 `feature_dir/feature_summary.json`。若模型迁移后该目录不可访问，应先恢复或显式迁移其清单；程序不会跳过协议核对。默认 chr19 与原训练染色体重合，因此结果应称为独立 WGA 文库测试，不能自动解释为未见基因组位点泛化。

更完整的实验原理、指标解释和判读边界见 [Test1 方案](../README.md)。
