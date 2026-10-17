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
| 配置 | [extraction_config.py](extraction_config.py) | [config.yaml](config.yaml) | Step2 模型选择、参数解析及架构核对 |
| 测试 | [test_step2_config.py](test_step2_config.py) | [run_all.sh](run_all.sh) | 配置优先级、架构兼容及可选三主干小样本提取测试 |

每个 Python 入口支持 `--help`。每个 Bash 脚本中已注释参数含义，支持环境变量覆盖和追加 Python 命令行参数。所有入口都可以从任意目录调用。

## Step2：统一配置不同版本和隐藏层

主要修改 [config.yaml](config.yaml)：

- `model`：当前选择的模型名称。
- `extraction`：语料路径、输出位置、设备、批次、缓存精度、窗口长度等通用参数。
- `models.<名称>`：架构、权重路径和该架构的隐藏状态选项。

当前配置为 V011 → Diffusion、V543 → BERT、V610 → GPT。模型名称不再决定提取器；由 `architecture` 选择 `diffusion`、`bert` 或 `gpt`，也可设为 `auto` 根据主干 `config.json` 判断。支持与现有 PoreDLM、PoreBERT、GPT/OLMo 提取协议兼容的更新权重；结构或信号/token 协议发生变化时仍需要对应适配。

默认 `extraction.data_dir` 已与当前 Step1 脚本一致，指向 `eval/Test1/data/step1_out/chr19`。所有 YAML 内部相对路径以项目根目录为基准；CLI 传入的相对路径以调用目录为基准。

例如未来增加一个 BERT 版本，只需在 `models` 下新增：

```yaml
model: V554
models:
  V554:
    architecture: bert
    model_root: train/backbone/V554-BERT
    encoder_layer: 0
    hidden_layer: -1
    token_offset: 128
```

上面是需合并到完整配置中的片段。`config.yaml` 中也提供了新 Diffusion / GPT 版本的注释模板。`encoder_dir`、`decoder_dir` 可以分别指定，不要求新 checkpoint 与旧版本使用同样的目录名。主干路径优先寻找 `model_root/base`；没有 base 且存在多个候选时必须明确指定 `decoder_dir`。当前 V011 已在 YAML 中明确指向原实验使用的 step65000 权重，防止自动选错。

三个路径独立生效：如果条目中已经显式设置 `decoder_dir`，只修改 `model_root` 不会取消这个主干路径。切换 checkpoint 时应同步修改对应路径，并先用 dry-run 检查最终结果。

隐藏层参数的索引约定不同：

| 架构 | 参数 | 含义 |
| --- | --- | --- |
| BERT | `hidden_layer` | `hidden_states` 索引：0 是 embedding，1 是 Transformer 第 0 层输出，-1 是最终隐藏状态 |
| GPT | `decoder_layers` | `last` 为最终隐藏状态；整数 8 表示 Transformer 第 8 层，从 0 开始编号 |
| GPT | `selected_layer_name` | 多选如 `[3, 8]` 时，指定实际落盘的一层，如 `layer_8` |
| Diffusion | `hidden_source` 和 ODE 参数 | 选择 `ode_hidden`、`context_hidden`、`last_hidden` 等原提取器支持的表征来源 |

每次输出保存一个目标层/表征来源。比较同一模型的多个隐藏层时，使用不同 `out_dir` 或 `FEATURES_DIR`。改变主干、层或 ODE 协议后，需要使用在对应特征上训练的分类头；评测器会核对训练协议。

```bash
# 不加载权重、不创建特征，只展示最终解析配置并核对模型 config 路径。
DRY_RUN=1 bash eval/Test1/workflow/step2_extract_features.sh

# 临时切换 YAML 已配置的模型。
VERSION=V543 DRY_RUN=1 bash eval/Test1/workflow/step2_extract_features.sh

# 临时选择 BERT 的 hidden_states[6]，并单独命名输出目录。
VERSION=V543 HIDDEN_LAYER=6 \
FEATURES_DIR=/path/to/new/V543_hidden6 \
  bash eval/Test1/workflow/step2_extract_features.sh

# 使用另外一份完整 YAML。
EXTRACTION_CONFIG=/path/to/experiment.yaml \
  bash eval/Test1/workflow/step2_extract_features.sh
```

优先级为：**追加的命令行参数 > 显式环境变量 > YAML 模型条目/通用设置 > 架构默认值**。启动器不会用 `common.sh` 中自动补齐的默认值覆盖 YAML。最终配置同时保存在 `resolved_extraction_config.json` 和 `feature_summary.json` 中，便于复核实际用的权重路径、层和参数。

## Step2 推理优化配置

性能选项直接放在现有 `config.yaml` 的 `extraction` 下，也可在 `models.<名称>` 中覆盖。
没有增加新的提取脚本，三类原提取器复用现有 `common.py` 中的性能辅助函数。

```yaml
extraction:
  batch_size: 32
  codec_batch_size: 1
  codec_buffer_size: 1024
  pooling_device: model
```

| 参数 | 作用和选择 |
| --- | --- |
| `batch_size` | 主干推理批次。Diffusion/GPT 当前设为 32；V543/V554 在各自模型条目覆盖为 8，保留原 BERT 批次形状。显存不足会自动拆小。 |
| `codec_batch_size` | 默认 1，沿用原逐条 codec 编码。大于 1 时按**完全相同的信号长度**分组批量编码，不补零，随后恢复输入顺序。批量编码会改变卷积数值运算，可能改变离散 token；需作为单独实验验证后使用。 |
| `codec_buffer_size` | 只有批量 codec 启用时使用，限制暂存的有效窗口数量；末尾不足一批也会处理。当前为 1024，不缓存整套语料。 |
| `pooling_device` | `model` 在模型设备上按原 span 区间进行 FP32 均值聚合，再一次性传回 `[B,15,H]`；`cpu` 保留原先完整 token 隐藏状态传回 CPU 后逐窗口聚合的路径。模型在 CPU 时，两种模式都可运行。 |

**为什么没有默认打开批量 codec：**扩大到 1024 个真实窗口后，V543 的等长批量 codec（32/1024）有 21 个窗口各出现一个不同 token。定位到卷积批次形状引起的数值差异跨过量化边界，并非信号补零或顺序错误。此前 256 个窗口的测试没有覆盖该现象，不能据此保证全量一致。因此正式对比默认保持 `codec_batch_size: 1`。V543 主干从批次 8 改为 32 时也观察到最大约 0.0844 的特征绝对差异（原 BERT `high` 计算精度），当前 BERT 默认保留 8。

GPU 聚合使用前缀和计算相同区间的均值，浮点求和顺序不同，因此不承诺逐位相同。隐藏层选择、特殊 token、原有主干计算精度、ODE 参数、截断/跳过规则、输出格式和元数据顺序均沿用原协议。`dtype` 仍只控制落盘精度，未启用额外混合精度或调整 TF32 开关。小样本检查不能代替全量下游预测一致性验证。

环境变量可覆盖上述 YAML 设置：

```bash
# 只预览最终配置，不加载模型或产生特征。
VERSION=V011 DRY_RUN=1 bash eval/Test1/workflow/step2_extract_features.sh

# 回到原执行方式；命令执行后会启动提取，正式运行时自行调用。
BATCH_SIZE=8 CODEC_BATCH_SIZE=1 POOLING_DEVICE=cpu \
  bash eval/Test1/workflow/step2_extract_features.sh
```

对应 CLI 为 `--codec-batch-size`、`--codec-buffer-size`、`--pooling-device`。原始 `workflow/step4_*.py` 入口也支持这些选项，默认仍为逐条 codec 和 CPU 聚合。

每个任务在启动时解析一次 YAML，并保存 `resolved_extraction_config.json`；运行中修改 YAML 只影响后续任务，不再因此误报 `Dataset changed during feature extraction`。输入窗口、数据清单、审计索引和模型 `config.json` 的文件签名仍会检查，变化时报告具体路径。优化不会替换已存在的正式特征目录，需要另选输出目录；已启动的旧进程也不会自动切换到新实现。

### 本次优化验证（2026-10-17）

A800 80 GB，CUDA 0，4 个 CPU 线程，同一份 WGA 输入的前 1024 个窗口。先预热两条路径，再以优化→原始的顺序计时，包含读取、codec、主干、聚合和临时特征写盘；不包含权重加载和 Step2 的 SQLite 元数据补充。基准为主干批次 8、codec 逐条、CPU 聚合；优化为当前 YAML 的批次设置、codec 逐条、模型设备聚合。数值对照使用 FP32 落盘。

| 模型 | 原始耗时（s） | 优化耗时（s） | 速度比 | 特征最大绝对差异 |
| --- | ---: | ---: | ---: | ---: |
| V011 | 6.465 | 4.455 | 1.45× | 4.48e-05 |
| V543 | 2.998 | 2.758 | 1.09× | 3.05e-05 |
| V554 | 3.020 | 2.815 | 1.07× | 1.72e-05 |
| V610 | 4.084 | 3.408 | 1.20× | 1.14e-05 |

四个模型的元数据、样本顺序和标签逐字节一致。上述速度来自单轮小样本对照，受长窗口分布、磁盘和其他 GPU 进程影响，不保证全量倍率，也未验证全量下游分类器预测完全一致。等长批量 codec（32/1024）的单独对照中，V543/V554 各有 21/1024 个窗口 token 不同，V011/V610 各有 14/1024，因此默认关闭。

另外测试了缓冲区尾批和顺序、非有限特征拒绝、模拟 CUDA OOM 拆批、Diffusion 截断跨度检查、GPT 超长窗口 skip/error、配置覆盖与非法参数。测试使用临时目录，未运行正式全量提取。

三类架构还分别通过了 17 个真实窗口的 Bash → Step2 完整入口测试：验证 FP16 特征、审计字段补全、`partial_test` 标记及 Step3 中心裁剪兼容性；启动后修改临时 YAML，任务仍按启动快照正常完成。所有临时特征已自动清理。

## 只做测试

注意：当前工作区缺少下文历史命令引用的 `test_step2_config.py`，因此 `run_all.sh --test` 暂不可用。本次优化按要求未新增测试脚本，使用隔离的临时测试验证；`DRY_RUN=1` 配置预览仍可直接使用。

默认一键入口只测试，测试数据、日志和缓存位于 `/tmp/dna5mc-test1-*`；成功后自动清理，不生成正式 `output/` 结果：

```bash
bash /home/shj/ZJ-lab/Modification/DNA-5mC/eval/Test1/workflow/run_all.sh --test
```

该测试先验证配置解析与下游协议检查，再使用少量真实 WGA 窗口和三个真实预训练主干验证新版本名、隐藏层选择及缓存格式，默认使用 CPU。测试使用已有权重模拟新配置项，不代表已经验证未来尚未提供的模型权重。

不加载真实主干的快速联调，可在已经激活项目环境后执行：

```bash
python eval/Test1/workflow/test_step2_config.py
```

快速模式只进行配置、参数转发、旧分类器协议兼容和错误拒绝测试，不加载模型权重。完整小样本测试还需要原提取脚本和三个主干权重；不会扫描完整 WGA 文件。

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

`DRY_RUN=1` 不创建输出，不检查完整数据和权重是否能执行。Step2 会额外解析 YAML、检查模型 config 路径并显示最终设置；其他步骤仍只显示命令。

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
| `EXTRACTION_CONFIG` | `workflow/config.yaml` | Step2 配置文件 |
| `VERSION` | Step2 采用 YAML 的 `model` | 模型名称，允许新增版本 |
| `MODEL_ARCH` | 采用 YAML 模型条目 | Step2 架构：diffusion / bert / gpt / auto |
| `DEVICE` / `CPU_THREADS` | `auto` / `4` | 设备和 CPU 算子线程 |
| `BATCH_SIZE` | YAML：Diffusion/GPT 32，BERT 8 | 主干提取批次 |
| `CODEC_BATCH_SIZE` | YAML：1 | 等长信号编码批次；大于 1 可能改变离散 token |
| `CODEC_BUFFER_SIZE` | YAML：1024 | codec 分组缓冲窗口数 |
| `POOLING_DEVICE` | YAML：model | model / cpu，选择均值聚合位置 |
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

当前 Step1 已单独设置了数据输出目录，因此自定义整套输出位置时还应同步 `DATA_DIR` 或 Step2 YAML 的 `extraction.data_dir`。新增模型参加整个评测时，应提供相应训练 run，并通过 `VERSIONS` 指定要循环的模型名；Step2 的配置文件只控制特征提取。

## 输出和计算规则

```text
output/
├── data/
│   ├── test.jsonl.gz
│   ├── sample_audit.sqlite
│   ├── sites.tsv
│   └── dataset_manifest.json
├── features/<VERSION>/                # [N,15,768]
│   └── resolved_extraction_config.json # 实际提取配置；新模型维度由其配置决定
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

当前测试工具检查 YAML/环境变量/命令行优先级、新模型名称、独立权重路径、架构推断、隐藏层参数转发及错误参数拒绝，并验证旧分类器协议仍能核对。真实主干测试仅使用受限窗口，不用于判断模型真实假阳性水平。

2026-10-16 的早期流程验证已通过：12 个已有 seed1 分类器的实际权重加载与前向推理；三个真实预训练主干分别处理 4 个用户 WGA 窗口；完整的 test 裁剪、分类及共同样本汇总；已知概率的统计核对；缺失 read 后的交集和覆盖度重算；融合头分支；以及三个版本原 test 文件前 16 条保存概率的复现。

本次配置重构后，额外通过了 10 项配置/协议测试，以及三个架构使用新配置名的真实提取测试。每个主干仅提取 1 个真实 WGA 窗口；其中 BERT 选择 `hidden_states[6]`，GPT 选择 `layer_8`，Diffusion 保持原 ODE 设置。测试使用已有权重模拟新版本条目，没有验证尚未提供的新权重。真实模型测试使用 CPU，未验证 GPU 路径；成功测试的临时结果已清理，没有启动全量评测。

训练协议核对依赖原 run 保存的 `feature_dir/feature_summary.json`。若模型迁移后该目录不可访问，应先恢复或显式迁移其清单；程序不会跳过协议核对。默认 chr19 与原训练染色体重合，因此结果应称为独立 WGA 文库测试，不能自动解释为未见基因组位点泛化。

更完整的实验原理、指标解释和判读边界见 [Test1 方案](../README.md)。
