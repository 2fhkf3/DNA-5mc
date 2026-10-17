# DNA-5mC

基于 Nanopore 信号预训练模型表征的 DNA 5mC 二分类实验流程，包含信号预处理、按位点构建数据集、类别平衡、特征提取、中心窗口裁剪，以及多种分类头的训练与评估。

训练阶段读取预先提取的特征，只训练分类头。BERT、Diffusion 和 GPT 主干在 Step4 中用于提取表征，不在分类训练中更新。

![DNA-5mC Workflow：从信号预处理、位点数据集构建、特征提取和中心裁剪，到分类头训练与评估](docs/assets/dna-5mc-workflow.png)

```text
带比对及碱基—信号对应关系的 JSONL + bedMethyl
  → Step1 信号预处理
  → Step2 构建带标签的碱基窗口，按位点划分数据集
  → Step3 类别平衡 / 可选筛选
  → Step4 encoder + BERT / Diffusion / GPT → [N, W, H] 特征
  → Step5 可选中心裁剪，例如 15 → 5 / 7 / 9
  → 分类头训练 → read/window 与 site 两个层面的评估
```

本仓库从已有 JSONL 开始，不包含从 POD5/FAST5 生成比对 JSONL 的完整 basecalling 流程。预训练权重、原始数据、特征缓存和训练结果需单独准备，不随 Git 仓库分发。

## 1. 目录结构

```text
DNA-5mC/
├── README.md
├── workflow/
│   ├── paths.sh                               # Bash 入口共用路径设置
│   ├── step1_preprocess_signal_jsonl.{py,sh}
│   ├── step2_build_chrom19_dataset.{py,sh}
│   ├── step3_balance_chr19.py
│   ├── step3_run_balance_chr19.sh
│   ├── step3.1_extract_chr19_small.py           # 可选：抽取小数据集
│   ├── step3.2_filter_chr19_mapq.py             # 可选：提高 MAPQ / 覆盖度要求
│   ├── step4_extract_features.{py,sh}   # 统一 YAML 配置入口
│   ├── step4_config.yaml
│   ├── step4_extract_window_hidden_states_BERT-js.{py,sh}
│   ├── step4_extract_window_hidden_states_Diffusion-kx.{py,sh}
│   ├── step4_extract_window_hidden_states_GPT.{py,sh}
│   └── step5_center_crop_features.{py,sh}
├── train/
│   ├── train.py                               # 训练、任务调度及评估
│   ├── wandb_tracking.py                      # W&B 记录模块
│   ├── model/                                 # 分类头实现
│   ├── run/                                   # YAML 配置和启动脚本
│   │   ├── example/
│   │   ├── V011-v2/
│   │   ├── V543-v2/
│   │   ├── V554-v1/
│   │   ├── V610-v2/
│   │   └── compare-jingyue/
│   ├── backbone/                              # 外部模型或软链接，Git 忽略
│   ├── data/                                  # 裁剪特征，Git 忽略
│   └── output/                                # 分类训练输出，Git 忽略
└── output/                                    # Step1–Step4 输出，Git 忽略
```

脚本中的 `chr19` 是历史命名，实际处理的染色体由参数决定。GitHub 仓库名为 `DNA-5mc`，下面示例将本地目录命名为 `DNA-5mC`。

## 2. 环境与路径

使用 Linux、Bash 和 Python 3.10 或更高版本。以下版本来自当前开发环境，供搭建环境时参考；不同主干还可能需要其自带模型代码声明的依赖。

| 依赖 | 当前环境版本 | 用途 |
| --- | --- | --- |
| Python | 3.10.20 | 运行脚本 |
| PyTorch | 2.6.0 | 特征提取、分类训练、读取 `.pt` 清单 |
| Transformers | 4.48.3 | 加载 encoder 和主干模型 |
| NumPy | 1.26.4 | 特征数组、memmap 和指标计算 |
| PyYAML | 6.0.3 | 配置文件 |
| safetensors | 0.7.0 | 预训练权重 |
| tqdm | 4.67.3 | 部分模型代码的进度显示 |
| wandb | 0.28.0 | 可选的实验记录 |

先安装与机器 CUDA 环境匹配的 PyTorch，再安装其余依赖。可选安装 `orjson` 加速 JSON 处理；Step1 的压缩相关操作可使用系统中的 `pigz`。

```bash
git clone https://github.com/2fhkf3/DNA-5mc.git DNA-5mC
cd DNA-5mC

# 在已准备好的 Python / Conda 环境中执行。
export PROJECT_DIR="$(pwd)"
export PYTHON_BIN="$(command -v python)"
"$PYTHON_BIN" -m pip install transformers==4.48.3 numpy==1.26.4 \
  PyYAML==6.0.3 safetensors==0.7.0 tqdm==4.67.3
# 使用 W&B 时再安装。
"$PYTHON_BIN" -m pip install wandb==0.28.0
```

所有 Bash 入口通过 [workflow/paths.sh](workflow/paths.sh) 定位项目，不依赖启动时的工作目录。本文命令统一在项目根目录执行；`/path/to/...` 均需替换为实际路径。

| 环境变量 | 默认值 / 行为 |
| --- | --- |
| `PYTHON_BIN` | 显式指定优先；否则使用已激活的非 base Conda 环境，再尝试服务器已有的 PoreDLM 环境；均不可用时使用当前 Conda Python 或 `python3` |
| `BACKBONE_ROOT` | `项目根目录/train/backbone` |
| `OUTPUT_ROOT` | `项目根目录/output`，影响工作流的默认输入输出路径 |
| `HF_MODULES_CACHE` | `OUTPUT_ROOT/.hf_modules` |
| `DRY_RUN=1` | Bash 入口只显示解析后的路径和命令，不开始处理或训练 |

**训练 YAML 的相对路径始终以 `train/` 为基准**，与 YAML 自身位置、启动目录无关。例如 `feature_dir: data/V543-V2` 指向 `train/data/V543-V2`；`output_dir: output` 指向 `train/output`。修改 `OUTPUT_ROOT` 不会自动修改 YAML。

若启动时报 `No module named 'numpy'` 或 `No module named 'torch'`，应检查实际使用的 Python。当前服务器的 `base` 环境缺少训练依赖，Bash 入口在 `base` 下优先使用已有 PoreDLM；手动设置 `PYTHON_BIN` 时，请确保对应环境包含这些依赖。

## 3. 准备预训练模型

将已有模型放在 `train/backbone/`，或创建同名软链接。以下名称与现有脚本的默认值对应；只需准备计划使用的模型。

```bash
mkdir -p "$PROJECT_DIR/train/backbone"
ln -s /path/to/V011 "$PROJECT_DIR/train/backbone/V011-Diffusion"
ln -s /path/to/V543 "$PROJECT_DIR/train/backbone/V543-BERT"
ln -s /path/to/V554 "$PROJECT_DIR/train/backbone/V554-BERT"
ln -s /path/to/V610 "$PROJECT_DIR/train/backbone/V610-GPT"
# Step1 原有预处理流程使用的模型。
ln -s /path/to/V003 "$PROJECT_DIR/train/backbone/V003-Preprocess"
```

| 本地目录名 | 需要的子目录 | 对应入口 |
| --- | --- | --- |
| `V003-Preprocess` | `encoder/`，包含预处理器配置及代码 | Step1 |
| `V543-BERT`、`V554-BERT` | `encoder/`、`base/` | Step4 BERT |
| `V011-Diffusion` | `encoder/`、`hf_dlm_elf_mixed_with_cls_token_step65000-unsharded/` | Step4 Diffusion |
| `V610-GPT` | `encoder/`、`base/` | Step4 GPT |

模型目录中的配置、tokenizer、自定义 Python 文件与权重应配套保存。各主干的 token 映射和特殊 token 约定不同，分别使用对应提取脚本。Diffusion/GPT 的 Bash 入口默认启用 Hugging Face 离线模式，需要本地模型文件完整。

## 4. 已有特征时：裁剪后开始训练

如果已有 Step4 的完整输出，可直接从这里开始。下面以 V543 的中心 5-mer 为例，输入目录必须同时包含 `train`、`valid`、`test` 三个 split。

```bash
INPUT_DIR="$PROJECT_DIR/output/step4_output/V543" \
OUT_DIR="$PROJECT_DIR/train/data/V543-V2" \
TRAIN_CONFIG="$PROJECT_DIR/train/run/V543-v2/config.yaml" \
WINDOW_SIZE=5 EXPERIMENT=V543-v2 \
  bash workflow/step5_center_crop_features.sh

# 验证真实输入、元数据和任务配置，不启动训练。
bash train/run/V543-v2/run.sh --dry-run

# 小规模试跑：一个分类头、一个 seed、单张 GPU、最多两轮。
# W&B 离线记录仍需要安装 wandb；也可在 YAML 中设置 enabled: false。
WANDB_MODE=offline bash train/run/V543-v2/run.sh \
  --strategies center_5mer_mlp_raw768 --seeds 1 --device cuda:0 \
  --epochs 2 --experiment V543-smoke

# 完整实验：使用 YAML 中设置的分类头、seed、设备和早停配置。
WANDB_MODE=offline bash train/run/V543-v2/run.sh
```

`DRY_RUN=1 bash ...` 只预览命令；`bash ... --dry-run` 则调用 Python 校验真实输入，数据量大时仍需要扫描元数据。

## 5. 从 JSONL 构建数据集

### Step1：信号预处理

输入应已包含信号、序列、比对和碱基—信号对应信息。Step1 使用模型的 `AutoFeatureExtractor` 处理 `signal`，保留其他字段，并检查处理前后的长度。

```bash
INPUT_FILE=/path/to/reads.genome_signal.jsonl \
OUTPUT_FILE="$PROJECT_DIR/output/step1_output/reads.preprocessed.jsonl" \
MODEL_DIR="$PROJECT_DIR/train/backbone/V003-Preprocess/encoder" \
MODE=single \
  bash workflow/step1_preprocess_signal_jsonl.sh
```

可用 `LIMIT=10000` 先处理少量记录；需要并行预处理时，可使用 `MODE=shard JOBS=4`。若输入已按目标预处理器处理，先确认其处理方式和坐标对应关系，再决定是否跳过 Step1，避免重复处理信号。

### Step2：生成窗口和标签，按位点划分

Step2 从一个输入数据集及 bedMethyl 构建正负样本，不需要分别提供 MOD/UNMOD 输入。当前规则包括：

- 中心参考碱基为 `C`，read 和 BED 均使用正链，默认修饰类型为 `m`。
- bedMethyl 修饰百分比为 `100` 的位点标记为 `1`，为 `0` 的标记为 `0`；中间百分比不进入数据集。
- 默认窗口为中心左右各 7 个碱基，共 15 nt；要求窗口内 query 序列与参考序列一致，信号跨度有效。
- 分别检查 BED 覆盖度、提取后的实际位点覆盖度和 read MAPQ。
- train/valid 按 `chrom + position` 确定性划分，同一位点的窗口归入同一个 split；`--test-only` 将保留数据全部写入 test。

以下示例使用 chr19 构建 train/valid，chr16 构建独立 test。覆盖度和 MAPQ 是示例实验设置，应按实际数据需求选择。

```bash
GENOME_JSONL="$PROJECT_DIR/output/step1_output/reads.preprocessed.jsonl"
BED_METHYL=/path/to/labels.bed.gz

"$PYTHON_BIN" workflow/step2_build_chrom19_dataset.py \
  --jsonl "$GENOME_JSONL" --bed-gz "$BED_METHYL" \
  --out-dir "$PROJECT_DIR/output/step2_output/chr19_cov10" \
  --chrom chr19 --flank 7 --min-coverage 10 --min-actual-coverage 10 \
  --min-mapq 30 --train-ratio 0.9 --valid-ratio 0.1

"$PYTHON_BIN" workflow/step2_build_chrom19_dataset.py \
  --jsonl "$GENOME_JSONL" --bed-gz "$BED_METHYL" \
  --out-dir "$PROJECT_DIR/output/step2_output/chr16_cov20_test" \
  --chrom chr16 --flank 7 --min-coverage 20 --min-actual-coverage 20 \
  --min-mapq 30 --test-only
```

这里直接调用 Python，以明确区分训练集和测试集设置。[Step2 的 Bash 入口](workflow/step2_build_chrom19_dataset.sh) 当前默认是 **chr16、覆盖度 20、test-only**；只修改染色体参数不会取消 `test-only`。

上游 JSONL 除 `signal`、`seq`、`chrom`、`strand`、`mapq` 和原始 read ID 外，还需要 `base_sample_span_seq` 及 `per_base`。后者包含 `genome_pos0`、`query_pos_signal`、`signal_span`、`query_base`、`ref_base` 等映射信息。仅有序列和信号数组不足以构建参考位点窗口；准确规则见 [Step2 实现](workflow/step2_build_chrom19_dataset.py)。

### Step3：类别平衡与可选筛选

平衡输入使用完整的 Step2 输出目录，保留其清单和校验信息。按 read/window 样本数平衡两类，同时维持最小位点覆盖度，保持已有 split 归属。

```bash
INPUT_DIR="$PROJECT_DIR/output/step2_output/chr19_cov10" \
OUT_DIR="$PROJECT_DIR/output/step3_output/chr19" \
MIN_COVERAGE=10 SPLITS="train valid" \
  bash workflow/step3_run_balance_chr19.sh

INPUT_DIR="$PROJECT_DIR/output/step2_output/chr16_cov20_test" \
OUT_DIR="$PROJECT_DIR/output/step3_output/chr16" \
MIN_COVERAGE=20 SPLITS=test \
  bash workflow/step3_run_balance_chr19.sh

# Step4 从同一个目录读取三个 split；这里使用软链接汇集。
mkdir -p "$PROJECT_DIR/output/step3_output/new"
ln -s "$PROJECT_DIR/output/step3_output/chr19/train.jsonl.gz" \
  "$PROJECT_DIR/output/step3_output/new/train.jsonl.gz"
ln -s "$PROJECT_DIR/output/step3_output/chr19/valid.jsonl.gz" \
  "$PROJECT_DIR/output/step3_output/new/valid.jsonl.gz"
ln -s "$PROJECT_DIR/output/step3_output/chr16/test.jsonl.gz" \
  "$PROJECT_DIR/output/step3_output/new/test.jsonl.gz"
```

`SITE_POLICY=auto` 在必要时可能移除部分多数类位点；`preserve_all` 要求保留所有位点，无法同时满足平衡与覆盖度时会报错。

可选使用 [Step3.1](workflow/step3.1_extract_chr19_small.py) 抽取小数据集，或 [Step3.2](workflow/step3.2_filter_chr19_mapq.py) 按更高 MAPQ 和实际覆盖度筛选。筛选后的类别比例需重新检查。各 Bash 入口保存了具体实验的默认路径，衔接时应显式设置 `INPUT_DIR`、`OUT_DIR`，不要假定所有默认值可直接串联。

## 6. Step4：提取主干特征

推荐使用新增的统一入口 [step4_extract_features.sh](workflow/step4_extract_features.sh)，参数集中在 [step4_config.yaml](workflow/step4_config.yaml)。Python 入口 [step4_extract_features.py](workflow/step4_extract_features.py) 复用原有三类提取器，支持任意兼容的新版本名；只复用 Test1 的通用配置/I/O 工具，不要求 WGA 标签、`dataset_manifest.json` 或 `sample_audit.sqlite`。

```bash
# 预览：不加载权重、不创建输出，显示缺失的输入 split。
DRY_RUN=1 bash workflow/step4_extract_features.sh

# 正式：先在 YAML 中填写实际 data_dir，再选择模型。
MODEL=V548 bash workflow/step4_extract_features.sh
MODEL=V011 bash workflow/step4_extract_features.sh
MODEL=V610 bash workflow/step4_extract_features.sh

# 小样本：limit 对每个选中的 split 分别生效；必须选择新的输出目录。
MODEL=V548 SPLITS="train valid test" LIMIT=17 \
DATA_DIR=/path/to/windows OUT_DIR=/tmp/my_step4_smoke \
  bash workflow/step4_extract_features.sh

# 只处理 test 或选择另一层，不影响其他模型的配置。
MODEL=V543 SPLITS=test HIDDEN_LAYER=6 OUT_DIR=/path/to/new/V543_layer6 \
  bash workflow/step4_extract_features.sh
```

配置分为 `model`、`extraction`、`models`：

- `model` 默认 V548；已提供 V011、V543、V548、V554、V610 条目。
- `extraction` 控制输入、输出、`splits`、设备、批次、落盘精度和性能开关。默认 `splits: [train, valid, test]`，输入支持 `.jsonl.gz` 和 `.jsonl`。
- `models.<名称>` 设置架构、encoder/base 路径、隐藏层及 ODE/token 协议，也可覆盖通用设置。未来模型直接新增条目即可。
- 优先级：命令行 > 显式环境变量 > 当前模型条目 > `extraction` > 默认值。`MODEL` 和 `VERSION` 都可选模型，同时设置时 `MODEL` 优先。`EXTRACTION_CONFIG` 指定其他 YAML；`PYTHON_BIN` 指定解释器。
- YAML 相对路径以项目根目录为基准；CLI/环境变量相对路径以启动目录为基准。`OUTPUT_ROOT` 在统一入口中表示 Step4 特征根目录。

**输入路径需填写实际位置：**统一配置沿用 `output/step3_output/new` 的原约定；创建本入口时该目录不存在。先运行预览检查 `missing_inputs`，再修改 `extraction.data_dir`。默认输出为 `output/step4_output/<模型名>`，其中 V011 默认是 `V011`，而非旧 Diffusion Shell 的 `V011-ode097`；需要旧名称时明确设置 `out_dir`。已存在的输出目录会被拒绝，避免覆盖已有实验。

统一入口默认落盘 FP16；BERT 主干批次 8，Diffusion/GPT 32；逐条 codec、模型设备 FP32 聚合。三类主干仍保持原来的计算精度与层选择规则。`pooling_device: cpu` 恢复原 CPU 聚合；`codec_batch_size > 1` 可尝试等长批量编码，但已有真实测试发现其卷积舍入误差可能改变量化 token，正式对比默认保持 1。所有字段均在 YAML 中注释。

输出仍为 `streaming_memmap_v1`，可供原 Step5 裁剪和训练读取。一次加载模型后依次处理所选 split，所有 split 成功后发布输出目录，并保存 `resolved_extraction_config.json` 和 `feature_summary.json`。运行中的 YAML 修改只影响后续任务；输入窗口或模型配置发生变化会报错。`limit` 非空时汇总标记 `partial_extraction: true`。只提取 test 的目录不能直接用于要求 train/valid/test 齐全的训练版 Step5。

2026-10-17 小样本验证：五个配置模型均通过 GPU 1 上的 `train/valid/test` 提取，每个 split 17 个窗口。测试使用真实 WGA 信号并人工交替设置标签，仅用于检查通用标签读取，不用于评估准确率；已核对样本顺序、FP16 缓存、Step5 清单读取、YAML 启动快照、已有输出拒绝覆盖、失败清理及输入变化检测。所有临时特征已清理，未运行全量提取。

下面保留原有三个架构入口的用法及默认值，旧入口仍可使用。

提取流程先将窗口信号通过 encoder 转成 token，再提取对应主干的隐藏状态，按碱基的信号跨度聚合为 `[N, W, H]`。`N` 为窗口数，`W` 为窗口碱基数，`H` 来自模型配置，常见为 768。

```bash
# BERT：默认模型是 V554；指定 MODEL_ROOT 可切换到 V543。
MODEL_ROOT="$PROJECT_DIR/train/backbone/V543-BERT" \
DATA_DIR="$PROJECT_DIR/output/step3_output/new" \
OUT_DIR="$PROJECT_DIR/output/step4_output/V543" \
DEVICE=cuda:0 \
  bash workflow/step4_extract_window_hidden_states_BERT-js.sh

# V554 使用相同 BERT 入口。
MODEL_ROOT="$PROJECT_DIR/train/backbone/V554-BERT" \
OUT_DIR="$PROJECT_DIR/output/step4_output/V554" \
  bash workflow/step4_extract_window_hidden_states_BERT-js.sh

# Diffusion 和 GPT 使用各自的入口。
DEVICE=cuda:0 bash workflow/step4_extract_window_hidden_states_Diffusion-kx.sh
DEVICE=cuda:0 bash workflow/step4_extract_window_hidden_states_GPT.sh
```

| 入口 | 默认输出目录，相对项目根目录 | 关键默认参数 |
| --- | --- | --- |
| BERT | `output/step4_output/V554` | `ENCODER_LAYER=0`、`HIDDEN_LAYER=-1`、`BATCH_SIZE=8`、`DTYPE=float16` |
| Diffusion | `output/step4_output/V011-ode097` | `ENCODER_LAYER=0`、`HIDDEN_SOURCE=ode_hidden`、`ODE_STEPS=1`、`ODE_START_T=0.97`、`ODE_SELF_COND_CFG_SCALE=0.5` |
| GPT | `output/step4_output/V610` | `ENCODER_LAYER=1`、`DECODER_LAYERS=last`、`TOKEN_ID_MODE=auto`、`SPECIAL_TOKENS=none` |

三个入口默认 `WINDOW_SIZE=15`、`PREPROCESS=none`。Diffusion/GPT 默认 `BATCH_SIZE=4`、`DTYPE=float32`。调整 batch、层选择或 token 设置时，使用新的输出目录，并记录相应配置。

先用 `LIMIT=20 OUT_DIR=/path/to/new_smoke_features` 小规模验证。正式提取完成后，检查 `feature_summary.json` 的记录数、跳过原因和截断统计；不同模型可能因长度限制保留不同样本，比较结果前应核对样本集合。

每个 split 的输出包括：

```text
train.pt                    # streaming_memmap_v1 小清单，不是全部特征张量
train.features.f16.bin      # float32 时为 train.features.f32.bin
train.labels.f32.bin
metadata_train.tsv
valid.pt / valid.* / metadata_valid.tsv
test.pt / test.* / metadata_test.tsv
feature_summary.json
```

移动特征时应保留 `.pt` 引用的 `.bin` 和对应 TSV。仅复制 `.pt` 无法训练；`.partial` 文件也不能作为已完成输入。Step4 的 JSONL 窗口应包含非空有限值信号、二分类标签、序列/参考序列、碱基信号跨度以及 read ID、染色体和中心位置等元数据。

## 7. Step5：中心裁剪为 5-mer、7-mer 等

[裁剪脚本](workflow/step5_center_crop_features.py) 读取三种提取入口输出的 `streaming_memmap_v1` 格式，分块写出新特征。裁剪长度必须为不小于 5、且不超过原窗口长度的奇数。

裁剪同时更新 TSV 的 `seq/ref`，保留原序列及裁剪偏移，保持标签、样本顺序和原始基因组/信号坐标。它保留原主干上下文，仅截取输出表征，不等同于将更短信号重新输入主干。

```bash
INPUT_DIR="$PROJECT_DIR/output/step4_output/V543" \
OUT_DIR="$PROJECT_DIR/train/data/V543_center7" \
TRAIN_CONFIG="$PROJECT_DIR/train/run/V543-v2/config.yaml" \
WINDOW_SIZE=7 EXPERIMENT=V543-center7 \
  bash workflow/step5_center_crop_features.sh

# 使用生成的配置，其中 feature_dir 和 fusion_window_size 已同步更新。
WANDB_MODE=offline bash train/run/example/run.sh \
  "$PROJECT_DIR/train/data/V543_center7/train_config.yaml"
```

`TRAIN_CONFIG` 是训练超参数模板，不是模型配置；脚本生成新目录中的 `train_config.yaml`，不会修改模板。Bash 默认输入、输出和模板均对应 V610；切换模型时应同时指定这三个路径。V011 通常对应 `output/step4_output/V011-ode097`、`train/data/V011-V2` 和 `train/run/V011-v2/config.yaml`。

名称含 `center_5mer` 的 Linear、Gaussian Pool、Learned Pool、MLP、DWConv 头固定使用中心 5 个位置；`center_base` 只使用中心位置。融合头 `center_5mer_emb8_none` 按 `fusion_window_size` 取窗口，因此使用生成配置可比较 5/7/9-mer 融合实验。

裁剪期间不要修改源特征、元数据或模板 YAML。脚本结束前会检查这些文件是否变化，变化时拒绝发布输出；每个 split 的 `[DONE]` 不代表整个任务完成，以最终 `[SUCCESS]` 为准。输出目录必须不存在，已有目录不会被覆盖。

## 8. 分类训练配置

完整参数及中文注释见 [example/config.yaml](train/run/example/config.yaml)。启动脚本支持一个可选的 YAML 路径，其后的参数转交 `train.py`；命令行参数优先于 YAML。

| 配置入口 | 默认特征目录，相对 `train/` | 分类头数 × seed 数 | 最大轮数 / 早停 patience |
| --- | --- | --- | --- |
| [V011-v2](train/run/V011-v2/config.yaml) | `data/V011-V2` | 5 × 3 | 80 / 6 |
| [V543-v2](train/run/V543-v2/config.yaml) | `data/V543-V2` | 5 × 3 | 80 / 6 |
| [V610-v2](train/run/V610-v2/config.yaml) | `data/V610-V2` | 5 × 3 | 80 / 6 |
| [V554-v1](train/run/V554-v1/config.yaml) | `../output/step4_output/V554` | 4 × 3 | 80 / 12 |
| [example](train/run/example/config.yaml) | `../output/step4_output/V554` | 6 × 3 | 80 / 12 |

V554-v1 的默认 `experiment` 名为 `V554`。上述配置默认使用 `cuda:0` 和 `cuda:1`；单卡机器可用 `--device cuda:0` 覆盖。`devices` 表示在不同 GPU 上调度独立的“分类头 × seed”任务，每个任务占一个 GPU，不是同一个模型的 DDP 训练。

### 支持的分类头

| `strategy` | 输入与结构 |
| --- | --- |
| `center_base_linear_raw768` | 中心一个碱基表征，LayerNorm + Linear |
| `center_5mer_linear_raw768` | 中心 5 个表征，归一化后展开 + Linear |
| `center_5mer_gaussian_pool_linear_raw768` | 中心 5 个表征，固定高斯权重池化 + Linear |
| `center_5mer_learned_pool_linear_raw768` | 中心 5 个表征，可学习的全局位置权重池化 + Linear |
| `center_5mer_mlp_raw768` | 中心 5 个表征，MLP |
| `center_5mer_residual_dwconv_raw768` | 中心 5 个表征，残差深度卷积分类头 |
| `center_5mer_emb8_none` | 信号表征分支 + 8 维碱基 Embedding 分支，BiLSTM 融合 |

`raw768` 是策略的历史名称，实际隐藏维度从输入特征读取。融合头不依赖额外统计量 TSV，碱基来自元数据中的 `ref` 或 `seq`，由 `base_source` 决定。

### 训练、早停和读取性能

- `batch_size`、`eval_batch_size` 分别控制每个任务的训练和评估 batch。`strategy_overrides` 可为特定分类头覆盖训练参数；现有配置对融合头使用更小的 batch。
- `batch_read: true` 启用批量 memmap 读取。随机训练 batch 按磁盘行号排序读取，再恢复采样顺序；评估通常按连续切片读取。
- `num_workers` 是每个训练任务的 DataLoader worker 数，`eval_num_workers` 单独控制评估。`prefetch_factor` 和 `persistent_workers` 在 worker 数大于 0 时生效。多 GPU 任务会分别占用 CPU、预取内存和磁盘带宽，worker 数应结合实际吞吐调整。
- `log_every_steps` 控制训练日志间隔，并额外输出首尾 batch；设为 `0` 关闭 step 日志，仍保留 epoch 记录。日志中的 `data_wait_s` 是当前日志区间等待 batch 的累计时间，`eta_s` 估计当前 epoch 剩余训练时间。
- 每轮在 valid 的 read/window 层面评估。默认按 `monitor_metric: auroc` 保存最佳模型；连续 `patience` 轮没有严格提升即早停，相同分数也计为未提升。选择 `loss` 时越低越好；指定指标无法计算时回退到验证 loss。
- `threshold: 0.5` 用于概率转标签，影响 accuracy、F1 等，不影响 AUROC、AUPRC。需要选阈值时在 valid 上选择，再固定用于 test。
- `check_site_overlap: true` 检查三个 split 是否有相同染色体和中心坐标。三组数据的窗口长度、隐藏维度及特征/元数据行数必须匹配。
- `skip_completed: true` 仅在配置、输入签名和代码匹配时跳过完整实验。`best.pt` 保存最佳分类头权重，不含用于断点续训的完整优化器状态。更改实验设置时使用新的 `experiment`。

### 输出与指标

```text
train/output/<experiment>/
├── scheduling.json
├── summary.tsv                         # 每个分类头、每个 seed 的结果
├── aggregate.tsv                       # 跨 seed 的均值、样本标准差和方差
└── runs/<strategy>_seed<N>/
    ├── resolved_config.yaml
    ├── config.json
    ├── train.log
    ├── history.jsonl
    ├── best.pt
    ├── metrics.json
    ├── test_predictions.tsv            # save_predictions: test 时
    └── test_site_predictions.tsv
```

最终评估使用最佳验证轮次的权重，默认评估 valid 和 test；`evaluate_train: true` 可额外评估 train。read/window 层面以每条窗口为样本；site 层面按位点汇总窗口预测概率的均值。`metrics.json` 包含对应的二分类指标；汇总表记录 accuracy、F1、AUROC 和 AUPRC。跨 seed 的标准差/方差使用 `ddof=1`。

## 9. W&B 实验记录

[wandb_tracking.py](train/wandb_tracking.py) 由训练入口调用，应与训练代码一起保留。V011-v2、V543-v2、V610-v2 默认启用 W&B，仓库配置中的 entity 是原实验账号；其他账号运行时应修改为有权限的个人或团队 entity。

```yaml
wandb:
  enabled: true
  project: DNA-5mc-classifier
  entity: null             # 使用当前登录账号，或填自己的团队名
  name: null               # 默认 experiment/strategy_seedN
  group: null              # 默认 experiment
  mode: online             # online、offline、disabled
  tags: [V543]
```

```bash
# 在线模式：先登录，再使用自己的 entity 配置运行。
"$PYTHON_BIN" -m wandb login
bash train/run/V543-v2/run.sh

# 离线模式：覆盖 mode，训练记录保存在本地。
WANDB_MODE=offline bash train/run/V543-v2/run.sh
```

每个分类头/seed 对应一个 run，记录 step loss、学习率、吞吐、等待时间，以及 epoch 验证指标和最终评估结果。运行目录中另有 `wandb/` 和 `wandb_run.json`。`wandb.enabled: false` 可完全关闭记录，此时无需安装 W&B SDK；关闭后仅设置 `WANDB_MODE` 不会重新启用记录。

## 10. 运行前检查与常见问题

```bash
# 只检查路径和即将执行的命令，不需要准备完整数据。
DRY_RUN=1 bash workflow/step4_extract_window_hidden_states_BERT-js.sh
DRY_RUN=1 bash workflow/step5_center_crop_features.sh
DRY_RUN=1 bash train/run/V543-v2/run.sh

# 已准备数据后，执行真实输入检查。
bash train/run/V543-v2/run.sh --dry-run
```

- **找不到 config.yaml**：检查指定文件是否存在，并区分项目名中的 `DNA-5mC` 与旧目录 `DNA-5mc`。Step5 的 `TRAIN_CONFIG` 应指向本仓库内的训练 YAML。
- **裁剪过程中源文件发生变化**：报错路径可能是特征、TSV，也可能是模板 YAML。保持这些输入不变后重新裁剪；需要编辑训练参数时，可先复制模板，并将 `TRAIN_CONFIG` 指向固定副本。
- **启动后长时间没有训练 step**：启动阶段会读取清单、核对标签/元数据、检查位点交集；这部分发生在首个 batch 之前。根据日志区分输入检查、训练和验证阶段。
- **训练等待数据时间高**：检查特征所在磁盘、并发读任务和可用内存。优先使用中心裁剪缓存、本地较快磁盘和 `batch_read: true`，再比较适量 worker 的吞吐。
- **GPU 编号不可用**：用 `--device cuda:0` 或 `--device cpu` 覆盖多 GPU 配置；CPU 可用于小规模验证。
- **W&B 初始化失败**：核对登录状态和 entity 权限，或使用 `WANDB_MODE=offline` / `enabled: false`。

`.gitignore` 已排除外部模型目录、数据、特征、权重、日志及运行结果。迁移到新机器时，需要重新准备这些文件，并检查路径；克隆代码本身不会恢复本地模型软链接或实验数据。
