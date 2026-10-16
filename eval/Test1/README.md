# Test1：使用约 5× WGA UNMOD 数据评估 DNA 5mC 假阳性

本方案适用于本仓库已经训练完成的 V011、V543、V610 分类模型。输入为通过 WGA 制备的未修饰 DNA 测序数据，**5× 指总体测序深度，不是每个位点固定有 5 条 read**。窗口语料沿用 `workflow/step2_build_chrom19_dataset.sh` 所调用脚本的 JSONL 输出格式。

本次实验回答两个问题：未修饰 DNA 上有多少 read–位点被误报为 5mC；将同一位点的多条 read 汇总后，有多少阴性位点被误报为阳性。

**主指标为 read/window 级 FPR 和 site 级 FPR，同时报告有效 read 数、位点数、覆盖度分布和置信区间。全阴性数据不能单独评估检出率，也不能用 AUROC 作为主要结论。**

本文是评测设计和实施说明，更新日期为 2026-10-16。五步代码、带参数注释的 Bash 脚本和隔离测试入口已经放入 [workflow/](workflow/README.md)，原训练流程保持不变。当前只开展小规模测试，文中的结果表均为模板，不代表已经完成全量 WGA 评测。

默认输入已设置为用户提供的 chr19 WGA 预处理 JSONL。默认一键入口仅测试：`bash eval/Test1/workflow/run_all.sh --test`；只有显式 `--run` 才启动全量流程。测试写入 `/tmp` 并在成功后清理，正式输出位置为 `eval/Test1/output/`。

## 1. 项目的整体逻辑

项目将 Nanopore 信号的表征提取和分类器训练分开。Step4 使用已有预训练主干提取隐藏状态，`train/train.py` 读取固定特征，只更新分类头。

```text
已有全 read 的信号、序列、比对和碱基—信号映射
    │
    ├─ Step1：按原实验方式预处理信号
    │
    ├─ Step2：提取以参考 C 为中心的 15 nt 窗口，附加标签和元数据
    │
    ├─ Step3：训练数据类别平衡，可选 MAPQ / 覆盖度筛选
    │
    ├─ Step4：信号 → encoder / codec → 预训练主干 → [N, 15, 768]
    │
    ├─ Step5：裁剪输出表征的中心 5 个位置 → [N, 5, 768]
    │
    └─ train.py：训练分类头 → valid 选择 best.pt → read/site 评测
```

| 环节 | 当前实现及意义 | 对 Test1 的处理 |
| --- | --- | --- |
| 上游数据 | 仓库从已有 JSONL 开始，不包含完整 POD5/FAST5 到比对 JSONL 流程 | WGA 数据需先具有相同的信号、比对和 span 信息 |
| Step1 | 预处理信号；Step4 当前训练特征使用 `preprocess: none` | 与训练保持一致，避免重复预处理 |
| Step2 | 正链 read、正链标签、中心 C；默认 15 nt；要求窗口 `seq == ref` | 保持相同窗口和质量条件；阴性标签改由 WGA 实验身份确定 |
| Step3 | 按两类窗口数量平衡，并维持最小位点覆盖度 | **跳过类别平衡**；全阴性测试不需要构造两类 |
| Step3.2 | 可再次提高 MAPQ 和覆盖度；Bash 默认 MAPQ 60、覆盖度 10 | 不直接使用默认值；按预先确定的质量规则处理 |
| Step4 | 三种主干分别提取特征，主干权重冻结 | 每个主干只提取一次 WGA 特征，再供相应多个分类头复用 |
| Step5 | 15 → 5 是输出表征裁剪，保留原 15 nt 输入上下文 | 保持此语义；不能直接将原始信号改成 5 nt 后提取 |
| 分类头 | 中心碱基 Linear、5-mer Linear、MLP、DWConv，以及可选融合头 | 用已完成训练的 `best.pt` 做前向推理，不重新训练 |
| read/window 评测 | 每一条 read 上的一个目标位点构成一条预测 | 是本次低深度阴性评测的主分析单位 |
| site 评测 | 当前按 `chrom + position + strand + source` 分组，取概率均值 | 保持相同汇总方式，额外按实际覆盖度分层 |

对应代码：

- [Step2 窗口和标签构建](../../workflow/step2_build_chrom19_dataset.py)
- [类别平衡](../../workflow/step3_balance_chr19.py)
- [MAPQ / 覆盖度筛选](../../workflow/step3.2_filter_chr19_mapq.py)
- [BERT 特征提取](../../workflow/step4_extract_window_hidden_states_BERT-js.py)
- [Diffusion 特征提取](../../workflow/step4_extract_window_hidden_states_Diffusion-kx.py)
- [GPT 特征提取](../../workflow/step4_extract_window_hidden_states_GPT.py)
- [中心裁剪](../../workflow/step5_center_crop_features.py)
- [分类头定义](../../train/model/__init__.py)
- [训练、概率计算和位点汇总](../../train/train.py)

## 2. Test1 的实验范围

### 2.1 主分析设置

| 项目 | 设置 |
| --- | --- |
| 阴性对照 | 独立 WGA 制备的 UNMOD DNA |
| 总体测序深度 | 约 5×；另行统计过滤后的实际深度 |
| 链 | 与现有训练一致，主分析仅使用正向比对 read 和相应正链 C |
| 输入窗口 | 15 nt，中心两侧各 7 nt |
| 分类器特征 | 与对应训练 run 一致；当前 v2 数据为中心 5 个位置、隐藏维度 768 |
| 质量筛选 | 沿用训练数据实际采用的 MAPQ、窗口匹配和 span 有效性规则 |
| 实际位点覆盖度 | 构建阶段保留至少 1 条有效独立 read；评测时分层 |
| 数据 split | 独立 `test`，不产生新的训练集或验证集 |
| checkpoint | 原训练 run 的 `best.pt` |
| 默认 read 阈值 | `t_read = 0.5`，沿用当前模型设置 |
| 默认 site 阈值 | `t_site = 0.5`，用于同一位点的平均预测概率 |
| 模型选择 | 在 WGA 结果揭盲前，按原 valid 规则确定比较对象 |

本项目当前训练数据仅含正链，因此 Test1 首先测量正链表现。负链可作为后续独立扩展，不应直接混入本次主结果。

如果约 5× 是两条链合计的总体深度，正向比对 read 的深度通常会进一步降低；并不保证恰好减半。WGA 扩增、比对质量、15 nt 完全匹配和 span 筛选也会改变实际覆盖度。**主分析不能要求每个位点至少 5 条 read，更不能继续使用原测试集的 20 条 read 门槛。**

### 2.2 染色体和位点集合

当前用户提供的是 chr19 的 WGA 数据，因此可执行脚本默认处理 chr19。它属于独立阴性文库评测，但 chr19 与原训练染色体重合，不能将其写成未见基因组位点的泛化评测。后续如有 chr16 WGA 数据，可再加入原独立测试染色体的对照。

目标位点集合应在查看模型概率之前确定，可采用参考基因组中的 CpG 集合或与原训练任务一致的候选位点集合。当前 Step2 只要求中心 C，没有在代码中强制 CpG；是否属于 CpG 取决于上游候选 BED。评测前必须核对实际训练位点上下文，不能未经检查将“所有 C”和“CpG”混为同一个任务。

如扩展到 chr19 等训练染色体，应单列“已见坐标 / 未见坐标”结果。新的 WGA read 可以是独立测序样本，但相同基因组坐标不自动构成未见位点。

### 2.3 WGA 阴性对照的解释边界

以 WGA 制备阴性对照有文献依据。Rockfish 使用 WGA 样本作为 negative control，并对 WGA 样本进行 M.SssI 处理制备 positive control，见 [Rockfish，Nature Communications，2024，Methods](https://www.nature.com/articles/s41467-024-49847-0)。

Test1 将经过实验质控、确认用于阴性对照的 WGA 文库目标碱基标为 `label = 0`。需要记录 WGA 批次、建库和测序条件；如果存在原始模板残留或污染，观察到的阳性调用不能全部无条件归因为算法错误。没有额外实验核验时，应将结果描述为“WGA 阴性对照上的观察误报率”。

该实验检验的是当前化学体系和处理流程下的阴性误报，不单独证明模型在天然 DNA 上的全部泛化能力。

## 3. 数据准备：保留 Step2 格式，改变标签来源

### 3.1 当前 Step2 不能原样承担 WGA 阴性标签构建

当前实现从 bedMethyl 第 11 列读取修饰百分比：

- `0%` → `label = 0`。
- `100%` → `label = 1`。
- 其余百分比被移除。
- 默认只接受正链 BED 和正向比对 read。
- `source` 被硬编码为 `mod`。

这些规则适合原有 BED 标签实验，却不能直接定义 WGA 阴性评测集合。

**不要先运行某个甲基化检测工具，再只保留其报告为 0% 的 WGA 位点来做 Test1。** 这样会优先删除阴性对照中的疑难位点和部分误报位点，低估假阳性。也不要把原天然 DNA 的正负标签直接用于 WGA：天然 DNA 中被修饰的位点在 WGA 阴性文库中仍应按照阴性对照身份标注。

### 3.2 推荐的构建方式

建议为窗口构建器增加独立的 WGA 阴性模式，或新增专用构建入口，输出仍符合 Step2 JSONL 合同：

1. 从预先确定的参考位点集合选择目标 C，沿用已有质量和窗口完整性要求。
2. 从 WGA read 提取真实信号和真实碱基—信号映射。
3. 将保留的目标窗口全部标记为 `label = 0`，写入 `source = unmod` 或明确的 WGA 文库标识。
4. 按原始 read ID 和目标位点去重；同一原始 read 的切片不能重复计数。
5. 将最小实际覆盖度设为 1，保留低深度窗口。
6. 全部写入 `test.jsonl.gz`，记录候选数、过滤数和过滤原因。

上述构建要求已由 [step1_build_unmod_windows.py](workflow/step1_build_unmod_windows.py) 实现，复用原 Step2 的窗口验证逻辑，独立确定 WGA 阴性标签。原项目 Step2 没有增加 `--unmod` 或 `--constant-label` 选项。

也可以使用独立生成的、明确标记为阴性的候选 BED 复用现有提取器，但候选位置必须来自预先确定的参考集合，不能来自模型阴性调用；用于满足旧 BED 字段要求的数值不能冒充真实测得的修饰率或覆盖度。优先采用专用 WGA 模式，避免这种字段语义混用。

如果已经制备好本节要求的窗口 JSONL，并确认没有在上游按甲基化预测结果筛选，直接从 Step4 开始即可，不需要重做 Step1/Step2。

### 3.3 窗口输入合同

每条记录至少应保留以下信息：

| 字段 | 要求 |
| --- | --- |
| `read_id` | 可追溯到独立原始 read；不要用窗口 ID 替代原始 read 身份 |
| `chrom`、`center_genome_pos0` | 与参考版本一致的 0-based 中心坐标 |
| `strand` | 主分析为 `+` |
| `seq`、`ref` | 15 nt，中心为目标 C，与训练窗口要求一致 |
| `signal` | 对应 15 nt 窗口的有限数值信号，保持原处理尺度 |
| `base_sample_span_ref` / `base_sample_span_seq` | 每个碱基在局部窗口信号内的有效半开区间 |
| `signal_len` | 等于局部信号长度 |
| `label` | 全部为整数 0 |
| `source` | 建议为 `unmod`；单独保留具体 `library_id` |
| `mapq` | 保留以便检查筛选条件 |
| `split` | `test` |
| `split_key` | 与原流程兼容，例如 `chr16:123456` |
| `actual_coverage` | 如填写，应来自去重和质量过滤后的实际覆盖度 |

Step4 会重新判断部分窗口是否可提取，所以 site 级分母和覆盖度还应以**最终成功产生预测的独立 read**重新计算，不能只沿用输入中的 `actual_coverage`。

当前 Step4 的元数据 TSV 不保证保留所有原始质量字段。若要按 MAPQ、原始信号长度或文库批次分层，应在构建窗口时另存样本审计表，或扩展元数据保留字段，再按 read–位点键连接，不能假定最终 TSV 自动包含这些信息。

`source = unmod` 是来源标识，`label = 0` 才是分类真值。当前模型不会把 `source` 作为输入，但现有位点汇总会使用它分组。同一物理位点如因多个来源标识被拆成多个组，不应在主分析中被重复计为多个独立位点。多个文库优先分别评测。

### 3.4 5× 数据的筛选策略

| 条件 | 主分析建议 | 原因 |
| --- | --- | --- |
| MAPQ | 以原 run 的实际训练数据来源为准 | 当前 Bash 默认值不一定等于已训练模型的数据筛选值 |
| 窗口匹配 | 沿用 `seq == ref` | 保持与现有模型评估域一致 |
| 原始信号和 span | 严格验证 | 防止错误窗口产生伪误报 |
| 最小实际覆盖度 | 1 | 保留 5× 数据中的低覆盖位点 |
| 原 BED 覆盖度 | WGA 专用模式不依赖原天然 DNA BED 的覆盖度 | 它不代表 WGA 上的有效深度 |
| 类别平衡 | 不执行 | WGA 全阴性集合没有需要平衡的阳性类 |
| 按模型置信度过滤 | 主结果不执行 | 防止通过丢弃疑难预测降低 FPR |

如果另设高置信度调用或拒绝判断机制，需要同时报告保留比例 / call rate；被拒绝的样本不能计作 TN。无覆盖、比对失败和特征提取失败也不能记为正确阴性。

## 4. 固定模型及特征提取协议

### 4.1 先冻结模型，再看 WGA 结果

本实验不使用 WGA 数据继续训练、不用它选择最佳 epoch，也不根据其 FPR 挑选最佳 seed。

从每个训练 run 读取：

```text
train/output/<experiment>/runs/<strategy>_seed<N>/
├── best.pt
├── config.json
├── resolved_config.yaml
└── metrics.json
```

使用 `best.pt` 内保存的模型状态和训练配置，并核对该 run 的 `config.json`。不要使用此刻可能已经改动的 `train/run/*/config.yaml` 反推旧模型的真实结构。

建议主比较采用三个版本共有的分类头和共同完成的 seeds，例如 seed1、seed2、seed3，逐 seed 输出结果。融合头仅在对应 `best.pt` 和完整训练结果存在时纳入。不同 seed 的变化反映训练随机性，不等同于独立 WGA 生物或实验重复。

### 4.2 与现有 v2 特征一致的关键设置

Step2 现在通过 [workflow/config.yaml](workflow/config.yaml) 配置模型名称、架构、权重路径和隐藏层，启动方式见 [工作流说明](workflow/README.md)。下表描述原 v2 分类头使用的基线协议；新版本或新隐藏层可通过新增配置提取，但应配套在相应特征上训练的分类头。

以下依据当前 `train/data/*/feature_summary.json` 核对。正式运行仍需核对具体 checkpoint、tokenizer、模型配置和提取实现；目录名称相同不保证权重相同。

| 版本 | 主干特征协议 |
| --- | --- |
| V011 | `encoder_layer=0`；`hidden_source=ode_hidden`；`ode_steps=1`；`ode_start_t=0.97`；`ode_self_cond_cfg_scale=0.5`；`last_t=1.0`；保留 CLS/BOS/EOS 约定 |
| V543 | 指定 V543 权重；`encoder_layer=0`；BERT 最后隐藏层 `hidden_layer=-1`；保持 token offset 和特殊 token 约定 |
| V610 | `encoder_layer=1`；GPT 最后一层；当前脚本 `DECODER_LAYERS=last` 对应 summary 的 `[-2]`；`TOKEN_ID_MODE=auto`，核对实际解析为 tokenizer 映射；`SPECIAL_TOKENS=none` |
| 三者共通 | 输入 15 nt 信号窗口；`PREPROCESS=none`，前提是信号已经按训练流程处理；隐藏维度 768；与旧数据一致的 pooling 和截断策略；当前缓存存储为 float16 |

注意 V543 脚本的 `MODEL_ROOT` 默认指向 V554，必须显式指定 V543。部分历史 summary 中 `model_family` / `feature_source` 使用了旧名称，实际模型应以权重来源、配置和提取参数确认。

特征清单见：

- [V011 feature_summary.json](../../train/data/V011-V2/feature_summary.json)
- [V543 feature_summary.json](../../train/data/V543-V2/feature_summary.json)
- [V610 feature_summary.json](../../train/data/V610-V2/feature_summary.json)

这些数据文件受 Git 忽略，仓库迁移后可能不存在。

### 4.3 推荐目录

```text
eval/Test1/
├── README.md                         # 实验方案
├── workflow/                         # 五步代码、启动脚本和测试
└── output/                           # 正式运行时生成，当前未启动全量评测
    ├── data/                         # test.jsonl.gz、审计索引和数据清单
    ├── features/<VERSION>/           # 原 15 nt 输出表征
    ├── center5/<VERSION>/            # 中心 5 个输出位置
    ├── results/<VERSION>/<run>/      # 每个 checkpoint 的结果
    └── comparison/                   # 相同样本上的模型比较
```

### 4.4 当前可用的 test-only 特征提取命令

以下原始提取器示例从项目根目录执行。先激活项目所用环境，准备好上一节规定的 `eval/Test1/output/data/test.jsonl.gz`，确认 `train/backbone` 中链接指向原实验相同权重。实际五步工作流优先使用新增启动脚本，以便自动补齐审计字段和协议清单。

```bash
cd /home/shj/ZJ-lab/Modification/DNA-5mC
```

V011 当前支持仅处理 test：

```bash
DATA_DIR="$PWD/eval/Test1/output/data" \
OUT_DIR="$PWD/eval/Test1/output/features/V011-original-example" \
MODEL_ROOT="$PWD/train/backbone/V011-Diffusion" \
SPLITS=test DEVICE=cuda:0 WINDOW_SIZE=15 \
ENCODER_LAYER=0 HIDDEN_SOURCE=ode_hidden \
ODE_STEPS=1 ODE_START_T=0.97 ODE_SELF_COND_CFG_SCALE=0.5 LAST_T=1.0 \
PREPROCESS=none DTYPE=float16 \
  bash workflow/step4_extract_window_hidden_states_Diffusion-kx.sh
```

V610 当前支持仅处理 test：

```bash
DATA_DIR="$PWD/eval/Test1/output/data" \
OUT_DIR="$PWD/eval/Test1/output/features/V610-original-example" \
MODEL_ROOT="$PWD/train/backbone/V610-GPT" \
SPLITS=test DEVICE=cuda:0 WINDOW_SIZE=15 \
ENCODER_LAYER=1 DECODER_LAYERS=last TOKEN_ID_MODE=auto \
SPECIAL_TOKENS=none PREPROCESS=none DTYPE=float16 \
  bash workflow/step4_extract_window_hidden_states_GPT.sh
```

初次运行可增加 `LIMIT=20`，但必须使用独立的 smoke 输出目录；正式评测去掉 LIMIT。`DRY_RUN=1` 只预览命令，不能验证权重、全部输入或推理结果。

**原 V543 入口没有 test-only 参数。** 新增的 [step2_extract_features.py](workflow/step2_extract_features.py) 直接复用其模型加载和单 split 提取函数，仅处理 test；原 BERT 脚本不做修改。三个主干统一使用新 [Step2 启动脚本](workflow/step2_extract_features.sh)，无需复制测试数据来凑齐三个 split。

### 4.5 中心裁剪的两种实施方式

现有 Step5 同样固定要求三个 split，没有仅裁剪 test 的参数。推荐在独立评测器中对 `[N, 15, 768]` 做批次内中心裁剪：

```python
# 示例逻辑：15 个输出位置取中心 5 个，索引 5:10。
# 应对每个 batch 执行，不要一次把全部特征加载到内存。
start = (source_window_size - checkpoint_window_size) // 2
x = x[:, start:start + checkpoint_window_size, :]
```

`checkpoint_window_size` 读取该 run 的 `source_window_size`，当前 v2 为 5。若是融合头，还需按同一偏移裁剪 `ref` 或 `seq`，并遵循 checkpoint 的 `base_source` / `fusion_window_size`。不能只裁剪特征、不裁剪配套碱基输入。

新增的 [step3_center_crop.py](workflow/step3_center_crop.py) 已支持只裁剪 test。一键流程默认使用它提前生成中心 5-mer 缓存；独立评测器也支持直接读取 15 nt 特征并按 checkpoint 裁剪。两者对同一输入应产生一致的分类概率。

## 5. 加载训练好的分类器进行推理

### 5.1 当前仓库缺少独立评测入口

`train/train.py` 当前没有 `--eval-only`、`--checkpoint` 或 `--resume-for-eval` 参数。它的主入口会准备三组数据、建立优化器并启动训练。

**不要把 WGA 特征目录换进训练 YAML，再运行 `train/run/*/run.sh` 作为评测。** 正式 Test1 需要一个只加载 checkpoint、只前向推理的入口。

独立入口已实现为 [workflow/step4_evaluate.py](workflow/step4_evaluate.py)，复用已有模型定义，支持单 run 和多 run，只进行前向推理；启动脚本为 [step4_evaluate.sh](workflow/step4_evaluate.sh)。

### 5.2 评测器的实现要求

1. 接受一个训练 run 目录和一个外部 WGA test 特征目录。
2. 加载 `best.pt` 和该 run 保存的配置，重建对应分类头，严格匹配权重键。
3. 校验主干身份、隐藏维度、输入窗口协议、元数据行数和行顺序。
4. 确认 WGA `label` 全部为 0；发现阳性标签时直接拒绝按全阴性模式出结果。
5. 批次读取 memmap，按 checkpoint 需要进行中心裁剪。
6. 使用 `model.eval()` 和关闭梯度的前向推理，不建立优化器、不更新权重。
7. 普通分类头使用 `sigmoid(logits)`；融合头使用 `softmax(logits)[:, 1]`。
8. 保存每条 read–位点的分数与标签，保留原始 read ID、链和文库来源。
9. 按位点重新汇总有效独立 read，计算覆盖度、平均概率和阈值调用比例。
10. 输出全阴性专用指标、覆盖度分层、保留比例和运行记录。

当前 `train.py` 的 `predict()` 已经有关闭梯度和评估模式的处理，可以复用；但 `binary_metrics()` 在分母为零时会对部分指标填 0，不能直接把这些 0 解释成全阴性数据上的有效性能指标。

### 5.3 已实现的独立推理接口

以下命令可在特征准备完成后运行。当前没有启动正式全量评测：

```text
python eval/Test1/workflow/step4_evaluate.py
  --run-dir train/output/V610-v2/runs/center_5mer_mlp_raw768_seed1
  --feature-dir eval/Test1/output/center5/V610
  --version V610
  --threshold 0.5
  --site-threshold 0.5
  --out-dir eval/Test1/output/results/V610/center_5mer_mlp_raw768_seed1
```

本入口固定处理 test，无需指定 split。上面采用分行的接口说明格式；实际可执行的 Bash 命令见 [workflow/README.md](workflow/README.md)。中心裁剪参数从 checkpoint 读取，不使用 WGA 测试结果来修改模型输入协议。

## 6. 主指标的严格定义

### 6.1 read/window 级假阳性率

这里的“一条样本”是某条原始 read 上的一个目标碱基窗口。一个长 read 可以贡献多个目标窗口，因此它不等于独立整条 read 的数量。

对全阴性窗口，设甲基化预测概率为 `p_i`，固定阈值为 `t_read`：

```text
预测阳性：p_i >= t_read
FP_read = 被预测为阳性的阴性窗口数
TN_read = 被预测为阴性的阴性窗口数
N_read  = FP_read + TN_read

FPR_read = FP_read / N_read
Specificity_read = TN_read / N_read = 1 - FPR_read
每百万次目标碱基调用的误报数 = FPR_read × 1,000,000
```

例如 1,000,000 个阴性窗口中有 8,000 个阳性调用，则 FPR 为 0.8%，特异度为 99.2%，每百万次调用有 8,000 次误报。此数值只是公式示例。

FPR 的分母是所有实际阴性样本，不是所有预测阳性样本。`FP / (TP + FP)` 属于另一种指标，不能作为这里的 FPR。定义参见 [scikit-learn 官方模型评估文档](https://scikit-learn.org/stable/modules/model_evaluation.html)。

### 6.2 site 级假阳性率

与原项目保持一致，先对同一物理位点汇总：

```text
n_j = 位点 j 成功预测的独立 read 数
s_j = sum(p_ij) / n_j

FP_site = 满足 s_j >= t_site 的阴性位点数
TN_site = 满足 s_j <  t_site 的阴性位点数
N_site  = FP_site + TN_site

FPR_site = FP_site / N_site
```

默认 `t_site = 0.5`。site 级分母中每个位点计一次，不因 read 更多而获得更多权重。无覆盖、被质量规则排除或没有成功预测的位点不计为 TN，应单独报告为不可评估。

主表包含 `n_j >= 1` 的可评估位点。另列 `n_j >= 3` 和 `n_j >= 5` 的结果，但这两者的分母更小、集合不同，不能代替全部低深度位点结果。

### 6.3 背景阳性调用比例：与 site FPR 分开报告

对每个位点另计算：

```text
q_j = count(p_ij >= t_read) / n_j
```

`q_j` 是该位点被调用为甲基化的 read 比例。在 WGA 阴性对照中，它表示背景阳性调用比例。建议报告位点等权的 `q_j` 均值、P50、P95，以及高尾部位点数。

需要区分三个量：

- `p_ij`：单个 read–位点的模型分数 / 概率输出。
- `s_j = mean(p_ij)`：原项目用于 site 分类的平均预测概率。
- `q_j = mean(p_ij >= t_read)`：阈值调用后的阳性 read 比例。

**`s_j` 不能未经校准就直接解释为真实甲基化频率。** 例如 5 条 read 的概率都为 0.3，`s_j = 0.3`，但在阈值 0.5 下 `q_j = 0`。

低覆盖下 `q_j` 的步长很粗：2 条 read 时为 50%，5 条 read 时为 20%。因此“某位点出现 20% 阳性调用”不一定表示稳定的低水平甲基化，必须结合覆盖度解释。

全阴性条件下可以计算 `q_j` 相对真值 0 的 MAE / RMSE 作为背景误差描述，但它们依赖调用阈值，不能替代独立混合样本的甲基化比例定量评估。

### 6.4 推荐报告的指标

| 指标 | 用途 | 优先级 |
| --- | --- | --- |
| `N_read`、`FP_read`、`FPR_read`、95% CI | 单碱基窗口误报规模与比例 | 主指标 |
| `N_site`、`FP_site`、`FPR_site`、95% CI | 汇总后有多少阴性位点被错判 | 主指标 |
| read/site specificity | FPR 的补充表达 | 必报，注意与 FPR 冗余 |
| 有效覆盖度分布、可评估位点比例 | 说明结果覆盖哪些数据 | 必报 |
| 每个覆盖度分层的 site FPR | 分析低深度行为 | 必报 |
| `q_j` 分布 | 描述背景阳性调用比例 | 推荐 |
| `p_i`、`s_j` 的 P50 / P95 / P99 和分布 | 查看整体背景和高分尾部 | 推荐 |
| BCE / Brier score | 描述阴性分数偏离 0 的程度 | 辅助，不作为主要模型选择标准 |

若报告 Brier score，在全阴性数据上即 `mean(p_i²)`；若报告 BCE，则为 `mean(-log(1-p_i))`，实现时需稳定处理边界概率。它们不是假阳性率。

### 6.5 不作为 Test1 结论的指标

| 指标 | 原因 |
| --- | --- |
| AUROC | 缺少阳性类，无法比较正负排序；记为 N/A / null |
| AUPRC / AP | 没有阳性可检出，库函数返回的退化数值不构成有效检出性能 |
| 阳性 recall / sensitivity | 没有真实阳性，分母为零 |
| Precision、F1、MCC | 在本数据上为零或退化，不能区分有用的分类能力；无阳性预测时 precision 还会出现零分母 |
| Accuracy | 全阴性时等于 specificity，不必重复作为主要成绩 |
| 与真值甲基化比例的 Pearson / Spearman 相关 | 真值恒为 0，没有变异，相关性不成立 |

不要将 N/A 填成 0 或 1 来参与模型均值排名。

## 7. 低深度分析、阈值和公平比较

### 7.1 覆盖度分层

至少输出以下 site 分层：

```text
n_reads = 1
n_reads = 2
n_reads = 3
n_reads = 4
5 <= n_reads <= 9
n_reads >= 10
```

每一层同时列出位点数和 FPR。若某层没有位点，报告样本数 0、FPR 为 N/A，不能报告为零误报率。

如比较“相同位点在 1、2、3、5 条 read 时的表现”，应先固定具有足够覆盖度的共同位点，再按 read 身份下采样，并使用相同抽样列表比较模型。此实验只代表高覆盖子集，不能替代总体 5× 数据的主结果。不同模型在同一位点采用不同 read 子集，会引入额外偏差。

纳米孔甲基化基准研究会在固定候选位点或控制混合比例的条件下分析覆盖度影响，见 [Systematic benchmarking，Nature Communications，2021](https://www.nature.com/articles/s41467-021-23778-6)。本方案据此将总体表现、覆盖度分层和固定集合下采样分开。

### 7.2 阈值预先固定

主结果使用当前训练协议的 `t_read = 0.5` 和 `t_site = 0.5`。如果采用其他阈值，需在原独立 valid 集或专用校准集上确定后冻结。

可以附加展示 `FPR(t)` 随阈值变化的曲线或预先约定的多个阈值，但不得在本次 WGA 测试上选择“FPR 最低”的阈值，再将其当作无偏测试成绩。

**FPR 更低不自动表示模型整体更好。** 永远输出阴性的模型也能得到 FPR=0。因此模型比较还应并列原独立阳性/混合测试集的检出率，或在独立验证集上将各模型阈值设为相同 sensitivity 后，再比较 WGA FPR。

例如将“验证集 sensitivity 95%”设为共同工作点时，各模型阈值可以不同，但工作点、read/site 层级和校准数据必须预先约定。WGA 本身不能提供 sensitivity。

### 7.3 使用共同样本比较主干

不同主干的超长输入和截断处理可能使成功提取的样本略有差异。建议同时保留：

1. **每个模型的完整可评估集合**：反映实际处理覆盖范围。
2. **模型共同的 read–位点交集**：用于直接比较 FPR。

read–位点的匹配键至少含 `library_id + 原始 read_id + chrom + center_genome_pos0 + strand`。不能按文件行号直接连接不同模型的预测。

site 级公平比较不仅需要相同位点，还应尽量使用相同的成功预测 read 集合；否则平均后的差异可能部分来自覆盖度不同。应同时报告各模型的提取失败数和共同集合的覆盖度。

### 7.4 置信区间和重复

输出每个 FPR 的分子、分母，并给出 95% 置信区间。read 窗口并非完全独立：同一 read 可覆盖多个位点，同一位点也可贡献多个 read 窗口。

建议主分析以预先定义的基因组区块进行配对 bootstrap：例如固定 1 Mb 区块、重复 1000 次并记录随机种子；模型比较时对相同区块配对重采样，计算 FPR 差值区间。区块尺度是本方案的分析选择，不是通用标准，应结合实际相关范围检查，必要时做尺度敏感性分析。仅有少量区块时，不宜过度解读区间。

也可附加 Wilson 区间作为简单比例的参考，但应注明其将观测视为独立的近似前提。即便观察到零次误报，也需报告样本数和区间，不能表述为“绝无假阳性”。

零误报时，直接对全零观测做普通百分位 bootstrap 会得到退化的 `[0, 0]`，不能把它作为真实 FPR 必为零的证据。此时可另报在独立观测近似下的单侧二项上界，并明确相关性限制；独立试验数为 N、零误报时，95% 单侧上界为 `1 - 0.05^(1/N)`，N 较大时约为 `3/N`。窗口有相关性时不能直接把所有窗口数当作 N 来宣称严格的误报上界。

有独立 WGA 制备或测序重复时，分别报告每个文库结果，再汇总；不能将不同训练 seed 当作独立文库重复来扩大样本量。

## 8. 结果文件和汇总表

### 8.1 每个 checkpoint 的结果

```text
results/<version>/<strategy>_seed<N>/
├── run_manifest.json
├── metrics.json
├── read_predictions.tsv.gz
├── site_predictions.tsv.gz
├── coverage_metrics.tsv
├── threshold_metrics.tsv
└── false_positive_sites.tsv.gz
```

推荐字段：

| 文件 | 关键内容 |
| --- | --- |
| `run_manifest.json` | 文库、参考版本、checkpoint 路径和校验标识、训练 run、提取协议、评测参数、代码版本、样本交集策略 |
| `metrics.json` | read/site 的 FP、TN、N、FPR、specificity、区间和不可计算指标说明 |
| `read_predictions.tsv.gz` | 原始 read ID、文库、染色体、坐标、链、标签、概率、阈值调用、用于分层的上下文 |
| `site_predictions.tsv.gz` | 位点键、标签、`n_reads`、`prob_mean`、阳性 read 数、`q_j`、site 调用 |
| `coverage_metrics.tsv` | 每个覆盖度层级的位点数、FP、FPR、区间 |
| `threshold_metrics.tsv` | 预设阈值对应的 FPR，read 和 site 分开 |
| `false_positive_sites.tsv.gz` | site 阈值下的所有误报位点及上下文，便于追查 |

### 8.2 主比较表模板

`read_FPR` / `site_FPR` 建议统一存为 0–1 的数值，展示时转换为百分数；表头必须注明单位。

| 主干 | 分类头 | Seed | 阈值来源 | 阴性窗口数 | FP 窗口数 | read FPR % [95% CI] | 可评估位点数 | FP 位点数 | site FPR % [95% CI] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| V011 | 待填 | 1 | 固定 0.5 / 独立校准 | 待评测 | 待评测 | 待评测 | 待评测 | 待评测 | 待评测 |
| V543 | 待填 | 1 | 固定 0.5 / 独立校准 | 待评测 | 待评测 | 待评测 | 待评测 | 待评测 | 待评测 |
| V610 | 待填 | 1 | 固定 0.5 / 独立校准 | 待评测 | 待评测 | 待评测 | 待评测 | 待评测 | 待评测 |

另输出跨共同 seeds 的均值和样本标准差，保留逐 seed 明细。若将多个 seed 的概率平均，属于新增 ensemble，应单独命名并预先确定，不能与单模型结果混用。

### 8.3 必要的配套表

- 样本流转表：原始候选 → 正链 → MAPQ 通过 → 窗口通过 → 去重 → 特征提取成功 → 最终可评估。
- 覆盖度分层表：`1 / 2 / 3 / 4 / 5–9 / >=10`。
- 质量分层表：MAPQ、信号长度或其他预先约定的质量条件。
- 上下文分层表：与任务一致的 CpG / 非 CpG 或中心 k-mer；稀疏分组注明样本数。
- 独立检出率表：来自原混合/阳性测试集，与 WGA FPR 并列解读。

推荐图形为阴性分数分布、FPR–阈值曲线、FPR–覆盖度图。全阴性数据不画 ROC 作为主要性能图。本文没有生成实际图形或结果文件。

## 9. Test1 功能实现状态

| 功能 | 当前状态 | Test1 所需工作 |
| --- | --- | --- |
| Step2 格式窗口 | 已实现 | Test1 Step1；标签来自 WGA，保留独立来源字段 |
| 低深度窗口保留 | 已实现 | 去重后的最小覆盖度默认 1 |
| V011 / V610 只提取 test | 已实现 | Test1 Step2 调用原主干提取器 |
| V543 只提取 test | 已实现 | Test1 Step2 复用原单 split 函数 |
| 单 split 中心裁剪 | 已实现 | Test1 Step3；推理器也可自动中心裁剪 |
| 已训练分类头独立推理 | 已实现 | Test1 Step4，加载 best.pt 和保存的配置 |
| FPR / specificity | 已实现 | 显式分子分母、全阴性 null、区块区间 |
| 覆盖度分层和共同样本比较 | 已实现 | Test1 Step4 / Step5，按原始 read 和位点匹配 |

代码、脚本、参数解释和测试方式见 [workflow/README.md](workflow/README.md)。小规模可运行测试不能替代全量实验，也不构成模型假阳性水平的评估结果。

## 10. 建议的执行顺序和验收条件

1. **冻结实验设置。** 确定 WGA 文库、参考版本、chr16 / 全基因组范围、目标 motif、正链策略、MAPQ、共同 seeds 和阈值来源。
2. **准备全阴性窗口。** 复用 Step2 输出格式，全部 `label=0`，不按甲基化预测筛选，不执行类别平衡，保留至少 1 条有效 read 的位点。
3. **完成窗口审计。** 核对真实信号方向、15 nt 长度、中心 C、span、去重、split 和标签；记录样本流转。
4. **核对独立评测入口。** 使用已实现的五步工作流，确认 checkpoint-only 推理、中心裁剪和融合头碱基输入协议。
5. **小规模联调。** 使用独立 smoke 目录提取少量特征并推理；验证 checkpoint 成功加载、无参数更新、样本和元数据对齐、阴性指标正确。
6. **提取正式特征。** 每个主干使用相同 WGA 语料和对应原协议，记录失败窗口及保留比例；多个分类头复用特征。
7. **运行逐模型评测。** 使用已冻结的 checkpoint、阈值和共同 seeds，保存逐窗口及逐位点预测。
8. **做公平比较。** 在完整集合和共同 read–位点集合上分别计算结果，输出覆盖度分层、置信区间和原独立测试集 sensitivity。
9. **追查误报。** 查看误报是否集中于特定 k-mer、低 MAPQ、低覆盖或异常信号窗口；这些是误差分析，不反过来删除主结果中的误报。

评测器联调还应包含一次与原训练评测的复现：将同一 checkpoint 用于原 test 特征，确认概率计算、中心裁剪和 site 分组能够在合理数值容差内复现原 `metrics.json`。这一步用于验证新推理入口；复现通过后再运行独立 WGA 测试。

正式结果至少应满足：

- 预测输入全部来自已记录的 WGA 阴性对照，标签全 0，测试 read 未参与分类头训练。
- 候选位点和阈值在查看本次模型结果之前固定。
- 5× 总体深度与最终正链实际覆盖度分别报告。
- read/site 指标分开，FPR 分子分母可复算；不可评估位点不计为 TN。
- 特征协议与训练一致，特别是 15 nt 上下文和中心 5 个输出位置。
- 所有比较模型使用可核对的共同样本；额外保留各自完整集合结果。
- 没有以全阴性 AUROC、F1 或相关系数作为性能结论。
- 说明低 FPR 对应的独立检出率，避免将一味预测阴性误认为模型更好。

## 11. 参考依据

1. **Rockfish: A transformer-based model for accurate 5-methylcytosine prediction from nanopore sequencing.** Nature Communications, 2024. [论文与 Methods](https://www.nature.com/articles/s41467-024-49847-0)。支持 WGA 阴性对照、独立 read/site 评估及链特异性分析的设计背景。
2. **Systematic benchmarking of tools for CpG methylation detection from nanopore sequencing.** Nature Communications, 2021. [论文](https://www.nature.com/articles/s41467-021-23778-6)。采用阴阳性对照和控制混合，分析个体 read、位点频率、覆盖度和阈值的影响。
3. **scikit-learn：Metrics and scoring.** [官方文档](https://scikit-learn.org/stable/modules/model_evaluation.html)。用于核对 FPR、specificity、precision、recall 和 ROC 的定义。

本方案中的主分析覆盖度门槛、分层方式、固定阈值、输出文件和 bootstrap 设置，是结合本仓库与约 5× WGA 条件提出的具体实验设计，不是上述论文统一规定的标准。
