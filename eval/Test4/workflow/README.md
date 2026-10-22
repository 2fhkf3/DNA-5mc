# Test4 workflow：连续甲基化频率与独立 WGBS 的相关性

本目录实现 [Test4 实验方案](../README.md)。流程参考 Test3，但输入是独立 WGBS 连续 beta 和完整 chr16 read 信号；不复用只保留 0%/100%、经过类别平衡的 Test3 子集。

**只有 `--run` 才正式执行；默认无参数运行临时合成测试。** 当前尚未启动全量 1.969 TiB 信号处理，也未产生真实 Test4 相关性结果。

## 1. 步骤与入口

| 步骤 | Python / Shell | 作用 |
| --- | --- | --- |
| Step0 | `step0_audit_inputs.py/.sh` | 审计 WGBS、参考 FASTA、原分类头与特征协议，冻结 90 个运行及输入/权重内容哈希 |
| Step1 | `step1_build_windows.py/.sh` | 单次扫描原 JSONL，构建无逐 read 标签的 15nt 窗口分片，原始 read–位点去重，断点续跑 |
| Step2 | `step2_predict_reads.py/.sh` | 每主干分片提取一次特征，中心 5×768 float16 特征供所有分类头/seed 复用；保存概率与有效掩码 |
| Step3 | `step3_aggregate_sites.py/.sh` | 所有主干共同 read 交集、共同覆盖度筛选、WGBS 配对、硬调用频率和概率均值 |
| Step4 | `step4_evaluate_correlation.py/.sh` | Pearson/Spearman、MAE/RMSE/Bias/斜率/截距、固定分层及配对空间区块 bootstrap |
| Step5 | `step5_compare.py/.sh` | 逐 seed 明细、均值/样本方差、模型间差异、HTML/CSV 和可选 PNG/PDF |
| 公共模块 | `common.py`、`provenance.py`、`inference.py`、`common.sh` | 配置、校验、历史协议解析和无标签推理适配 |
| 测试 | `test_workflow.py` | `/tmp` 中的合成数据全流程与 CPU 分类头测试；没有生产用的模拟预测后端 |

## 2. 先测试与查看计划

```bash
# 默认也是--test；临时数据自动清理，不扫描真实大文件、不使用GPU。
bash /home/shj/ZJ-lab/Modification/DNA-5mC/eval/Test4/workflow/run_all.sh --test

# 只读发现真实已完成的模型/分类头/seed，显示六步计划及缺失的输入依据。
bash /home/shj/ZJ-lab/Modification/DNA-5mC/eval/Test4/workflow/run_all.sh --dry-run
```

依赖 NumPy、PyYAML、PyTorch；模型提取使用原项目的 Transformers、safetensors、SciPy、einops 等已有依赖。测试用 SciPy 核对统计量，图形测试还用 PyMuPDF；图形导出用 matplotlib。`orjson` 为大 JSONL 解析的可选加速，无此包时使用标准库。无需 pyarrow；中间数据采用 gzip JSONL、SQLite、NPY/NPZ、TSV，与 Test3 的轻量文件风格一致。

解释器由项目 `workflow/paths.sh` 选择，优先已激活的非 base 环境，否则使用已有 PoreDLM；可显式设置 `PYTHON_BIN`。Matplotlib 缓存放在可写 `/tmp`，不写用户只读配置目录。

## 3. 正式运行前要填写的配置

所有参数位于 [config.yaml](config.yaml)。当前六模型、五分类头、全部已完成 seed 的真实只读发现为 **90 个运行**。

以下信息未获核实，配置中保留为 pending；Step0 会明确拒绝将它们当作正式已确认信息：

- `reference.build`：WGBS 与原 BAM 的实际参考基因组版本。
- `reference.fasta`：匹配版本且包含 `chr16` 的 FASTA，可为 `.gz`；本流程不猜测 hg19/hg38，也不自动 liftover。
- `reference.coordinate_evidence`：该 11 列 BED 为 0-based 分链位点的转换依据。当前仅支持已验证的 `bed0 / strand_specific`；合链参考、Bismark 原始六列等需先按明确规则转换。
- `preprocessing.status/evidence`：核对上游日志或原训练同 read/位点信号后，使用 `training_matched` 并记录证据。不能只根据文件名或均值/方差认定已正确 Apple 化。

WGBS 来源已按用户确认记为 `independent_wgbs`，仍应补充原始样本、文库及导出流程。Step0 本身不会重新归一化信号，也不会用“相关性最高”的偏移去猜坐标。

填写并检查计划后，正式执行：

```bash
bash /home/shj/ZJ-lab/Modification/DNA-5mC/eval/Test4/workflow/run_all.sh --run
```

默认输出到 `eval/Test4/output/hg002_chr16_wgbs_correlation_v1/`。全部中间数据也放在同一个协议目录，以便整体迁移和审计；可以将整个协议根目录覆盖到容量足够的 NAS：

```bash
PROTOCOL_DIR=/absolute/path/new_test4_protocol \
bash /home/shj/ZJ-lab/Modification/DNA-5mC/eval/Test4/workflow/run_all.sh --run
```

`CONFIG=/absolute/path/config.yaml` 或 `--config` 选择其他配置。相同协议的每次调用必须使用同一配置和输出根目录。输出路径建议在配置中直接固定；CLI 覆盖后也会进入冻结配置。

## 4. 独立步骤与续跑

```bash
# 单步正式运行。
bash eval/Test4/workflow/step0_audit_inputs.sh
bash eval/Test4/workflow/step1_build_windows.sh

# 两张 GPU 分别处理不同主干，保留冻结配置中的 cuda:0，用 CUDA_VISIBLE_DEVICES 映射物理卡。
CUDA_VISIBLE_DEVICES=0 VERSION=V554 bash eval/Test4/workflow/step2_predict_reads.sh
CUDA_VISIBLE_DEVICES=1 VERSION=V610 bash eval/Test4/workflow/step2_predict_reads.sh

# 所有启用主干的预测都完成后才能聚合共同位点。
bash eval/Test4/workflow/step3_aggregate_sites.sh
VERSION=V554 bash eval/Test4/workflow/step4_evaluate_correlation.sh
bash eval/Test4/workflow/step5_compare.sh

# 从某个阶段继续；已完成且通过校验的结果会复用。
bash eval/Test4/workflow/run_all.sh --run --from-step 2
```

GPU 映射会影响实际物理设备，但不改变逻辑提取参数；如果需要严格追踪跨 GPU 数值差异，使用独立协议。上例中每个主干拥有独立锁；同一主干不能并发启动两份提取。

Step1 使用 `.windows.work/` 中的 SQLite 状态与已提交窗口分片，保存**完整 read 边界**的原文件字节 offset。中断后重新执行相同步骤即可继续，已有 committed 分片先校验哈希；只清理该工作目录中未提交的自有孤立分片。SQLite 事务与扫描状态在同一次提交中更新，避免再次计数已提交窗口。

Step2 以主干/窗口分片为单位原子发布，已完成分片核对配置、源窗口及概率文件哈希后复用。未完成分片重算，不重新解析原 2 TB 文件。Step3 在临时目录完成后整体发布，中断则重新聚合，但不重新提取主干。Step4 以 run 为单位完成，Step5 整体发布。任何冻结输入、模型代码或配置改变都拒绝复用，需新建协议；没有 `--force` 覆盖正式结果。

### 限量联调

使用独立临时 YAML/输出根目录，将 `allow_partial_results: true`，`scan.max_records` 设为正整数。也可对所有单步传相同 `MAX_RECORDS`：

```bash
CONFIG=/tmp/test4_smoke_config.yaml PROTOCOL_DIR=/tmp/test4_smoke MAX_RECORDS=100 \
bash eval/Test4/workflow/step0_audit_inputs.sh
# 后续各单步使用同样的 CONFIG、PROTOCOL_DIR、MAX_RECORDS。
```

前 N 条完整 read 不保证任何位点自然有 5 条共同 read。共同集合不足时 Step3 会报错；若仅做接口测试，可在独立临时配置降低覆盖门槛，但不能将其指标当作正式 ≥5× 实验。一键正式入口拒绝环境变量 `MAX_RECORDS`，避免误带前缀限制；显式临时配置内的限量会保留 partial 标记。

## 5. 实际产物结构

```text
<protocol_dir>/
├── protocol.yaml
├── audit/
│   ├── reference.sqlite             # 全部有效连续WGBS beta，不做0/100筛选
│   ├── inventory.json / run_inventory.tsv
│   ├── environment.json
│   └── manifest.json                # 模型/分类头/摘要/输入及代码指纹
├── windows/
│   ├── part_*.jsonl.gz              # 无label、无wgbs_beta的15nt信号窗口
│   ├── observations.sqlite          # 原始read–位点唯一身份与内容校验；原行号保存在窗口分片
│   ├── scan_summary.json
│   └── manifest.json
├── predictions/<版本>/
│   ├── part_*/probabilities.npz      # indices、valid、probabilities[N,该主干run数]
│   ├── part_*/manifest.json
│   └── manifest.json
├── common/
│   ├── fixed_sites.tsv / fixed_sites.npz
│   ├── excluded_sites.tsv
│   ├── common_indices_*.npz         # 模型共同有效观测，最终还需join fixed_sites过滤低coverage位点
│   ├── scores/<版本>/<运行>.npz     # 甲基化read整数计数、概率均值，与fixed_sites同序
│   ├── bootstrap_weights.npz        # 所有模型/seed/估计器共享的区块抽样权重
│   └── manifest.json
├── metrics/<版本>/<运行>/
│   ├── metrics.json
│   ├── stratified_metrics.tsv
│   ├── bootstrap_metrics.npz
│   └── manifest.json
└── summary/
    ├── correlation_by_seed.tsv
    ├── correlation_mean_variance.tsv
    ├── model_differences.tsv
    ├── data_funnel.json
    ├── compare/correlation_comparison.csv / .html
    ├── figures/<版本>/<运行>.png / .pdf
    ├── README.md
    └── manifest.json
```

`common/scores` 不为每个模型重复保存同一份 WGBS/坐标数组；关联 `fixed_sites` 即得到完整逐位点表。未保存巨大的全量 embedding，特征在分片内用 float16 模拟原训练缓存，再转 float32 输入分类头。各 run 的概率为 float32；位点频率汇总和统计使用 float64。

## 6. 关键统计与协议检查

- 原始 read–位点去重；相同键但窗口内容冲突时中止，不平均冲突样本。
- WGBS 11 列 BED 的百分比必须有限且在0～100，coverage为非负整数。相同位置重复记录只能完全一致；冲突不合并。参考不是C时中止检查坐标，参考为C但非CpG时单独统计排除。
- WGBS beta 不进入推理窗口，不给每条 read 虚构0/1标签，不按beta平衡样本。
- 所有主干完成后，对相同window ID取有效掩码交集，再重算不同read的共同覆盖度。主干推理过长窗口不静默截断，标记不可评测并进入排除统计。
- `beta_call = methylated_read_count / n_common_reads`，逐read `p>=0.5`计为甲基化；`beta_prob=mean(p)`；二者都报告，不按相关性挑选主指标。
- Pearson/Spearman、MAE/RMSE/Bias/斜率/截距对固定共同站点等权计算。空集合/常量的相关系数为NA；Spearman并列值用平均秩。
- 区块bootstrap先合并区块的充分统计量再重算全局Pearson/误差，绝不平均区块相关系数。区块数不足时CI留空；Spearman区块CI未计算并明确标记，不套用Pearson CI。
- 每个seed单独计算，再求均值和样本方差ddof=1。模型差值为B−A，空间抽样使用相同区块权重；不要求不同分类头使用同一seed编号，也不称为训练seed配对实验。
- PNG/PDF为每个run的两种估计器图；相同固定位点按同一确定性索引显示至多50,000点，相关性/误差始终使用全部位点。图上颜色范围固定，显示y=x和完整N。

## 7. 历史特征协议的实际适配

`provenance.py` 从各版本现有 `feature_summary.json` 及嵌套来源记录恢复参数，避免照搬统一入口的新默认。当前只读核对包括：

- V011：ODE steps=1、start_t=0.97、CFG=0.5，历史batch=64、CPU逐碱基池化。
- V013：旧分类头为steps=1、start_t=0.97、CFG=0.5；不是新统一YAML的2步/0.98。
- V543：最后层、历史batch=64、CPU池化。
- V548：最后层、历史batch=8、模型设备池化。
- V554：最后层、历史batch=8、CPU池化。
- V610：最后层、历史batch=64、Codec layer=1、**无BOS/EOS**、CPU池化；不采用近期独立GPT脚本的新默认。

分类头检查原配置与checkpoint的strategy、seed、fingerprint、模型结构、维度、训练配置及best epoch，严格加载state_dict。其冻结特征协议仍需用历史重叠窗口做数值回归；当前完整合成测试不等价于证明所有真实GPU/旧缓存逐字节相同。

大原信号只冻结大小/mtime，并在实际读取时记录带offset的SHA256分块链；续跑不为恢复单一SHA256状态而重读前面的TB数据。该链记录实际已扫描内容，不等价于预先验证过整个来源文件的历史SHA256。权重、源码和小摘要文件采用内容SHA256。

## 8. 验证范围

23项CPU测试已通过（2026-10-20，79.647秒）：与SciPy交叉核对相关性、并列分数、常量/空值/非法beta、区块充分统计与显式重采样等价、真实PyTorch分类头、六步串联、连续beta保留、跨模型不同可评测read、覆盖度筛选、重复与冲突、窗口和预测分片中断续跑、哈希/配置损坏拒绝、partial标记、区块不足及PNG/PDF导出。另覆盖非CpG胞嘧啶排除、概率恰为0.5时计为甲基化，以及未知配置/不安全前缀配置拒绝。示例图已检查排版。

合成主干只存在于测试mock，生产入口只调用真实BERT/ELF/Flow/GPT实现。实际六模型/90运行已完成只读计划检查，不代表正式WGBS相关性已计算。输入基因组/预处理pending项未补齐时，正式Step0会报出具体缺项。

另外，从真实信号首条read取两个无标签15nt窗口，六个主干均已完成CPU接口检查：V011/V013/V543/V548/V554/V610均产生有限的 `[2,5,768]` float16特征，各自真实MLP seed1分类头严格加载后产生合法概率；覆盖CPU池化和V548的模型设备池化分支。这只是小规模接口检查，未验证真实GPU批次结果或历史缓存逐字节一致，也没有将这两个窗口作为正式WGBS相关性样本。
