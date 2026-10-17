# Test3 workflow：5 / 10 / 20 reads per site 鲁棒性

本目录实现 [Test3 方案](../README.md)：保留原 chr16 test 的正负两类，固定跨主干、分类头和训练 seed 的共同测试位点；每次抽样共享同一 read 排名，分别取前5、10、20条，用平均正类概率计算 site-level 性能。

**默认复用已保存预测，无需 GPU，也无需重新提取隐藏状态。** 5×/10×/20×指每个位点恰好使用的 read 数，不是整条染色体的平均测序深度。

## 1. 文件与顺序

| 步骤 | Python / Shell | 作用 |
| --- | --- | --- |
| Step1 | `step1_audit_predictions.py/.sh` | 冻结已完成 runs；校验原始完整 test 和概率；保存索引及概率向量 |
| Step2 | `step2_build_common_pool.py/.sh` | 求所有冻结 run 的共同观测，重算 coverage，固定≥20位点；可选一次性类别平衡 |
| Step3 | `step3_sample_site_reads.py/.sh` | 对每个位点、repeat确定性随机排名，无放回生成共享嵌套抽样 |
| Step4 | `step4_evaluate_coverage.py/.sh` | 每个run的50次×3深度位点指标、Full-common、上下文与稳定性 |
| Step5 | `step5_summarize.py/.sh` | 跨seed汇总、深度损失、模型差值、区块bootstrap及PNG/PDF曲线 |
| 测试 | `test_workflow.py` | `/tmp`下的合成小数据CPU集成测试，不评测正式全部数据 |

`config.yaml` 为统一配置；`common.py/.sh` 为公共工具；`run_all.sh` 为一键入口。各Shell和YAML内均有中文参数注释。

## 2. 先测试和检查计划

```bash
# 默认不加参数也运行测试，所有测试数据和结果位于临时目录。
bash /home/shj/ZJ-lab/Modification/DNA-5mC/eval/Test3/workflow/run_all.sh --test

# 只读显示五步计划，不生成正式数据或模型成绩。
bash /home/shj/ZJ-lab/Modification/DNA-5mC/eval/Test3/workflow/run_all.sh --dry-run
```

解释器沿用项目 `workflow/paths.sh`，优先使用已激活的非base环境，其次使用本机PoreDLM。可通过 `PYTHON_BIN` 指定。运行依赖为 NumPy、PyYAML、PyTorch、标准库SQLite；图形输出需matplotlib。自动测试额外使用scikit-learn交叉核对指标。`threadpoolctl` 可用时会按 `cpu_threads` 限制已加载的BLAS线程。

所有路径按脚本位置定位，可从任意工作目录调用。

## 3. 正式运行

先修改 [config.yaml](config.yaml)。默认模型为V011/V543/V554/V610，各模型下全部已完成分类头和seed；默认5/10/20、50次重复、共同coverage≥20。

```bash
bash /home/shj/ZJ-lab/Modification/DNA-5mC/eval/Test3/workflow/run_all.sh --run
```

或者依次执行：

```bash
bash /home/shj/ZJ-lab/Modification/DNA-5mC/eval/Test3/workflow/step1_audit_predictions.sh
bash /home/shj/ZJ-lab/Modification/DNA-5mC/eval/Test3/workflow/step2_build_common_pool.sh
bash /home/shj/ZJ-lab/Modification/DNA-5mC/eval/Test3/workflow/step3_sample_site_reads.sh
bash /home/shj/ZJ-lab/Modification/DNA-5mC/eval/Test3/workflow/step4_evaluate_coverage.sh
bash /home/shj/ZJ-lab/Modification/DNA-5mC/eval/Test3/workflow/step5_summarize.sh
```

使用另一份配置和独立实验目录：

```bash
CONFIG=/absolute/path/test3_config.yaml \
PROTOCOL_DIR=/absolute/path/test3_protocol \
bash /home/shj/ZJ-lab/Modification/DNA-5mC/eval/Test3/workflow/run_all.sh --run
```

也支持 `--config` 和 `--protocol-dir`。同一次实验所有步骤必须使用相同配置及目录。Step1将配置与run清单冻结；后续修改模型范围、平衡方式、抽样次数、阈值等会被拒绝，需建立新实验目录。

输出采用临时目录完成后发布，已有目录不覆盖。若前3步已经成功，可继续：

```bash
bash /home/shj/ZJ-lab/Modification/DNA-5mC/eval/Test3/workflow/run_all.sh --run --from-step 4
```

若Step4部分run已经完成，重复整步会拒绝覆盖。应只处理缺少的run，例如：

```bash
VERSION=V011 RUN_NAME=center_5mer_mlp_raw768_seed1 \
bash /home/shj/ZJ-lab/Modification/DNA-5mC/eval/Test3/workflow/step4_evaluate_coverage.sh
```

该过滤只选择冻结清单中的计算任务，固定共同位点和抽样不会改变。所有run完成后再执行Step5；不能把缺失的run静默省略。

## 4. 参数说明与正负平衡

| YAML参数 | 含义 |
| --- | --- |
| `enabled_versions` / `models` | 选择主干及各自原test、训练输出路径；不同主干特征不可混用 |
| `strategies` | 空列表=全部已有已完成策略；各主干必须有相同策略集合 |
| `training_seeds` | 空列表=全部已有seed，不强制1/2/3；不同主干允许不同seed列表 |
| `prediction_source` | `saved`导入旧预测；`inference`使用各run冻结权重对原缓存推理一次 |
| `minimum_common_reads` | 最终共同read池的最低数量，默认20，必须覆盖最大depth |
| `depths` | 严格递增、无重复的每位点read数，默认[5,10,20] |
| `repeats` / `sampling_seed` | 抽样次数和主种子；不是训练seed，也不是生物重复 |
| `site_balance` / `balance_seed` | `none`保留全部合格位点；`equal`一次性确定等量正负位点 |
| `site_threshold` | site平均概率≥阈值判阳性；默认0.5 |
| `site_chunk_size` | 抽样时每块位点数，控制内存，不改变所选read |
| `cpu_threads` | CPU算子/BLAS线程上限；Shell启动默认4，可用TEST3_CPU_THREADS限制导入前线程 |
| `device` / `batch_size` | 仅inference路线使用的设备和分类头批次；saved和Step2–5无需GPU |
| `save_all_site_scores` | 另存所有repeat/depth/site的float64分数，默认关闭以控制磁盘 |
| `save_site_stability` | 保存逐site分数SD、调用不稳定性、错误频率及翻转比例，默认开启 |
| `bootstrap.block_size_bp/repeats/seed` | 默认1Mb、1000次、42；次数0禁用 |
| `bootstrap.metrics` | BA/FPR/FNR区块区间，不用区块混淆计数近似AUROC/AP |
| `plots.enabled` | 输出AUROC/BA/FPR/FNR曲线PNG/PDF；带为抽样经验范围 |
| `allow_partial_results` | 默认false；仅隔离联调配置可true |

主实验保留全部共同coverage≥20的site，再按site等权评测；三种深度有完全相同的正负位点数。当前原test本来接近1:1，不需要通过复制read平衡。

需要严格1:1时，复制配置，设置 `site_balance: equal`，选择新的 `protocol_id/output_dir`。Step2按 `balance_seed` 从多数类合格位点中固定选取少数类数量的位点，之后所有深度、重复和模型共享同一批位点；不每次重新平衡，也不复制少数类read。

## 5. 各步具体校验

**Step1** 每个主干只建立一次完整观测索引。逐run核对所有预测的原始ID、chr16、坐标、正链、source=mod、test标签和seq/ref；核对二进制标签、数组大小、训练特征签名、checkpoint配置/epoch。全量saved导入还要复现原read/site的TP/TN/FP/FN。重复、缺行、额外行、非法概率、错位均中止。

同一主干所有run的观测顺序和身份经过逐行核对，所以只存一份观测数据库，各run另存float64概率向量，避免为每个分类头重复保存read ID。不同主干在Step2按物理观测键交集，不能用行号直接跨主干匹配。

旧TSV没有内嵌checkpoint哈希，来源只能做一致性核对；清单明确记 `historical_prediction_binding=consistency_checks_only`。`inference`则记录实际加载权重的SHA256，采用eval/inference_mode和float32分类头推理。CPU/GPU数值可能略有差异，复用saved能保持原结果概率。

**Step2** 检查相同位点label/ref一致、相同read–位点label/seq/ref一致，之后才计算共同read数并筛选。一个模型少一条read导致共同coverage19，该site退出三个深度。输出排除原因、最终正负数量与原始行号映射。

**Step3** 对 `[sampling_seed,repeat_id,library_id,chrom,position,strand,read_id]` 的规范JSON做SHA256排序，再取前20。模型名、策略、训练seed、depth、label和概率不参与排名，因此顺序/进程/批次变化不会改变抽样。5/10读取同一20条清单的前缀。

**Step4** float64汇总概率，q≥阈值判阳性，所有指标按site等权。Full-common使用相同固定site上的全部共同read。每次检查抽中索引没有重复、没有越出所属site。

**Step5** 首先检查全部冻结run、所有repeat/depth完整且位点及正负分母一致；bootstrap对所有模型/深度共享区块权重，合并区块混淆计数后重算全局指标。结果之间的共同集合、抽样和配置哈希必须相同。

## 6. 实际输出结构

```text
<output_dir>/
├── protocol.yaml                         # 冻结配置
├── manifests/
│   ├── run_inventory.tsv                 # 冻结的主干/策略/seed清单
│   ├── audit_manifest.json                # 来源、概率哈希、原指标核对、partial状态
│   └── environment.json                  # 实际Python/NumPy/Torch/PyYAML版本
├── audit/
│   ├── versions/<version>/
│   │   ├── observations.sqlite           # 原始观测键、标签、seq/ref及原行号
│   │   ├── site_indices.npy
│   │   ├── site_labels.npy
│   │   ├── sites.tsv
│   │   └── source_manifest.json
│   ├── runs/<version>/<run>/
│   │   ├── probabilities.npy             # 与本主干原始行号对齐的float64概率
│   │   └── run_manifest.json
│   └── manifest.json
├── common/
│   ├── fixed_sites.tsv / excluded_sites.tsv
│   ├── pool.sqlite                       # 统一obs_index/site_index及原始ID
│   ├── mapping_<version>.npy             # 统一观测索引→该主干原始行号
│   ├── offsets.npy / labels.npy / site_indices.npy
│   └── manifest.json
├── sampling/
│   ├── selected_indices.npy              # [repeats,N_sites,max_depth]，整数观测索引
│   └── manifest.json
├── results/<version>/<run>/
│   ├── metrics_by_repeat.tsv
│   ├── metrics_full_common.json
│   ├── block_confusion.npz                # [repeat,depth,block,4]，顺序TP/TN/FP/FN
│   ├── context_metrics.tsv
│   ├── stability_summary.json
│   ├── site_stability.tsv.gz              # 可关闭
│   ├── site_scores.npy                    # 可选，默认不保存
│   └── manifest.json
└── summary/
    ├── seed_summary.tsv / model_summary.tsv
    ├── depth_differences.tsv / model_differences.tsv
    ├── full_common_summary.tsv / stability_summary.tsv
    ├── context_metrics.tsv
    ├── bootstrap_model_draws.npz
    ├── reporting_notes.json
    ├── figures/                           # PNG/PDF
    └── manifest.json
```

使用`.npy/.npz/SQLite`保存大索引、矩阵和区块计数，减少重复文本；主要结果表仍为TSV。位置列 `position` 对应原 `center_genome_pos0`，都是0-based；ref5为保存ref的中心5-mer。

各阶段manifest保存配置及输出SHA256；后续步骤校验文件未被修改。输入运行中变化时终止，旧结果不覆盖。

## 7. 如何读结果

- `seed_summary.tsv`：一个训练seed、一个深度、一个指标一行。mean/SD/经验分位数跨抽样repeat。
- `model_summary.tsv`：mean先跨抽样再跨全部冻结训练seed；`training_seed_sd`是各seed抽样均值之间的样本SD；`sampling_sd_rms`是各seed内抽样SD的均方根；`sampling_q025/q975`是每次repeat先跨seed平均后的经验范围。
- `depth_differences.tsv`：低深度相对最大深度的损失。高值更好指标用Z(max)−Z(d)，错误率用E(d)−E(max)。`sd`是已训练seed平均后随抽样变化的SD；`training_seed_sd`单列训练随机性。
- `model_differences.tsv`：差值统一B−A，主差值使用各模型全部冻结seed。另列实际共有训练seed的差值，seed不相同不会导致整个策略被省略或强行配对。
- `full_common_summary.tsv / stability_summary.tsv`：全部共同read基线和逐site稳定性的跨训练seed汇总；SD跨训练seed，不增加生物重复数。
- `ci_low/ci_high`：仅BA/FPR/FNR及其差值有基因组区块区间。AUROC/AP的空间CI留空，它们仍有抽样范围；零事件、少区块、禁用或退化区间带状态说明。
- `context_metrics.tsv`：Step4的上下文FPR/FNR/BA为固定上下文位点上的抽样平均；Step5的SD跨训练seed。空类指标留空，不把NA填成0。

主性能包括AUROC、AP、BA、Acc、F1、MCC、FPR/FNR、Recall/Specificity、Brier/BCE。AP沿用average precision，概率恰等于0.5判阳性。

抽样SD、训练seed SD、区块CI均不代表独立生物重复。50次抽样不增加真实文库数；曲线稳定也可能始终预测错误，应同时检查绝对性能、低深度损失和逐site错误频率。

## 8. 测试与小前缀联调

自动测试构造具有19、20、21等不同read数的位点，覆盖5/10/20真实无放回抽样、不同训练seed、跨主干缺失read、严格平衡、旧预测与CPU推理一致、阈值等号、并列分数、区块计数汇总、缺失/非法/重复输入、配置冻结、文件损坏与拒绝覆盖，并交叉核对scikit-learn指标及图形导出。

2026-10-18 已通过18项CPU自动测试、Python/Bash语法及默认四主干60个run的只读计划检查。另对V011/V543/V554/V610各自MLP seed1的真实test前128行完成五步接口联调，共同集合为128个位点（73阴性、55阳性）；该前缀每site仅1条read，因此使用独立临时配置的1 read/site，不能视为真实数据5/10/20性能验证。5/10/20及coverage不足排除由有充足read的可控测试样本验证。所有测试均在临时目录，正式output中没有生成评测文件。

单步Step1支持：

```bash
CONFIG=/absolute/path/isolated_smoke_config.yaml \
PROTOCOL_DIR=/tmp/test3-prefix-smoke MAX_RECORDS=128 \
bash /home/shj/ZJ-lab/Modification/DNA-5mC/eval/Test3/workflow/step1_audit_predictions.sh
```

`MAX_RECORDS`表示原test输入前N行，**不是前N个位点的全部read**。小前缀可能每site仅1条read，因此不能期望它自然支持5/10/20；没有coverage足够的site时Step2会明确中止。只验证真实接口时可在独立配置把depths与minimum_common_reads暂设1，设置allow_partial_results=true；该结果不是正式Coverage成绩。

默认一键入口拒绝MAX_RECORDS，以免误将限量实验当正式评测。临时数据与正式协议不能混合。
