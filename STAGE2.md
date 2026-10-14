# 第二阶段：扩大训练覆盖，按更新步数验证

保持 **冻结 encoder + 单层线性 CTC head**，每 2 秒使用累计音频，回滚策略不变。本阶段检验独立验证集上的泛化，以及 CTC 相对旧草稿的增量价值。

## 为什么换一轮实验

设备日志中，27 个训练事件经过 5000 次更新后，TER 从 474.0% 降到 13.72%，完整序列正确 10/27，平均正确前缀 5.78 token。这说明该配置可以学习这些缓存特征，但尚未完全拟合，更不能证明泛化或加速。

原 pilot 的 828 个训练事件、8 卡、每卡 batch 8，每轮只有 13 次更新；15 轮共 195 次，所选 epoch 2 只有 26 次更新。同时 dev token OOV 达 38.72%。第二阶段同时解决训练预算不透明和训练词汇覆盖不足的问题。

## 1. 准备扩大版音频包

更新仓库，在联网机器执行（仅依赖 Python 标准库）：

```bash
git pull --ff-only
python prepare_aishell.py build --download \
  --recipe aishell_expanded_recipe.json \
  --cache-dir data/aishell-cache \
  --output-dir data/aishell-ctc-expanded \
  --zip data/aishell-ctc-expanded.zip
```

- 固定同一 AISHELL 镜像 revision，所有源文件按长度和 SHA256 校验。
- 训练扩大到 **64 位说话人**，保留原训练 8 位，取每位所有符合 6–30 秒条件的录音。
- **dev/test 仍是原来的各 64 条录音**。配方嵌入原选择规则，先还原原 holdout，再排除扩大训练集与 holdout 重复的转写、音频哈希及说话人。
- 源文件总量约 **2.53 GB**（2,533,381,795 字节），已有缓存可复用。这是源压缩包大小，解包空间另计。
- 实际训练录音数、时长和 chunk 数以生成的 `stats.json` 为准；本次交付的是可复现配方，未预先生成扩大版音频或 teacher 标签。
- 输出目录与 ZIP 必须不存在。保留旧包、旧轨迹和旧实验目录。

将扩大包搬入内网、解压到固定位置，例如仓库下 `data/aishell-ctc-expanded`，然后校验：

```bash
python prepare_aishell.py verify data/aishell-ctc-expanded
```

这是镜像内自定义划分，仍不是 AISHELL 官方 benchmark；朗读短句也不能代替业务域长会话。

## 2. 采集 train/dev 的真实 token 轨迹

在已能运行 teacher 的 Qwen/vLLM 环境执行，模型路径替换为内网路径：

```bash
mkdir -p data/ctc-traces-expanded
for split in train dev; do
  python collect_teacher.py \
    --model /models/Qwen3-ASR-1.7B \
    --manifest data/aishell-ctc-expanded/${split}_full.jsonl \
    --language Chinese --chunk-seconds 2.0 \
    --unfixed-chunk-num 2 --unfixed-token-num 5 \
    --events-out data/ctc-traces-expanded/${split}_events.jsonl \
    --teacher-out data/ctc-traces-expanded/${split}_teacher.jsonl
done
```

音频每次从开头累计到当前，训练目标仍是当前累计音频的完整 teacher 内容 token 序列。不要把训练目标改成上一轮回滚 token，也不要用整句人工转写监督尚未到达的音频。

dev 音频内容保持原样，但新目录会改变轨迹中的绝对音频路径；上面命令重新采集对应路径的 dev 轨迹，避免新 events 与旧 teacher 错配。目标模型、tokenizer、teacher 推理设置及软件版本应保持一致并记录，不能把模型或版本变化混入 CTC 对比。输出文件须为新文件；失败后的部分轨迹不能作为完整数据使用。

暂不采集或查看新 test 结果。旧 test 已用于前一阶段报告，后续确认性业务评测还需新的独立留出集。

## 3. 在编码前检查词汇覆盖

```bash
python audit_coverage.py \
  --train data/ctc-traces-expanded/train_events.jsonl \
  --dev data/ctc-traces-expanded/dev_events.jsonl \
  --max-dev-oov 0.05 \
  --output data/ctc-traces-expanded/coverage.json
```

这个审计不加载模型，词表只从 **train 的 recorded token IDs** 构建。它报告 token OOV、完整序列可表示率、非空序列首 token 可表示率及前 8 个 token 的连续词表覆盖长度，并检查重复事件和 train/dev 同路径音频。

`5%` 是本轮人为设定的筛查门槛，不是理论阈值，也不保证 CTC 可用。OOV 超标时退出码为 2，仍保存报告；训练入口也可在加载 encoder 前拦截。64 位说话人能否达到门槛，须以实际采集结果为准。超标则先增加独立训练语料或检查标签来源，**不要把 dev/test token 加入训练词表**。覆盖指标只说明能否表示，不代表预测准确或可被 teacher 接受。

## 4. 8 卡训练，显式预算更新步数

在 CTC 训练环境中运行。新实验从头初始化线性 head；不加载 27 个事件的过拟合 head。

```bash
export PYTHON_EXEC="$(python -c 'import sys; print(sys.executable)')"
python -m torch.distributed.run --standalone --nproc_per_node=8 train.py \
  --streaming --chunk-seconds 2.0 --backend qwen \
  --target-token-source recorded --selection-metric hybrid --verify-budget 8 \
  --model /models/Qwen3-ASR-1.7B \
  --manifest data/ctc-traces-expanded/train_events.jsonl \
  --dev-manifest data/ctc-traces-expanded/dev_events.jsonl \
  --dev-teacher data/ctc-traces-expanded/dev_teacher.jsonl \
  --output data/ctc-runs/aishell-expanded-001 --device npu \
  --max-steps 10000 --eval-every-steps 250 --train-eval-samples 128 \
  --max-dev-oov 0.05 --batch-size 8 --learning-rate 3e-4
```

- `--max-steps` 覆盖 `--epochs`，指每个 rank 同步执行的 optimizer 更新数，**不乘卡数**。10000 步是有界的起始实验预算，不是收敛保证；8 卡常规全局 batch 为 64，epoch 末批次可能更小。
- `--eval-every-steps 250` 按更新步数评估，最后不足 250 步也会评估。未提供该参数仍逐 epoch 评估。
- `run.json` 保存计划步数、每 epoch 步数、词汇覆盖和固定训练评估子集索引。
- `history.jsonl` / `best_metrics.json` 增加 `global_step`；`train_loss` 是自上次评估以来的更新平均 loss。`train_subset` 是固定随机抽取的 128 个训练事件上的诊断指标，不参与选模。
- `best_head.pt` 仍只按 **dev hybrid 增量接受量 → hybrid 接受量 → TER** 排序选择。没有 optimizer 恢复功能；新运行必须使用新目录。
- 编码一次并缓存，随后只训练 head。8 卡实际收益须测量；CPU CTCLoss、缓存 I/O 和主 rank 验证都可能影响耗时。

建议先用新的临时输出目录跑 `--max-steps 5 --eval-every-steps 2` 检查本机多卡链路；它仍需完成全部特征缓存。正式运行用另一个新目录。

## 5. 如何判断下一步

看同一评估点的以下字段，不只看 loss：

| 指标 | 用途 |
| --- | --- |
| `global_step`、`train_subset.token_error_rate` | 判断当前优化预算下是否学到训练集；抽样指标不是全训练集统计 |
| 顶层 `token_error_rate`、`char_error_rate`、`dev_oov_token_rate` | 看独立 dev 泛化与词汇覆盖 |
| `rollback_comparison.after_first.mean_delta_accepted_at_8` | 相对原回滚草稿的增量接受量，主要路线指标 |
| `rollback_comparison.after_first.extension_proposed_rate` | 是否真正追加了 CTC 草稿 |
| `rollback_comparison.by_trigger.later` | 第 3 轮起稳定前缀匹配、回滚尾部及 CTC 扩展效果 |

训练子集 TER 降而 dev 不降，优先检查泛化/覆盖；两者都不降则回到特征、标签及优化诊断。若 TER 明显改善但 CTC 始终无法扩展，下一步才单独检验拼接门槛/对齐策略，保持这一轮结果作为基线。

固定模型与参数后，再对相同设置的 test 轨迹运行 `stream_draft.py` 和 `score_teacher.py --require-rollback --verify-budget 8`（见 [DATASET.md](DATASET.md)）。本阶段未实现线上 target 并行验证器，因此离线增量接受量不能换算为端到端加速。进入线上实验还需测真实 2 秒到达节奏下的延迟、P95、额外 head 成本和输出一致性。
