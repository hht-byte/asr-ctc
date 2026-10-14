# AISHELL-1 小规模流式训练数据

用途：检查冻结 Qwen3-ASR-1.7B 音频编码器、只训练 CTC 头能否产生有用的草稿。每 2 秒触发一次，输入累计音频 `[0, 2s]`、`[0, 4s]`、`[0, 6s]`……。

## 数据来源与划分

- 来源：[AISHELL-1 / OpenSLR 33](https://www.openslr.org/33/)，中文普通话朗读语音，16 kHz，Apache-2.0；数据包内附许可证和来源说明。
- 下载：[AISHELL/AISHELL-1 镜像](https://huggingface.co/datasets/AISHELL/AISHELL-1)，固定 revision `bbe295d530192a4cd41644b711c9aecd087df653`。
- 镜像仅提供部分说话人。本配方自定义训练/验证/测试划分，**不是 AISHELL 官方 benchmark 划分**；原始所属集合写在 `source_split`。
- 训练 256 条 / 8 位说话人，验证 64 条 / 2 位，测试 64 条 / 2 位；每位 32 条。集合间说话人互斥，并排除跨集合相同转写和相同 WAV 哈希。
- 筛选 6–30 秒原始录音，保证每条至少有三个完整触发；不裁剪、不拼接、不重采样。实际时长和触发数见 `stats.json`。
- 人工转写仅去除中文分词空格；不添加时间戳。同一录音的所有累计前缀必须留在同一集合。
- 说话人按可用长录音数量预选；录音按 `SHA256(seed:utterance_id)` 排序采样，seed 为 42。配方及源文件哈希见 `aishell_recipe.json`。

实际生成并校验的结果（详细统计见 `aishell_pilot_stats.json`）：

| 集合 | 录音 | 说话人 | 分钟 | 完整 2s 触发 | 第 3 次及之后触发 |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 256 | 8 | 30.86 | 828 | 316 |
| dev | 64 | 2 | 7.48 | 202 | 74 |
| test | 64 | 2 | 7.29 | 196 | 68 |
| 合计 | 384 | 12 | 45.62 | 1226 | 458 |

实际音频时长 6.00–11.89 秒，因此本轮只能检查短语句内的流式行为，不能检验长会话的累计编码开销。

## 1. 联网下载并打包

只需 **Python >= 3.10 标准库**，不需要 PyTorch、NPU 或模型。

```bash
python prepare_aishell.py build --download \
  --cache-dir ./data/aishell-cache \
  --output-dir ./data/aishell-ctc-pilot \
  --zip ./data/aishell-ctc-pilot.zip
```

源文件合计约 513 MB。缓存按文件续用，单个中断文件重新下载；长度和 SHA256 均须匹配。已有缓存时去掉 `--download` 即可离线重建；输出目录和 ZIP 必须尚不存在。脚本输出 ZIP SHA256。

```text
aishell-ctc-pilot/
  audio/{train,dev,test}/Sxxxx/*.wav
  train_full.jsonl
  dev_full.jsonl
  test_full.jsonl
  recipe.json
  stats.json
  DATASET_README.md
  LICENSE-AISHELL-1.txt
  SHA256SUMS
```

`audio` 为相对路径，可将整个目录或 ZIP 搬到内网。清单额外保留说话人、录音 ID、采样数、来源和音频哈希；现有训练入口读取所需字段。

## 2. 内网校验与 teacher 采集

假设解压后目录为 `/data/aishell-ctc-pilot`，代码仓库是当前目录：

```bash
python prepare_aishell.py verify /data/aishell-ctc-pilot

# 轨迹放在数据包外，保留原数据包的校验清单不变。
mkdir -p /data/ctc-traces
for split in train dev test; do
  python collect_teacher.py --model /models/Qwen3-ASR-1.7B \
    --manifest /data/aishell-ctc-pilot/${split}_full.jsonl \
    --language Chinese --chunk-seconds 2.0 \
    --unfixed-chunk-num 2 --unfixed-token-num 5 \
    --events-out /data/ctc-traces/${split}_events.jsonl \
    --teacher-out /data/ctc-traces/${split}_teacher.jsonl
done
```

采集依赖原始 Qwen checkpoint 和兼容的 `qwen-asr[vllm]` / Ascend vLLM 环境，见主 README。可以先采集 train/dev，选模冻结后再采集 test。输出文件必须是新文件；中途失败留下的部分文件不能当成完整轨迹，应改用新输出名重跑对应集合。

**不能把整句标注直接用于逐 chunk 训练。** 2 秒时尚未出现的词不应成为该时刻的监督目标。由目标 Qwen 模型为每个累计前缀产生蒸馏标签和真实 decoder token 轨迹；此数据包不包含尚未运行的 teacher 结果。

轨迹使用内网机器上的绝对音频路径，音频、采集、训练和评测应保持固定路径。当前 pilot 忽略末尾不足 2 秒的余音；不要将最后一个完整 chunk 的输出直接与**全句**人工转写计算 CER。人工全句标签适用于主 README 的整句对照基线。

## 3. 8 卡训练与独立测试

以下 15 epoch 是原小规模启动试验：828 个事件、8 卡、每卡 batch 8 仅有 195 次更新，不能据此判断已收敛。扩大训练覆盖、保留原 holdout 并按更新步数训练的流程见 [第二阶段实验](STAGE2.md)。

在能成功导入 `qwen_asr` 的 Python 环境中启动，显式固定 worker 解释器；裸 `torchrun` 可能来自系统环境，详见 [排障说明](TROUBLESHOOTING.md)。

```bash
export PYTHON_EXEC="$(python -c 'import sys; print(sys.executable)')"
python -m torch.distributed.run --standalone --nproc_per_node=8 train.py \
  --streaming --chunk-seconds 2.0 --backend qwen \
  --target-token-source recorded --selection-metric hybrid --verify-budget 8 \
  --model /models/Qwen3-ASR-1.7B \
  --manifest /data/ctc-traces/train_events.jsonl \
  --dev-manifest /data/ctc-traces/dev_events.jsonl \
  --dev-teacher /data/ctc-traces/dev_teacher.jsonl \
  --output /data/ctc-runs/aishell-pilot-001 --device npu \
  --epochs 15 --batch-size 8 --learning-rate 3e-4

# 训练和参数选择完成后，test 用于最终报告。
python stream_draft.py \
  --checkpoint /data/ctc-runs/aishell-pilot-001/best_head.pt \
  --manifest /data/ctc-traces/test_events.jsonl \
  --output /data/ctc-runs/aishell-pilot-001/test_drafts.jsonl --device npu
python score_teacher.py \
  --drafts /data/ctc-runs/aishell-pilot-001/test_drafts.jsonl \
  --teacher /data/ctc-traces/test_teacher.jsonl \
  --require-rollback --verify-budget 8 \
  --output /data/ctc-runs/aishell-pilot-001/test_metrics.json
```

按 [回滚复用协议](ROLLBACK.md)，第 1 轮无草稿，第 2 轮完整复用，第 3 轮起复用回滚尾部。CTC 仅尝试延长旧草稿。先看 loss、验证未见 token 比率和 `rollback_comparison.after_first.mean_delta_accepted_at_8`，再看第 3 次及之后分组。dev 用来选模型；不要根据 test 反复调参。已有音频包不变，若已采集旧版轨迹，需改用新的输出文件名重新采集 v2 teacher/events。8 卡的收益主要取决于特征缓存阶段，需实测。

## 适用范围

这是约一小时量级的路线初筛，只有 12 位说话人，且偏向有较长句子的说话人。朗读语音不能代替会议、噪声、口音和长会话评测，也无法保证与 Qwen 预训练数据无重合。初筛通过后应扩到更多说话人和更大词汇覆盖，再用业务域长音频验证；不能仅凭此小集合否定整个 CTC 路线，也不能据此前缀指标声称端到端加速。
