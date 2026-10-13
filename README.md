# Qwen3-ASR-1.7B 冻结音频编码器 + CTC 流式草稿试验

本目录是独立试验，不改动原 ASR 推理权重或 vLLM 代码。**目标场景是每 2 秒触发一次推理，并且每次重新输入从录音开始累计到当前的音频**，与 [Qwen 官方流式实现](https://github.com/QwenLM/Qwen3-ASR/blob/main/qwen_asr/inference/qwen3_asr.py) 一致。支持 [Transformers 原生 `Qwen/Qwen3-ASR-1.7B-hf`](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf) 和 [Qwen 官方原始 `Qwen/Qwen3-ASR-1.7B`](https://github.com/QwenLM/Qwen3-ASR)。CTC 头接在音频编码器最终 `last_hidden_state` 后；原始 checkpoint 的编码器输出维度为 `output_dim`，原生 `-hf` 版是 `d_model`。音频编码器始终 `eval()`、无梯度；训练转写文本使用**同一模型 tokenizer 的原始 token ID**。CTC 类别只覆盖训练集出现过的 token，类别 0 留给 blank，保存的草稿会映射回 Qwen token ID。

## 环境和数据

**现成小数据集：** 已提供 AISHELL-1 中文流式试验数据的固定版本配方、下载脚本和离线校验，见 [DATASET.md](DATASET.md)。其中 `*_full.jsonl` 是人工整句标注；流式训练前须在目标 Qwen 环境运行 teacher 采集。

使用与 910B4 的 CANN 版本匹配的 PyTorch / `torch_npu`。两种模型格式使用**不同 Python 环境**：

- `--backend qwen`：原始 Qwen checkpoint；按 [Qwen 官方安装说明](https://github.com/QwenLM/Qwen3-ASR) 安装 `qwen-asr`，再安装 `soundfile`、`numpy`。
- `--backend hf`：原生 `-hf` checkpoint；[模型卡](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf)要求 `transformers>=5.13.0`，安装方式如下。不要在同一环境强行混装与其不兼容的 `qwen-asr` 版本。

```bash
python -m pip install 'transformers>=5.13.0' accelerate soundfile numpy
```

音频文件须为 16 kHz、至少 0.5 秒，单声道或双声道。先挑选同一目标业务的短音频（建议不超过 30 秒）；把说话人、会话和录制场景按训练集与验证集隔离，避免泄漏。每行 JSONL：

```json
{"audio":"audio/001.wav","text":"今天的会议开始了"}
{"audio":"audio/002.wav","text":"请确认订单编号"}
```

音频路径相对于清单文件所在目录解析，也可使用绝对路径。建议先用约 500–2000 条训练音频和至少 100 条独立验证音频；8 卡是否比单卡快，取决于数据量及模型加载和缓存 I/O，需在目标机器测量。

先在一张 910B4 上做真实 checkpoint 契约检查：

```bash
python probe_encoder.py --backend qwen --model /models/Qwen3-ASR-1.7B \
  --audio /data/ctc/train/001.wav --device npu
```

预期输出 `encoder_frozen: true`、正的 `frames`，并且 `hidden_size` 与模型编码器输出一致。若使用 `-hf` 模型，将 `--backend` 改为 `hf` 并提供对应模型路径。

## 2 秒流式主试验

按**录音/说话人/会话**先划分训练集与验证集，再分别保存整段录音清单；两份清单不能包含相同录音。下面四个阶段可顺序运行。第 1 阶段使用官方 Qwen vLLM 流式方法，需要安装 `qwen-asr[vllm]` 和可用的 Ascend vLLM 环境；第 2、3 阶段运行冻结编码器和 CTC 头。

1. **采集每次 2 秒触发的 teacher 轨迹。** `--language` 强制文本输出，避免语言标记混进目标 token。采集脚本调用官方 `streaming_transcribe()`，读取它每次传给 vLLM 的 prompt 和实际 `generated token_ids`，分别写出逐 chunk 转写和 teacher token 轨迹。脚本仅处理完整 2 秒 chunk，录音末尾不足 2 秒的 `finish_streaming_transcribe()` 暂不纳入主试验。

```bash
python collect_teacher.py --model /models/Qwen3-ASR-1.7B \
  --manifest /data/ctc/train_full.jsonl --language Chinese \
  --events-out /data/ctc/train_events.jsonl \
  --teacher-out /data/ctc/train_teacher.jsonl
python collect_teacher.py --model /models/Qwen3-ASR-1.7B \
  --manifest /data/ctc/dev_full.jsonl --language Chinese \
  --events-out /data/ctc/dev_events.jsonl \
  --teacher-out /data/ctc/dev_teacher.jsonl
```

`*_events.jsonl` 的每行是 `{"audio":"/abs/a.wav","end_sample":32000,"text":"当前累计转写"}`，接着是同一录音的 `64000`、`96000` 等。`text` 来自该次 teacher 流式输出，可能在下个 chunk 被修订，因此它是蒸馏目标，不等于人工标注。`*_teacher.jsonl` 同时记录同一事件的 `prefix_token_ids`（回滚后 prompt 的文本前缀）和 `generated_token_ids`（vLLM 该次实际输出的内容 token）。如线上使用不同的 `unfixed_chunk_num` 或 `unfixed_token_num`，采集时用对应参数；默认是 2 和 5。

2. **用 8 卡缓存每个累计音频前缀并训练 CTC 头。** 每个事件的编码器输入严格截到 `end_sample`；同一录音的事件不会跨训练和验证集。事件数会显著多于录音数，可先用约 100–300 条训练录音和 30–50 条独立验证录音试跑。

```bash
torchrun --standalone --nproc_per_node=8 train.py \
  --streaming --chunk-seconds 2.0 --backend qwen \
  --model /models/Qwen3-ASR-1.7B \
  --manifest /data/ctc/train_events.jsonl \
  --dev-manifest /data/ctc/dev_events.jsonl \
  --dev-teacher /data/ctc/dev_teacher.jsonl \
  --output /data/ctc/runs/stream-001 --device npu \
  --epochs 15 --batch-size 8 --learning-rate 3e-4
```

提供 `--dev-teacher` 时，每个 epoch 都统计流式前缀指标，并优先按 `mean_accepted_at_4` 选择 `best_head.pt`；未提供时退回逐 chunk token 错误率选模。验证轨迹用于选模，因此若要给出无偏的最终结果，应另留一份未参与选模的测试录音。

3. **按真实触发顺序重放验证集。** 音频文件只读取一次；每次用累计音频重新运行编码器和 CTC，记录 `encoder_head_ms` 与 `head_ms`。前者包括特征提取、编码器、CPU/NPU 传输和 CTC 头，后者包括缓存特征送入 CTC 头的传输与计算。两者都不包括 vLLM verifier；如果线上能复用目标模型已经算出的编码器输出，`head_ms` 更接近分支增量开销，但跨运行时共享所需的搬运和同步还要单独测量。

```bash
python stream_draft.py \
  --checkpoint /data/ctc/runs/stream-001/best_head.pt \
  --manifest /data/ctc/dev_events.jsonl \
  --output /data/ctc/runs/stream-001/stream_drafts.jsonl --device npu
```

4. **与相同 chunk 的 teacher 后缀比较。** 只有 CTC 完整草稿的开头与当次回滚后稳定前缀 token **完全一致**，才会提出剩余草稿作为 verifier 候选；不一致时该次候选为空，防止把已确认前缀之外的错误 token 送入验证。报告给出前缀匹配率、平均候选长度、平均可接受 token、截断到 1/2/4/8 token 的接受量、首次/第二次/之后触发的分组结果，以及编码器加 CTC 超过 2 秒 chunk 预算的比例。

```bash
python score_teacher.py \
  --drafts /data/ctc/runs/stream-001/stream_drafts.jsonl \
  --teacher /data/ctc/dev_teacher.jsonl \
  --output /data/ctc/runs/stream-001/teacher_prefix.json
```

这份报告比整句 token 错误率更接近目标场景，但仍不是推测解码真实接受率：CTC 编码器与 vLLM decoder 的 tokenizer 边界可能不同，当前实现只用**完全一致的 token 前缀**提候选；也尚未把 CTC 输出接入 verifier、测 rejection/KV 回滚或比较完整端到端时延。若 `prefix_match_rate` 很低，先检查 token 化和稳定前缀拼接，而不要仅增加训练轮次。若 `mean_accepted_at_4/8` 不高，或编码器加 CTC 的时间已接近 2 秒预算，优先停止扩大试验。

## 整句对照基线

每卡负责约八分之一的音频编码和特征缓存。所有卡随后用 DDP 训练同一个小型线性头；每个 epoch 不再重复跑冻结编码器。CTC loss 在 CPU 上计算，梯度会回传到 NPU 线性头；这是优先保证可移植性的试验实现，性能须实测。8 卡训练每轮最多丢弃 7 条不足一组的样本，逐轮洗牌后可轮换覆盖；小于 8 条训练音频会直接报错。

```bash
torchrun --standalone --nproc_per_node=8 train.py \
  --backend qwen --model /models/Qwen3-ASR-1.7B \
  --manifest /data/ctc/train.jsonl \
  --dev-manifest /data/ctc/dev.jsonl \
  --output /data/ctc/runs/pilot-001 \
  --device npu --epochs 15 --batch-size 8 --learning-rate 3e-4
```

没有独立验证清单时，可省略 `--dev-manifest`，程序用 `--dev-fraction 0.2 --seed 42` 按音频随机切分；这个便捷方式**不会**保证说话人或会话隔离。单卡 CPU 冒烟运行将 `torchrun ...` 换成 `python train.py ... --device cpu`，但 1.7B 模型本身会很慢。每次运行需要新的空 `--output` 目录，避免混用旧缓存。输出包括 `run.json`、`history.jsonl`、`best_head.pt`、`best_metrics.json` 和 `best_predictions.jsonl`。

## 整句草稿与判定

```bash
python draft.py --checkpoint /data/ctc/runs/pilot-001/best_head.pt \
  --audio /data/ctc/dev/001.wav --device npu
```

输出是 JSON，包含 `token_ids` 和 `text`。`token_ids` 是原 Qwen tokenizer 的 ID，可作为后续推测验证实验的输入；尚未接入 vLLM verifier。

`best_metrics.json` 提供 token 错误率、去空格 CER、平均草稿长度、与**人工转写 token 化结果**一致的前缀长度，以及验证集未见 token 比率。人工转写的 token 序列与原 Qwen 自回归解码序列可能不同，所以这些前缀指标只用于初筛，不能当作真实接受率。

要检验真实草稿前缀，从原 Qwen ASR 解码器保存验证集的**转写内容 token ID**（去掉 `language ...<asr_text>` 等前缀和 EOS），格式如下：

```json
{"audio":"/data/ctc/dev/001.wav","token_ids":[123,456,789]}
```

`audio` 必须与 `best_predictions.jsonl` 中的绝对路径一致，然后运行：

```bash
python score_teacher.py \
  --drafts /data/ctc/runs/pilot-001/best_predictions.jsonl \
  --teacher /data/ctc/teacher_tokens.jsonl \
  --output /data/ctc/runs/pilot-001/teacher_prefix.json
```

重点看验证集 OOV 比例、CER、`prefix_at_least_2/4/8_rate` 和 `mean_accepted_at_2/4/8`。若低频词和跨场景音频上的前缀很短，继续增加纯线性 CTC 头的训练量可能意义不大；可先尝试更大的目标 token 词表、带上下文的轻量 CTC 分支或其他草稿方案。即使前缀可用，是否加速仍需在 910B4 上把**编码 + CTC 草稿 + verifier**的端到端时延与原 ASR 比较。当前脚本没有设备实测结果，也没有验证 speculative rejection/KV 回滚。

本机检查：`python -m unittest discover -s tests -v`。其中 CPU smoke test 用假编码器执行完整的一轮 CTC 训练；不代替真实 checkpoint 与 910B4 验收。
