# Qwen3-ASR-1.7B 冻结音频编码器 + CTC 流式草稿试验

本目录是独立试验，不改动原 ASR 推理权重或 vLLM 代码。**目标场景是每 2 秒触发一次推理，并且每次重新输入从录音开始累计到当前的音频**，与 [Qwen 官方流式实现](https://github.com/QwenLM/Qwen3-ASR/blob/main/qwen_asr/inference/qwen3_asr.py) 一致。支持 [Transformers 原生 `Qwen/Qwen3-ASR-1.7B-hf`](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf) 和 [Qwen 官方原始 `Qwen/Qwen3-ASR-1.7B`](https://github.com/QwenLM/Qwen3-ASR)。CTC 头接在音频编码器最终 `last_hidden_state` 后；原始 checkpoint 的编码器输出维度为 `output_dim`，原生 `-hf` 版是 `d_model`。音频编码器始终 `eval()`、无梯度；训练转写文本使用**同一模型 tokenizer 的原始 token ID**。CTC 类别只覆盖训练集出现过的 token，类别 0 留给 blank，保存的草稿会映射回 Qwen token ID。

## 环境和数据

**回滚复用协议已更新：** 第 1 轮无草稿，第 2 轮复用上一轮完整输出，第 3 轮起复用回滚尾部；CTC 从第 2 轮起仅尝试延长旧草稿。训练保留完整累计序列 CTC loss，使用采集的实际 token ID；按相对旧草稿的增量接受量选模。详细协议、指标及兼容性见 [ROLLBACK.md](ROLLBACK.md)。已有音频包可继续使用，旧版 teacher/events 需重新采集。

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

多进程训练使用当前环境的 `python -m torch.distributed.run`。启动前确认该 Python 能导入 `qwen_asr`（原始模型后端），并固定 worker 解释器，避免 PATH 中系统 `torchrun` 或遗留 `PYTHON_EXEC` 指向其他环境：

```bash
python -c 'import sys, torch, qwen_asr; print(sys.executable); print(sys.prefix); print(torch.__file__); print(qwen_asr.__file__)'
export PYTHON_EXEC="$(python -c 'import sys; print(sys.executable)')"
```

HF 后端的导入检查应使用 `transformers`，无需 `qwen_asr`。解释器不一致，以及 `qwen-asr 0.0.6 + transformers 5.5.4` 初始化时缺少 `thinker_config` 的局部修复与配置检查，见 [训练启动排障](TROUBLESHOOTING.md)。

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

`*_events.jsonl` 保存音频、累计截止采样点、当前文本，以及 `target_token_ids`（实际稳定前缀 ID + 当前实际生成 ID）。`text` 来自该次 teacher 流式输出，可能在下个 chunk 被修订，因此是蒸馏目标，不等于人工标注。新版 `*_teacher.jsonl` 使用 `schema_version: 2`，还保存上一轮输出、旧草稿和 chunk 序号。前两轮稳定前缀为空，第 2 轮完整复用第 1 轮输出；第 3 轮起复用实际回滚尾部。`unfixed_chunk_num` 固定为 2；`unfixed_token_num` 默认 5，按线上回滚长度设置。

2. **用 8 卡缓存每个累计音频前缀并训练 CTC 头。** 每个事件的编码器输入严格截到 `end_sample`；同一录音的事件不会跨训练和验证集。事件数会显著多于录音数，可先用约 100–300 条训练录音和 30–50 条独立验证录音试跑。

```bash
python -m torch.distributed.run --standalone --nproc_per_node=8 train.py \
  --streaming --chunk-seconds 2.0 --backend qwen \
  --target-token-source recorded --selection-metric hybrid --verify-budget 8 \
  --model /models/Qwen3-ASR-1.7B \
  --manifest /data/ctc/train_events.jsonl \
  --dev-manifest /data/ctc/dev_events.jsonl \
  --dev-teacher /data/ctc/dev_teacher.jsonl \
  --output /data/ctc/runs/stream-001 --device npu \
  --epochs 15 --batch-size 8 --learning-rate 3e-4
```

使用新版 teacher 和 `--selection-metric hybrid` 时，按 `rollback_comparison.after_first.mean_delta_accepted_at_8` 选择 `best_head.pt`，相同时比较组合接受量、再比较 token 错误率。第 1 轮不参与组合收益选模；验证预算可通过 `--verify-budget` 修改。`auto` 对新版轨迹自动使用组合指标，对旧轨迹仍使用 CTC 单独指标；无 teacher 时按 token 错误率选模。验证用于选模，最终结果应另留独立测试录音。

3. **按真实触发顺序重放验证集。** 音频文件只读取一次；每次用累计音频重新运行编码器和 CTC，记录 `encoder_head_ms` 与 `head_ms`。前者包括特征提取、编码器、CPU/NPU 传输和 CTC 头，后者包括缓存特征送入 CTC 头的传输与计算。两者都不包括 vLLM verifier；如果线上能复用目标模型已经算出的编码器输出，`head_ms` 更接近分支增量开销，但跨运行时共享所需的搬运和同步还要单独测量。

```bash
python stream_draft.py \
  --checkpoint /data/ctc/runs/stream-001/best_head.pt \
  --manifest /data/ctc/dev_events.jsonl \
  --output /data/ctc/runs/stream-001/stream_drafts.jsonl --device npu
```

4. **与仅回滚复用比较。** 第 1 轮两种策略都无草稿；第 2 轮起仅当 CTC 完整输出与“稳定前缀 + 旧草稿”逐 token 一致时，追加剩余 CTC token，否则保持旧草稿。报告 `rollback_comparison` 比较相同预算下旧草稿、CTC 单独和组合候选的接受量、增量及第 1/2/3+ 轮分组。顶层旧版 CTC 指标仍作为诊断保留，不代表部署策略。若稳定前缀无法从上一轮实际 ID 序列切出，该轮禁用旧草稿与延长并报告前缀不可复用。

```bash
python score_teacher.py \
  --drafts /data/ctc/runs/stream-001/stream_drafts.jsonl \
  --teacher /data/ctc/dev_teacher.jsonl \
  --require-rollback --verify-budget 8 \
  --output /data/ctc/runs/stream-001/teacher_prefix.json
```

该报告是相同 prompt 下贪心解码的单次验证代理指标，不含 target bonus token、EOS 和后续验证轮次，也尚未把 CTC 接入真实 verifier 或测 KV 回滚与端到端时延。原始 CTC 输出仍逐轮计算用于诊断，但组合策略第 1 轮不提出任何草稿。先看 `after_first.mean_delta_accepted_at_8`、`extension_proposed_rate`、`old_prefix_reusable_rate`；旧草稿占满预算时，追加部分本次不会被验证，不能据此认定后续验证轮次没有收益。

## 整句对照基线

每卡负责约八分之一的音频编码和特征缓存。所有卡随后用 DDP 训练同一个小型线性头；每个 epoch 不再重复跑冻结编码器。CTC loss 在 CPU 上计算，梯度会回传到 NPU 线性头；这是优先保证可移植性的试验实现，性能须实测。8 卡训练每轮最多丢弃 7 条不足一组的样本，逐轮洗牌后可轮换覆盖；小于 8 条训练音频会直接报错。

```bash
python -m torch.distributed.run --standalone --nproc_per_node=8 train.py \
  --backend qwen --model /models/Qwen3-ASR-1.7B \
  --manifest /data/ctc/train.jsonl \
  --dev-manifest /data/ctc/dev.jsonl \
  --output /data/ctc/runs/pilot-001 \
  --device npu --epochs 15 --batch-size 8 --learning-rate 3e-4
```

没有独立验证清单时，可省略 `--dev-manifest`，程序用 `--dev-fraction 0.2 --seed 42` 按音频随机切分；这个便捷方式**不会**保证说话人或会话隔离。单卡 CPU 冒烟运行使用 `python train.py ... --device cpu`，但 1.7B 模型本身会很慢。每次运行需要新的空 `--output` 目录，避免混用旧缓存。输出包括 `run.json`、`history.jsonl`、`best_head.pt`、`best_metrics.json` 和 `best_predictions.jsonl`。

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
