# 训练启动排障

## 训练样本也识别很差：先做缓存特征过拟合诊断

若选中的模型在训练子集上 token/字符错误率接近 100%，且训练子集 OOV 为 0，词表覆盖不足不是唯一问题。仅凭早期 checkpoint 的这个结果，不能证明最后一轮也没学会，更不能直接判断冻结 encoder 路线不可行。

小数据配八卡时应检查更新次数。例如 828 个训练事件、8 卡、每卡 batch=8，当前 sampler 每卡每轮取 103 个事件，DataLoader 每轮更新 13 次；15 轮共 195 次，第 2 轮 checkpoint 只有 26 次更新。`train_samples` 是累计音频事件数，不是独立录音数。

`overfit_cached.py` 只读取已有 `run.json` 中的训练划分、recorded 目标 ID 和 `features/` 缓存。不加载 encoder、tokenizer 或音频，不访问 dev/test 缓存；保留原始完整 CTC 词表，随机选取完整训练录音，重新初始化同形状线性头，并使用与主训练相同的 AdamW 和 CPU CTCLoss（梯度回传到设备）。它用于检验少量训练样本的可学习性，不是正式重新训练或选模。

```bash
python overfit_cached.py \
  --run data/ctc-runs/aishell-pilot-004 \
  --output data/ctc-runs/aishell-overfit-001 \
  --device npu --recordings 8 --steps 1000 \
  --batch-size 8 --learning-rate 3e-4 --eval-every 50
```

请用能运行原训练脚本的 Python，单进程运行，不使用八卡 torchrun。输出目录必须新建或为空，原训练结果不会被覆盖。CPU 对照可改为 `--device cpu` 并使用另一个输出目录；相同 seed、参数和源缓存会使用相同样本、初始化及 batch 顺序。

输出包括 `run.json`、`history.jsonl`、`summary.json`、`predictions.jsonl` 和仅供诊断的 `last_head.pt`。日志第 0 步是随机初始化基线，第 1 步及每次评测记录：

- `token_error_rate`、`exact_sequence_rate`、`mean_prefix_tokens`：整个所选训练子集的识别指标。
- `blank_frame_rate`：被 argmax 判为 blank 的帧比例；高 blank 比例需结合识别指标判断，CTC 本身允许大量 blank。
- `batch_loss`：当前更新 batch 的 loss，不是整个子集的平均 loss。
- `gradient_norm`、`weight_update_max_abs`：本次反向传播及权重变化；首步无梯度/无更新会直接报错。

达到训练 token 错误率 ≤1% 且序列完全匹配率 ≥95% 时，标记 `memorized_subset` 并提前停止；否则到指定步数标记 `budget_exhausted`。这是诊断判据，不是模型质量或理论可行性的统一门槛。若预算耗尽，先看完整曲线、梯度、更新量、blank 比例和缓存特征统计，再决定继续训练或检查特征/标签/线性头容量。即使通过，也只证明该训练子集可以被拟合，不证明验证集泛化、910B 性能或投机加速收益。

验证集 OOV 高时，应另行扩大独立训练数据并检查覆盖；不要把 dev/test 标签加入训练词表来改善评测，也不要把这个仅在训练子集上拟合的头用于部署。

## 单进程可导入 qwen_asr，八卡却 ModuleNotFoundError

先对比成功命令和失败 worker 的 Python。虚拟环境中没有自己的 `torchrun` 入口时，shell 可能从 PATH 后面找到系统安装的入口；它的 shebang 会选择系统 Python，即使提示符显示已经激活虚拟环境。

典型证据：单独 `python -c 'import qwen_asr'` 成功；训练 traceback 的 launcher 却位于系统目录，worker 报 `of binary: /.../system/python3`，随后 `ModuleNotFoundError: No module named 'qwen_asr'`。

```bash
type -a python torchrun
head -n 1 "$(command -v torchrun)"
python -c 'import sys, torch, qwen_asr; print(sys.executable); print(sys.prefix); print(torch.__file__); print(qwen_asr.__file__)'
```

如果虚拟环境通过 system-site-packages 使用系统 PyTorch，`torch.__file__` 位于系统目录本身不表示有问题；重点是 worker 的 `sys.executable` / `sys.prefix` 和是否能导入所需包。`python` 是 alias/function 时，应直接使用已确认的虚拟环境解释器路径。

在可导入模型包的环境中，用同一个 Python 启动 launcher 和 worker：

```bash
ASR_PY=/path/to/myenv/bin/python
"$ASR_PY" -c 'import sys, torch, qwen_asr; print(sys.executable); print(sys.prefix); print(qwen_asr.__file__)'

# 后面接 README 中原有 train.py 参数，包括新的空 --output 目录。
PYTHON_EXEC="$ASR_PY" "$ASR_PY" -m torch.distributed.run \
  --standalone --nproc_per_node=8 train.py --help
```

这里的 `--help` 只验证命令入口，不加载模型或验证 NPU 训练。实际训练时用 README/DATASET 的完整参数替换 `--help`。`ASR_PY` 应保留虚拟环境入口路径，不要用 `readlink -f` 将它替换为环境外的底层 Python 二进制。

[PyTorch 官方文档](https://docs.pytorch.org/docs/stable/elastic/run.html)说明 `torchrun` 等价于 `python -m torch.distributed.run`，worker 默认使用 launcher 的 `sys.executable`，也可由 `PYTHON_EXEC` 覆盖。上述命令同时固定两者，不需要为了解决解释器选错而重装整套 PyTorch/CANN。

## Transformers 5.5.4：文本 RoPE 报 KeyError: 'default'

若配置初始化后，traceback 到达 `Qwen3ASRThinkerTextRotaryEmbedding` 的 `ROPE_INIT_FUNCTIONS[self.rope_type]` 并报 `KeyError: 'default'`，这是另一个文本模型 API 兼容问题。[Qwen 代码](https://github.com/QwenLM/Qwen3-ASR/blob/main/qwen_asr/core/transformers_backend/modeling_qwen3_asr.py)查找 `default` 初始化函数，而 [Transformers 5.5.4 的注册表](https://github.com/huggingface/transformers/blob/v5.5.4/src/transformers/modeling_rope_utils.py)已不包含该项。前面的 `Unrecognized keys in rope_parameters` 是警告，本次中止点是这个字典查找。

新版 `--backend qwen` 的 CTC 加载路径直接读取原始 checkpoint 的 `thinker_config.audio_config`，构造官方 `Qwen3ASRAudioEncoder`，通过 safetensors 逐个加载 `thinker.audio_tower.*` 张量，包括编码器内部的 `proj1/proj2`。它不再调用 `Qwen3ASRModel.from_pretrained()`，也不构造根配置、thinker 或文本模型。所有音频权重键、形状和浮点类型必须匹配；缺失或多余的音频权重会报错，不会带着随机权重继续训练。

预处理器直接加载同一 checkpoint 的 `WhisperFeatureExtractor` 和 `Qwen2TokenizerFast`。训练 tokenizer 同样直接加载，避免 `AutoTokenizer` 再次创建复合配置。音频仍按原有累计前缀输入，CTC loss、teacher token ID 和回滚草稿协议不变。

更新仓库后，用原来失败的单卡 probe 重试即可，不必再打 RoPE 补丁，也不必为此降级整个 vLLM/Ascend 环境：

```bash
git pull --ff-only
ASR_PY=/path/to/myenv/bin/python
MODEL=/models/Qwen3-ASR-1.7B
"$ASR_PY" probe_encoder.py --backend qwen --model "$MODEL" \
  --audio /path/to/actual-16khz.wav --device npu
```

成功结果包含 `encoder_only: true`、`text_decoder_loaded: false`、`encoder_frozen: true`；公开 1.7B 模型还应有 `loaded_audio_tensors: 397`、`hidden_size: 2048` 和正的 `frames`。通过后再以同一解释器启动八卡训练，使用新的空输出目录。

验证范围：已在 Transformers 5.5.4 / Hub 1.27.0 下核对公开 1.7B 配置对应的全部 397 个音频权重键；官方 tokenizer 的测试文本 ID 与 AutoTokenizer 一致，2 秒输入得到 `[1,128,200]` 特征。使用官方 0.0.6 音频类定义、缩小配置和合成权重，在 CPU BF16 下验证加载前后前向结果逐元素相同。为隔离下载版源码中的文本装饰器兼容问题，该本地测试只执行音频相关类定义，没有验证整个 `qwen_asr` 包导入。你的环境已能导入该包；真实 1.7B 权重、Python 3.12、PyTorch 2.10 / NPU 和八卡运行仍需设备验证。

该修复仅覆盖本仓库 CTC 的 encoder 加载，不修改 teacher 采集或线上 vLLM 的完整模型路径。之前应用的配置初始化补丁可保留；新版 CTC 路径不依赖它。完整模型若仍需 Transformers 兼容修复，应独立验证。

## Transformers 5.5.4：配置缺少 thinker_config

**新版 CTC 路径已绕过复合配置初始化。** 先更新仓库并运行上面的单卡 probe。以下保留之前的配置诊断与恢复方法，单独通过此检查不能证明完整模型加载成功。

已复现组合：`qwen-asr 0.0.6` 的公开配置源码、`transformers 5.5.4`、`huggingface-hub 1.27.0`。典型调用链是 `super().__init__` → `validate_token_ids` → `get_text_config` → `AttributeError: 'Qwen3ASRConfig' object has no attribute 'thinker_config'`。

[Qwen 0.0.6 的依赖声明](https://github.com/QwenLM/Qwen3-ASR/blob/main/pyproject.toml)固定 `transformers==4.57.6`。5.5.4 的基类初始化会校验 token ID，并调用子类的 `get_text_config()`；此时 Qwen 尚未建立 `thinker_config`，导致失败。这个错误发生在加载权重前，与训练数据或八卡 batch size 无关。

已有 vLLM/Ascend 环境需要保留现有依赖时，可以显式应用本仓库的局部源码补丁。它只调整 `Qwen3ASRConfig` 和 `Qwen3ASRThinkerConfig` 的构造顺序：先建立子配置，再调用 `super().__init__(**kwargs)`，保留基类校验。补丁会备份文件、拒绝不识别的构造结构，并支持恢复；不会自动修改包版本。

先停止使用该 Qwen 源码的训练进程，在 asr-ctc 目录执行（替换为实际路径）：

```bash
ASR_PY=/path/to/myenv/bin/python
QWEN_CONFIG=/path/to/Qwen3-ASR/qwen_asr/core/transformers_backend/configuration_qwen3_asr.py
MODEL=/models/Qwen3-ASR-1.7B

# 预览改动；不写文件。
"$ASR_PY" patch_qwen_config.py --file "$QWEN_CONFIG"
# 在源码旁保存 .asr-ctc-init-order.bak，然后原子替换源码。
"$ASR_PY" patch_qwen_config.py --file "$QWEN_CONFIG" --apply
# 只导入配置模块，读取本地 config.json，不加载权重或导入 vLLM。
"$ASR_PY" patch_qwen_config.py --file "$QWEN_CONFIG" --check-model "$MODEL"
```

预期 `config_load` 和 `config_roundtrip` 都为 `ok`。公开 1.7B 配置的结果是 `audio_d_model: 1024`、`audio_output_dim: 2048`、`text_vocab_size: 151936`。本地已在上述精确 Transformers/Hub 版本下复现原错误，并验证补丁后的配置加载及序列化往返；测试使用 Python 3.14、公开配置文件，没有加载真实权重，也没有验证 NPU。

随后用一条实际音频进行单卡检查，确认模型加载、编码器前向和输出契约：

```bash
"$ASR_PY" probe_encoder.py --backend qwen --model "$MODEL" \
  --audio /path/to/actual-16khz.wav --device npu
```

通过后才重试八卡训练，仍用 `PYTHON_EXEC="$ASR_PY" "$ASR_PY" -m torch.distributed.run`，并选择新的空 `--output` 目录。配置检查通过只说明当前初始化错误已解决，不代表 Qwen 的所有模型代码都兼容 Transformers 5.5.4；后续若出现新的模型 API 错误，应保留完整 traceback 继续定位。配置检查中可能出现 RoPE 参数键警告，也不能据此认为模型前向已验证。

恢复原文件：

```bash
"$ASR_PY" patch_qwen_config.py --file "$QWEN_CONFIG" --restore
```

若补丁后手动改过源码，恢复命令会拒绝覆盖，需自行比较备份。升级或重新安装 Qwen 后，应重新检查源码及兼容性，不要盲目复用旧备份。

## 重试提示输出目录不是空目录

训练会在 encoder 加载前创建 `run.json` 和 `features/`。即使在导入模型包时失败，原 `--output` 也可能已被占用。保留失败目录供检查，重试使用一个尚不存在的新目录，例如把 `aishell-pilot-001` 改成 `aishell-pilot-002`。音频和已采集的 teacher/events 无需因这类启动失败重新生成。

## 如何看待日志里的其他 warning/error

若独立导入最终打印成功，之前出现的 docstring `cache_position` 检查消息未阻止这次导入。训练退出原因应以 worker traceback 为准。hostname、LD_PRELOAD、CANN padding 或 barrier 警告应结合后续实际异常分析，不能只凭它们推断本次包导入失败由通信或驱动导致。修复解释器后，若出现新的完整 traceback，应从新失败位置继续定位；导入成功不等于真实 checkpoint/NPU 训练已经通过。
