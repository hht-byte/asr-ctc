# 训练启动排障

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

## Transformers 5.5.4：配置缺少 thinker_config

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
