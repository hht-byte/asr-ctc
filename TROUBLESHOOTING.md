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

## 重试提示输出目录不是空目录

训练会在 encoder 加载前创建 `run.json` 和 `features/`。即使在导入模型包时失败，原 `--output` 也可能已被占用。保留失败目录供检查，重试使用一个尚不存在的新目录，例如把 `aishell-pilot-001` 改成 `aishell-pilot-002`。音频和已采集的 teacher/events 无需因这类启动失败重新生成。

## 如何看待日志里的其他 warning/error

若独立导入最终打印成功，之前出现的 docstring `cache_position` 检查消息未阻止这次导入。训练退出原因应以 worker traceback 为准。hostname、LD_PRELOAD、CANN padding 或 barrier 警告应结合后续实际异常分析，不能只凭它们推断本次包导入失败由通信或驱动导致。修复解释器后，若出现新的完整 traceback，应从新失败位置继续定位；导入成功不等于真实 checkpoint/NPU 训练已经通过。
