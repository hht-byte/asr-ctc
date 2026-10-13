"""Opt-in, backed-up fix for Qwen ASR composite config initialization with TF 5.5.4.

Only moves super().__init__(**kwargs) after child construction in two named classes.
Does not change model weights, dependencies, validation methods, or model forward.
"""
from __future__ import annotations

import argparse
import ast
import difflib
import importlib.util
import json
import os
import stat
import sys
import tempfile
from pathlib import Path

CLASSES = {"Qwen3ASRConfig": {"thinker_config"},
           "Qwen3ASRThinkerConfig": {"audio_config", "text_config"}}
BACKUP_SUFFIX = ".asr-ctc-init-order.bak"


def patch_source(source: str) -> str:
    tree = ast.parse(source)
    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    changes = []
    for name, fields in CLASSES.items():
        if name not in classes:
            raise ValueError(f"missing expected class {name}; refusing to patch")
        methods = [n for n in classes[name].body if isinstance(n, ast.FunctionDef) and n.name == "__init__"]
        if len(methods) != 1:
            raise ValueError(f"unsupported constructor in {name}")
        init = methods[0]
        body = init.body
        if isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
            body = body[1:]
        calls = [n for n in body if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                 and isinstance(n.value.func, ast.Attribute) and n.value.func.attr == "__init__"
                 and isinstance(n.value.func.value, ast.Call)
                 and isinstance(n.value.func.value.func, ast.Name)
                 and n.value.func.value.func.id == "super"]
        if len(calls) != 1 or ast.dump(calls[0].value) != ast.dump(ast.parse("super().__init__(**kwargs)").body[0].value):
            raise ValueError(f"unsupported super initializer in {name}")
        call = calls[0]
        assignments = {n.attr for n in ast.walk(init) if isinstance(n, ast.Attribute)
                       and isinstance(n.ctx, ast.Store) and isinstance(n.value, ast.Name)
                       and n.value.id == "self"}
        if not fields <= assignments or any(isinstance(n, (ast.Return, ast.Yield, ast.YieldFrom)) for n in ast.walk(init)):
            raise ValueError(f"unsupported child initialization in {name}")
        if call is body[-1]:
            continue
        if call is not body[0]:
            raise ValueError(f"unsupported initializer position in {name}")
        # Do not rewrite/unparse the module: preserve all other code and comments.
        changes.append((call.lineno - 1, call.end_lineno, init.end_lineno))
    lines = source.splitlines(keepends=True)
    for start, end, after in sorted(changes, reverse=True):
        statement = lines[start:end]
        lines[start:after] = lines[end:after] + statement
    patched = "".join(lines)
    compile(patched, "configuration_qwen3_asr.py", "exec")
    return patched


def atomic_write(path, data):
    mode = stat.S_IMODE(path.stat().st_mode)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def apply_patch_file(path):
    path = Path(path).resolve()
    original = path.read_bytes()
    patched = patch_source(original.decode("utf-8")).encode("utf-8")
    backup = path.with_name(path.name + BACKUP_SUFFIX)
    if patched == original:
        return backup
    if backup.exists():
        if backup.read_bytes() != original:
            raise ValueError(f"backup already exists for different source: {backup}")
    else:
        with backup.open("xb") as stream:
            stream.write(original)
    if path.read_bytes() != original:
        raise ValueError("source changed while preparing patch; no overwrite performed")
    atomic_write(path, patched)
    return backup


def restore_patch_file(path):
    path = Path(path).resolve()
    backup = path.with_name(path.name + BACKUP_SUFFIX)
    original = backup.read_bytes()
    current = path.read_bytes()
    if current == original:
        return
    if current != patch_source(original.decode("utf-8")).encode("utf-8"):
        raise ValueError("source changed after patch; refusing to overwrite subsequent edits")
    atomic_write(path, original)


def check_model(config_source, model_path):
    """Import config source only; avoid importing qwen_asr/vLLM or allocating weights."""
    import transformers
    import huggingface_hub
    config_path = Path(model_path)
    if config_path.is_dir():
        config_path = config_path / "config.json"
    raw = json.loads(config_path.read_text("utf-8"))
    if raw.get("model_type") != "qwen3_asr":
        raise ValueError("expected original Qwen qwen3_asr checkpoint config.json")
    name = "_asr_ctc_checked_qwen_config"
    spec = importlib.util.spec_from_file_location(name, config_source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    config = module.Qwen3ASRConfig.from_dict(raw)
    text = config.get_text_config(decoder=True)
    audio = config.thinker_config.audio_config
    roundtrip = module.Qwen3ASRConfig.from_dict(config.to_dict())
    if text.vocab_size != raw["thinker_config"]["text_config"]["vocab_size"]:
        raise ValueError("text vocabulary changed during config construction")
    if roundtrip.to_dict() != config.to_dict():
        raise ValueError("config serialization round trip changed values")
    return {"config_load": "ok", "config_roundtrip": "ok", "python": sys.executable,
            "transformers": transformers.__version__, "huggingface_hub": huggingface_hub.__version__,
            "config_source": str(Path(config_source).resolve()),
            "audio_d_model": audio.d_model, "audio_output_dim": audio.output_dim,
            "text_vocab_size": text.vocab_size, "model_weights_loaded": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path, required=True,
                        help="Installed/source Qwen configuration_qwen3_asr.py")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--apply", action="store_true", help="Back up then patch source in place")
    action.add_argument("--restore", action="store_true", help="Restore only if no later edits would be lost")
    action.add_argument("--check-model", type=Path, help="Check local model config only, without weights")
    args = parser.parse_args()
    if args.check_model:
        print(json.dumps(check_model(args.file, args.check_model), ensure_ascii=False, indent=2))
    elif args.apply:
        backup = apply_patch_file(args.file)
        print(f"Initialization order ready: {args.file}")
        print(f"Backup: {backup}" if backup.exists() else "Source was already in the supported order; no backup created")
    elif args.restore:
        restore_patch_file(args.file)
        print(f"Restored: {args.file}")
    else:
        source = args.file.read_bytes().decode("utf-8")
        patched = patch_source(source)
        print("".join(difflib.unified_diff(source.splitlines(True), patched.splitlines(True),
                                         fromfile=str(args.file), tofile=str(args.file) + " (patched)")), end="")
        if patched == source:
            print("Already in the supported initialization order")


if __name__ == "__main__":
    main()
