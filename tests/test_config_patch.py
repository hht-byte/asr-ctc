"""A narrow source patch must preserve validation and support safe rollback."""
import tempfile
import unittest
from pathlib import Path

from patch_qwen_config import patch_source, apply_patch_file, restore_patch_file


SOURCE = '''class Base:
    def __init__(self, **kwargs):
        self.checked = self.get_text_config()

class Qwen3ASRThinkerConfig(Base):
    def __init__(self, audio_config=None, text_config=None, **kwargs):
        super().__init__(**kwargs)
        self.audio_config = audio_config
        self.text_config = text_config
    def get_text_config(self):
        return self.text_config

class Qwen3ASRConfig(Base):
    def __init__(self, thinker_config=None, **kwargs):
        super().__init__(**kwargs)
        self.thinker_config = Qwen3ASRThinkerConfig(**(thinker_config or {}))
    def get_text_config(self):
        return self.thinker_config.get_text_config()
'''


class ConfigPatchTests(unittest.TestCase):
    def test_reproduces_then_repairs_initialization_without_skipping_validation(self):
        namespace = {}
        exec(SOURCE, namespace)
        with self.assertRaisesRegex(AttributeError, "thinker_config"):
            namespace["Qwen3ASRConfig"]()
        patched = patch_source(SOURCE)
        exec(patched, namespace)
        config = namespace["Qwen3ASRConfig"](thinker_config={"text_config": "text"})
        self.assertEqual(config.checked, "text")
        self.assertEqual(config.thinker_config.checked, "text")
        self.assertEqual(patch_source(patched), patched)

    def test_backup_restore_and_repeat_application(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            path = Path(directory) / "configuration_qwen3_asr.py"
            path.write_bytes(SOURCE.replace("\n", "\r\n").encode())
            original = path.read_bytes()
            backup = apply_patch_file(path)
            self.assertEqual(backup.read_bytes(), original)
            self.assertEqual(apply_patch_file(path), backup)
            restore_patch_file(path)
            self.assertEqual(path.read_bytes(), original)

    def test_restore_refuses_to_overwrite_subsequent_edits(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            path = Path(directory) / "configuration_qwen3_asr.py"
            path.write_text(SOURCE, encoding="utf-8")
            apply_patch_file(path)
            path.write_text(path.read_text("utf-8") + "# user edit\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "changed"):
                restore_patch_file(path)

    def test_unknown_constructor_shape_is_not_modified(self):
        changed = SOURCE.replace("super().__init__(**kwargs)", "super().__init__(foo=1, **kwargs)", 1)
        with self.assertRaisesRegex(ValueError, "unsupported"):
            patch_source(changed)
        with self.assertRaisesRegex(ValueError, "missing"):
            patch_source(SOURCE.replace("class Qwen3ASRConfig", "class OtherConfig"))


if __name__ == "__main__":
    unittest.main()
