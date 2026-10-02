import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]


class SAMDownloadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.fp = types.SimpleNamespace(models_dir=self.tmp.name,
            get_filename_list=Mock(return_value=[]), get_full_path=Mock(return_value=None),
            get_folder_paths=Mock(return_value=[]))
        spec = importlib.util.spec_from_file_location("sam_loader_test", ROOT / "sam_loader.py")
        self.mod = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"folder_paths": self.fp}):
            spec.loader.exec_module(self.mod)
        self.download = Mock(side_effect=self.fake_download)
        self.hub = patch.dict(sys.modules, {"huggingface_hub": types.SimpleNamespace(hf_hub_download=self.download)})
        self.hub.start()
        self.addCleanup(self.hub.stop)

    def fake_download(self, **kwargs):
        path = Path(kwargs["local_dir"]) / kwargs["filename"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"test checkpoint")
        return str(path)

    def test_missing_downloads_to_checkpoints_then_reuses_offline(self):
        path = self.mod.ensure_sam_checkpoint()
        self.assertEqual(Path(path), Path(self.tmp.name) / "checkpoints" / self.mod.SAM_FILE)
        self.assertEqual(self.mod.ensure_sam_checkpoint(), path)
        self.download.assert_called_once_with(repo_id="Comfy-Org/sam3.1",
            filename="checkpoints/" + self.mod.SAM_FILE, local_dir=self.tmp.name)

    def test_extra_model_path_is_reused_without_network(self):
        existing = Path(self.tmp.name) / "existing.safetensors"
        existing.write_bytes(b"existing")
        self.fp.get_filename_list.return_value = ["sam/" + self.mod.SAM_FILE]
        self.fp.get_full_path.return_value = str(existing)
        self.assertEqual(self.mod.ensure_sam_checkpoint(), str(existing))
        self.download.assert_not_called()

    def test_failed_download_can_be_retried(self):
        self.download.side_effect = OSError("connection lost")
        with self.assertRaisesRegex(RuntimeError, "download failed"):
            self.mod.ensure_sam_checkpoint()
        self.assertFalse((Path(self.tmp.name) / "checkpoints" / self.mod.SAM_FILE).exists())
        self.download.side_effect = self.fake_download
        self.assertTrue(Path(self.mod.ensure_sam_checkpoint()).is_file())

    def test_empty_file_does_not_skip_download(self):
        target = Path(self.tmp.name) / "checkpoints" / self.mod.SAM_FILE
        target.parent.mkdir()
        target.touch()
        self.mod.ensure_sam_checkpoint()
        self.download.assert_called_once()

    def test_loader_passes_downloaded_path_to_comfy(self):
        sd = types.ModuleType("comfy.sd")
        sd.load_checkpoint_guess_config = Mock(return_value=("model", "clip", None))
        comfy = types.ModuleType("comfy")
        comfy.sd = sd
        with patch.dict(sys.modules, {"comfy": comfy, "comfy.sd": sd}):
            self.assertEqual(self.mod.MiniMaxH3SAMLoader().load(), ("model", "clip"))
        self.assertTrue(Path(sd.load_checkpoint_guess_config.call_args.args[0]).is_file())

    def test_auto_choice_available_without_installed_models(self):
        choices = self.mod.MiniMaxH3SAMLoader.INPUT_TYPES()["required"]["ckpt_name"][0]
        self.assertEqual(choices, [self.mod.AUTO])
        self.download.assert_not_called()


if __name__ == "__main__":
    unittest.main()
