"""SAM checkpoint loader with an on-demand, fixed Hugging Face download."""
import os
import threading

import folder_paths

SAM_REPO = "Comfy-Org/sam3.1"
SAM_FILE = "sam3.1_multiplex_fp16.safetensors"
AUTO = "Auto (download SAM 3.1 if missing)"
_DOWNLOAD_LOCK = threading.Lock()


def ensure_sam_checkpoint():
    """Reuse registered checkpoints, including extra model paths, before network I/O."""
    with _DOWNLOAD_LOCK:
        for name in folder_paths.get_filename_list("checkpoints"):
            if os.path.basename(name.replace("\\", "/")) == SAM_FILE:
                path = folder_paths.get_full_path("checkpoints", name)
                if path and os.path.getsize(path) > 0:
                    return path
        target = os.path.join(folder_paths.models_dir, "checkpoints", SAM_FILE)
        if os.path.isfile(target) and os.path.getsize(target) > 0:
            return target
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise RuntimeError("Auto Mask download needs huggingface_hub. Install this node's requirements.txt.") from exc
        print(f"[MiniMaxH3] Downloading {SAM_FILE} from {SAM_REPO}; progress follows in the console.", flush=True)
        try:
            # Hub preserves checkpoints/ under models/, stages incomplete files,
            # and locks the download across processes. No duplicate Hub-cache copy.
            path = hf_hub_download(repo_id=SAM_REPO, filename="checkpoints/" + SAM_FILE,
                                   local_dir=folder_paths.models_dir)
        except Exception as exc:
            raise RuntimeError("SAM 3.1 download failed. Check the internet connection, free disk space and "
                               "write access to models/checkpoints, then retry Auto Mask. "
                               "You can also place the checkpoint there manually.") from exc
        if not os.path.isfile(path) or os.path.getsize(path) == 0:
            raise RuntimeError("SAM 3.1 download did not produce a complete checkpoint; retry Auto Mask.")
        return path


class MiniMaxH3SAMLoader:
    CATEGORY = "conditioning/video_models"
    DESCRIPTION = "Loads SAM for Auto Mask; downloads the default SAM 3.1 checkpoint only when missing."
    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "load"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"ckpt_name": ([AUTO] + folder_paths.get_filename_list("checkpoints"),)}}

    def load(self, ckpt_name=AUTO):
        import comfy.sd
        if ckpt_name == AUTO:
            path = ensure_sam_checkpoint()
        else:
            path = folder_paths.get_full_path("checkpoints", ckpt_name)
            if not path or not os.path.isfile(path):
                raise FileNotFoundError(f"SAM checkpoint not found: {ckpt_name}. Select Auto to download SAM 3.1.")
        result = comfy.sd.load_checkpoint_guess_config(
            path, output_vae=False, output_clip=True,
            embedding_directory=folder_paths.get_folder_paths("embeddings"))
        return result[:2]


NODE_CLASS_MAPPINGS = {"MiniMaxH3SAMLoader": MiniMaxH3SAMLoader}
NODE_DISPLAY_NAME_MAPPINGS = {"MiniMaxH3SAMLoader": "Fantastic H3 SAM Loader"}
