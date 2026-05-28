"""
ComfyUI API client — submits the SD-inpainting workflow via the ComfyUI REST API.

ComfyUI runs on port 8188 by default.
API docs: https://github.com/comfyanonymous/ComfyUI/blob/master/server.py
"""

import base64
import io
import json
import logging
import time
import uuid
from pathlib import Path

import requests
from PIL import Image

log = logging.getLogger(__name__)

WORKFLOW_NODE_IDS = {
    "load_image":    "10",   # LoadImage
    "load_mask":     "11",   # LoadImageMask
    "checkpoint":    "1",    # CheckpointLoaderSimple
    "positive":      "4",    # CLIPTextEncode (positive)
    "negative":      "5",    # CLIPTextEncode (negative)
    "vae_encode":    "6",    # VAEEncodeForInpaint
    "ksampler":      "7",    # KSampler
    "vae_decode":    "8",    # VAEDecode
    "save_image":    "9",    # SaveImage
}


class ComfyUIClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8188"):
        self.base_url  = base_url.rstrip("/")
        self._session  = requests.Session()
        self.client_id = str(uuid.uuid4())

    # ── Health ────────────────────────────────────────────────
    def health(self) -> str:
        try:
            r = self._session.get(f"{self.base_url}/system_stats", timeout=3)
            return "ok" if r.status_code == 200 else f"http {r.status_code}"
        except Exception:
            return "unreachable"

    # ── Run workflow ─────────────────────────────────────────
    def run_inpaint_workflow(
        self,
        image_path:    Path,
        mask_path:     Path,
        output_path:   Path,
        prompt:        str  = "seamless decorative background, no text",
        negative_prompt: str = "text, words, letters, watermark, blurry",
        workflow_path: Path = None,
        steps:         int  = 20,
        cfg:           float = 7.0,
        seed:          int  = -1,
        timeout:       int  = 300,
    ):
        """
        1. Upload image + mask to ComfyUI /upload/image
        2. Patch the workflow JSON with the uploaded filenames + prompts
        3. POST to /prompt
        4. Poll /history until done
        5. Download result to output_path
        """
        # ── Upload source image (unique filename per session to avoid collisions) ──
        # Use the session-id prefix from the parent dir (e.g. /temp/<uuid>/source.png)
        session_prefix = image_path.parent.name[:8]
        img_name  = self._upload_image(image_path, "", prefix=f"{session_prefix}_src_")
        mask_name = self._upload_image(mask_path,  "", prefix=f"{session_prefix}_msk_")

        # ── Load & patch workflow ─────────────────────────────
        if workflow_path and workflow_path.exists():
            with open(workflow_path) as f:
                workflow = json.load(f)
        else:
            workflow = self._default_workflow()

        # Remove top-level metadata keys (e.g. "_comment") that aren't node IDs
        workflow = {k: v for k, v in workflow.items()
                    if isinstance(v, dict) and "class_type" in v}

        # Patch node inputs
        workflow = self._patch_workflow(
            workflow, img_name, mask_name, prompt, negative_prompt,
            steps, cfg, seed
        )

        # ── Submit prompt ─────────────────────────────────────
        prompt_id = self._submit_prompt(workflow)
        log.info("ComfyUI prompt submitted: %s", prompt_id)

        # ── Poll for completion ───────────────────────────────
        start = time.time()
        while time.time() - start < timeout:
            history = self._get_history(prompt_id)
            if prompt_id in history:
                entry = history[prompt_id]
                if entry.get("status", {}).get("completed"):
                    # Find the output image
                    outputs = entry.get("outputs", {})
                    image_data = self._extract_output_image(outputs)
                    output_path.write_bytes(image_data)
                    log.info("ComfyUI result saved → %s", output_path)
                    return
                if entry.get("status", {}).get("status_str") == "error":
                    raise RuntimeError(f"ComfyUI workflow failed: {entry}")
            time.sleep(2)

        raise TimeoutError(f"ComfyUI did not finish within {timeout}s")

    # ── Internal helpers ─────────────────────────────────────
    def _upload_image(
        self, path: Path, subfolder: str = "", prefix: str = ""
    ) -> str:
        """
        Upload image to ComfyUI's input folder, return the server-side
        filename that should be passed to LoadImage / LoadImageMask nodes.

        ComfyUI's /upload/image puts files in `input/<subfolder>/<filename>`
        and returns {"name": "<filename>", "subfolder": "<subfolder>", "type": "input"}.
        LoadImage accepts either `filename` (if subfolder empty) or
        `subfolder/filename` (if subfolder set).
        """
        upload_name = (prefix + path.name) if prefix else path.name
        with open(path, "rb") as f:
            r = self._session.post(
                f"{self.base_url}/upload/image",
                files={"image": (upload_name, f, "image/png")},
                data={
                    "subfolder": subfolder,
                    "type":      "input",
                    "overwrite": "true",
                },
                timeout=30,
            )
        r.raise_for_status()
        data = r.json()
        # Returned filename (may differ from upload_name if conflict)
        returned = data.get("name", upload_name)
        sub      = data.get("subfolder", "") or subfolder
        full = f"{sub}/{returned}" if sub else returned
        log.info("ComfyUI upload: %s → %s (subfolder=%r)", path.name, returned, sub)
        return full

    def _submit_prompt(self, workflow: dict) -> str:
        payload = {"prompt": workflow, "client_id": self.client_id}
        r = self._session.post(
            f"{self.base_url}/prompt", json=payload, timeout=30
        )
        r.raise_for_status()
        return r.json()["prompt_id"]

    def _get_history(self, prompt_id: str) -> dict:
        r = self._session.get(
            f"{self.base_url}/history/{prompt_id}", timeout=10
        )
        r.raise_for_status()
        return r.json()

    def _extract_output_image(self, outputs: dict) -> bytes:
        """Get the first PNG output from a ComfyUI history outputs dict."""
        for node_id, node_out in outputs.items():
            images = node_out.get("images", [])
            for img_info in images:
                fname     = img_info["filename"]
                subfolder = img_info.get("subfolder", "")
                ftype     = img_info.get("type", "output")
                url = (
                    f"{self.base_url}/view"
                    f"?filename={fname}&subfolder={subfolder}&type={ftype}"
                )
                r = self._session.get(url, timeout=30)
                r.raise_for_status()
                return r.content
        raise RuntimeError("No output image found in ComfyUI result.")

    @staticmethod
    def _patch_workflow(
        workflow: dict,
        img_name:  str,
        mask_name: str,
        prompt:    str,
        neg_prompt: str,
        steps:     int,
        cfg:       float,
        seed:      int,
    ) -> dict:
        """Inject runtime values into the workflow dict."""
        ids = WORKFLOW_NODE_IDS
        import random

        def _set(node_id, key, value):
            if node_id in workflow:
                workflow[node_id]["inputs"][key] = value

        _set(ids["load_image"],  "image",    img_name)
        _set(ids["load_mask"],   "image",    mask_name)
        _set(ids["positive"],    "text",     prompt)
        _set(ids["negative"],    "text",     neg_prompt)
        _set(ids["ksampler"],    "steps",    steps)
        _set(ids["ksampler"],    "cfg",      cfg)
        _set(ids["ksampler"],    "denoise",  1.0)
        # Always randomize seed unless caller specified one
        _set(ids["ksampler"], "seed", seed if seed >= 0 else random.randint(0, 2**31 - 1))

        # Strip comment fields ComfyUI doesn't like
        for node in workflow.values():
            if isinstance(node, dict):
                node.pop("_comment", None)

        return workflow

    def _default_workflow(self) -> dict:
        """Return a minimal SD-inpainting workflow if no JSON file is found."""
        # This mirrors workflows/sd_inpaint.json
        return {
            "1":  {"class_type": "CheckpointLoaderSimple",
                   "inputs": {"ckpt_name": "sd-v1-5-inpainting.ckpt"}},
            "4":  {"class_type": "CLIPTextEncode",
                   "inputs": {"text": "", "clip": ["1", 1]}},
            "5":  {"class_type": "CLIPTextEncode",
                   "inputs": {"text": "text, watermark", "clip": ["1", 1]}},
            "6":  {"class_type": "VAEEncodeForInpaint",
                   "inputs": {"pixels": ["10", 0], "vae": ["1", 2],
                               "mask": ["11", 0], "grow_mask_by": 6}},
            "7":  {"class_type": "KSampler",
                   "inputs": {"model": ["1", 0], "positive": ["4", 0],
                               "negative": ["5", 0], "latent_image": ["6", 0],
                               "seed": 42, "steps": 20, "cfg": 7.0,
                               "sampler_name": "euler", "scheduler": "normal",
                               "denoise": 1.0}},
            "8":  {"class_type": "VAEDecode",
                   "inputs": {"samples": ["7", 0], "vae": ["1", 2]}},
            "9":  {"class_type": "SaveImage",
                   "inputs": {"images": ["8", 0], "filename_prefix": "inpainted"}},
            "10": {"class_type": "LoadImage",
                   "inputs": {"image": "source.png", "upload": "image"}},
            "11": {"class_type": "LoadImageMask",
                   "inputs": {"image": "mask.png", "channel": "red", "upload": "image"}},
        }
