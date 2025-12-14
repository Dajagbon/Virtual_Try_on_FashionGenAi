"""RunPod serverless handler for Virtual Try-On (StableVITON style).
- load_model() runs once on cold start: load heavy models to GPU/CPU.
- handler(job) runs per request: download assets, run inference, upload result, return URL.
Replace TODOs with your actual StableVITON loading/inference code.
"""

import os
import io
import json
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

import requests
import runpod

# Local fallback context store (used when rp_context is unavailable)
_LOCAL_CONTEXT = {}

try:
    from runpod.serverless.utils.rp_context import get_context
except Exception:
    def get_context():
        return _LOCAL_CONTEXT

# Optional: set larger request timeout for downloads
requests.adapters.DEFAULT_RETRIES = 3


def load_model() -> Dict[str, Any]:
    """Load all heavy models once per worker cold start."""
    print("[load_model] Loading StableVITON pipeline into memory...")

    # TODO: replace with actual weights/paths
    # Example placeholders
    stable_viton_model = "<stable_viton_model_object>"
    preprocessor = "<preprocessor_object>"  # e.g., DensePose + segmentation

    # TODO: load real model objects here
    # stable_viton_model = StableVITON.load_from_checkpoint(...)  # pseudo-code
    # preprocessor = Preprocessor(...)

    print("[load_model] Models ready.")
    return {
        "StableVITON_Model": stable_viton_model,
        "PreProcessor": preprocessor,
    }


def _download_image(url: str) -> bytes:
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    resp = requests.get(url, timeout=30, headers=headers)
    resp.raise_for_status()
    return resp.content


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """RunPod per-request handler for StableVITON VTON.
    
    Simplified input format:
    - human_img: Full body photo URL (user's body photo)
    - cloth_img: Clothing item URL (garment to try on)
    - category: Garment category (upper_body, lower_body, dresses)
    """
    job_input = job.get("input", {}) or {}

    human_img_url = job_input.get("human_img")
    cloth_img_url = job_input.get("cloth_img")
    category = job_input.get("category", "upper_body")

    if not human_img_url or not cloth_img_url:
        return {
            "error": "Missing required inputs: human_img, cloth_img"
        }

    # Get preloaded models
    model_ctx = get_context() or _LOCAL_CONTEXT or {}
    vton_model = model_ctx.get("StableVITON_Model")
    preprocessor = model_ctx.get("PreProcessor")

    # Fallback stubs for local smoke tests
    if vton_model is None:
        vton_model = "<dummy_model>"
    if preprocessor is None:
        preprocessor = "<dummy_preprocessor>"

    try:
        # 1) Download human and garment images
        print("[handler] Downloading full body photo and garment image...")
        human_bytes = _download_image(human_img_url)
        garment_bytes = _download_image(cloth_img_url)

        # 2) Preprocess - Generate pose and mask automatically from human image
        # TODO: replace with real preprocessing
        # pose_map, agnostic_mask = preprocessor.generate_conditions(human_bytes, category)

        # 3) Inference with StableVITON
        # TODO: replace with real inference call
        # result_image = vton_model.generate(
        #     human=human_bytes,
        #     garment=garment_bytes,
        #     category=category
        # )
        
        # For now, return garment bytes as placeholder output
        result_image_bytes = garment_bytes

        # 4) Upload result to Supabase Storage
        final_image_url = _upload_to_supabase(result_image_bytes, "")

        # If upload disabled or failed, fall back to stub URL
        if not final_image_url:
            final_image_url = "mock://not-implemented"

        return {"output_url": final_image_url}
    except Exception as e:
        return {"error": f"inference failed: {e}"}


# Start the worker; load_model runs once per cold start
if __name__ == "__main__":
    runpod.serverless.start({
        "handler": handler,
        "concurrency_controller": load_model,
    })


# -----------------------------
# Supabase upload helper
# -----------------------------

def _upload_to_supabase(image_bytes: bytes, explicit_path: str = "") -> Optional[str]:
    """Upload image bytes to Supabase Storage and return the public URL.

    Requires env vars:
    - SUPABASE_URL
    - SUPABASE_SERVICE_ROLE_KEY (or a scoped service key with Storage write)
    - SUPABASE_BUCKET
    Optional:
    - OUTPUT_PREFIX (path prefix inside the bucket)
    """

    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    bucket = os.getenv("SUPABASE_BUCKET")
    prefix = os.getenv("OUTPUT_PREFIX", "")

    if not supabase_url or not supabase_key or not bucket:
        return None

    if not image_bytes:
        return None

    # Build object path
    if explicit_path:
        object_path = explicit_path.lstrip("/")
    else:
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        unique = uuid.uuid4().hex[:10]
        object_path = f"{prefix}/{timestamp}_{unique}.png" if prefix else f"{timestamp}_{unique}.png"
    object_path = object_path.replace("//", "/")

    upload_url = f"{supabase_url}/storage/v1/object/{bucket}/{object_path}"

    headers = {
        "Authorization": f"Bearer {supabase_key}",
        "Content-Type": "image/png",
        "x-upsert": "true",
    }

    try:
        resp = requests.post(upload_url, data=image_bytes, headers=headers, timeout=60)
        resp.raise_for_status()
    except Exception as exc:
        print(f"[upload] failed to upload to Supabase: {exc}")
        return None

    # Public URL pattern (if bucket is public). For RLS buckets, generate signed URLs separately.
    return f"{supabase_url}/storage/v1/object/public/{bucket}/{object_path}"
