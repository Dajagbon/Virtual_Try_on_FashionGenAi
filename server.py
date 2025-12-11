"""
FastAPI server for Virtual Try-On FashionGenAI model.

Exposes an HTTP endpoint that accepts:
- image_url: URL of user's photo
- prompt: Text description of outfit/clothing
- part: 'upper' or 'lower' body clothing
- resolution: 256, 512, 1024, or 2048
- num_steps: Diffusion steps (default 5-10)
- guidance_scale: Guidance scale (default 7.5)
- rembg: Whether to remove background first

Returns:
- output: Base64-encoded PNG image
"""

# Windows multiprocessing fix for torch
import os
if os.name == 'nt':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import torch
import os
import sys
import base64
import io
from PIL import Image
import requests
import warnings
import cv2
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
from diffusers import StableDiffusionInpaintPipeline
from rembg import remove

# Add main directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'main'))

from base_dataset import Normalize_image
from utils.saving_utils import load_checkpoint_mgpu
from networks import U2NET

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Initialize FastAPI app
app = FastAPI(title="Virtual Try-On API", version="1.0.0")

# Global models (loaded once at startup)
device = "cuda" if torch.cuda.is_available() else "cpu"
u2net_model = None
inpainting_pipeline = None

print(f"[INFO] Using device: {device}")


class Parts:
    UPPER = 1
    LOWER = 2


class TryOnRequest(BaseModel):
    image_url: str
    prompt: str
    part: str = "upper"
    resolution: int = 512
    num_steps: int = 5
    guidance_scale: float = 7.5
    rembg: bool = True


class TryOnResponse(BaseModel):
    output: str  # Base64-encoded PNG image


def load_u2net_model():
    """Load U2NET segmentation model."""
    global u2net_model
    if u2net_model is not None:
        return u2net_model
    
    print("[INFO] Loading U2NET segmentation model...")
    checkpoint_path = os.path.join(
        os.path.dirname(__file__), "main", "trained_checkpoint", "../cloth_segm_u2net_latest.pth"
    )
    
    net = U2NET(in_ch=3, out_ch=4)
    net = load_checkpoint_mgpu(net, checkpoint_path)
    net = net.to(device)
    net = net.eval()
    u2net_model = net
    print("[INFO] U2NET model loaded successfully")
    return u2net_model


def load_inpainting_model():
    """Load Stable Diffusion Inpainting pipeline."""
    global inpainting_pipeline
    if inpainting_pipeline is not None:
        return inpainting_pipeline
    
    print("[INFO] Loading Stable Diffusion Inpainting pipeline...")
    inpainting_pipeline = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        revision="fp16",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    ).to(device)
    print("[INFO] Inpainting pipeline loaded successfully")
    return inpainting_pipeline


def download_image(url: str) -> Image.Image:
    """Download image from URL."""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        img = Image.open(io.BytesIO(response.content))
        return img
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download image: {str(e)}")


def change_bg_color(rgba_image: Image.Image, color: str) -> Image.Image:
    """Change background color of RGBA image."""
    new_image = Image.new("RGBA", rgba_image.size, color)
    new_image.paste(rgba_image, (0, 0), rgba_image)
    return new_image.convert("RGB")


def segment_clothing(
    image: Image.Image,
    part: str,
    resolution: int,
    rembg_enabled: bool
) -> Image.Image:
    """Segment clothing region using U2NET."""
    net = load_u2net_model()
    
    # Prepare transforms
    transforms_list = [transforms.ToTensor(), Normalize_image(0.5, 0.5)]
    transform_rgb = transforms.Compose(transforms_list)
    
    # Preprocess image
    img = image.convert("RGB")
    img = img.resize((resolution, resolution))
    
    # Remove background if requested
    if rembg_enabled:
        img_with_bg = remove(img)
        img_with_bg = change_bg_color(img_with_bg, color="GREEN")
        img_with_bg = img_with_bg.convert("RGB")
    else:
        img_with_bg = img
    
    # Segment clothing
    image_tensor = transform_rgb(img_with_bg)
    image_tensor = image_tensor.unsqueeze(0)
    
    with torch.no_grad():
        output_tensor = net(image_tensor.to(device))
        output_tensor = F.log_softmax(output_tensor[0], dim=1)
        output_tensor = torch.max(output_tensor, dim=1, keepdim=True)[1]
        output_tensor = torch.squeeze(output_tensor, dim=0)
        output_tensor = torch.squeeze(output_tensor, dim=0)
    
    output_arr = output_tensor.cpu().numpy()
    
    # Create mask for specific body part
    mask_code = eval(f"Parts.{part.upper()}")
    mask = (output_arr == mask_code)
    output_arr[mask] = 1
    output_arr[~mask] = 0
    output_arr *= 255
    
    mask_pil = Image.fromarray(output_arr.astype("uint8"), mode="L")
    
    return img_with_bg, mask_pil


def generate_tryon(
    image: Image.Image,
    mask: Image.Image,
    prompt: str,
    resolution: int,
    num_steps: int,
    guidance_scale: float
) -> Image.Image:
    """Generate try-on image using Stable Diffusion Inpainting."""
    pipeline = load_inpainting_model()
    
    print(f"[INFO] Generating try-on with prompt: {prompt}")
    
    with torch.no_grad():
        result = pipeline(
            prompt=prompt,
            image=image,
            mask_image=mask,
            width=resolution,
            height=resolution,
            guidance_scale=guidance_scale,
            num_inference_steps=num_steps
        ).images[0]
    
    # Remove background from result
    result = remove(result)
    result = change_bg_color(result, "WHITE")
    
    return result.convert("RGB")


def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()
    return img_base64


@app.on_event("startup")
async def startup_event():
    """Load models on server startup."""
    print("[INFO] Warming up models...")
    try:
        load_u2net_model()
        load_inpainting_model()
        print("[INFO] All models loaded and ready!")
    except Exception as e:
        print(f"[ERROR] Failed to load models: {str(e)}")
        raise


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "device": device,
        "cuda_available": torch.cuda.is_available()
    }


@app.post("/tryon", response_model=TryOnResponse)
async def try_on(request: TryOnRequest):
    """
    Generate virtual try-on image.
    
    Args:
        image_url: URL of user's photo
        prompt: Text description of outfit/clothing
        part: 'upper' or 'lower' body clothing
        resolution: Image resolution (256, 512, 1024, 2048)
        num_steps: Diffusion steps (1-50, default 5)
        guidance_scale: Guidance scale (default 7.5)
        rembg: Whether to remove background (default True)
    
    Returns:
        Base64-encoded PNG image in 'output' field
    """
    try:
        # Validate inputs
        if request.part not in ["upper", "lower"]:
            raise HTTPException(status_code=400, detail="part must be 'upper' or 'lower'")
        
        if request.resolution not in [256, 512, 1024, 2048]:
            raise HTTPException(
                status_code=400,
                detail="resolution must be one of: 256, 512, 1024, 2048"
            )
        
        if not (1 <= request.num_steps <= 50):
            raise HTTPException(status_code=400, detail="num_steps must be between 1 and 50")
        
        print(f"\n[INFO] Processing try-on request:")
        print(f"  - URL: {request.image_url[:50]}...")
        print(f"  - Part: {request.part}")
        print(f"  - Resolution: {request.resolution}")
        print(f"  - Prompt: {request.prompt}")
        
        # Download and process image
        print("[INFO] Downloading image...")
        image = download_image(request.image_url)
        
        # Segment clothing
        print("[INFO] Segmenting clothing region...")
        img_with_bg, mask = segment_clothing(
            image,
            request.part,
            request.resolution,
            request.rembg
        )
        
        # Generate try-on
        print("[INFO] Generating try-on image...")
        result = generate_tryon(
            img_with_bg,
            mask,
            request.prompt,
            request.resolution,
            request.num_steps,
            request.guidance_scale
        )
        
        # Convert to base64
        print("[INFO] Encoding result to base64...")
        base64_image = image_to_base64(result)
        
        print("[INFO] Try-on generation completed successfully!")
        
        return TryOnResponse(output=base64_image)
    
    except HTTPException as e:
        print(f"[ERROR] HTTP error: {e.detail}")
        raise
    except Exception as e:
        print(f"[ERROR] Try-on generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Try-on generation failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    
    # Run server
    # By default, listens on 0.0.0.0:8000
    # For production, use: gunicorn -w 1 -k uvicorn.workers.UvicornWorker server:app
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
