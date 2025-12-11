#!/usr/bin/env python
"""Test script to verify torch and models load correctly."""

import os
import sys

# Windows torch fix
if os.name == 'nt':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

print("[TEST] Python version:", sys.version)
print("[TEST] Python executable:", sys.executable)

try:
    print("[TEST] Importing torch...")
    import torch
    print("[TEST] Torch version:", torch.__version__)
    print("[TEST] CUDA available:", torch.cuda.is_available())
    print("[TEST] ✓ Torch loaded successfully")
except Exception as e:
    print(f"[ERROR] Failed to import torch: {e}")
    sys.exit(1)

try:
    print("[TEST] Importing FastAPI...")
    from fastapi import FastAPI
    from uvicorn import __version__ as uvicorn_version
    print("[TEST] FastAPI loaded successfully")
    print("[TEST] Uvicorn version:", uvicorn_version)
except Exception as e:
    print(f"[ERROR] Failed to import FastAPI: {e}")
    sys.exit(1)

print("[TEST] ✓ All imports successful!")
print("[TEST] Ready to run server.py")
