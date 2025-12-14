"""Local smoke test for rp_handler.py.
- Loads models via load_model()
- Seeds runpod context
- Invokes handler() with test_input.json
"""
import json
import runpod
from rp_handler import handler, load_model, _LOCAL_CONTEXT

# Prevent server start if runpod.serverless.start is called indirectly
runpod.serverless.start = lambda cfg: None

# Load models and seed local context fallback used by handler
context = load_model()
_LOCAL_CONTEXT.update(context)

with open("test_input.json", "r", encoding="utf-8") as f:
    job = json.load(f)

result = handler(job)
print(json.dumps(result, indent=2))
