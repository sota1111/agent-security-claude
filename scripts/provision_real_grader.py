#!/usr/bin/env python3
"""Provision and smoke-test the 4-bit Gemma 3 grading-model approximation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download

MODEL_ID = "unsloth/gemma-3-4b-it-bnb-4bit"
MODEL_REVISION = "eb03c885bc2cc913fe792994bc766006f14ad72d"
DEFAULT_PATH = Path(".models/gemma-3-4b-it-bnb-4bit")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_PATH)
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()

    snapshot_download(
        MODEL_ID,
        revision=MODEL_REVISION,
        local_dir=args.model_path,
    )
    if args.download_only:
        return 0

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU is required for the real-grader smoke test")
    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        device_map="auto",
        local_files_only=True,
    )
    inputs = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "text", "text": "Reply only OK."}]}],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=8, do_sample=False)
    response = processor.decode(
        generated[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True
    ).strip()
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    print(json.dumps({
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "quantization": "bitsandbytes-nf4-double-quant",
        "gpu_name": properties.name,
        "gpu_vram_bytes": properties.total_memory,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "smoke_response": response,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
