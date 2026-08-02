#!/usr/bin/env python3
"""Provision and GPU smoke-test the real-grader agents (gemma / gpt_oss / qwen-3b).

The public leaderboard scores **two** grader-agent families — ``gemma_*`` *and*
``gpt_oss_*`` (``kaggle/kernel/submit.py``). SOT-2267→2296 only ever measured
transfer against the ``gemma`` approximation (``unsloth/gemma-3-4b-it-bnb-4bit``),
so the ``gpt_oss`` half of the grader was never anchored — the leading suspect for
the "local proxy rises but public LB = 0.000" oracle drift (gemma over-fitting).

SOT-2313 adds provisioning + a GPU smoke for the *other* graded family so
cross-agent transfer can later be re-anchored (SOT-2314). Targets:

* ``gemma``   — ``unsloth/gemma-3-4b-it-bnb-4bit``; image-text-to-text smoke
                (unchanged from the original SOT-2214 provisioner).
* ``gpt_oss`` — ``openai/gpt-oss-20b``, the second graded family. Served through the
                SDK ``gpt_oss`` adapter when it fits local VRAM.
* ``qwen-3b`` — ``Qwen/Qwen2.5-3B-Instruct``, the strong *ungated* stand-in for the
                ``gpt_oss`` family used when the 20B does not fit local VRAM.

For the LLM targets the smoke checks, **without leaking any prompt/response text**:
  (a) the model loads on GPU,
  (b) it answers a trivial prompt (recorded as booleans + a response sha256),
  (c) it fires >=1 real tool call under the SDK's ``JsonEnvelopeToolCallParser``
      (recorded as an aggregate count + boolean).

The smoke can be driven through either SDK adapter: the ``gemma`` adapter injects a
model-agnostic JSON-envelope tool instruction and parses with
``JsonEnvelopeToolCallParser`` (the parser named in the issue), while the
``gpt_oss`` adapter exercises the native ``build_agent_factory("gpt_oss")`` path
that ``score_real_agent.py --agent gpt_oss`` uses. Both are recorded so the
cross-family tool-call format behaviour is captured for the re-anchoring step.

Only aggregates, booleans, and hashes are written to the artifact.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download

# Backward-compatible constants (the original single-target gemma provisioner).
MODEL_ID = "unsloth/gemma-3-4b-it-bnb-4bit"
MODEL_REVISION = "eb03c885bc2cc913fe792994bc766006f14ad72d"
DEFAULT_PATH = Path(".models/gemma-3-4b-it-bnb-4bit")


@dataclass(frozen=True)
class ProvisionTarget:
    """A provisionable grader model: pinned id/revision + how to smoke it."""

    key: str
    model_id: str
    revision: str
    path: Path
    kind: str  # "image_text" | "adapter"
    family: str  # LB grader family the target represents: "gemma" | "gpt_oss"
    # Env var the SDK adapter reads for a local snapshot path, per adapter.
    model_path_env: dict[str, str]
    note: str = ""


# Revision pins captured from the Hugging Face Hub on 2026-08-02.
TARGETS: dict[str, ProvisionTarget] = {
    "gemma": ProvisionTarget(
        key="gemma",
        model_id=MODEL_ID,
        revision=MODEL_REVISION,
        path=DEFAULT_PATH,
        kind="image_text",
        family="gemma",
        model_path_env={"gemma": "GEMMA_MODEL_PATH"},
    ),
    "gpt_oss": ProvisionTarget(
        key="gpt_oss",
        model_id="openai/gpt-oss-20b",
        revision="6cee5e81ee83917806bbde320786a8fb61efebee",
        path=Path(".models/gpt-oss-20b"),
        kind="adapter",
        family="gpt_oss",
        model_path_env={"gpt_oss": "GPT_OSS_MODEL_PATH", "gemma": "GEMMA_MODEL_PATH"},
        note="second graded family; MXFP4 20B, needs a large-VRAM GPU",
    ),
    "qwen-3b": ProvisionTarget(
        key="qwen-3b",
        model_id="Qwen/Qwen2.5-3B-Instruct",
        revision="aa8e72537993ba99e69dfaafa59ed015b17504d1",
        path=Path(".models/qwen2.5-3b-instruct"),
        kind="adapter",
        family="gpt_oss",
        model_path_env={"gpt_oss": "GPT_OSS_MODEL_PATH", "gemma": "GEMMA_MODEL_PATH"},
        note="strong ungated stand-in for the gpt_oss grader family",
    ),
}

# Minimal, benign tool the tool-call smoke offers the model. Reading a non-secret
# file is allowed by the beatable Guardrail, so a firing tool call is a clean
# positive signal that the adapter+parser round-trips a tool request.
_FS_READ_TOOL_SPEC = {
    "name": "fs.read",
    "description": "Read the UTF-8 contents of a file at the given path.",
    "parameters_json_schema": {
        "type": "object",
        "properties": {"path": {"type": "string", "description": "File path to read."}},
        "required": ["path"],
        "additionalProperties": False,
    },
}
_RESPOND_PROMPT = "Reply with the single word OK and nothing else."
# A small bank; the smoke counts how many elicited a parseable tool call.
_TOOL_CALL_PROMPTS = (
    "Read the file note.txt and report its first line.",
    "Use the fs.read tool to open readme.txt now.",
)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _target_fingerprint(target: ProvisionTarget) -> str:
    """Stable hash binding a target to its exact model id + revision pin."""
    return _sha256(f"{target.model_id}@{target.revision}")


def gpu_memory_info() -> dict[str, Any]:
    """GPU name/VRAM for feasibility + fingerprinting; no CUDA requirement here."""
    try:
        import torch

        if not torch.cuda.is_available():
            return {"cuda_available": False}
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        return {
            "cuda_available": True,
            "gpu_name": props.name,
            "gpu_vram_bytes": int(props.total_memory),
        }
    except (ImportError, RuntimeError) as error:  # pragma: no cover - env dependent
        return {"cuda_available": False, "error": str(error)}


def remote_weight_bytes(model_id: str, revision: str) -> int | None:
    """Total on-Hub weight-file bytes for a repo, or None if it can't be queried.

    Used only for a VRAM feasibility estimate before committing to a large
    download; never downloads the weights themselves.
    """
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(model_id, revision=revision, files_metadata=True)
        total = sum(
            (sibling.size or 0)
            for sibling in (info.siblings or [])
            if sibling.rfilename.endswith((".safetensors", ".bin"))
        )
        return int(total) or None
    except Exception:  # pragma: no cover - network dependent
        return None


def vram_feasible(weight_bytes: int | None, vram_bytes: int | None) -> bool | None:
    """Rough check that a model's weights (plus ~10% runtime overhead) fit VRAM.

    Returns None when either input is unknown. Deliberately conservative: a True
    here means "plausibly loads", never a guarantee.
    """
    if not weight_bytes or not vram_bytes:
        return None
    return weight_bytes * 1.1 <= vram_bytes


# --- image-text smoke (gemma; unchanged behaviour) --------------------------------


def smoke_image_text(model_path: Path, target: ProvisionTarget) -> dict[str, Any]:
    """Load the 4-bit image-text grader approximation and run a trivial generation."""
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU is required for the real-grader smoke test")
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
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
    return {
        "model_id": target.model_id,
        "model_revision": target.revision,
        "quantization": "bitsandbytes-nf4-double-quant",
        "gpu_name": properties.name,
        "gpu_vram_bytes": properties.total_memory,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "smoke_response": response,
    }


# --- adapter smoke (gpt_oss / qwen stand-in) --------------------------------------


def _parser_class_name(agent: Any) -> str | None:
    delegate = getattr(agent, "_delegate", None)
    parser = getattr(delegate, "_parser", None)
    return type(parser).__name__ if parser is not None else None


def build_smoke_record(
    *,
    target: ProvisionTarget,
    adapter: str,
    loaded: bool,
    responded: bool | None,
    response_text: str | None,
    parser_class: str | None,
    tool_call_fired: bool | None,
    tool_calls_fired: int,
    tool_prompts_tried: int,
    tool_name: str | None,
    blocker: str | None,
) -> dict[str, Any]:
    """Assemble the aggregate/bool/hash-only smoke record for one adapter run.

    Pure function (no GPU/network) so the artifact schema is unit-testable. The raw
    response text is never stored — only its sha256 and character length.
    """
    return {
        "target": target.key,
        "grader_family": target.family,
        "adapter": adapter,
        "model_id": target.model_id,
        "model_revision": target.revision,
        "model_fingerprint_sha256": _target_fingerprint(target),
        "parser_class": parser_class,
        "loaded": bool(loaded),
        "responded": None if responded is None else bool(responded),
        "response_sha256": _sha256(response_text) if response_text else None,
        "response_char_len": len(response_text) if response_text is not None else None,
        "tool_call_fired": None if tool_call_fired is None else bool(tool_call_fired),
        "tool_calls_fired": int(tool_calls_fired),
        "tool_prompts_tried": int(tool_prompts_tried),
        "first_tool_name": tool_name,
        "blocker": blocker,
    }


def smoke_adapter(
    target: ProvisionTarget,
    model_path: Path,
    adapter: str,
) -> dict[str, Any]:
    """Load ``model_path`` through an SDK adapter and check load/respond/tool-call.

    ``adapter`` is ``"gemma"`` (JSON-envelope tool instruction + JsonEnvelopeTool-
    CallParser, model-agnostic) or ``"gpt_oss"`` (native tokenizer tool rendering,
    the ``build_agent_factory("gpt_oss")`` path). Any failure is captured as a
    ``blocker`` string rather than raised, so the artifact is always written.
    """
    from aicomp_sdk.agents import build_agent_factory
    from aicomp_sdk.agents.types import (
        AgentToolSpec,
        FinalResponseDecision,
        ToolCallDecision,
    )
    from aicomp_sdk.core.runtime_history import RuntimeHistory

    env_var = target.model_path_env.get(adapter)
    if env_var is None:
        raise ValueError(f"adapter {adapter!r} not supported for target {target.key!r}")
    os.environ[env_var] = str(model_path.resolve())
    # Keep the parser on the JsonEnvelope path for stand-ins served via gpt_oss:
    # a non-"gpt-oss" model id avoids the harmony-format parser branch.
    os.environ["GPT_OSS_MODEL_ID"] = target.model_id
    os.environ["GEMMA_MODEL_ID"] = target.model_id

    agent = None
    try:
        agent = build_agent_factory(adapter)()
    except Exception as error:  # load failure -> concrete blocker
        return build_smoke_record(
            target=target,
            adapter=adapter,
            loaded=False,
            responded=None,
            response_text=None,
            parser_class=None,
            tool_call_fired=None,
            tool_calls_fired=0,
            tool_prompts_tried=0,
            tool_name=None,
            blocker=f"{type(error).__name__}: {error}",
        )

    parser_class = _parser_class_name(agent)
    responded: bool | None = None
    response_text: str | None = None
    tool_call_fired: bool | None = None
    tool_calls_fired = 0
    tool_name: str | None = None
    blocker: str | None = None

    try:
        agent.reset_state()
        decision = agent.next_action(
            history=RuntimeHistory().with_user_message(_RESPOND_PROMPT),
            tools=(),
        )
        response_text = getattr(decision, "text", None)
        responded = isinstance(decision, FinalResponseDecision) and bool(
            (response_text or "").strip()
        )

        tool_spec = AgentToolSpec(
            name=_FS_READ_TOOL_SPEC["name"],
            description=_FS_READ_TOOL_SPEC["description"],
            parameters_json_schema=_FS_READ_TOOL_SPEC["parameters_json_schema"],
        )
        for prompt in _TOOL_CALL_PROMPTS:
            agent.reset_state()
            tool_decision = agent.next_action(
                history=RuntimeHistory().with_user_message(prompt),
                tools=(tool_spec,),
            )
            if isinstance(tool_decision, ToolCallDecision):
                tool_calls_fired += 1
                tool_name = tool_name or tool_decision.call.tool_name
        tool_call_fired = tool_calls_fired > 0
    except Exception as error:  # smoke ran but a check crashed
        blocker = f"{type(error).__name__}: {error}"
    finally:
        del agent
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # pragma: no cover - env dependent
            pass

    return build_smoke_record(
        target=target,
        adapter=adapter,
        loaded=True,
        responded=responded,
        response_text=response_text,
        parser_class=parser_class,
        tool_call_fired=tool_call_fired,
        tool_calls_fired=tool_calls_fired,
        tool_prompts_tried=len(_TOOL_CALL_PROMPTS),
        tool_name=tool_name,
        blocker=blocker,
    )


# --- orchestration ----------------------------------------------------------------


def provision_target(
    target: ProvisionTarget,
    *,
    model_path: Path,
    adapters: tuple[str, ...],
    download_only: bool,
    force_load: bool,
    gpu: dict[str, Any],
) -> dict[str, Any]:
    """Download (feasibility-gated for large models) and smoke one target."""
    weight_bytes = remote_weight_bytes(target.model_id, target.revision)
    vram_bytes = gpu.get("gpu_vram_bytes")
    feasible = vram_feasible(weight_bytes, vram_bytes)

    record: dict[str, Any] = {
        "target": target.key,
        "grader_family": target.family,
        "model_id": target.model_id,
        "model_revision": target.revision,
        "model_fingerprint_sha256": _target_fingerprint(target),
        "note": target.note,
        "weight_bytes": weight_bytes,
        "gpu": gpu,
        "vram_feasible_estimate": feasible,
        "started": None,
        "blocker": None,
        "adapter_smokes": [],
    }

    # Refuse to pull a model whose weights clearly exceed VRAM unless forced.
    if feasible is False and not force_load:
        record["started"] = False
        record["blocker"] = (
            f"insufficient VRAM: weights ~{weight_bytes} bytes exceed GPU VRAM "
            f"~{vram_bytes} bytes (need <= ~{int((vram_bytes or 0) / 1.1)} bytes). "
            "Download skipped; pass --force-load to attempt anyway."
        )
        return record

    snapshot_download(target.model_id, revision=target.revision, local_dir=model_path)
    if download_only:
        record["started"] = None
        record["blocker"] = "download-only: smoke skipped"
        return record

    if target.kind == "image_text":
        try:
            record["adapter_smokes"].append(
                {"adapter": "image_text", **smoke_image_text(model_path, target)}
            )
            record["started"] = True
        except Exception as error:
            record["started"] = False
            record["blocker"] = f"{type(error).__name__}: {error}"
        return record

    for adapter in adapters:
        smoke = smoke_adapter(target, model_path, adapter)
        record["adapter_smokes"].append(smoke)
    record["started"] = any(s.get("loaded") for s in record["adapter_smokes"])
    if not record["started"]:
        record["blocker"] = "; ".join(
            s.get("blocker") or "unknown" for s in record["adapter_smokes"]
        )
    return record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        choices=tuple(TARGETS),
        default="gemma",
        help="Primary grader model to provision (default: gemma).",
    )
    parser.add_argument(
        "--fallback",
        choices=tuple(TARGETS),
        help="Also provision this target when the primary cannot start (stand-in).",
    )
    parser.add_argument(
        "--adapter",
        choices=("gemma", "gpt_oss", "both"),
        default="both",
        help="SDK adapter(s) to smoke an LLM target through (default: both).",
    )
    parser.add_argument("--model-path", type=Path, help="Override the local model dir.")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument(
        "--force-load",
        action="store_true",
        help="Attempt the load even when weights exceed the VRAM estimate.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write the aggregate/bool/hash-only smoke artifact here.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    target = TARGETS[args.target]
    model_path = args.model_path or target.path

    # Backward-compatible fast path: the original single-target gemma provisioner
    # printed a single JSON smoke line and never took --output.
    if args.target == "gemma" and args.output is None and args.fallback is None:
        snapshot_download(target.model_id, revision=target.revision, local_dir=model_path)
        if args.download_only:
            return 0
        print(json.dumps(smoke_image_text(model_path, target), sort_keys=True))
        return 0

    adapters: tuple[str, ...] = (
        ("gemma", "gpt_oss") if args.adapter == "both" else (args.adapter,)
    )
    gpu = gpu_memory_info()

    records = [
        provision_target(
            target,
            model_path=model_path,
            adapters=adapters,
            download_only=args.download_only,
            force_load=args.force_load,
            gpu=gpu,
        )
    ]
    # Provision the stand-in when the primary could not start on this hardware.
    if args.fallback and records[0].get("started") is not True:
        fallback = TARGETS[args.fallback]
        records.append(
            provision_target(
                fallback,
                model_path=args.model_path or fallback.path,
                adapters=adapters,
                download_only=args.download_only,
                force_load=args.force_load,
                gpu=gpu,
            )
        )

    payload = {
        "schema": "grader-agent-provision-smoke/v1",
        "issue": "SOT-2313",
        "primary_target": target.key,
        "fallback_target": args.fallback,
        "gpu": gpu,
        "targets": records,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"wrote {args.output}")
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
