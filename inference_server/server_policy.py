from __future__ import annotations

import base64
import io
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel, ConfigDict


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


MODEL_PATH = Path(
    os.getenv("PI_MODEL_PATH")
    or os.getenv("PI0_MODEL_PATH")
    or os.getenv("PI05_MODEL_PATH")
    or "/data/robot_project/models/pi0_yellow_cube_green_scoop_full_20260708_091450/pretrained_model"
)
POLICY_TYPE = os.getenv("PI_POLICY_TYPE")
DEVICE = os.getenv("PI_DEVICE") or os.getenv("PI0_DEVICE") or os.getenv("PI05_DEVICE") or (
    "cuda" if torch.cuda.is_available() else "cpu"
)
TASK_DEFAULT = (
    os.getenv("PI_DEFAULT_TASK")
    or os.getenv("PI0_DEFAULT_TASK")
    or os.getenv("PI05_DEFAULT_TASK")
    or "Pick up the yellow cube and place it into the green scoop."
)
N_ACTION_STEPS = int(
    os.getenv("PI_N_ACTION_STEPS")
    or os.getenv("PI0_N_ACTION_STEPS")
    or os.getenv("PI05_N_ACTION_STEPS")
    or "10"
)
COMPILE_MODEL = env_bool("PI_COMPILE_MODEL", False)
GRADIENT_CHECKPOINTING = env_bool("PI_GRADIENT_CHECKPOINTING", False)
LOAD_ON_STARTUP = env_bool("PI_LOAD_ON_STARTUP", env_bool("PI05_LOAD_ON_STARTUP", True))

STATE_KEY = "observation.state"
FRONT_KEY = "observation.images.front"
WRIST_KEY = "observation.images.wrist"
ACTION_NAMES = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    task: str | None = None
    state: list[float] | None = None
    images: dict[str, Any] | None = None
    observation: dict[str, Any] | None = None


class PolicyRuntime:
    def __init__(self) -> None:
        self.policy = None
        self.preprocess = None
        self.postprocess = None
        self.policy_type: str | None = None
        self.loaded_at: float | None = None
        self.load_ms: float | None = None
        self.load_error: str | None = None

    @property
    def loaded(self) -> bool:
        return self.policy is not None and self.preprocess is not None and self.postprocess is not None

    def load(self) -> None:
        if self.loaded:
            return

        if not MODEL_PATH.is_dir():
            raise RuntimeError(f"Model directory not found: {MODEL_PATH}")
        required = ["config.json", "model.safetensors", "policy_preprocessor.json", "policy_postprocessor.json"]
        missing = [name for name in required if not (MODEL_PATH / name).exists()]
        if missing:
            raise RuntimeError(f"Checkpoint is missing required files: {missing}")

        start = time.perf_counter()
        try:
            from lerobot.configs import PreTrainedConfig
            from lerobot.policies.factory import make_pre_post_processors
            from lerobot.policies.factory import get_policy_class

            torch.set_grad_enabled(False)
            if torch.cuda.is_available():
                torch.backends.cuda.matmul.allow_tf32 = True

            config = PreTrainedConfig.from_pretrained(str(MODEL_PATH), local_files_only=True)
            policy_type = POLICY_TYPE or getattr(config, "type", None)
            if not policy_type:
                with (MODEL_PATH / "config.json").open("r", encoding="utf-8") as f:
                    policy_type = json.load(f).get("type")
            if not policy_type:
                raise RuntimeError(f"Cannot determine policy type from {MODEL_PATH / 'config.json'}")

            config.device = DEVICE
            if hasattr(config, "compile_model"):
                config.compile_model = COMPILE_MODEL
            if hasattr(config, "gradient_checkpointing"):
                config.gradient_checkpointing = GRADIENT_CHECKPOINTING
            if 0 < N_ACTION_STEPS <= getattr(config, "chunk_size", N_ACTION_STEPS):
                config.n_action_steps = N_ACTION_STEPS

            policy_class = get_policy_class(str(policy_type))
            policy = policy_class.from_pretrained(
                str(MODEL_PATH),
                config=config,
                local_files_only=True,
            ).to(DEVICE).eval()
            if 0 < N_ACTION_STEPS <= policy.config.chunk_size:
                policy.config.n_action_steps = N_ACTION_STEPS
                policy.reset()
            preprocess, postprocess = make_pre_post_processors(
                policy.config,
                str(MODEL_PATH),
                preprocessor_overrides={"device_processor": {"device": str(DEVICE)}},
                postprocessor_overrides={"device_processor": {"device": "cpu"}},
            )

            self.policy = policy
            self.preprocess = preprocess
            self.postprocess = postprocess
            self.policy_type = str(policy_type)
            self.loaded_at = time.time()
            self.load_ms = (time.perf_counter() - start) * 1000.0
            self.load_error = None
        except Exception as exc:
            self.load_error = repr(exc)
            raise

    def predict(self, payload: PredictRequest) -> dict[str, Any]:
        self.load()
        assert self.policy is not None
        assert self.preprocess is not None
        assert self.postprocess is not None

        task = payload.task or TASK_DEFAULT
        start = time.perf_counter()
        action_queue = getattr(self.policy, "_queues", {}).get("action")
        cache_hit = bool(action_queue)
        if cache_hit:
            # This action is already part of the previous SmolVLA chunk. Avoid
            # JPEG decoding and preprocessing until that chunk is exhausted.
            raw_action = action_queue.popleft()
        else:
            batch = build_batch(payload, task)
            with torch.inference_mode():
                processed = self.preprocess(batch)
                raw_action = self.policy.select_action(processed)

        with torch.inference_mode():
            action = self.postprocess(raw_action)
        latency_ms = (time.perf_counter() - start) * 1000.0

        action_array = tensor_to_action_list(action)
        return {
            "ok": True,
            "action": action_array,
            "action_names": ACTION_NAMES,
            "task": task,
            "latency_ms": latency_ms,
            "cache_hit": cache_hit,
            "actions_remaining": len(getattr(self.policy, "_queues", {}).get("action", ())),
            "model_path": str(MODEL_PATH),
            "policy_type": self.policy_type,
            "device": str(DEVICE),
        }

    def predict_chunk(self, payload: PredictRequest) -> dict[str, Any]:
        """Generate one complete action chunk for a real-time client-side queue."""
        self.load()
        assert self.policy is not None
        assert self.preprocess is not None
        assert self.postprocess is not None

        task = payload.task or TASK_DEFAULT
        batch = build_batch(payload, task)
        start = time.perf_counter()
        with torch.inference_mode():
            # Chunk mode owns the policy action queue, so a previous request
            # can never leak actions into the next visual replan.
            self.policy.reset()
            processed = self.preprocess(batch)
            raw_actions = self.policy.predict_action_chunk(processed)
            actions = self.postprocess(raw_actions)
        latency_ms = (time.perf_counter() - start) * 1000.0

        return {
            "ok": True,
            "actions": tensor_to_action_chunk(actions, N_ACTION_STEPS),
            "task": task,
            "latency_ms": latency_ms,
            "n_action_steps": N_ACTION_STEPS,
            "model_path": str(MODEL_PATH),
            "policy_type": self.policy_type,
            "device": str(DEVICE),
        }


runtime = PolicyRuntime()
app = FastAPI(title="SO101 Policy Inference Server", version="1.1")


def decode_image(value: Any) -> torch.Tensor:
    if isinstance(value, dict):
        for key in ("jpeg_base64", "base64", "data", "image"):
            if key in value:
                value = value[key]
                break

    if isinstance(value, str):
        if "," in value and value.strip().lower().startswith("data:"):
            value = value.split(",", 1)[1]
        raw = base64.b64decode(value)
        image = Image.open(io.BytesIO(raw)).convert("RGB")
        arr = np.asarray(image, dtype=np.uint8)
    elif isinstance(value, list):
        arr = np.asarray(value)
        if arr.dtype != np.uint8:
            max_value = float(np.nanmax(arr)) if arr.size else 0.0
            if max_value <= 1.0:
                arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
            else:
                arr = arr.clip(0, 255).astype(np.uint8)
    else:
        raise ValueError(f"Unsupported image payload type: {type(value)!r}")

    if arr.ndim != 3:
        raise ValueError(f"Expected image with 3 dimensions, got shape={arr.shape}")
    if arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        chw = arr[:3]
    else:
        chw = np.transpose(arr[:, :, :3], (2, 0, 1))
    return torch.from_numpy(np.ascontiguousarray(chw)).float().div(255.0)


def get_image_payload(payload: PredictRequest, short_name: str, full_key: str, aliases: tuple[str, ...] = ()) -> Any:
    names = (short_name, full_key, *aliases)
    if payload.images:
        for name in names:
            if name in payload.images:
                return payload.images[name]

    if payload.observation:
        obs_images = payload.observation.get("images")
        if isinstance(obs_images, dict):
            for name in names:
                if name in obs_images:
                    return obs_images[name]
        for name in names:
            if name in payload.observation:
                return payload.observation[name]

    extra = payload.model_extra or {}
    for name in names:
        if name in extra:
            return extra[name]

    raise ValueError(f"Missing image payload for {short_name}/{full_key}")


def get_state(payload: PredictRequest) -> list[float]:
    if payload.state is not None:
        state = payload.state
    elif payload.observation and "state" in payload.observation:
        state = payload.observation["state"]
    elif payload.observation and STATE_KEY in payload.observation:
        state = payload.observation[STATE_KEY]
    elif payload.model_extra and STATE_KEY in payload.model_extra:
        state = payload.model_extra[STATE_KEY]
    else:
        raise ValueError("Missing 6D robot state")

    if isinstance(state, dict):
        state = [state[name] for name in ACTION_NAMES]
    if len(state) != 6:
        raise ValueError(f"Expected state length 6, got {len(state)}")
    return [float(v) for v in state]


def build_batch(payload: PredictRequest, task: str) -> dict[str, Any]:
    return {
        STATE_KEY: torch.tensor(get_state(payload), dtype=torch.float32),
        FRONT_KEY: decode_image(
            get_image_payload(payload, "front", FRONT_KEY, aliases=("top", "observation.images.top"))
        ),
        WRIST_KEY: decode_image(get_image_payload(payload, "wrist", WRIST_KEY)),
        "task": task,
    }


def tensor_to_action_list(action: Any) -> list[float]:
    if isinstance(action, torch.Tensor):
        arr = action.detach().cpu().float().numpy()
    else:
        arr = np.asarray(action, dtype=np.float32)

    arr = np.squeeze(arr)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    if arr.shape[0] < 6:
        raise ValueError(f"Policy returned too few action values: shape={arr.shape}")
    return [float(v) for v in arr[:6]]


def tensor_to_action_chunk(actions: Any, count: int) -> list[list[float]]:
    if isinstance(actions, torch.Tensor):
        arr = actions.detach().cpu().float().numpy()
    else:
        arr = np.asarray(actions, dtype=np.float32)

    arr = np.squeeze(arr)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[1] < len(ACTION_NAMES):
        raise ValueError(f"Policy returned invalid action chunk shape: {arr.shape}")
    return [[float(v) for v in row[: len(ACTION_NAMES)]] for row in arr[:count]]


def gpu_status() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"cuda": False}
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return {
        "cuda": True,
        "name": torch.cuda.get_device_name(0),
        "allocated_gb": torch.cuda.memory_allocated() / 1024**3,
        "reserved_gb": torch.cuda.memory_reserved() / 1024**3,
        "free_gb": free_bytes / 1024**3,
        "total_gb": total_bytes / 1024**3,
    }


@app.on_event("startup")
def startup_load_policy() -> None:
    if LOAD_ON_STARTUP:
        runtime.load()


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": runtime.loaded and runtime.load_error is None,
        "loaded": runtime.loaded,
        "load_ms": runtime.load_ms,
        "load_error": runtime.load_error,
        "model_path": str(MODEL_PATH),
        "policy_type": runtime.policy_type or POLICY_TYPE,
        "task_default": TASK_DEFAULT,
        "n_action_steps": N_ACTION_STEPS,
        "device": str(DEVICE),
        "gpu": gpu_status(),
    }


@app.post("/predict")
def predict(payload: PredictRequest) -> dict[str, Any]:
    try:
        return runtime.predict(payload)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=repr(exc)) from exc


@app.post("/predict_chunk")
def predict_chunk(payload: PredictRequest) -> dict[str, Any]:
    try:
        return runtime.predict_chunk(payload)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=repr(exc)) from exc


@app.post("/reset")
def reset() -> dict[str, Any]:
    if runtime.policy is None:
        return {"ok": True, "note": "policy not loaded yet"}
    runtime.policy.reset()
    return {"ok": True, "n_action_steps": runtime.policy.config.n_action_steps}


@app.post("/warmup")
def warmup() -> dict[str, Any]:
    black = np.zeros((480, 640, 3), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(black).save(buffer, format="JPEG", quality=90)
    image_b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
    payload = PredictRequest(
        task=TASK_DEFAULT,
        state=[0.0, 0.0, 0.0, 0.0, 0.0, 50.0],
        images={"front": image_b64, "wrist": image_b64},
    )
    return runtime.predict(payload)
