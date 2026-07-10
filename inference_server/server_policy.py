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
    return default if value is None else value.strip().lower() in {"1", "true", "yes", "on"}


def required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} must be set before starting the SmolVLA server")
    return value


def camera_map_from_env() -> dict[str, str]:
    raw = required_env("SMOLVLA_CAMERA_MAP")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("SMOLVLA_CAMERA_MAP must be a JSON object") from exc
    if not isinstance(value, dict) or not all(isinstance(key, str) and isinstance(item, str) for key, item in value.items()):
        raise RuntimeError("SMOLVLA_CAMERA_MAP must map model image keys to client image-key strings")
    sources = list(value.values())
    if len(set(sources)) != len(sources):
        raise RuntimeError("SMOLVLA_CAMERA_MAP must use a distinct client camera for each model input")
    return value


MODEL_PATH = Path(required_env("SMOLVLA_MODEL_PATH"))
DEVICE = os.getenv("SMOLVLA_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
TASK_DEFAULT = os.getenv("SMOLVLA_DEFAULT_TASK", "Pick up the red screwdriver and drop it on the yellow container.")
N_ACTION_STEPS = int(os.getenv("SMOLVLA_N_ACTION_STEPS", "10"))
LOAD_ON_STARTUP = env_bool("SMOLVLA_LOAD_ON_STARTUP", True)
CAMERA_MAP = camera_map_from_env()

STATE_KEY = "observation.state"
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


def decode_image(value: Any) -> torch.Tensor:
    if isinstance(value, dict):
        for key in ("jpeg_base64", "base64", "data", "image"):
            if key in value:
                value = value[key]
                break

    if isinstance(value, str):
        if value.strip().lower().startswith("data:"):
            value = value.split(",", 1)[1]
        image = Image.open(io.BytesIO(base64.b64decode(value))).convert("RGB")
        array = np.asarray(image, dtype=np.uint8)
    elif isinstance(value, list):
        array = np.asarray(value)
        if array.dtype != np.uint8:
            scale = 255.0 if array.size and float(np.nanmax(array)) <= 1.0 else 1.0
            array = (array * scale).clip(0, 255).astype(np.uint8)
    else:
        raise ValueError(f"Unsupported image payload type: {type(value)!r}")

    if array.ndim != 3:
        raise ValueError(f"Expected an HWC or CHW colour image, got shape={array.shape}")
    if array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
        chw = array[:3]
    else:
        chw = np.transpose(array[:, :, :3], (2, 0, 1))
    return torch.from_numpy(np.ascontiguousarray(chw)).float().div(255.0)


def get_state(payload: PredictRequest) -> list[float]:
    state: Any = payload.state
    if state is None and payload.observation:
        state = payload.observation.get("state", payload.observation.get(STATE_KEY))
    if state is None and payload.model_extra:
        state = payload.model_extra.get(STATE_KEY)
    if state is None:
        raise ValueError("Missing 6D robot state")
    if isinstance(state, dict):
        state = [state[name] for name in ACTION_NAMES]
    if not isinstance(state, list) or len(state) != len(ACTION_NAMES):
        raise ValueError(f"Expected a 6D robot state, got {state!r}")
    return [float(value) for value in state]


def get_image(payload: PredictRequest, client_key: str, model_key: str) -> Any:
    candidates = (client_key, model_key)
    if payload.images:
        for key in candidates:
            if key in payload.images:
                return payload.images[key]
    if payload.observation:
        images = payload.observation.get("images")
        if isinstance(images, dict):
            for key in candidates:
                if key in images:
                    return images[key]
        for key in candidates:
            if key in payload.observation:
                return payload.observation[key]
    if payload.model_extra:
        for key in candidates:
            if key in payload.model_extra:
                return payload.model_extra[key]
    raise ValueError(f"Missing required camera {client_key!r} for model input {model_key!r}")


def action_chunk_to_list(actions: Any, count: int) -> list[list[float]]:
    array = actions.detach().cpu().float().numpy() if isinstance(actions, torch.Tensor) else np.asarray(actions)
    array = np.squeeze(array)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[1] < len(ACTION_NAMES):
        raise ValueError(f"Policy returned invalid action chunk shape: {array.shape}")
    return [[float(value) for value in row[: len(ACTION_NAMES)]] for row in array[:count]]


def action_to_list(action: Any) -> list[float]:
    array = action.detach().cpu().float().numpy() if isinstance(action, torch.Tensor) else np.asarray(action)
    array = np.asarray(array).squeeze().reshape(-1)
    if array.shape[0] < len(ACTION_NAMES):
        raise ValueError(f"Policy returned invalid action shape: {array.shape}")
    return [float(value) for value in array[: len(ACTION_NAMES)]]


class SmolVLARuntime:
    def __init__(self) -> None:
        self.policy: Any = None
        self.preprocess: Any = None
        self.postprocess: Any = None
        self.visual_inputs: list[str] = []
        self.load_error: str | None = None
        self.load_ms: float | None = None

    @property
    def loaded(self) -> bool:
        return self.policy is not None and self.preprocess is not None and self.postprocess is not None

    def load(self) -> None:
        if self.loaded:
            return
        if not MODEL_PATH.is_dir():
            raise RuntimeError(f"SmolVLA checkpoint directory does not exist: {MODEL_PATH}")
        required_files = ("config.json", "model.safetensors", "policy_preprocessor.json", "policy_postprocessor.json")
        missing = [name for name in required_files if not (MODEL_PATH / name).is_file()]
        if missing:
            raise RuntimeError(f"Checkpoint is missing required files: {missing}")

        started = time.perf_counter()
        try:
            from lerobot.configs import PreTrainedConfig
            from lerobot.policies.factory import get_policy_class, make_pre_post_processors

            config = PreTrainedConfig.from_pretrained(str(MODEL_PATH), local_files_only=True)
            if getattr(config, "type", None) != "smolvla":
                raise RuntimeError(f"Expected a smolvla checkpoint, found {getattr(config, 'type', None)!r}")
            self.visual_inputs = [
                name
                for name, feature in config.input_features.items()
                if str(getattr(feature, "type", "")).upper().endswith("VISUAL")
            ]
            unmapped = [name for name in self.visual_inputs if name not in CAMERA_MAP]
            if unmapped:
                raise RuntimeError(
                    "SMOLVLA_CAMERA_MAP has no source for checkpoint visual input(s): " + ", ".join(unmapped)
                )

            config.device = DEVICE
            if 0 < N_ACTION_STEPS <= config.chunk_size:
                config.n_action_steps = N_ACTION_STEPS
            torch.set_grad_enabled(False)
            policy = get_policy_class("smolvla").from_pretrained(str(MODEL_PATH), config=config, local_files_only=True)
            policy = policy.to(DEVICE).eval()
            policy.config.n_action_steps = min(N_ACTION_STEPS, policy.config.chunk_size)
            preprocess, postprocess = make_pre_post_processors(
                policy.config,
                str(MODEL_PATH),
                preprocessor_overrides={"device_processor": {"device": str(DEVICE)}},
                postprocessor_overrides={"device_processor": {"device": "cpu"}},
            )
            self.policy, self.preprocess, self.postprocess = policy, preprocess, postprocess
            self.load_ms = (time.perf_counter() - started) * 1000.0
            self.load_error = None
        except Exception as exc:
            self.load_error = repr(exc)
            raise

    def build_batch(self, payload: PredictRequest) -> dict[str, Any]:
        batch: dict[str, Any] = {STATE_KEY: torch.tensor(get_state(payload), dtype=torch.float32)}
        for model_key in self.visual_inputs:
            batch[model_key] = decode_image(get_image(payload, CAMERA_MAP[model_key], model_key))
        batch["task"] = payload.task or TASK_DEFAULT
        return batch

    def predict_chunk(self, payload: PredictRequest) -> dict[str, Any]:
        self.load()
        assert self.policy is not None and self.preprocess is not None and self.postprocess is not None
        started = time.perf_counter()
        with torch.inference_mode():
            self.policy.reset()
            actions = self.postprocess(self.policy.predict_action_chunk(self.preprocess(self.build_batch(payload))))
        return {
            "ok": True,
            "actions": action_chunk_to_list(actions, self.policy.config.n_action_steps),
            "action_names": ACTION_NAMES,
            "task": payload.task or TASK_DEFAULT,
            "latency_ms": (time.perf_counter() - started) * 1000.0,
            "n_action_steps": self.policy.config.n_action_steps,
            "model_path": str(MODEL_PATH),
            "policy_type": "smolvla",
            "device": str(DEVICE),
        }

    def predict(self, payload: PredictRequest) -> dict[str, Any]:
        """Return one action; subsequent calls use SmolVLA's cached action chunk."""
        self.load()
        assert self.policy is not None and self.preprocess is not None and self.postprocess is not None
        started = time.perf_counter()
        with torch.inference_mode():
            action = self.postprocess(self.policy.select_action(self.preprocess(self.build_batch(payload))))
        queue = getattr(self.policy, "_queues", {}).get("action", ())
        return {
            "ok": True,
            "action": action_to_list(action),
            "action_names": ACTION_NAMES,
            "task": payload.task or TASK_DEFAULT,
            "latency_ms": (time.perf_counter() - started) * 1000.0,
            "actions_remaining": len(queue),
            "model_path": str(MODEL_PATH),
            "policy_type": "smolvla",
            "device": str(DEVICE),
        }


runtime = SmolVLARuntime()
app = FastAPI(title="SO101 SmolVLA Inference Server", version="1.0")


@app.on_event("startup")
def load_on_startup() -> None:
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
        "policy_type": "smolvla",
        "n_action_steps": N_ACTION_STEPS,
        "camera_map": CAMERA_MAP,
        "checkpoint_visual_inputs": runtime.visual_inputs,
    }


@app.post("/predict_chunk")
def predict_chunk(payload: PredictRequest) -> dict[str, Any]:
    try:
        return runtime.predict_chunk(payload)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=repr(exc)) from exc


@app.post("/predict")
def predict(payload: PredictRequest) -> dict[str, Any]:
    try:
        return runtime.predict(payload)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=repr(exc)) from exc


@app.post("/reset")
def reset() -> dict[str, Any]:
    if runtime.policy is not None:
        runtime.policy.reset()
    return {"ok": True}
