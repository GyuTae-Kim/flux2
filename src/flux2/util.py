import base64
import io
import os
import shutil
import sys
from pathlib import Path

import huggingface_hub
import torch
from PIL import Image
from safetensors.torch import load_file as load_sft

from .autoencoder import AutoEncoder, AutoEncoderParams
from .model import Flux2, Flux2Params, Klein4BParams, Klein9BParams
from .text_encoder import load_mistral_small_embedder, load_qwen3_embedder

KLEIN4B_TEXT_ENCODER_GGUF_REPO = "Cordux/flux2-klein-4B-uncensored-text-encoder"
KLEIN4B_TEXT_ENCODER_GGUF_FILENAME = "qwen3-4b-abl-q4_0.gguf"
KLEIN4B_TEXT_ENCODER_GGUF_LOCAL_DEFAULT = "models/text_encoders/qwen3-4b-abl-q4_0.gguf"
KLEIN4B_TEXT_ENCODER_GGUF_LOCAL_ALT = "models/unet/qwen3-4b-abl-q4_0.gguf"
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_local_path(path: str) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = _PROJECT_ROOT / candidate
    return candidate


def resolve_klein4b_text_encoder_gguf_path() -> str:
    env_override = os.environ.get("KLEIN4B_TEXT_ENCODER_GGUF_PATH")
    candidate_paths = []
    if env_override:
        candidate_paths.append(_resolve_local_path(env_override))
    candidate_paths.extend(
        [
            _resolve_local_path(KLEIN4B_TEXT_ENCODER_GGUF_LOCAL_DEFAULT),
            _resolve_local_path(KLEIN4B_TEXT_ENCODER_GGUF_LOCAL_ALT),
        ]
    )

    for candidate in candidate_paths:
        if candidate.exists():
            print(f"Using Klein 4B GGUF text encoder file: {candidate}")
            return str(candidate)

    target_path = candidate_paths[0]
    target_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        downloaded = huggingface_hub.hf_hub_download(
            repo_id=KLEIN4B_TEXT_ENCODER_GGUF_REPO,
            filename=KLEIN4B_TEXT_ENCODER_GGUF_FILENAME,
            repo_type="model",
        )
    except Exception:
        print(
            "Klein 4B GGUF text encoder file was not found locally and download failed. "
            f"Place {KLEIN4B_TEXT_ENCODER_GGUF_FILENAME} in "
            f"{_resolve_local_path(KLEIN4B_TEXT_ENCODER_GGUF_LOCAL_DEFAULT)} "
            "or set KLEIN4B_TEXT_ENCODER_GGUF_PATH.",
            file=sys.stderr,
        )
        raise

    downloaded_path = Path(downloaded)
    if downloaded_path.resolve() != target_path.resolve():
        shutil.copy2(downloaded_path, target_path)

    print(f"Using Klein 4B GGUF text encoder file: {target_path}")
    return str(target_path)


FLUX2_MODEL_INFO = {
    "flux.2-klein-4b": {
        "repo_id": "black-forest-labs/FLUX.2-klein-4B",
        "ae_repo_id": "black-forest-labs/FLUX.2-dev",
        "filename": "flux-2-klein-4b.safetensors",
        "filename_ae": "ae.safetensors",
        "params": Klein4BParams(),
        "text_encoder_load_fn": lambda device="cuda": load_qwen3_embedder(
            model_spec=KLEIN4B_TEXT_ENCODER_GGUF_REPO,
            gguf_file=KLEIN4B_TEXT_ENCODER_GGUF_FILENAME,
            tokenizer_spec="Qwen/Qwen3-4B",
            device=device,
        ),
        "model_path": "KLEIN_4B_MODEL_PATH",
        "defaults": {"guidance": 1.0, "num_steps": 4},
        "fixed_params": {"guidance", "num_steps"},
        "guidance_distilled": True,
    },
    "flux.2-klein-9b": {
        "repo_id": "black-forest-labs/FLUX.2-klein-9B",
        "ae_repo_id": "black-forest-labs/FLUX.2-dev",
        "filename": "flux-2-klein-9b.safetensors",
        "filename_ae": "ae.safetensors",
        "params": Klein9BParams(),
        "text_encoder_load_fn": lambda device="cuda": load_qwen3_embedder(variant="8B", device=device),
        "model_path": "KLEIN_9B_MODEL_PATH",
        "defaults": {"guidance": 1.0, "num_steps": 4},
        "fixed_params": {"guidance", "num_steps"},
        "guidance_distilled": True,
    },
    "flux.2-klein-base-4b": {
        "repo_id": "black-forest-labs/FLUX.2-klein-base-4B",
        "ae_repo_id": "black-forest-labs/FLUX.2-dev",
        "filename": "flux-2-klein-base-4b.safetensors",
        "filename_ae": "ae.safetensors",
        "params": Klein4BParams(),
        "text_encoder_load_fn": lambda device="cuda": load_qwen3_embedder(
            model_spec=KLEIN4B_TEXT_ENCODER_GGUF_REPO,
            gguf_file=KLEIN4B_TEXT_ENCODER_GGUF_FILENAME,
            tokenizer_spec="Qwen/Qwen3-4B",
            device=device,
        ),
        "model_path": "KLEIN_4B_BASE_MODEL_PATH",
        "defaults": {"guidance": 4.0, "num_steps": 50},
        "fixed_params": {},
        "guidance_distilled": False,
    },
    "flux.2-klein-base-9b": {
        "repo_id": "black-forest-labs/FLUX.2-klein-base-9B",
        "ae_repo_id": "black-forest-labs/FLUX.2-dev",
        "filename": "flux-2-klein-base-9b.safetensors",
        "filename_ae": "ae.safetensors",
        "params": Klein9BParams(),
        "text_encoder_load_fn": lambda device="cuda": load_qwen3_embedder(variant="8B", device=device),
        "model_path": "KLEIN_9B_BASE_MODEL_PATH",
        "defaults": {"guidance": 4.0, "num_steps": 50},
        "fixed_params": {},
        "guidance_distilled": False,
    },
    "flux.2-dev": {
        "repo_id": "black-forest-labs/FLUX.2-dev",
        "ae_repo_id": "black-forest-labs/FLUX.2-dev",
        "filename": "flux2-dev.safetensors",
        "filename_ae": "ae.safetensors",
        "params": Flux2Params(),
        "text_encoder_load_fn": load_mistral_small_embedder,
        "model_path": "FLUX2_MODEL_PATH",
        "defaults": {"guidance": 4.0, "num_steps": 50},
        "fixed_params": {},
        "guidance_distilled": True,
    },
}


def load_flow_model(model_name: str, debug_mode: bool = False, device: str | torch.device = "cuda") -> Flux2:
    config = FLUX2_MODEL_INFO[model_name.lower()]

    if debug_mode:
        config["params"].depth = 1
        config["params"].depth_single_blocks = 1
    else:
        if config["model_path"] in os.environ:
            weight_path = os.environ[config["model_path"]]
            assert os.path.exists(weight_path), f"Provided weight path {weight_path} does not exist"
        else:
            # download from huggingface
            try:
                weight_path = huggingface_hub.hf_hub_download(
                    repo_id=config["repo_id"],
                    filename=config["filename"],
                    repo_type="model",
                )
            except huggingface_hub.errors.RepositoryNotFoundError:
                print(
                    f"Failed to access the model repository. Please check your internet "
                    f"connection and make sure you've access to {config['repo_id']}."
                    "Stopping."
                )
                sys.exit(1)

    if not debug_mode:
        with torch.device("meta"):
            model = Flux2(FLUX2_MODEL_INFO[model_name.lower()]["params"]).to(torch.bfloat16)
        print(f"Loading {weight_path} for the FLUX.2 weights")
        sd = load_sft(weight_path, device=str(device))
        if "img_in.weight" not in sd and any(k.startswith("model.diffusion_model.") for k in sd):
            print("Detected ComfyUI key format; remapping state_dict keys")
            prefix = "model.diffusion_model."
            sd = {k[len(prefix) :]: v for k, v in sd.items() if k.startswith(prefix)}
        model.load_state_dict(sd, strict=True, assign=True)
        return model.to(device)
    else:
        with torch.device(device):
            return Flux2(FLUX2_MODEL_INFO[model_name.lower()]["params"]).to(torch.bfloat16)


def load_text_encoder(model_name: str, device: str | torch.device = "cuda"):
    config = FLUX2_MODEL_INFO[model_name.lower()]
    return config["text_encoder_load_fn"](device=device)


def load_ae(model_name: str, device: str | torch.device = "cuda") -> AutoEncoder:
    config = FLUX2_MODEL_INFO[model_name.lower()]
    ae_repo_id = config.get("ae_repo_id", config["repo_id"])

    if "AE_MODEL_PATH" in os.environ:
        weight_path = os.environ["AE_MODEL_PATH"]
        assert os.path.exists(weight_path), f"Provided weight path {weight_path} does not exist"
    else:
        # download from huggingface
        try:
            weight_path = huggingface_hub.hf_hub_download(
                repo_id=ae_repo_id,
                filename=config["filename_ae"],
                repo_type="model",
            )
        except huggingface_hub.errors.EntryNotFoundError:
            fallback_repo_id = "black-forest-labs/FLUX.2-dev"
            if ae_repo_id != fallback_repo_id:
                print(
                    f"AutoEncoder weights not found in {ae_repo_id}. "
                    f"Falling back to {fallback_repo_id}."
                )
                weight_path = huggingface_hub.hf_hub_download(
                    repo_id=fallback_repo_id,
                    filename=config["filename_ae"],
                    repo_type="model",
                )
            else:
                raise
        except huggingface_hub.errors.RepositoryNotFoundError:
            print(
                f"Failed to access the model repository. Please check your internet "
                f"connection and make sure you've access to {ae_repo_id}."
                "Stopping."
            )
            sys.exit(1)

    if isinstance(device, str):
        device = torch.device(device)
    with torch.device("meta"):
        ae = AutoEncoder(AutoEncoderParams())

    print(f"Loading {weight_path} for the AutoEncoder weights")
    sd = load_sft(weight_path, device=str(device))
    ae.load_state_dict(sd, strict=True, assign=True)
    return ae.to(device)


def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str
