import base64
import io
import random
import threading
from pathlib import Path

import torch
from einops import rearrange
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fire import Fire
from pydantic import BaseModel, Field
from PIL import ExifTags, Image

from flux2.sampling import batched_prc_img, batched_prc_txt, denoise, denoise_cfg, get_schedule, scatter_ids
from flux2.util import FLUX2_MODEL_INFO, load_ae, load_flow_model, load_text_encoder

HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>FLUX.2 Web</title>
  <style>
    body { font-family: sans-serif; max-width: 960px; margin: 24px auto; padding: 0 12px; }
    textarea { width: 100%; min-height: 88px; }
    input { width: 120px; margin-right: 8px; }
    .row { margin: 10px 0; }
    .muted { color: #555; font-size: 14px; }
    button { padding: 8px 14px; cursor: pointer; }
    img { max-width: 100%; border: 1px solid #ddd; margin-top: 14px; }
    pre { background: #f7f7f7; padding: 10px; overflow-x: auto; }
  </style>
</head>
<body>
  <h1>FLUX.2 Web Server</h1>
  <div class="muted">Model runs on the server GPU. This page calls <code>POST /generate</code>.</div>
  <div class="row">
    <label for="prompt">Prompt</label><br/>
    <textarea id="prompt">a cinematic photo of a glass observatory in the snow at blue hour, ultra detailed</textarea>
  </div>
  <div class="row">
    <label>Width</label>
    <input id="width" type="number" value="1360" min="256" step="16" />
    <label>Height</label>
    <input id="height" type="number" value="768" min="256" step="16" />
    <label>Steps</label>
    <input id="steps" type="number" value="4" min="1" />
    <label>Guidance</label>
    <input id="guidance" type="number" value="1.0" step="0.1" />
  </div>
  <div class="row">
    <label>Seed (optional)</label>
    <input id="seed" type="number" />
    <button onclick="run()">Generate</button>
  </div>
  <div id="status" class="muted"></div>
  <pre id="meta"></pre>
  <img id="result" />
  <script>
    async function run() {
      const prompt = document.getElementById("prompt").value;
      const width = Number(document.getElementById("width").value);
      const height = Number(document.getElementById("height").value);
      const num_steps = Number(document.getElementById("steps").value);
      const guidance = Number(document.getElementById("guidance").value);
      const seedRaw = document.getElementById("seed").value.trim();
      const body = { prompt, width, height, num_steps, guidance };
      if (seedRaw !== "") body.seed = Number(seedRaw);
      document.getElementById("status").textContent = "Generating...";
      document.getElementById("meta").textContent = "";
      try {
        const res = await fetch("/generate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body)
        });
        const payload = await res.json();
        if (!res.ok) throw new Error(payload.detail || "Request failed");
        document.getElementById("status").textContent = "Done";
        document.getElementById("meta").textContent = JSON.stringify({
          seed: payload.seed,
          output_path: payload.output_path
        }, null, 2);
        document.getElementById("result").src = "data:image/png;base64," + payload.image_base64;
      } catch (err) {
        document.getElementById("status").textContent = "Error: " + err.message;
      }
    }
  </script>
</body>
</html>
"""


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1)
    width: int = Field(default=1360, ge=256)
    height: int = Field(default=768, ge=256)
    num_steps: int | None = Field(default=None, ge=1)
    guidance: float | None = Field(default=None)
    seed: int | None = None


def _validate_dims(width: int, height: int) -> None:
    if width % 16 != 0 or height % 16 != 0:
        raise HTTPException(status_code=400, detail="width/height must be multiples of 16")


def _validate_model_params(model_name: str, num_steps: int, guidance: float) -> None:
    model_info = FLUX2_MODEL_INFO[model_name]
    defaults = model_info.get("defaults", {})
    fixed_params = model_info.get("fixed_params", set())

    if "num_steps" in fixed_params and num_steps != defaults["num_steps"]:
        raise HTTPException(status_code=400, detail=f"Model requires num_steps={defaults['num_steps']}")
    if "guidance" in fixed_params and guidance != defaults["guidance"]:
        raise HTTPException(status_code=400, detail=f"Model requires guidance={defaults['guidance']}")


def create_app(
    model_name: str = "flux.2-klein-9b",
    device: str = "cuda:0",
):
    model_name = model_name.lower()
    if model_name not in FLUX2_MODEL_INFO:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(FLUX2_MODEL_INFO.keys())}")

    torch_device = torch.device(device)
    model_info = FLUX2_MODEL_INFO[model_name]
    defaults = model_info.get("defaults", {})
    default_num_steps = defaults.get("num_steps", 50)
    default_guidance = defaults.get("guidance", 4.0)

    print(f"Loading text encoder on {torch_device}")
    text_encoder = load_text_encoder(model_name, device=torch_device)
    print(f"Loading model weights for {model_name} on {torch_device}")
    model = load_flow_model(model_name, device=torch_device)
    print("Loading autoencoder")
    ae = load_ae(model_name, device=torch_device)

    text_encoder.eval()
    model.eval()
    ae.eval()

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    infer_lock = threading.Lock()

    app = FastAPI(title="FLUX.2 Web Server", version="0.1.0")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return HTML_PAGE

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "model": model_name,
            "device": str(torch_device),
        }

    @app.post("/generate")
    def generate(req: GenerateRequest):
        if not infer_lock.acquire(blocking=False):
            raise HTTPException(status_code=429, detail="Server is busy. Try again shortly.")
        try:
            _validate_dims(req.width, req.height)
            num_steps = req.num_steps if req.num_steps is not None else default_num_steps
            guidance = req.guidance if req.guidance is not None else default_guidance
            _validate_model_params(model_name, num_steps=num_steps, guidance=guidance)
            seed = req.seed if req.seed is not None else random.randrange(2**31)

            with torch.no_grad():
                if model_info["guidance_distilled"]:
                    ctx = text_encoder([req.prompt]).to(torch.bfloat16)
                else:
                    ctx_empty = text_encoder([""]).to(torch.bfloat16)
                    ctx_prompt = text_encoder([req.prompt]).to(torch.bfloat16)
                    ctx = torch.cat([ctx_empty, ctx_prompt], dim=0)
                ctx, ctx_ids = batched_prc_txt(ctx)

                shape = (1, 128, req.height // 16, req.width // 16)
                generator = torch.Generator(device=str(torch_device)).manual_seed(seed)
                randn = torch.randn(shape, generator=generator, dtype=torch.bfloat16, device=torch_device)
                x, x_ids = batched_prc_img(randn)

                timesteps = get_schedule(num_steps, x.shape[1])
                if model_info["guidance_distilled"]:
                    x = denoise(
                        model,
                        x,
                        x_ids,
                        ctx,
                        ctx_ids,
                        timesteps=timesteps,
                        guidance=guidance,
                        img_cond_seq=[],
                        img_cond_seq_ids=[],
                    )
                else:
                    x = denoise_cfg(
                        model,
                        x,
                        x_ids,
                        ctx,
                        ctx_ids,
                        timesteps=timesteps,
                        guidance=guidance,
                        img_cond_seq=[],
                        img_cond_seq_ids=[],
                    )
                x = torch.cat(scatter_ids(x, x_ids)).squeeze(2)
                x = ae.decode(x).float()

            x = x.clamp(-1, 1)
            x = rearrange(x[0], "c h w -> h w c")
            img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())

            output_name = output_dir / f"web_{len(list(output_dir.glob('web_*.png')))}.png"
            exif_data = Image.Exif()
            exif_data[ExifTags.Base.Software] = "AI generated;flux2-web"
            exif_data[ExifTags.Base.Make] = "Black Forest Labs"
            img.save(output_name, exif=exif_data, quality=95, subsampling=0)

            buf = io.BytesIO()
            img.save(buf, format="PNG")
            image_base64 = base64.b64encode(buf.getvalue()).decode()

            return {
                "seed": seed,
                "output_path": str(output_name),
                "image_base64": image_base64,
            }
        except HTTPException:
            raise
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")
        finally:
            infer_lock.release()

    return app


def main(
    host: str = "0.0.0.0",
    port: int = 7860,
    model_name: str = "flux.2-klein-9b",
    device: str = "cuda:0",
):
    import uvicorn

    app = create_app(model_name=model_name, device=device)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    Fire(main)
