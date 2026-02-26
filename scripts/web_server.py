import base64
import io
import json
import random
import secrets
import threading
from pathlib import Path

import torch
from einops import rearrange
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, PlainTextResponse
from fire import Fire
from pydantic import BaseModel, Field
from PIL import Image

from flux2.sampling import (
    batched_prc_img,
    batched_prc_txt,
    denoise,
    denoise_cfg,
    encode_image_refs,
    get_schedule,
    scatter_ids,
)
from flux2.util import FLUX2_MODEL_INFO, load_ae, load_flow_model, load_text_encoder

HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>FLUX.2 Interactive Web</title>
  <style>
    :root {
      --bg: radial-gradient(circle at 20% 20%, #1b2033 0%, #101522 40%, #0b0f17 100%);
      --panel: #141a2a;
      --panel-soft: #1a2134;
      --text: #f1f4fb;
      --muted: #9ba7c3;
      --line: #2a344f;
      --accent: #4ad7a7;
      --accent-strong: #2bb98b;
      --danger: #ff6a88;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      font-family: "Space Grotesk", "Noto Sans KR", "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
      padding: 18px;
    }
    .wrap {
      max-width: 1400px;
      margin: 0 auto;
      display: grid;
      grid-template-columns: 360px 1fr;
      gap: 14px;
    }
    .panel {
      background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.01));
      border: 1px solid var(--line);
      border-radius: 14px;
      backdrop-filter: blur(8px);
    }
    .sidebar { padding: 14px; }
    .title { margin: 2px 0 4px; font-size: 19px; font-weight: 700; }
    .sub { color: var(--muted); font-size: 12px; margin-bottom: 12px; }
    .row { margin-bottom: 10px; }
    .row label { display: block; font-size: 12px; color: var(--muted); margin-bottom: 4px; }
    input, textarea, button {
      font: inherit;
      border-radius: 10px;
      border: 1px solid var(--line);
      background: var(--panel-soft);
      color: var(--text);
    }
    input[type="number"], input[type="text"], textarea {
      width: 100%;
      padding: 9px 10px;
    }
    textarea { resize: vertical; min-height: 88px; }
    .grid2 {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }
    .checks { display: grid; gap: 6px; margin-top: 8px; }
    .checks label {
      display: flex;
      align-items: center;
      gap: 7px;
      font-size: 13px;
      color: var(--text);
      margin: 0;
    }
    .checks input { width: auto; }
    .actions { display: flex; gap: 8px; margin-top: 10px; }
    button {
      cursor: pointer;
      padding: 8px 12px;
      font-weight: 600;
      transition: transform .12s ease, border-color .12s ease, background .12s ease;
    }
    button:hover { transform: translateY(-1px); border-color: var(--accent); }
    .btn-primary { background: linear-gradient(135deg, var(--accent), var(--accent-strong)); color: #07140f; border: none; }
    .btn-secondary { background: var(--panel-soft); }
    .status {
      margin-top: 10px;
      min-height: 18px;
      color: var(--muted);
      font-size: 13px;
    }

    .main {
      display: grid;
      grid-template-rows: auto 1fr auto;
      min-height: calc(100vh - 36px);
    }
    .main-head {
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 10px;
    }
    .main-head .hint { color: var(--muted); font-size: 12px; }
    .chat-log {
      padding: 12px;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 12px;
      min-height: 0;
    }
    .msg {
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 10px;
      background: rgba(255,255,255,0.03);
    }
    .msg-user { align-self: flex-end; max-width: 82%; background: rgba(74,215,167,0.10); }
    .msg-assistant { align-self: stretch; }
    .msg-meta { font-size: 11px; color: var(--muted); margin-bottom: 6px; }
    .msg-text { white-space: pre-wrap; line-height: 1.4; }
    .result {
      width: 100%;
      margin-top: 8px;
      border-radius: 10px;
      border: 1px solid var(--line);
      background: #0a0d14;
      display: block;
    }
    .composer {
      border-top: 1px solid var(--line);
      padding: 12px;
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 8px;
      align-items: end;
    }
    .composer textarea { min-height: 74px; }

    @media (max-width: 1020px) {
      .wrap { grid-template-columns: 1fr; }
      .main { min-height: 70vh; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <aside class="panel sidebar">
      <div class="title">FLUX.2 Interactive</div>
      <div class="sub">Chat-style text-to-image + image editing (single/multi-reference)</div>

      <div class="row">
        <label for="inputImages">Reference Images (editing)</label>
        <input id="inputImages" type="file" accept="image/*" multiple />
      </div>

      <div class="grid2">
        <div class="row">
          <label for="width">Width</label>
          <input id="width" type="number" value="1360" min="256" step="16" />
        </div>
        <div class="row">
          <label for="height">Height</label>
          <input id="height" type="number" value="768" min="256" step="16" />
        </div>
      </div>

      <div class="grid2">
        <div class="row">
          <label for="steps">Steps</label>
          <input id="steps" type="number" value="4" min="1" />
        </div>
        <div class="row">
          <label for="guidance">Guidance</label>
          <input id="guidance" type="number" value="1.0" step="0.1" />
        </div>
      </div>

      <div class="grid2">
        <div class="row">
          <label for="seed">Seed (optional)</label>
          <input id="seed" type="number" />
        </div>
        <div class="row">
          <label for="matchImageSize">Match Ref Size Index</label>
          <input id="matchImageSize" type="number" min="0" placeholder="0" />
        </div>
      </div>

      <div class="checks">
        <label><input id="useHistory" type="checkbox" checked /> Use recent chat prompts as context</label>
        <label><input id="reuseLast" type="checkbox" /> Reuse last generated image as reference</label>
        <label><input id="enableThinking" type="checkbox" /> Enable thinking in text-encoder chat template</label>
      </div>

      <div class="actions">
        <button class="btn-secondary" onclick="clearConversation()">Clear Chat</button>
      </div>
      <div id="status" class="status"></div>
    </aside>

    <section class="panel main">
      <div class="main-head">
        <div>
          <div style="font-weight:700">Conversation</div>
          <div class="hint">Ctrl+Enter to generate</div>
        </div>
        <div class="hint" id="health">Checking health...</div>
      </div>

      <div id="chatLog" class="chat-log"></div>

      <div class="composer">
        <textarea id="prompt" placeholder="Describe what to generate or how to edit the reference image..."></textarea>
        <button class="btn-primary" id="sendBtn" onclick="sendPrompt()">Generate</button>
      </div>
    </section>
  </div>

  <script>
    const chatLog = document.getElementById("chatLog");
    const statusEl = document.getElementById("status");
    const promptEl = document.getElementById("prompt");
    const sendBtn = document.getElementById("sendBtn");
    const healthEl = document.getElementById("health");

    const promptHistory = [];
    let lastImageBase64 = null;
    let pending = false;

    function appendMessage(role, text, meta = null, imageBase64 = null) {
      const box = document.createElement("div");
      box.className = `msg ${role === "user" ? "msg-user" : "msg-assistant"}`;

      const roleLabel = document.createElement("div");
      roleLabel.className = "msg-meta";
      roleLabel.textContent = role === "user" ? "You" : "FLUX.2";
      box.appendChild(roleLabel);

      if (text) {
        const body = document.createElement("div");
        body.className = "msg-text";
        body.textContent = text;
        box.appendChild(body);
      }

      if (meta) {
        const m = document.createElement("div");
        m.className = "msg-meta";
        m.style.marginTop = "8px";
        m.textContent = meta;
        box.appendChild(m);
      }

      if (imageBase64) {
        const img = document.createElement("img");
        img.className = "result";
        img.src = "data:image/png;base64," + imageBase64;
        box.appendChild(img);
      }

      chatLog.appendChild(box);
      chatLog.scrollTop = chatLog.scrollHeight;
    }

    function clearConversation() {
      promptHistory.length = 0;
      lastImageBase64 = null;
      chatLog.innerHTML = "";
      statusEl.textContent = "Conversation cleared";
    }

    function fileToBase64(file) {
      return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => {
          const result = String(reader.result || "");
          const b64 = result.includes(",") ? result.split(",", 2)[1] : result;
          resolve(b64);
        };
        reader.onerror = reject;
        reader.readAsDataURL(file);
      });
    }

    async function getReferenceImages() {
      const refs = [];
      const files = Array.from(document.getElementById("inputImages").files || []);
      for (const file of files) {
        refs.push(await fileToBase64(file));
      }
      if (document.getElementById("reuseLast").checked && lastImageBase64) {
        refs.unshift(lastImageBase64);
      }
      return refs;
    }

    async function sendPrompt() {
      if (pending) return;

      const prompt = promptEl.value.trim();
      if (!prompt) {
        statusEl.textContent = "Prompt is empty";
        return;
      }

      pending = true;
      sendBtn.disabled = true;
      statusEl.textContent = "Preparing request...";

      try {
        const refs = await getReferenceImages();
        appendMessage("user", prompt, refs.length ? `references: ${refs.length}` : "text-to-image");

        const body = {
          prompt,
          width: Number(document.getElementById("width").value),
          height: Number(document.getElementById("height").value),
          num_steps: Number(document.getElementById("steps").value),
          guidance: Number(document.getElementById("guidance").value),
          input_images_base64: refs,
          enable_thinking: document.getElementById("enableThinking").checked,
        };

        const seedRaw = document.getElementById("seed").value.trim();
        if (seedRaw) body.seed = Number(seedRaw);

        const matchRaw = document.getElementById("matchImageSize").value.trim();
        if (matchRaw !== "") body.match_image_size = Number(matchRaw);

        if (document.getElementById("useHistory").checked && promptHistory.length) {
          body.history_prompts = promptHistory.slice(-6);
        }

        statusEl.textContent = "Generating...";

        const res = await fetch("/generate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });

        const payload = await res.json();
        if (!res.ok) throw new Error(payload.detail || "Request failed");

        promptHistory.push(prompt);
        lastImageBase64 = payload.image_base64;

        const meta = `seed=${payload.seed} | ${payload.width}x${payload.height} | refs=${payload.num_input_images}`;
        appendMessage("assistant", payload.prompt_used, meta, payload.image_base64);
        statusEl.textContent = "Done";
      } catch (err) {
        statusEl.textContent = "Error: " + (err?.message || err);
      } finally {
        pending = false;
        sendBtn.disabled = false;
      }
    }

    promptEl.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && event.ctrlKey) {
        event.preventDefault();
        sendPrompt();
      }
    });

    (async function checkHealth() {
      try {
        const res = await fetch("/health");
        const payload = await res.json();
        if (!res.ok) throw new Error(payload.detail || "unavailable");
        healthEl.textContent = `${payload.status} | ${payload.model} | ${payload.device}`;
      } catch (e) {
        healthEl.textContent = "health unavailable";
      }
    })();
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
    input_images_base64: list[str] = Field(default_factory=list)
    match_image_size: int | None = Field(default=None, ge=0)
    history_prompts: list[str] = Field(default_factory=list)
    enable_thinking: bool | None = None


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


def _load_auth_credentials(auth_file: str) -> tuple[str, str]:
    auth_path = Path(auth_file)
    if not auth_path.exists():
        raise ValueError(
            f"Auth file not found: {auth_file}. "
            "Create a JSON file with {'id': '...', 'pw': '...'}"
        )

    data = json.loads(auth_path.read_text(encoding="utf-8"))
    user = data.get("id") or data.get("username")
    password = data.get("pw") or data.get("password")

    if not isinstance(user, str) or not user:
        raise ValueError(f"Invalid auth file {auth_file}: missing non-empty 'id' (or 'username')")
    if not isinstance(password, str) or not password:
        raise ValueError(f"Invalid auth file {auth_file}: missing non-empty 'pw' (or 'password')")
    return user, password


def _unauthorized_response() -> Response:
    return PlainTextResponse(
        "Unauthorized",
        status_code=401,
        headers={"WWW-Authenticate": 'Basic realm="FLUX2"'},
    )


def _decode_base64_image(value: str) -> Image.Image:
    raw = value.strip()
    if raw.startswith("data:"):
        _, raw = raw.split(",", 1)
    try:
        data = base64.b64decode(raw, validate=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image payload: {exc}") from exc

    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image content: {exc}") from exc
    return image


def _compose_prompt(prompt: str, history_prompts: list[str]) -> str:
    clean_prompt = prompt.strip()
    clean_history = [p.strip() for p in history_prompts if isinstance(p, str) and p.strip()]
    if not clean_history:
        return clean_prompt

    tail = clean_history[-6:]
    return "\n".join([*tail, clean_prompt])


def create_app(
    model_name: str = "flux.2-klein-4b",
    device: str = "cuda:0",
    auth_file: str = "secrets/web_auth.json",
):
    model_name = model_name.lower()
    if model_name not in FLUX2_MODEL_INFO:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(FLUX2_MODEL_INFO.keys())}")
    auth_user, auth_password = _load_auth_credentials(auth_file)
    print(f"Basic auth enabled with credentials file: {auth_file}")

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
    model_dtype = next(model.parameters()).dtype
    ae_dtype = next(ae.parameters()).dtype

    infer_lock = threading.Lock()

    app = FastAPI(title="FLUX.2 Web Server", version="0.2.0")

    @app.middleware("http")
    async def auth_middleware(request: Request, call_next):
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Basic "):
            return _unauthorized_response()

        encoded = auth_header.split(" ", 1)[1]
        try:
            decoded = base64.b64decode(encoded).decode("utf-8")
            provided_user, provided_password = decoded.split(":", 1)
        except Exception:
            return _unauthorized_response()

        if not (
            secrets.compare_digest(provided_user, auth_user)
            and secrets.compare_digest(provided_password, auth_password)
        ):
            return _unauthorized_response()

        return await call_next(request)

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
            prompt_used = _compose_prompt(req.prompt, req.history_prompts)

            img_ctx = [_decode_base64_image(x) for x in req.input_images_base64]

            width = req.width
            height = req.height
            if req.match_image_size is not None:
                if req.match_image_size >= len(img_ctx):
                    raise HTTPException(
                        status_code=400,
                        detail=f"match_image_size index out of range for {len(img_ctx)} input images",
                    )
                width, height = img_ctx[req.match_image_size].size

            _validate_dims(width, height)
            num_steps = req.num_steps if req.num_steps is not None else default_num_steps
            guidance = req.guidance if req.guidance is not None else default_guidance
            _validate_model_params(model_name, num_steps=num_steps, guidance=guidance)
            seed = req.seed if req.seed is not None else random.randrange(2**31)

            with torch.no_grad():
                if req.enable_thinking is not None and hasattr(text_encoder, "set_enable_thinking"):
                    text_encoder.set_enable_thinking(req.enable_thinking)

                ref_tokens, ref_ids = encode_image_refs(ae, img_ctx)
                if ref_tokens is not None:
                    ref_tokens = ref_tokens.to(model_dtype)

                if model_info["guidance_distilled"]:
                    ctx = text_encoder([prompt_used]).to(model_dtype)
                else:
                    ctx_empty = text_encoder([""]).to(model_dtype)
                    ctx_prompt = text_encoder([prompt_used]).to(model_dtype)
                    ctx = torch.cat([ctx_empty, ctx_prompt], dim=0)
                ctx, ctx_ids = batched_prc_txt(ctx)

                shape = (1, 128, height // 16, width // 16)
                generator = torch.Generator(device=str(torch_device)).manual_seed(seed)
                randn = torch.randn(shape, generator=generator, dtype=model_dtype, device=torch_device)
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
                        img_cond_seq=ref_tokens,
                        img_cond_seq_ids=ref_ids,
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
                        img_cond_seq=ref_tokens,
                        img_cond_seq_ids=ref_ids,
                    )
                x = torch.cat(scatter_ids(x, x_ids)).squeeze(2)
                x = ae.decode(x.to(ae_dtype)).float()

            x = x.clamp(-1, 1)
            x = rearrange(x[0], "c h w -> h w c")
            img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())

            buf = io.BytesIO()
            img.save(buf, format="PNG")
            image_base64 = base64.b64encode(buf.getvalue()).decode()

            return {
                "seed": seed,
                "image_base64": image_base64,
                "prompt_used": prompt_used,
                "num_input_images": len(img_ctx),
                "width": width,
                "height": height,
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
    model_name: str = "flux.2-klein-4b",
    device: str = "cuda:0",
    auth_file: str = "secrets/web_auth.json",
):
    import uvicorn

    app = create_app(model_name=model_name, device=device, auth_file=auth_file)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    Fire(main)
