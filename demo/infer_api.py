#!/usr/bin/env python3
# demo/infer_api.py
# FastAPI server for VibeVoice inference (non-streaming), single-file, flags-only (no env).

import time
import uuid
import argparse
import logging
import traceback
from typing import List, Optional, Tuple
from pathlib import Path
import random

import numpy as np
import torch
import soundfile as sf
import librosa
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import uvicorn

from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

# ---------- logging (constant INFO) ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d - %(message)s",
)
log = logging.getLogger("vibevoice.api")

app = FastAPI(title="VibeVoice Inference API")

# ---------- request/response models ----------
class GenerateInput(BaseModel):
    # Provide EITHER script OR txt_path (script takes precedence if both given)
    script: Optional[str] = None
    txt_path: Optional[str] = None

    # Speaker names in order of appearance (e.g., ["Alice","Frank"])
    speaker_names: List[str]

    # Optional overrides (fallback to server defaults)
    cfg_scale: Optional[float] = None
    inference_steps: Optional[int] = None

    # Optional explicit save location (otherwise auto path)
    output_path: Optional[str] = None

    # Optional seed for determinism (overrides --default_seed if set)
    seed: Optional[int] = None


class GenerateOutput(BaseModel):
    status: str
    output_path: Optional[str]
    duration_s: Optional[float]
    rtf: Optional[float]
    tokens: Optional[dict]
    message: str


# ---------- tiny utilities ----------
def read_audio(audio_path: str, target_sr: int = 24000) -> np.ndarray:
    wav, sr = sf.read(audio_path)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != target_sr:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
    return wav.astype(np.float32)


def parse_txt_script(txt_content: str) -> Tuple[List[str], List[str]]:
    """Return (scripts, speaker_numbers) for lines like 'Speaker 1: ...'."""
    import re
    lines = txt_content.strip().split("\n")
    scripts, speaker_numbers = [], []
    pat = r"^Speaker\s+(\d+):\s*(.*)$"
    current_speaker, current_text = None, ""
    for line in lines:
        line = line.strip()
        if not line:
            continue
        m = re.match(pat, line, re.IGNORECASE)
        if m:
            if current_speaker and current_text:
                scripts.append(f"Speaker {current_speaker}: {current_text.strip()}")
                speaker_numbers.append(current_speaker)
            current_speaker = m.group(1).strip()
            current_text = m.group(2).strip()
        else:
            current_text = (current_text + " " + line).strip() if current_text else line
    if current_speaker and current_text:
        scripts.append(f"Speaker {current_speaker}: {current_text.strip()}")
        speaker_numbers.append(current_speaker)
    return scripts, speaker_numbers


class VoiceMapper:
    """Map speaker name -> voice file path (adds simple aliases)."""
    def __init__(self, voices_dir: Path):
        self.voices_dir = voices_dir.resolve()
        self.voice_presets = {}
        self._scan()

    def _scan(self):
        if not self.voices_dir.exists():
            log.warning("Voices directory not found at %s", self.voices_dir)
            return
        exts = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac")
        for f in sorted(self.voices_dir.iterdir()):
            if f.is_file() and f.suffix.lower() in exts:
                name = f.stem
                self.voice_presets[name] = str(f)
                # simple aliases: strip prefix/suffix tokens
                alias = name
                if "_" in alias:
                    alias = alias.split("_")[0]
                if "-" in alias:
                    alias = alias.split("-")[-1]
                self.voice_presets.setdefault(alias, str(f))
        if not self.voice_presets:
            log.warning("No voice files found in %s", self.voices_dir)

    def get(self, speaker_name: str) -> str:
        # exact
        if speaker_name in self.voice_presets:
            return self.voice_presets[speaker_name]
        # partial
        q = speaker_name.lower()
        for n, p in self.voice_presets.items():
            if n.lower() in q or q in n.lower():
                return p
        # fallback
        if self.voice_presets:
            k = next(iter(self.voice_presets))
            log.warning("No voice preset for '%s'; using '%s'", speaker_name, k)
            return self.voice_presets[k]
        raise RuntimeError("No voices available")


# ---------- app state ----------
class PipelineState:
    def __init__(self, model_path: str, device: str, voices_dir: Path,
                 inference_steps: int, cfg_scale: float, default_seed: Optional[int]):
        self.model_path = model_path
        self.device = device
        self.default_steps = inference_steps
        self.default_cfg = cfg_scale
        self.default_seed = default_seed
        self.voice_map = VoiceMapper(voices_dir)
        self.processor = None
        self.model = None
        self._init_model()

    def _init_model(self):
        dev = self.device
        if dev.lower() == "mpx":
            dev = "mps"
        if dev == "mps" and not torch.backends.mps.is_available():
            log.warning("MPS not available, falling back to CPU.")
            dev = "cpu"

        # dtype/attn (constants by device)
        if dev == "mps":
            torch_dtype, attn_impl = torch.float32, "sdpa"
        elif dev == "cuda":
            torch_dtype, attn_impl = torch.bfloat16, "flash_attention_2"
        else:
            torch_dtype, attn_impl = torch.float32, "sdpa"

        log.info("Loading processor & model from %s (device=%s, dtype=%s, attn=%s)",
                 self.model_path, dev, torch_dtype, attn_impl)
        self.processor = VibeVoiceProcessor.from_pretrained(self.model_path)

        try:
            if dev == "mps":
                self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                    self.model_path, torch_dtype=torch_dtype, attn_implementation=attn_impl, device_map=None
                )
                self.model.to("mps")
            elif dev == "cuda":
                self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                    self.model_path, torch_dtype=torch_dtype, device_map="cuda", attn_implementation=attn_impl
                )
            else:
                self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                    self.model_path, torch_dtype=torch_dtype, device_map="cpu", attn_implementation=attn_impl
                )
        except Exception as e:
            if attn_impl == "flash_attention_2":
                log.error("FlashAttention load failed: %s\n%s", e, traceback.format_exc())
                log.warning("Falling back to SDPA.")
                self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                    self.model_path,
                    torch_dtype=torch_dtype,
                    device_map=(dev if dev in ("cuda", "cpu") else None),
                    attn_implementation="sdpa",
                )
                if dev == "mps":
                    self.model.to("mps")
            else:
                raise

        self.model.eval()
        # match demo scheduler (constants)
        self.model.model.noise_scheduler = self.model.model.noise_scheduler.from_config(
            self.model.model.noise_scheduler.config,
            algorithm_type="sde-dpmsolver++",
            beta_schedule="squaredcos_cap_v2",
        )
        self.model.set_ddpm_inference_steps(num_steps=self.default_steps)
        if hasattr(self.model.model, "language_model"):
            log.info("LM attention: %s", self.model.model.language_model.config._attn_implementation)

    def _apply_seed(self, seed: Optional[int]):
        if seed is None:
            return
        torch.manual_seed(seed)
        np.random.seed(seed % (2**32 - 1))
        random.seed(seed)

    def run(self, script_text: str, speaker_names: List[str],
            cfg_scale: Optional[float], inference_steps: Optional[int],
            output_path: Optional[str], seed: Optional[int]) -> GenerateOutput:

        # seed precedence: request seed > default_seed
        effective_seed = seed if seed is not None else self.default_seed
        self._apply_seed(effective_seed)

        # build voices for unique speakers in order of first appearance
        _, speaker_numbers = parse_txt_script(script_text)
        uniq_nums, seen = [], set()
        for sp in speaker_numbers:
            if sp not in seen:
                uniq_nums.append(sp); seen.add(sp)

        # map "Speaker 1/2/..." -> user names list (1-based)
        number_to_name = {str(i): name for i, name in enumerate(speaker_names, 1)}
        voice_arrays = []
        for sp in uniq_nums:
            name = number_to_name.get(sp, f"Speaker {sp}")
            vpath = self.voice_map.get(name)
            voice_arrays.append(read_audio(vpath))

        # inputs
        full_script = script_text.replace("’", "'")
        inputs = self.processor(
            text=[full_script],
            voice_samples=[voice_arrays],
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )

        target_device = self.device if self.device in ("cuda", "mps") else "cpu"
        for k, v in inputs.items():
            if torch.is_tensor(v):
                inputs[k] = v.to(target_device)

        # optional overrides
        steps = int(inference_steps) if inference_steps is not None else self.default_steps
        if steps != self.default_steps:
            self.model.set_ddpm_inference_steps(num_steps=steps)
        cfg = float(cfg_scale) if cfg_scale is not None else self.default_cfg

        # output path
        if output_path:
            out_path = Path(output_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            out_dir = Path.cwd() / f"vibe_out_{uuid.uuid4().hex[:8]}"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "output.wav"

        # generate
        t0 = time.time()
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=None,
            cfg_scale=cfg,
            tokenizer=self.processor.tokenizer,
            generation_config={"do_sample": False},
            verbose=True,
        )
        dt = time.time() - t0

        # metrics
        sr = 24000
        if outputs.speech_outputs and outputs.speech_outputs[0] is not None:
            samples = outputs.speech_outputs[0].shape[-1] if outputs.speech_outputs[0].ndim > 0 else len(outputs.speech_outputs[0])
            duration = samples / sr
            rtf = dt / duration if duration > 0 else None
        else:
            duration, rtf = None, None

        # tokens
        try:
            in_tokens = int(inputs["input_ids"].shape[1])
            out_tokens = int(outputs.sequences.shape[1])
            gen_tokens = out_tokens - in_tokens
            tok = {"prefill": in_tokens, "generated": gen_tokens, "total": out_tokens}
        except Exception:
            tok = None

        # save
        try:
            self.processor.save_audio(outputs.speech_outputs[0], output_path=str(out_path))
        except Exception as e:
            raise RuntimeError(f"Failed to save audio: {e}")

        return GenerateOutput(
            status="success",
            output_path=str(out_path),
            duration_s=duration,
            rtf=rtf,
            tokens=tok,
            message=f"OK in {dt:.2f}s (cfg={cfg}, steps={steps}, seed={effective_seed if effective_seed is not None else 'none'})",
        )


# ---------- middleware ----------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.time()
    try:
        response = await call_next(request)
        log.info("%s %s -> %s (%.1f ms)", request.method, request.url.path, response.status_code, (time.time()-t0)*1000)
        return response
    except Exception:
        log.error("Unhandled %s %s\n%s", request.method, request.url.path, traceback.format_exc())
        raise


# ---------- endpoints ----------
@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/voices")
async def list_voices():
    state: PipelineState = app.state.state  # type: ignore
    return {"voices_dir": str(state.voice_map.voices_dir),
            "voices": sorted(state.voice_map.voice_presets.keys())}

@app.post("/generate", response_model=GenerateOutput)
async def generate(body: GenerateInput):
    state: PipelineState = app.state.state  # type: ignore
    try:
        script = body.script
        if not script and body.txt_path:
            p = Path(body.txt_path)
            if not p.exists():
                raise HTTPException(status_code=400, detail=f"txt_path not found: {p}")
            script = p.read_text(encoding="utf-8")
        if not script or not script.strip():
            raise HTTPException(status_code=400, detail="Provide non-empty 'script' or valid 'txt_path'.")

        if not body.speaker_names:
            raise HTTPException(status_code=400, detail="speaker_names cannot be empty.")

        result = state.run(
            script_text=script,
            speaker_names=body.speaker_names,
            cfg_scale=body.cfg_scale,
            inference_steps=body.inference_steps,
            output_path=body.output_path,
            seed=body.seed,
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        log.error("Generation failed: %s\n%s", e, traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


# ---------- main ----------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, default=str((Path(__file__).resolve().parents[1] / "weights" / "VibeVoice-Large")))
    ap.add_argument("--device", type=str,
                    default=("cuda" if torch.cuda.is_available()
                             else ("mps" if torch.backends.mps.is_available() else "cpu")))
    ap.add_argument("--voices_dir", type=str, default=str((Path(__file__).resolve().parent / "voices")))
    ap.add_argument("--inference_steps", type=int, default=10)
    ap.add_argument("--cfg_scale", type=float, default=1.3)
    ap.add_argument("--default_seed", type=int, default=None, help="Default seed if request omits 'seed'")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()

    try:
        app.state.state = PipelineState(
            model_path=args.model_path,
            device=args.device,
            voices_dir=Path(args.voices_dir),
            inference_steps=args.inference_steps,
            cfg_scale=args.cfg_scale,
            default_seed=args.default_seed,
        )
    except Exception as e:
        log.error("Init failed: %s\n%s", e, traceback.format_exc())
        raise

    log.info("Serving on 127.0.0.1:%d | device=%s | model=%s | voices=%s | default_seed=%s",
             args.port, args.device, args.model_path, args.voices_dir, str(args.default_seed))
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="info")
