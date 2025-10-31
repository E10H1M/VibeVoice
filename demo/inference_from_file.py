#!/usr/bin/env python3
# inference_from_file.py

import argparse
import os
import re
import time
import traceback
from typing import List, Tuple
import numpy as np
import torch
import soundfile as sf
import librosa

from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
from transformers.utils import logging

logging.set_verbosity_info()
logger = logging.get_logger(__name__)


class VoiceMapper:
    """Maps speaker names to voice file paths"""
    def __init__(self):
        self.setup_voice_presets()
        # add simplified aliases from filename tokens
        new_dict = {}
        for name, path in self.voice_presets.items():
            alias = name
            if "_" in alias:
                alias = alias.split("_")[0]
            if "-" in alias:
                alias = alias.split("-")[-1]
            new_dict[alias] = path
        self.voice_presets.update(new_dict)

    def setup_voice_presets(self):
        voices_dir = os.path.join(os.path.dirname(__file__), "voices")
        if not os.path.exists(voices_dir):
            print(f"Warning: Voices directory not found at {voices_dir}")
            self.voice_presets = {}
            self.available_voices = {}
            return

        exts = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac")
        files = [f for f in os.listdir(voices_dir)
                 if f.lower().endswith(exts) and os.path.isfile(os.path.join(voices_dir, f))]

        self.voice_presets = {}
        for f in files:
            name = os.path.splitext(f)[0]
            self.voice_presets[name] = os.path.join(voices_dir, f)

        self.voice_presets = dict(sorted(self.voice_presets.items()))
        self.available_voices = {n: p for n, p in self.voice_presets.items() if os.path.exists(p)}

        print(f"Found {len(self.available_voices)} voice files in {voices_dir}")
        if self.available_voices:
            print(f"Available voices: {', '.join(self.available_voices.keys())}")

    def get_voice_path(self, speaker_name: str) -> str:
        # exact
        if speaker_name in self.voice_presets:
            return self.voice_presets[speaker_name]
        # partial (case-insensitive)
        q = speaker_name.lower()
        for name, path in self.voice_presets.items():
            if name.lower() in q or q in name.lower():
                return path
        # default
        default_voice = list(self.voice_presets.values())[0]
        print(f"Warning: No voice preset found for '{speaker_name}', using default: {default_voice}")
        return default_voice


def read_audio(audio_path: str, target_sr: int = 24000) -> np.ndarray:
    """Load audio -> mono float32 @ target_sr."""
    wav, sr = sf.read(audio_path)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != target_sr:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
    return wav.astype(np.float32)


def parse_txt_script(txt_content: str) -> Tuple[List[str], List[str]]:
    """
    Parse 'Speaker X: text' lines into joined segments.
    Returns (scripts, speaker_numbers).
    """
    lines = txt_content.strip().split("\n")
    scripts, speaker_numbers = [], []
    pat = r"^Speaker\s+(\d+):\s*(.*)$"

    current_speaker = None
    current_text = ""
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


def parse_args():
    p = argparse.ArgumentParser(description="VibeVoice TXT-file inference (multi-speaker)")
    p.add_argument("--model_path", type=str, default="weights/VibeVoice-Large")
    p.add_argument("--txt_path", type=str, default="demo/text_examples/1p_abs.txt")
    p.add_argument("--speaker_names", type=str, nargs="+", default=["Andrew"])
    p.add_argument("--output_dir", type=str, default="./outputs")
    p.add_argument("--device", type=str,
                   default=("cuda" if torch.cuda.is_available()
                            else ("mps" if torch.backends.mps.is_available() else "cpu")),
                   help="cuda | mps | cpu")
    p.add_argument("--cfg_scale", type=float, default=1.3)
    p.add_argument("--inference_steps", type=int, default=10)
    return p.parse_args()


def main():
    args = parse_args()

    if args.device.lower() == "mpx":
        print("Note: device 'mpx' detected, treating it as 'mps'.")
        args.device = "mps"
    if args.device == "mps" and not torch.backends.mps.is_available():
        print("Warning: MPS not available. Falling back to CPU.")
        args.device = "cpu"
    print(f"Using device: {args.device}")

    vm = VoiceMapper()

    if not os.path.exists(args.txt_path):
        print(f"Error: txt file not found: {args.txt_path}")
        return

    print(f"Reading script from: {args.txt_path}")
    with open(args.txt_path, "r", encoding="utf-8") as f:
        txt = f.read()

    scripts, speaker_numbers = parse_txt_script(txt)
    if not scripts:
        print("Error: No valid 'Speaker X:' lines found.")
        return

    print(f"Found {len(scripts)} segments:")
    for i, (s, sp) in enumerate(zip(scripts, speaker_numbers), 1):
        print(f"  {i}. Speaker {sp} :: {s[:100]}...")

    # Map numbers -> provided names
    provided = args.speaker_names if isinstance(args.speaker_names, list) else [args.speaker_names]
    number_to_name = {str(i): name for i, name in enumerate(provided, 1)}
    print("\nSpeaker mapping:")
    for sp in sorted(set(speaker_numbers), key=lambda x: int(x)):
        print(f"  Speaker {sp} -> {number_to_name.get(sp, f'Speaker {sp}')}")
    # unique speakers in order of first appearance
    uniq_nums, seen = [], set()
    for sp in speaker_numbers:
        if sp not in seen:
            uniq_nums.append(sp); seen.add(sp)

    # Load voice samples as arrays (NOT paths)
    voice_arrays = []
    actual_speakers = []
    for sp in uniq_nums:
        name = number_to_name.get(sp, f"Speaker {sp}")
        vpath = vm.get_voice_path(name)
        voice_arrays.append(read_audio(vpath))
        actual_speakers.append(name)
        print(f"Speaker {sp} ('{name}') -> {os.path.basename(vpath)}")

    full_script = "\n".join(scripts).replace("’", "'")

    print(f"\nLoading processor & model from {args.model_path}")
    processor = VibeVoiceProcessor.from_pretrained(args.model_path)

    # dtype / attention
    if args.device == "mps":
        load_dtype, attn_impl = torch.float32, "sdpa"
    elif args.device == "cuda":
        load_dtype, attn_impl = torch.bfloat16, "flash_attention_2"
    else:
        load_dtype, attn_impl = torch.float32, "sdpa"
    print(f"torch_dtype={load_dtype}, attn_implementation={attn_impl}")

    try:
        if args.device == "mps":
            model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                args.model_path, torch_dtype=load_dtype, attn_implementation=attn_impl, device_map=None
            )
            model.to("mps")
        elif args.device == "cuda":
            model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                args.model_path, torch_dtype=load_dtype, device_map="cuda", attn_implementation=attn_impl
            )
        else:
            model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                args.model_path, torch_dtype=load_dtype, device_map="cpu", attn_implementation=attn_impl
            )
    except Exception as e:
        if attn_impl == "flash_attention_2":
            print(f"[ERROR] {type(e).__name__}: {e}")
            print(traceback.format_exc())
            print("Falling back to SDPA (may reduce audio quality).")
            model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                args.model_path,
                torch_dtype=load_dtype,
                device_map=(args.device if args.device in ("cuda", "cpu") else None),
                attn_implementation="sdpa",
            )
            if args.device == "mps":
                model.to("mps")
        else:
            raise

    model.eval()

    # Match demo: swap scheduler to SDE-DPM++ and set steps
    model.model.noise_scheduler = model.model.noise_scheduler.from_config(
        model.model.noise_scheduler.config,
        algorithm_type="sde-dpmsolver++",
        beta_schedule="squaredcos_cap_v2",
    )
    model.set_ddpm_inference_steps(num_steps=args.inference_steps)

    if hasattr(model.model, "language_model"):
        print(f"Language model attention: {model.model.language_model.config._attn_implementation}")

    # Prepare inputs (arrays)
    inputs = processor(
        text=[full_script],
        voice_samples=[voice_arrays],
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )
    target_device = args.device if args.device in ("cuda", "mps") else "cpu"
    for k, v in inputs.items():
        if torch.is_tensor(v):
            inputs[k] = v.to(target_device)

    print(f"\nStarting generation (cfg_scale={args.cfg_scale}, steps={args.inference_steps})")
    t0 = time.time()
    outputs = model.generate(
        **inputs,
        max_new_tokens=None,
        cfg_scale=args.cfg_scale,
        tokenizer=processor.tokenizer,
        generation_config={"do_sample": False},
        verbose=True,
    )
    gen_time = time.time() - t0
    print(f"Generation time: {gen_time:.2f}s")

    # Metrics
    sr = 24000
    if outputs.speech_outputs and outputs.speech_outputs[0] is not None:
        samples = outputs.speech_outputs[0].shape[-1] if outputs.speech_outputs[0].ndim > 0 else len(outputs.speech_outputs[0])
        duration = samples / sr
        rtf = gen_time / duration if duration > 0 else float("inf")
        print(f"Audio duration: {duration:.2f}s  |  RTF: {rtf:.2f}x")
    else:
        duration = 0.0
        print("No audio output generated")

    in_tokens = inputs["input_ids"].shape[1]
    out_tokens = outputs.sequences.shape[1]
    gen_tokens = out_tokens - in_tokens
    print(f"Prefill tokens: {in_tokens} | Generated tokens: {gen_tokens} | Total: {out_tokens}")

    # Save
    base = os.path.splitext(os.path.basename(args.txt_path))[0]
    os.makedirs(args.output_dir, exist_ok=True)
    out_wav = os.path.join(args.output_dir, f"{base}_generated.wav")
    processor.save_audio(outputs.speech_outputs[0], output_path=out_wav)
    print(f"Saved: {out_wav}")

    # Summary
    print("\n" + "=" * 50)
    print("GENERATION SUMMARY")
    print("=" * 50)
    print(f"Input file: {args.txt_path}")
    print(f"Output file: {out_wav}")
    print(f"Speakers provided: {args.speaker_names}")
    print(f"Unique speakers: {len(set(speaker_numbers))}")
    print(f"Segments: {len(scripts)}")
    print(f"Prefill tokens: {in_tokens}")
    print(f"Generated tokens: {gen_tokens}")
    print(f"Total tokens: {out_tokens}")
    print(f"Generation time: {gen_time:.2f}s")
    print(f"Audio duration: {duration:.2f}s")
    print("=" * 50)


if __name__ == "__main__":
    main()
