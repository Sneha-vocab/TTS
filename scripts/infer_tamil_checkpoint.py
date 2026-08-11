#!/usr/bin/env python3
"""Quick inference for a custom Tamil Matcha checkpoint (while training continues).

Example:
  python scripts/infer_tamil_checkpoint.py \\
    --checkpoint logs/train/tamil_codemix/runs/.../checkpoints/best-586.ckpt
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import soundfile as sf
import torch

from matcha.cli import load_matcha, load_vocoder, process_text, to_waveform
from matcha.utils.utils import assert_model_downloaded, get_user_data_dir

VOCODER_URLS = {
    "hifigan_T2_v1": "https://github.com/shivammehta25/Matcha-TTS-checkpoints/releases/download/v1.0/generator_v1",
    "hifigan_univ_v1": "https://github.com/shivammehta25/Matcha-TTS-checkpoints/releases/download/v1.0/g_02500000",
}

DEFAULT_TEXTS = [
    "வணக்கம் உங்களை வரவேற்கிறோம் தயவுசெய்து சொல்லுங்கள் நான் எப்படி help செய்ய முடியும்",
    "உங்கள் otp ஏழு ஒன்று நான்கு ஒன்பது ஐந்து பூஜ்ஜியம் verify செய்யுங்கள்",
    "rupees four hundred fifty refund delayக்கு நாங்கள் apologize செய்கிறோம் கவலைப்படாதீர்கள்",
    "emi issueக்கு நான் உங்கள் payment status check செய்கிறேன் தயவுசெய்து ஒரு minute holdல் இருங்கள்",
]


def parse_args():
    p = argparse.ArgumentParser(description="Infer Tamil Matcha checkpoint")
    p.add_argument("--checkpoint", required=True, type=str, help="Path to .ckpt")
    p.add_argument("--vocoder", default="hifigan_univ_v1", choices=VOCODER_URLS.keys())
    p.add_argument("--text", action="append", default=None, help="Text to synthesize (repeatable)")
    p.add_argument("--file", type=str, default=None, help="Optional filelist path|text lines")
    p.add_argument("--n_file", type=int, default=5, help="How many lines from --file")
    p.add_argument("--output_dir", type=str, default="outputs/tamil_infer")
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--temperature", type=float, default=0.667)
    p.add_argument("--speaking_rate", type=float, default=1.0)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--cleaner", default="basic_cleaners")
    return p.parse_args()


def main():
    args = parse_args()
    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Vocoder
    vocoder_path = get_user_data_dir() / args.vocoder
    assert_model_downloaded(vocoder_path, VOCODER_URLS[args.vocoder])
    vocoder, denoiser = load_vocoder(args.vocoder, vocoder_path, device)

    print(f"[+] Loading Matcha from {ckpt} on {device}")
    model = load_matcha("tamil_custom", str(ckpt), device)

    texts = []
    if args.text:
        texts.extend(args.text)
    if args.file:
        for i, line in enumerate(Path(args.file).read_text(encoding="utf-8").splitlines()):
            if i >= args.n_file:
                break
            if not line.strip():
                continue
            texts.append(line.split("|", 1)[1] if "|" in line else line.strip())
    if not texts:
        texts = DEFAULT_TEXTS

    print(f"[+] Synthesising {len(texts)} utterance(s) → {out_dir}")
    rtfs = []
    for i, text in enumerate(texts):
        text = text.strip()
        processed = process_text(i, text, device, [args.cleaner])
        t0 = time.perf_counter()
        output = model.synthesise(
            processed["x"],
            processed["x_lengths"],
            n_timesteps=args.steps,
            temperature=args.temperature,
            spks=None,
            length_scale=args.speaking_rate,
        )
        # clone: HiFi-GAN can't take inference-mode tensors
        mel = output["mel"].clone()
        wav = to_waveform(mel, vocoder, denoiser)
        dt = time.perf_counter() - t0
        audio = wav.squeeze().float().cpu().numpy()
        dur = len(audio) / 22050.0
        rtf = dt / max(dur, 1e-6)
        rtfs.append(rtf)

        stem = f"utt_{i:03d}"
        wav_path = out_dir / f"{stem}.wav"
        txt_path = out_dir / f"{stem}.txt"
        sf.write(wav_path, audio, 22050)
        txt_path.write_text(text + "\n", encoding="utf-8")
        print(f"[{i}] {wav_path.name}  dur={dur:.2f}s  wall={dt:.2f}s  RTF={rtf:.3f}")
        print(f"     {text[:100]}{'...' if len(text) > 100 else ''}")

    print("=" * 60)
    print(f"Mean RTF: {sum(rtfs)/len(rtfs):.3f}  ( <1.0 means faster than real-time )")
    print(f"Outputs: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
