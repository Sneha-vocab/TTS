# Text-to-Speech (TTS): Concepts & Matcha-TTS Architecture

A practical explanation of **what TTS is**, how modern neural TTS systems are built, and **how Matcha-TTS works** in this repository — including how that maps to our Tamil / multilingual training setup.

For the hands-on training recipe, see **[MULTILINGUAL_TRAINING_GUIDE.md](./MULTILINGUAL_TRAINING_GUIDE.md)**.

---

## Table of contents

1. [What is TTS?](#1-what-is-tts)
2. [The classic TTS pipeline](#2-the-classic-tts-pipeline)
3. [How speech is represented](#3-how-speech-is-represented)
4. [Evolution of TTS systems](#4-evolution-of-tts-systems)
5. [Autoregressive vs non-autoregressive](#5-autoregressive-vs-non-autoregressive)
6. [Acoustic model + vocoder](#6-acoustic-model--vocoder)
7. [Matcha-TTS architecture](#7-matcha-tts-architecture)
8. [Conditional Flow Matching (CFM)](#8-conditional-flow-matching-cfm)
9. [Training vs inference](#9-training-vs-inference)
10. [Text front-end: phonemes vs characters](#10-text-front-end-phonemes-vs-characters)
11. [Losses and alignments](#11-losses-and-alignments)
12. [What you control at synthesis time](#12-what-you-control-at-synthesis-time)
13. [Multilingual / Tamil implications](#13-multilingual--tamil-implications)
14. [Glossary](#14-glossary)

---

## 1. What is TTS?

**Text-to-Speech (TTS)** converts written text into spoken audio.

```text
  "வணக்கம்"  ──►  TTS system  ──►  waveform (.wav)
```

A good TTS system aims for:

| Goal | Meaning |
|------|---------|
| **Intelligibility** | Words are clear and understandable |
| **Naturalness** | Sounds human, not robotic |
| **Prosody** | Correct rhythm, stress, intonation |
| **Speaker identity** | Consistent voice (single- or multi-speaker) |
| **Speed / cost** | Fast enough for products (real-time or better) |

Matcha-TTS focuses on being **fast**, **compact**, and **natural** by using **conditional flow matching** instead of slow iterative diffusion.

---

## 2. The classic TTS pipeline

Almost every modern neural TTS system still follows this logical pipeline:

```text
┌─────────────┐    ┌──────────────────┐    ┌─────────────────┐    ┌──────────┐
│ Raw text    │ -> │ Text front-end   │ -> │ Acoustic model  │ -> │ Vocoder  │
│ (string)    │    │ (normalize,      │    │ (text → mel)    │    │ (mel →   │
│             │    │  tokenize)       │    │                 │    │  waveform)│
└─────────────┘    └──────────────────┘    └─────────────────┘    └──────────┘
```

### Stage A — Text front-end

- Normalize numbers, abbreviations (optional)  
- Clean punctuation / Unicode  
- Convert text → **tokens** the model knows:
  - **Characters** (what we use with `basic_cleaners`), or  
  - **Phonemes** (what English Matcha uses with `english_cleaners2` + eSpeak)

### Stage B — Acoustic model (core neural net)

Predicts an **intermediate speech representation**, usually a **mel spectrogram** (time × frequency energy).

This is where Matcha-TTS lives.

### Stage C — Vocoder

Turns the mel spectrogram into a **waveform** (samples you can play).  
Common choice: **HiFi-GAN** (used with Matcha).

> Matcha does **not** generate raw audio by itself. It generates mels; a vocoder finishes the job.

---

## 3. How speech is represented

### Waveform

- Time-domain signal: amplitude vs time  
- At 22.05 kHz: **22,050 samples per second**  
- Directly predicting waveforms is hard and expensive — so most TTS use a 2-stage approach.

### Spectrogram / Mel spectrogram

1. Cut audio into short frames (e.g. every **256** samples ≈ hop length)  
2. FFT each frame → frequency spectrum  
3. Warp frequencies onto the **mel scale** (closer to human hearing)  
4. Take log → **log-mel spectrogram**

In this repo’s default config:

| Parameter | Value | Role |
|-----------|-------|------|
| `sample_rate` | 22050 | Audio rate |
| `n_fft` | 1024 | FFT size |
| `hop_length` | 256 | Frame shift |
| `win_length` | 1024 | Window size |
| `n_feats` / mels | 80 | Mel channels |
| `f_min` / `f_max` | 0 / 8000 | Mel frequency range |

So the acoustic model predicts a matrix roughly:

```text
mel.shape = [80, T]
```

where `T` is the number of frames (duration of the utterance).

### Why normalize mels?

Training uses dataset-wide **mean** and **std** (`data_statistics`) so mel values are roughly standardized. At inference, Matcha denormalizes before the vocoder.

---

## 4. Evolution of TTS systems

| Generation | Idea | Pros | Cons |
|------------|------|------|------|
| **Concatenative** | Stitch recorded units | Natural when units match | Large DB, brittle |
| **Statistical parametric** (HMM) | Predict features, then vocode | Small footprint | Often muffled |
| **Neural AR** (Tacotron, Transformer TTS) | Seq2seq text → mel, autoregressive | Very natural | Slow, can skip/repeat |
| **Neural NAR + duration** (FastSpeech, Grad-TTS, Matcha) | Predict durations, generate mel in parallel / few steps | Fast, stable | Needs good duration modeling |
| **End-to-end / codec LMs** | Discrete audio tokens | Flexible | Heavy compute / data |

Matcha-TTS is in the **non-autoregressive + generative decoder** family, related to **Grad-TTS**, but uses **conditional flow matching** for faster, high-quality mel generation.

---

## 5. Autoregressive vs non-autoregressive

### Autoregressive (AR)

Generate mel **frame by frame**, each frame depending on previous ones.

```text
mel_1 → mel_2 → mel_3 → … → mel_T
```

- Natural prosody possible  
- Slow (O(T) network passes)  
- Attention can fail (repeat / skip words)

### Non-autoregressive (NAR)

1. Predict **how long** each text token lasts (duration)  
2. Expand text features to mel length  
3. Generate **all mel frames** together (or with few ODE steps)

```text
text tokens ──► durations ──► aligned features ──► full mel
```

- Fast and stable alignment  
- Duration model must be good  

**Matcha-TTS is NAR** with a duration predictor + flow-matching decoder.

---

## 6. Acoustic model + vocoder

```text
Text IDs  ──►  Matcha-TTS  ──►  Mel [80 × T]  ──►  HiFi-GAN  ──►  Wav
                 (acoustic)                         (vocoder)
```

| Module | Input | Output |
|--------|-------|--------|
| Matcha | Token IDs (+ optional speaker) | Mel spectrogram |
| HiFi-GAN | Mel spectrogram | 22.05 kHz waveform |

Training Matcha does **not** require training the vocoder at the same time if you use a pretrained HiFi-GAN compatible with 80-bin mels at 22.05 kHz. For a new language/voice, a universal or fine-tuned vocoder may improve quality.

---

## 7. Matcha-TTS architecture

High-level diagram:

```text
                         ┌─────────────────────────────────────────┐
 Text + lengths ───────► │              TEXT ENCODER               │
 (token IDs)             │  Embedding → Prenet → RoPE Transformer  │
                         │           + Duration Predictor          │
                         └───────────────┬─────────────────────────┘
                                         │
                    μ (prior mel mean), logw (durations), h (features)
                                         │
                         ┌───────────────▼─────────────────────────┐
                         │         ALIGN / EXPAND BY DURATION      │
                         │   (MAS alignment in train; predict      │
                         │    durations at inference)              │
                         └───────────────┬─────────────────────────┘
                                         │
                         ┌───────────────▼─────────────────────────┐
                         │     CFM DECODER (U-Net style estimator) │
                         │   Conditional Flow Matching: noise → mel│
                         └───────────────┬─────────────────────────┘
                                         │
                                         ▼
                                   Mel spectrogram
```

### 7.1 Text embedding

- Each character/phoneme ID → vector (`nn.Embedding`, size `n_vocab`)  
- Optional blank token interleaving (`add_blank: True`) helps monotonic alignment  

### 7.2 Text encoder (RoPE Encoder)

Configured in `configs/model/encoder/default.yaml`:

| Piece | Role |
|-------|------|
| Prenet | Conv + norm stack on embeddings |
| Transformer encoder | Contextualize text (RoPE = rotary position embeddings) |
| `proj_m` | Projects to prior mel mean \(\mu\) |
| Duration predictor (`proj_w`) | Predicts log-duration per token |

Typical sizes in this repo: ~7.2M params in the encoder.

### 7.3 Duration / alignment

Speech frames ≫ text tokens. The model must learn **which mel frames belong to which character**.

- **Training:** Monotonic Alignment Search (**MAS**) finds an alignment between encoder features and target mel (same idea as Glow-TTS / Grad-TTS).  
- **Inference:** Predicted durations expand encoder outputs to mel length.  
- **`length_scale`:** Multiply durations to speak slower/faster.

### 7.4 Decoder (CFM + U-Net estimator)

The decoder refines a coarse prior into a realistic mel using **conditional flow matching**.

Configured via:

- `configs/model/cfm/default.yaml` — CFM solver (`euler`), `sigma_min`  
- `configs/model/decoder/default.yaml` — U-Net channels, attention heads, `snakebeta` activations  

Typical size: ~11M params in the decoder. **Total Matcha ≈ 18.2M** parameters in the Tamil config.

### 7.5 Multi-speaker (optional)

If `n_spks > 1`, a speaker embedding is added so one model can hold many voices (e.g. VCTK).  
Our Tamil code-mix setup is **single speaker** (`n_spks: 1`).

---

## 8. Conditional Flow Matching (CFM)

### Intuition

Generative models learn to transform **simple noise** into **complex data** (here: mel spectrograms).

| Family | Idea | Typical cost |
|--------|------|----------------|
| Diffusion (e.g. Grad-TTS) | Many denoising steps | Tens–hundreds of steps |
| **Flow matching / CFM** | Learn a velocity field along a path from noise → data | **Few ODE steps** (e.g. 2–10) |

Matcha trains a network to predict the **flow** (how to move from noise toward the target mel), conditioned on the text encoder features.

### Why it matters

- **Faster synthesis** than classic diffusion TTS  
- Quality remains strong with a small number of Euler steps (`--steps` in CLI)  

Paper: [Matcha-TTS (ICASSP 2024)](https://arxiv.org/abs/2309.03199).

---

## 9. Training vs inference

### Training (what `matcha/train.py` does)

For each batch:

1. Load wav → compute mel → normalize with dataset mean/std  
2. Text → `basic_cleaners` / phonemizer → token IDs  
3. Encoder forward  
4. Align text ↔ mel (MAS) unless using precomputed durations  
5. Duration loss + prior loss + CFM (diffusion/flow) loss  
6. Backprop with Adam (`lr: 1e-4`), mixed precision (`16-mixed`)  
7. Validate; checkpoint best `loss/val`

### Inference (synthesis)

1. Text → tokens  
2. Encoder → predicted durations → expand  
3. Sample noise; run CFM ODE for `n_timesteps` steps → mel  
4. Denormalize mel  
5. Vocoder → wav  

```text
Training target: ground-truth mel from real audio
Inference output: predicted mel → vocoder → audio
```

---

## 10. Text front-end: phonemes vs characters

### English base Matcha

```text
Text → english_cleaners2 → eSpeak phonemes (IPA-like) → symbol IDs
```

- Pros: pronunciation rules handled by phonemizer  
- Cons: English-centric; bad fit for Tamil/Kannada orthography  

### Multilingual setup in this repo (`basic_cleaners`)

```text
Text → NFC + lowercase Latin + collapse spaces → character IDs
```

- Pros: keeps Tamil/Kannada script; simple; no English phonemizer  
- Cons: model must learn pronunciation from data; OOV chars must be in `symbols.py`  

**`n_vocab`** = number of symbols in `matcha/text/symbols.py` (pad + space + letters).  
Wrong vocab ↔ wrong embeddings ↔ training/inference break (`KeyError` on unknown chars).

---

## 11. Losses and alignments

Matcha optimizes (conceptually):

| Loss | Purpose |
|------|---------|
| **Duration loss** | Predicted log-durations match MAS (or precomputed) durations |
| **Prior loss** | Encoder prior \(\mu\) stays close to aligned target mel |
| **CFM / decoder loss** | Flow network learns to map noise → mel given text conditioning |

Logged validation metric used for checkpoints:

```text
loss/val  = sum of validation loss terms
```

We save **top-3** checkpoints with lowest `loss/val`.

### Optional: precomputed durations

After a first model exists, you can extract durations and retrain with `load_durations: true` (see upstream README). That can stabilize / speed training; not required for the first Tamil run.

---

## 12. What you control at synthesis time

| Control | Effect |
|---------|--------|
| **Text** | Content |
| **`steps` (ODE steps)** | More steps → usually slightly better mel, slower |
| **`temperature`** | Noise scale / variability |
| **`speaking_rate` / `length_scale`** | Global pace (via durations) |
| **Speaker ID** | Multi-speaker models only |
| **Vocoder** | Final timbre / artifacts |

Acoustic model quality (Matcha) and vocoder quality **both** affect what you hear.

---

## 13. Multilingual / Tamil implications

| Topic | Impact |
|-------|--------|
| Script in `symbols.py` | Defines what the embedding table can represent |
| `basic_cleaners` | Keeps native orthography |
| Code-mix Latin | Keep `a–z` in symbols |
| Mel stats | Must be recomputed per dataset |
| English pretrained ckpt | **Incompatible** after vocab change — train from scratch |
| Prosody / style | Learned from your recordings (call-center style, etc.) |
| Vocoder | Universal HiFi-GAN often works; fine-tune if buzz/artifacts |

Flow for our Tamil project:

```text
Tamil+English text
   → basic_cleaners + Tamil/Latin symbols
   → Matcha encoder + duration + CFM decoder
   → mel (22.05 kHz, 80 bins)
   → HiFi-GAN
   → wav
```

Same architecture for **Kannada** or any other language: change **data**, **symbols**, **mel stats** — not the network topology.

---

## 14. Glossary

| Term | Meaning |
|------|---------|
| **Mel spectrogram** | Compact time–frequency speech features |
| **Vocoder** | Mel → waveform neural net |
| **NAR** | Non-autoregressive (parallel / few-step generation) |
| **Duration predictor** | How many mel frames each token spans |
| **MAS** | Monotonic Alignment Search — finds text–mel alignment |
| **CFM** | Conditional Flow Matching — generative path noise→data |
| **ODE / Euler steps** | Numerical integration steps during CFM sampling |
| **RoPE** | Rotary positional encoding in the text transformer |
| **`n_vocab`** | Size of text embedding table |
| **`basic_cleaners`** | Light text cleanup without English phonemization |
| **Prior \(\mu\)** | Encoder’s coarse mel estimate before CFM refinement |
| **RTF** | Real-Time Factor (audio duration / compute time); &lt; 1 is faster than real time |

---

## Architecture map to this codebase

| Concept | Code / config |
|---------|----------------|
| Full model | `matcha/models/matcha_tts.py` |
| Text encoder + duration | `matcha/models/components/text_encoder.py` |
| CFM wrapper | `matcha/models/components/flow_matching.py` |
| U-Net decoder | `matcha/models/components/decoder.py` |
| Mel extraction | `matcha/utils/audio.py` + `TextMelDataset.get_mel` |
| Text → IDs | `matcha/text/` (`cleaners.py`, `symbols.py`) |
| Train loop | `matcha/train.py` + Lightning module |
| Model sizes | `configs/model/*.yaml` |
| HiFi-GAN | `matcha/hifigan/` |

---

## Further reading

- Matcha-TTS paper: https://arxiv.org/abs/2309.03199  
- Conditional Flow Matching: https://arxiv.org/abs/2210.02747  
- Grad-TTS (related NAR diffusion TTS): precursor ideas for alignment + generative mel decoding  
- Demo: https://shivammehta25.github.io/Matcha-TTS  

---

## Next step

When you understand the pipeline, follow the practical steps in:

**[MULTILINGUAL_TRAINING_GUIDE.md](./MULTILINGUAL_TRAINING_GUIDE.md)**

to prepare data and train Tamil, Kannada, or another language with the same architecture.
