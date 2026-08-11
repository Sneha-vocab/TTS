# Matcha-TTS Multilingual Training Guide

End-to-end guide to adapt **base (English) Matcha-TTS** to a new language (Tamil, Kannada, etc.) using `basic_cleaners` (character-level, no English phonemizer).

This document mirrors the Tamil code-mix setup in this repo. Follow the same steps for Kannada or any other language.

**Concepts & architecture (what TTS is, mel, vocoder, Matcha CFM):** see **[TTS_ARCHITECTURE.md](./TTS_ARCHITECTURE.md)**.

---

## Table of contents

1. [Overview](#1-overview)
2. [What changes from English](#2-what-changes-from-english)
3. [Environment setup](#3-environment-setup)
4. [Hardware check](#4-hardware-check)
5. [Prepare audio + transcripts](#5-prepare-audio--transcripts)
6. [Normalize text & build character set](#6-normalize-text--build-character-set)
7. [Train / val / test split](#7-train--val--test-split)
8. [Update Matcha for your language](#8-update-matcha-for-your-language)
9. [Compute mel statistics](#9-compute-mel-statistics)
10. [Checkpointing](#10-checkpointing)
11. [Train](#11-train)
12. [Inference (after training)](#12-inference-after-training)
13. [Checklist for a new language (e.g. Kannada)](#13-checklist-for-a-new-language-eg-kannada)
14. [Troubleshooting](#14-troubleshooting)
15. [Tamil reference paths](#15-tamil-reference-paths)

---



## 1. Overview


| Item                    | English (base)                        | Multilingual (this guide)                                             |
| ----------------------- | ------------------------------------- | --------------------------------------------------------------------- |
| Text pipeline           | `english_cleaners2` (eSpeak phonemes) | `basic_cleaners` (raw script + optional Latin)                        |
| Symbols                 | Latin + IPA (`n_vocab ≈ 178`)         | Your script (+ Latin if code-mix)                                     |
| Data config             | `configs/data/ljspeech.yaml`          | `configs/data/<lang>.yaml`                                            |
| Experiment              | `experiment=ljspeech`                 | `experiment=<lang>`                                                   |
| Pretrained English ckpt | Usable as-is                          | **Not compatible** after symbol/`n_vocab` change — train from scratch |


**Tamil example in this repo**

- Language: Tamil + English code-mix  
- Cleaner: `basic_cleaners`  
- Layout: `data/tamil_codemix/`  
- Train command: `python matcha/train.py experiment=tamil_codemix`

---



## 2. What changes from English

You must update:

1. `matcha/text/symbols.py` — character vocabulary for your script
2. `configs/model/matcha.yaml` — `n_vocab: <len(symbols)>`
3. `configs/data/<dataset>.yaml` — paths, `cleaners: [basic_cleaners]`, mel stats
4. `configs/experiment/<dataset>.yaml` — experiment entry point
5. **Filelists** — `path|text` lines
6. **(Optional)** dataloader resample if WAVs ≠ 22.05 kHz

You should **not** use `english_cleaners2` for Indic scripts (it phonemizes as English and will break or mangle text).

---



## 3. Environment setup

Always use the project conda env (not `(base)`):

```bash
conda activate matcha-tts
cd /path/to/Matcha-TTS

# Install this checkout as editable (important if multiple Matcha trees exist)
pip install -e . --no-build-isolation

# Avoid ~/.local PyTorch overriding the env (driver/CUDA mismatch)
export PYTHONNOUSERSITE=1
# (already set automatically on activate if the conda hook was installed)
```

Quick sanity check:

```bash
python -c "import torch, hydra, matcha; from matcha.text.symbols import symbols; print(torch.__version__, torch.cuda.is_available(), len(symbols), matcha.__file__)"
```

Expect: CUDA `True`, `matcha.__file__` pointing to **this** repo, and `len(symbols)` matching your language vocab.

---



## 4. Hardware check

Matcha single-GPU training is designed around ~24 GB class GPUs.


| Resource | Typical need                                                   |
| -------- | -------------------------------------------------------------- |
| GPU      | CUDA GPU, ~12–24 GB VRAM (batch 32; use `*_min_memory` if OOM) |
| RAM      | ≥ 16 GB free recommended                                       |
| Disk     | Dataset + checkpoints (tens of GB)                             |


Low VRAM:

```bash
python matcha/train.py experiment=<name>_min_memory
# or: data.batch_size=16
```

---



## 5. Prepare audio + transcripts



### Required layout

```text
data/<dataset_name>/
  wavs/                 # one wav per utterance
  train.txt
  val.txt
  test.txt              # optional but recommended
```



### Filelist format (LJSpeech-style)

Each line:

```text
data/<dataset_name>/wavs/<utt_id>.wav|<transcript>
```

Example:

```text
data/tamil_codemix/wavs/SAR_TN_F_SPK01_11780.wav|வணக்கம் pavithra அவர்களே insurance supportக்கு உங்களை வரவேற்கிறோம்
```



### Audio requirements

- Mono preferred  
- Matcha mel config expects **22.05 kHz**  
- If your WAVs are 48 kHz (e.g. Sarvam), either:
  - resample offline to 22.05 kHz, **or**
  - keep on-the-fly resample in `TextMelDataset.get_mel` (already enabled in this repo)



### Transcript tips

- One utterance ↔ one wav  
- Prefer NFC Unicode normalization  
- For code-mix: keep native script + Latin as spoken

---



## 6. Normalize text & build character set



### 6.1 Extract unique characters

```python
from collections import Counter
from pathlib import Path

chars = Counter()
for line in Path("metadata.csv").read_text(encoding="utf-8").splitlines():
    if "|" not in line:
        continue
    text = line.split("|", 1)[1]
    chars.update(text)

for c, n in sorted(chars.items(), key=lambda x: (-x[1], ord(x[0]))):
    print(repr(c), hex(ord(c)), n)
```



### 6.2 Typical cleanup for TTS

Remove punctuation that is not useful as symbols, e.g.:

- `,` `.` `?` `-` `;` `:` `"` `'` etc.

Also:

- Unicode **NFC** normalize  
- Collapse repeated whitespace  
- Do **not** strip language letters

`basic_cleaners` will additionally **lowercase Latin**; native scripts stay as-is.

### 6.3 Update `matcha/text/symbols.py`

Pattern used for Tamil (adapt for Kannada, etc.):

```python
_pad = "_"
_space = " "
_letters_latin = "abcdefghijklmnopqrstuvwxyz"  # keep if code-mix / English words
_letters_lang = "..."  # full orthography for your language

symbols = []
seen = set()
for ch in [_pad, _space] + list(_letters_latin) + list(_letters_lang):
    if ch not in seen:
        seen.add(ch)
        symbols.append(ch)

SPACE_ID = symbols.index(" ")
```

**Kannada tip:** put the Kannada Unicode block letters/matras in `_letters_lang` (and Latin if code-mix). Prefer the **full** orthography, not only characters seen once in a small dump.

### 6.4 Set `n_vocab`

```bash
python -c "from matcha.text.symbols import symbols; print(len(symbols))"
```

Set in `configs/model/matcha.yaml`:

```yaml
n_vocab: 77   # must equal len(symbols)
```



### 6.5 Validate every transcript encodes

```python
from pathlib import Path
from matcha.text import text_to_sequence

for split in ["train", "val", "test"]:
    for line in Path(f"data/<dataset>/{split}.txt").read_text(encoding="utf-8").splitlines():
        text = line.split("|", 1)[1]
        text_to_sequence(text, ["basic_cleaners"])
print("OK")
```

Any `KeyError` means a character is missing from `symbols.py`.

---



## 7. Train / val / test split

Recommended: **90% train / 5% val / 5% test**, shuffled with a fixed seed.

```python
import random
from pathlib import Path

rows = [ln.strip() for ln in Path("metadata_normalized.csv").read_text(encoding="utf-8").splitlines() if ln.strip()]
rng = random.Random(42)
rng.shuffle(rows)

n = len(rows)
n_test = int(n * 0.05)
n_val = int(n * 0.05)
test, val, train = rows[:n_test], rows[n_test:n_test + n_val], rows[n_test + n_val:]

out = Path("data/<dataset>")
out.mkdir(parents=True, exist_ok=True)
for name, data in [("train.txt", train), ("val.txt", val), ("test.txt", test)]:
    (out / name).write_text("\n".join(data) + "\n", encoding="utf-8")
```

Ensure wav paths in those files are relative to the **Matcha-TTS project root**.

---



## 8. Update Matcha for your language



### 8.1 Data config — `configs/data/<dataset>.yaml`

```yaml
_target_: matcha.data.text_mel_datamodule.TextMelDataModule
name: <dataset>
train_filelist_path: data/<dataset>/train.txt
valid_filelist_path: data/<dataset>/val.txt
batch_size: 32
num_workers: 8
pin_memory: True
cleaners: [basic_cleaners]
add_blank: True
n_spks: 1
n_fft: 1024
n_feats: 80
sample_rate: 22050
hop_length: 256
win_length: 1024
f_min: 0
f_max: 8000
data_statistics:          # fill after step 9
  mel_mean: 0.0
  mel_std: 1.0
seed: ${seed}
load_durations: false
```



### 8.2 Experiment config — `configs/experiment/<dataset>.yaml`

```yaml
# @package _global_

defaults:
  - override /data: <dataset>.yaml

tags: ["<lang>", "basic_cleaners"]
run_name: <dataset>
```

Optional low-memory variant (`configs/experiment/<dataset>_min_memory.yaml`):

```yaml
# @package _global_

defaults:
  - override /data: <dataset>.yaml

tags: ["<lang>", "min_memory"]
run_name: <dataset>_min

model:
  out_size: 172
```



### 8.3 Cleaners

Use:

```yaml
cleaners: [basic_cleaners]
```

`basic_cleaners` (in this repo): NFC normalize → lowercase Latin → collapse whitespace. No transliteration, no eSpeak.

### 8.4 Makefile (optional)

```makefile
train-<dataset>:
	python matcha/train.py experiment=<dataset>
```

---



## 9. Compute mel statistics

Mel mean/std must be computed on **your** train set (with the same sample rate / mel settings).

If Hydra helpers are installed:

```bash
python matcha/utils/generate_data_statistics.py -i <dataset>.yaml -b 32 -f
```

Or a standalone pass over `train.txt` (resample to 22.05 kHz if needed), then paste values into `configs/data/<dataset>.yaml`:

```yaml
data_statistics:
  mel_mean: -4.715458
  mel_std: 2.458284
```

**Tamil reference values (this repo):** `mel_mean: -4.715458`, `mel_std: 2.458284`.

---



## 10. Checkpointing

Best checkpoints by validation loss (`configs/callbacks/model_checkpoint.yaml`):

```yaml
model_checkpoint_best:
  _target_: lightning.pytorch.callbacks.ModelCheckpoint
  dirpath: ${paths.output_dir}/checkpoints
  filename: "best-{epoch:03d}"
  monitor: "loss/val"
  verbose: False
  save_last: false
  save_top_k: 3
  mode: "min"
  auto_insert_metric_name: False
  every_n_epochs: 1
```

Metric `loss/val` is logged by the Lightning module. Keep the top-3 lowest val-loss checkpoints.

---



## 11. Train

```bash
conda activate matcha-tts
cd /path/to/Matcha-TTS
export PYTHONNOUSERSITE=1

python matcha/train.py experiment=<dataset>
```

Tamil:

```bash
python matcha/train.py experiment=tamil_codemix
```

Notes:

- Default `trainer.max_epochs: -1` runs until you stop (Ctrl+C).  
- Quality for single-speaker setups is often discussed in **steps** (~200k–300k), not a fixed epoch count.  
- Prefer the best `loss/val` checkpoint over the latest step.  
- Logs/checkpoints: `logs/train/<run_name>/runs/<timestamp>/`

TensorBoard:

```bash
tensorboard --logdir logs/train
```

---



## 12. Inference (after training)

```bash
python -m matcha.cli \
  --checkpoint_path /path/to/best-XXX.ckpt \
  --text "உங்கள் வாக்கியம் இங்கே" \
  --cleaner basic_cleaners
```

You still need a vocoder (e.g. HiFi-GAN) matching 22.05 kHz mels for waveform output.

---



## 13. Checklist for a new language (e.g. Kannada)

Copy this list when starting Kannada / Hindi / Telugu / etc.

- [ ] Collect wavs + transcripts (`utt|text` or `path|text`)  
- [ ] Normalize text (NFC, strip unwanted punctuation)  
- [ ] Extract unique chars; build `symbols.py` for that script (+ Latin if code-mix)  
- [ ] Set `n_vocab = len(symbols)` in `configs/model/matcha.yaml`  
- [ ] Use `cleaners: [basic_cleaners]` (not `english_cleaners2`)  
- [ ] Split 90 / 5 / 5 → `data/<dataset>/{train,val,test}.txt`  
- [ ] Point paths from Matcha root; verify every wav exists  
- [ ] Confirm sample rate policy (22.05 kHz or resample)  
- [ ] Add `configs/data/<dataset>.yaml` + `configs/experiment/<dataset>.yaml`  
- [ ] Compute mel mean/std; paste into data yaml  
- [ ] `pip install -e .` from **this** repo (avoid wrong editable install)  
- [ ] `conda activate matcha-tts` + `PYTHONNOUSERSITE=1`  
- [ ] Train: `python matcha/train.py experiment=<dataset>`  
- [ ] Pick best `loss/val` checkpoint; listen to samples  



### Kannada-specific notes


| Topic              | Guidance                                                                  |
| ------------------ | ------------------------------------------------------------------------- |
| Script             | Kannada Unicode (U+0C80–U+0CFF) in `symbols.py`                           |
| Cleaner            | `basic_cleaners`                                                          |
| Code-mix           | Keep `a–z` in symbols if English words appear                             |
| Digits             | Prefer spoken word forms in transcripts, or add `0–9` to symbols if kept  |
| English pretrained | Do not fine-tune English Matcha after changing vocab — train from scratch |


---



## 14. Troubleshooting


| Error                                    | Fix                                                                                                        |
| ---------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| `ModuleNotFoundError: hydra`             | `conda activate matcha-tts` (not `base`)                                                                   |
| `NVIDIA driver is too old` / wrong torch | User-site torch override → `export PYTHONNOUSERSITE=1`; use env torch (~cu128 with driver 550 / CUDA 12.4) |
| `KeyError: 'ಅ'` / `'அ'`                  | Character missing in `symbols.py`, or editable install points to another Matcha tree                       |
| Wrong `matcha.__file__` path             | `pip uninstall matcha-tts` then `pip install -e .` from this repo                                          |
| `tostring_rgb` / matplotlib plot crash   | Use Agg backend + `buffer_rgba()` (fixed in `matcha/utils/utils.py`)                                       |
| `assert sr == sample_rate`               | Resample wavs or keep on-the-fly resample in dataloader                                                    |
| CUDA OOM                                 | `experiment=*_min_memory` or lower `data.batch_size`                                                       |


Verify which code is imported:

```bash
python -c "import matcha; from matcha.text.symbols import symbols; print(matcha.__file__); print(len(symbols))"
```

---



## 15. Tamil reference paths


| Item                | Path                                                               |
| ------------------- | ------------------------------------------------------------------ |
| Wavs                | `data/tamil_codemix/wavs/` (symlink to `data/wavs_tamil_code_mix`) |
| Splits              | `data/tamil_codemix/{train,val,test}.txt`                          |
| Normalized metadata | `data/metadata_tamil_code_mix_normalized.csv`                      |
| Char reports        | `data/unique_chars_*.txt`, `data/removed_chars_report.txt`         |
| Data config         | `configs/data/tamil_codemix.yaml`                                  |
| Experiment          | `configs/experiment/tamil_codemix.yaml`                            |
| Symbols             | `matcha/text/symbols.py`                                           |
| Mel stats JSON      | `data/tamil_codemix/data_statistics.json`                          |




### Tamil train command

```bash
conda activate matcha-tts
cd Matcha-TTS
python matcha/train.py experiment=tamil_codemix
```

---



## Flow summary

```text
Raw wavs + transcripts
        │
        ▼
 Text normalize (NFC, drop punct) + unique chars
        │
        ▼
 symbols.py + n_vocab + basic_cleaners
        │
        ▼
 Split 90/5/5 filelists under data/<dataset>/
        │
        ▼
 Data + experiment YAML + mel mean/std
        │
        ▼
 Train (best loss/val checkpoints)
        │
        ▼
 Inference with --cleaner basic_cleaners
```

For any new language: **same pipeline** — only `symbols.py`, filelists, and YAML names change.