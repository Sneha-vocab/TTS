#!/usr/bin/env python3

import os
import time
import wave
import base64
from sarvamai import SarvamAI

# =========================
# Configuration
# =========================
API_KEY        = os.environ.get("SARVAM_API_KEY", "")
if not API_KEY:
    raise SystemExit("Set SARVAM_API_KEY in the environment before running this script.")
LANG           = "TN"
LANG_CODE      = "ta-IN"
GENDER         = "F"
SPEAKER_ID     = "SPK01"
SARVAM_SPEAKER = "simran"
MODEL          = "bulbul:v3"
SAMPLE_RATE    = 48000
PACE           = 1.0
INPUT_FILE     = "code_mix_tamil_idformat.txt"
OUTPUT_FOLDER  = "wavs_tamil_code_mix"
METADATA_FILE  = "metadata_tamil_code_mix.csv"
RETRY_LIMIT    = 3
SLEEP_TIME     = 1.2

# =========================
# Init
# =========================
client = SarvamAI(api_subscription_key=API_KEY)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# =========================
# Read sentences
# =========================
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    lines = f.readlines()

# =========================
# Existing metadata
# =========================
existing_files = set()

if os.path.exists(METADATA_FILE):
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            wav_path = line.split("|", 1)[0].strip()

            existing_files.add(wav_path)

metadata = open(METADATA_FILE, "a", encoding="utf-8")

# =========================
# WAV helper
# =========================
def pcm_to_wav(
    pcm_bytes: bytes,
    sample_rate: int,
    output_path: str,
    num_channels: int = 1,
    bit_depth: int = 16,
):
    with wave.open(output_path, "wb") as wf:
        wf.setnchannels(num_channels)
        wf.setsampwidth(bit_depth // 8)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)

# =========================
# Counters
# =========================
total     = len([l for l in lines if l.strip()])
generated = 0
skipped   = 0
failed    = 0

# =========================
# Main loop
# =========================
for line in lines:
    line = line.strip()

    if not line or "|" not in line:
        continue

    utt_id, text = line.split("|", 1)

    utt_id = utt_id.strip()
    text   = text.strip()

    filename = f"SAR_{LANG}_{GENDER}_{SPEAKER_ID}_{utt_id}.wav"
    filepath = os.path.join(OUTPUT_FOLDER, filename)

    metadata_row = f"wavs/{filename}|{text}"
    wav_key      = f"wavs/{filename}"

    # ====================================
    # Skip existing wavs
    # ====================================
    if os.path.exists(filepath) and os.path.getsize(filepath) > 1000:
        print(f"Skipping: {filename}")

        if wav_key not in existing_files:
            metadata.write(metadata_row + "\n")
            metadata.flush()
            existing_files.add(wav_key)

        skipped += 1
        continue

    success = False
    attempt = 0

    # ====================================
    # Retry loop
    # ====================================
    while not success and attempt < RETRY_LIMIT:
        try:
            response = client.text_to_speech.convert(
                text=text,
                target_language_code=LANG_CODE,
                speaker=SARVAM_SPEAKER,
                pace=PACE,
                speech_sample_rate=SAMPLE_RATE,
                enable_preprocessing=True,
                model=MODEL,
            )

            if not response.audios:
                raise ValueError("Empty audio response")

            audio_base64 = response.audios[0]

            pcm_bytes = base64.b64decode(audio_base64)

            pcm_to_wav(
                pcm_bytes=pcm_bytes,
                sample_rate=SAMPLE_RATE,
                output_path=filepath,
            )

            file_size = os.path.getsize(filepath)

            if file_size <= 1000:
                raise ValueError(
                    f"Small wav generated ({file_size} bytes)"
                )

            if wav_key not in existing_files:
                metadata.write(metadata_row + "\n")
                metadata.flush()
                existing_files.add(wav_key)

            generated += 1

            print(
                f"[{generated+skipped}/{total}] "
                f"Generated: {filename} "
                f"({file_size/1024:.1f} KB)"
            )

            success = True

        except Exception as e:
            attempt += 1

            print(
                f"Attempt {attempt}/{RETRY_LIMIT} failed "
                f"for {utt_id}: {e}"
            )

            time.sleep(2)

    if not success:
        print(f"FAILED: {utt_id}")
        failed += 1

    time.sleep(SLEEP_TIME)

# =========================
# Done
# =========================
metadata.close()

print(f"""
=========================
Synthesis completed
=========================
Total lines : {total}
Generated   : {generated}
Skipped     : {skipped}
Failed      : {failed}
=========================
""")
