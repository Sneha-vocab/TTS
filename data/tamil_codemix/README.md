# Tamil code-mix dataset (Matcha-TTS)

- **Language**: Tamil + English code-mix  
- **Cleaners**: `basic_cleaners`  
- **Speaker**: single female (`SAR_TN_F_SPK01`)  
- **Sample rate**: source WAVs 48 kHz; dataloader resamples to 22.05 kHz  
- **Splits**: train 5383 / val 298 / test 298 (90% / 5% / 5%)  
- **Mel stats**: mean `-4.715458`, std `2.458284`  

Full step-by-step guide (English base → Tamil → reuse for Kannada / other languages):

**[../../MULTILINGUAL_TRAINING_GUIDE.md](../../MULTILINGUAL_TRAINING_GUIDE.md)**

TTS concepts & Matcha architecture:

**[../../TTS_ARCHITECTURE.md](../../TTS_ARCHITECTURE.md)**

## Train

```bash
conda activate matcha-tts
cd Matcha-TTS
python matcha/train.py experiment=tamil_codemix
# low VRAM: python matcha/train.py experiment=tamil_codemix_min_memory
```
