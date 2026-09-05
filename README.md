# AI-Powered Voice Cloning Detection

SIH 2026 inference backend for **AI-Powered Real-Time Detection and Prevention of Voice Cloning Impersonation Attacks**. It ships exactly one immutable, validated detector checkpoint: `models/AASIST-L-SIH-v3.pth`.

## Run inference

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m src.detector demo\demo_real.wav
```

```python
from src import VoiceDetector

detector = VoiceDetector()
result = detector.predict("audio.wav")
```

The result contains `prediction` (`REAL` or `AI_SPOOF`), `confidence`, `real_probability`, `fake_probability`, and `segments_analyzed`. Audio is mixed to mono, resampled to 16 kHz, repeat-padded when short, and analyzed in AASIST's 4.04-second windows. CUDA is selected when available; otherwise inference uses CPU. No datasets, downloads, or training code are required.

## Validated V3 baseline

| Metric | Result |
| --- | ---: |
| Validation Accuracy | 96.9% |
| In-the-Wild fixed benchmark | 92.93% |
| In-the-Wild Real Accuracy | 95.96% |
| In-the-Wild Fake Accuracy | 89.90% |
| FoR external test | 91.54% |
| FoR Real Accuracy | 84.01% |
| FoR Fake Accuracy | 99.08% |

## Layout

```text
models/AASIST.py                 # AASIST-L architecture required by V3
models/AASIST-L-SIH-v3.pth       # the only checkpoint
configs/finetune.conf            # immutable V3 architecture configuration
src/preprocessing.py             # mono conversion, 16 kHz resampling, windows
src/detector.py                  # VoiceDetector API and CLI
demo/                            # optional local smoke-test WAVs (git-ignored)
```

Install the appropriate PyTorch CPU/CUDA build for the deployment platform. The checkpoint is never modified or downloaded at runtime.
