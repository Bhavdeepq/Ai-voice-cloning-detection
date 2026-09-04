# AI Voice Cloning Detection

An AASIST based detector for classifying a 16 kHz WAV recording as **REAL** or
**SPOOF**. The repository contains the final FoR-tuned demo checkpoint and the
source required to run or retrain it.

## Run the detector

Create an environment and install the dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Then classify a WAV file:

```powershell
python -m src.detect path\to\audio.wav
```

The detector accepts 16 kHz WAV files. Stereo files are converted to mono and
audio shorter than four seconds is repeated before inference.

## Project layout

- `src/detect.py` — command-line inference entry point.
- `models/AASIST.py` — AASIST-L architecture.
- `models/AASIST.pth` — official full AASIST checkpoint.
- `models/AASIST-L-FoR-finetuned.pth` — final demo checkpoint.
- `models/AASIST-L.pth` — base checkpoint used for fine-tuning.
- `configs/` and `scripts/` — fine-tuning configuration and training utilities.

Training datasets, virtual environments, experiment outputs, logs, and temporary
model checkpoints are deliberately excluded from version control.

## Evaluate full AASIST

With ASVspoof 2019 LA available at the path in `configs/AASIST.conf`, run a
reproducible balanced development-set evaluation:

```powershell
python scripts/evaluate_aasist.py --samples-per-class 100
```
