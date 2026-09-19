# VoiceShield: AI-Powered Voice Cloning Detection

An SIH demonstration project for detecting AI-generated or voice-cloned speech. The production detector uses one model only: **AASIST-L-SIH-v4**.

The application accepts an uploaded audio file or short microphone recordings from the browser and returns one of two labels:

- `REAL` — the detector selected the genuine-speech class.
- `AI_SPOOF` — the detector selected the AI-generated/voice-cloned class.

This is a research prototype. A prediction is not proof of identity or fraud, and it is not direct telephone-call interception.

## Demo

<p align="center">
<img width="1131" height="1017" alt="image" src="https://github.com/user-attachments/assets/dcfc9a25-225d-498d-8b8c-e6e01e2000e2" />
</p>

## Production Architecture

```text
Audio upload or browser microphone
            |
            v
React + Vite frontend
            |
            | POST /detect (multipart field: file)
            v
FastAPI backend
            |
            v
VoiceDetector + AASIST-L-SIH-v4.pth
            |
            v
REAL / AI_SPOOF result

## Repository layout

```text
configs/finetune.conf             AASIST-L model configuration
frontend/                         React + Vite interface
models/AASIST.py                  AASIST-L architecture source
models/AASIST-L-SIH-v4.pth        Only production model checkpoint
src/preprocessing.py              Mono conversion, 16 kHz resampling, windowing
src/detector.py                   V4 inference and robust segment aggregation
src/api.py                        FastAPI GET / and POST /detect endpoints
tests/test_preprocessing.py       Relevant windowing unit tests
requirements.txt                  Runtime Python dependencies
```

## Setup (Windows / Git Bash)

```bash
git clone https://github.com/Bhavdeepq/Ai-voice-cloning-detection.git
cd Ai-voice-cloning-detection

python -m venv .venv
source .venv/Scripts/activate
python -m pip install -r requirements.txt

cd frontend
npm install
cd ..
```

The current runtime uses Python, Node.js, and a modern browser. CUDA is selected automatically when a compatible PyTorch/CUDA installation is available; otherwise inference uses CPU.

## Run the application

Start the backend from the repository root:

```bash
python -m uvicorn src.api:app --reload
```

Start the frontend in another terminal:

```bash
cd frontend
npm run dev
```

Open the Vite address shown in the terminal, normally `http://localhost:5173`.

## Use the application

### Upload Audio

Select **Upload Audio** and choose a WAV file (recommended) or another format supported by SoundFile. The frontend sends it to the local backend and displays the latest result.

### Microphone detection

Select **Start Detection**, grant microphone access, and speak normally. The browser submits approximately four-second WAV chunks until **Stop Detection** is selected.

### Result fields

- `confidence`: probability of the selected class.
- `real_probability` / `fake_probability`: V4 aggregate class probabilities.
- `segments_analyzed`: fixed AASIST windows processed for this request.
- `decision_margin`, `segment_agreement`, and `segment_fake_probability_std`: stability diagnostics for multi-window recordings.

For longer recordings, V4 uses 50%-overlapping end-aligned windows and robustly aggregates segment scores. The model decision boundary remains `0.5`; it has not been recalibrated without held-out validation evidence.

## API

### `GET /`

```json
{"status":"ok"}
```

### `POST /detect`

Send one audio file as `multipart/form-data` under `file`:

```bash
curl -X POST -F "file=@path/to/audio.wav" http://127.0.0.1:8000/detect
```

The API loads V4 once on startup, deletes each temporary upload after inference, and returns prediction data as JSON. Browser access is allowed from Vite's default local origins.

## Checks

Build the frontend:

```bash
cd frontend
npm run build
```

The preprocessing tests use pytest. Install it locally only when running tests:

```bash
python -m pip install pytest
python -m pytest -q
```

## Model credit

The model architecture is based on the original [clovaai/aasist](https://github.com/clovaai/aasist) project.
