# AI-Powered Voice Cloning Detection

An SIH project that analyzes an audio recording and classifies it as either **REAL** (likely genuine human speech) or **AI_SPOOF** (likely AI-generated or voice-cloned speech). It includes a Python/FastAPI inference service and a React browser demonstration.

Voice deepfake detection means looking for patterns in an audio signal that can help distinguish natural speech from speech synthesized or modified by an AI system. This is a research prototype, not a guarantee that every generated voice will be detected.

The live demonstration records short microphone chunks (about four seconds), uploads each chunk over HTTP to the local API, and displays the next result. It does **not** intercept a phone call or cellular call directly.

## What is in this project?

```text
Audio file or microphone
          |
          v
React + Vite frontend
          |
          | POST /detect (multipart form upload)
          v
FastAPI service
          |
          v
VoiceDetector
          |
          v
AASIST-L-SIH-v3 checkpoint
          |
          v
REAL or AI_SPOOF
          |
          v
Frontend result card
```

- **Audio file:** A user can upload a completed WAV (or another audio format supported by the backend decoder).
- **Microphone:** The browser captures microphone samples, creates WAV chunks, and sends approximately one chunk every four seconds.
- **Frontend:** Provides Start Detection, Stop Detection, Upload Audio, live status, result, confidence, probabilities, and segment/chunk counts.
- **FastAPI:** Receives the file in the `file` form field, saves it temporarily, calls the detector, returns JSON, then removes the temporary file.
- **VoiceDetector:** Loads the single V3 checkpoint once at API startup and runs inference.
- **AASIST-L-SIH-v3:** The current fine-tuned model baseline used to produce the two labels.

## Technology stack

- **Python** — backend and inference code.
- **PyTorch** — model loading and inference.
- **AASIST / AASIST-L** — the anti-spoofing model architecture.
- **FastAPI** and **Uvicorn** — local HTTP API and development server.
- **React + Vite** — browser user interface and frontend build tool.
- **JavaScript, HTML, and CSS** — microphone capture, WAV creation, and interface styling.
- **Git and GitHub** — source control and collaboration.

## Repository structure

```text
AI_Cloned_Voice_Detection/
├── configs/
│   └── finetune.conf              # JSON model configuration for the V3 checkpoint
├── frontend/
│   ├── src/
│   │   ├── App.jsx                # UI, microphone chunks, upload, and API calls
│   │   ├── main.jsx               # React entry point
│   │   ├── styles.css             # Responsive frontend styles
│   │   └── wav.js                 # Browser PCM-to-WAV conversion for microphone chunks
│   ├── index.html                 # Vite HTML entry point
│   ├── package.json               # Frontend scripts and dependencies
│   └── package-lock.json          # Locked frontend dependency tree
├── models/
│   ├── AASIST.py                  # AASIST-L model architecture
│   └── AASIST-L-SIH-v3.pth        # Current V3 model checkpoint
├── src/
│   ├── __init__.py                # Exposes VoiceDetector as a package API
│   ├── api.py                     # FastAPI app, CORS configuration, GET / and POST /detect
│   ├── detector.py                # VoiceDetector class and command-line inference entry point
│   └── preprocessing.py           # Mono conversion, 16 kHz resampling, audio windowing
├── tests/
│   └── test_preprocessing.py      # Tests for window splitting and final-window padding
├── requirements.txt               # Pinned Python runtime dependencies
├── .gitignore                     # Python/local-data ignore rules
└── README.md                      # This guide
```

`demo/` may contain local WAV files for manual experiments, but `demo/*.wav` is ignored by Git. Do not assume those files will be included in a fresh clone.

## Requirements

You need Git, Python with `pip`, Node.js with `npm`, and a modern desktop browser (for example, Chrome or Edge) for microphone capture.

The repository does not declare a minimum Python or Node.js version. The current development environment was created with **Python 3.14.4** and uses **Node.js v24.14.1**. Use a compatible current Python and Node.js installation if those exact versions are not available.

The Python dependencies are pinned in `requirements.txt`:

```text
annotated-doc==0.0.5
annotated-types==0.8.0
anyio==4.15.1
cffi==2.1.1
click==8.5.0
fastapi==0.141.1
filelock==3.32.5
fsspec==2026.7.0
h11==0.16.0
idna==3.19
Jinja2==3.1.6
MarkupSafe==3.0.3
mpmath==1.3.0
networkx==3.6.1
numpy==2.5.2
pycparser==3.0
pydantic==2.13.5
pydantic_core==2.46.5
python-multipart==0.0.32
scipy==1.18.1
setuptools==84.0.0
soundfile==0.14.0
starlette==1.6.0
sympy==1.14.0
torch==2.14.0
typing-inspection==0.4.4
typing_extensions==4.16.0
uvicorn==0.52.4
```

The frontend dependency manifest includes React, React DOM, Vite, and the Vite React plugin. `frontend/package-lock.json` records the resolved versions installed by npm.

`.venv/` and `frontend/node_modules/` are local machine folders. Do **not** commit them. The project ignore files already exclude them, along with the generated frontend `dist/` directory.

## Setup from zero (Windows with Git Bash)

Open **Git Bash** and run the following commands. The GitHub remote currently configured for this repository is used below.

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

If `python` is not recognized in Git Bash, install Python from the official Python installer and reopen the terminal. On some Windows installations, `py -m venv .venv` works instead of `python -m venv .venv`.

### Start the backend

In the repository root, with the virtual environment activated:

```bash
python -m uvicorn src.api:app --reload
```

The API starts at `http://127.0.0.1:8000`. Leave this terminal running.

### Start the frontend

Open a second Git Bash terminal:

```bash
cd Ai-voice-cloning-detection
cd frontend
npm run dev
```

Vite normally prints `http://localhost:5173`. Open that address in your browser. The backend currently permits browser POST requests from both `http://localhost:5173` and `http://127.0.0.1:5173`.

### Windows PowerShell activation note

If you use PowerShell instead of Git Bash, activate the environment with:

```powershell
.\.venv\Scripts\Activate.ps1
```

Git Bash uses forward slashes (`source .venv/Scripts/activate`); PowerShell commonly uses backslashes.

## How to use the application

### A. Upload Audio

1. Start the backend and frontend.
2. In the browser, choose **Upload Audio**.
3. Select a WAV file (recommended) or another audio type supported by the backend.
4. The UI changes to **Analyzing** while it sends the file to the local API.
5. Read the result in the Latest detection card.

This is for a completed recording and sends one request for the selected file.

### B. Microphone detection

1. Select **Start Detection**.
2. Allow the browser microphone permission request.
3. Speak normally near the microphone.
4. The browser captures audio and submits approximately four-second WAV chunks to the same local API.
5. The latest result appears after each completed request. Select **Stop Detection** to end microphone capture.

This is chunk-based microphone analysis over HTTP. It is not WebSocket/WebRTC streaming and is not phone-call interception.

### C. Understanding the result

- **Prediction:** `REAL` means the detector selected the genuine-speech class. `AI_SPOOF` means it selected the AI-generated/voice-cloned class.
- **Confidence:** The probability of the class selected as the prediction.
- **Real probability:** The model's mean probability for the `REAL` class.
- **AI/Fake probability:** The model's mean probability for the `AI_SPOOF` class.
- **Segments analyzed:** How many fixed model windows were taken from that uploaded file or chunk.
- **Chunks analyzed:** The frontend count of successful microphone requests; after a completed-file upload it is set to one.

Treat these values as model outputs, not as proof of identity or a guarantee of fraud.

## Backend API

The FastAPI application is defined in `src/api.py`.

### `GET /`

Health/status endpoint:

```bash
curl http://127.0.0.1:8000/
```

Response:

```json
{"status":"ok"}
```

### `POST /detect`

Accepts one uploaded audio file using `multipart/form-data`. The field name must be `file`.

```bash
curl -X POST \
  -F "file=@path/to/your-audio.wav" \
  http://127.0.0.1:8000/detect
```

Successful response shape:

```json
{
  "prediction": "REAL",
  "confidence": 0.0,
  "real_probability": 0.0,
  "fake_probability": 0.0,
  "segments_analyzed": 1
}
```

The numeric values and label shown above are an example of the response format, not a promised result. The service returns HTTP `400` for a missing, empty, or unreadable audio upload. It saves accepted uploads to a temporary file only for inference and deletes that file afterward.

## Running and testing the detector

Run one supported local audio file through the detector command-line interface:

```bash
python -m src.detector path/to/your-audio.wav
```

If you have local sample WAVs in `demo/`, for example:

```bash
python -m src.detector demo/demo_real.wav
```

A successful command prints JSON containing `prediction`, `confidence`, `real_probability`, `fake_probability`, and `segments_analyzed`. It may take longer on CPU than on CUDA.

The repository includes preprocessing tests in `tests/test_preprocessing.py`. They use `pytest`, which is **not** included in `requirements.txt`. To run that existing test locally, install pytest into your local virtual environment, then run it:

```bash
python -m pip install pytest
python -m pytest -q
```

A successful test run reports two passing tests. These tests check audio-window splitting and repeat padding; they do not evaluate model accuracy.

Build the frontend before a demo or deployment check:

```bash
cd frontend
npm run build
```

A successful build ends with `built in ...` and writes generated files to `frontend/dist/`.

## ML model and audio processing

This repository uses **AASIST-L**, an anti-spoofing neural-network architecture. The current project baseline is the fine-tuned checkpoint:

```text
models/AASIST-L-SIH-v3.pth
```

`VoiceDetector` only accepts that checkpoint name and loads its architecture values from `configs/finetune.conf`.

Before inference, the backend:

1. Loads the audio with SoundFile.
2. Mixes multiple channels down to mono.
3. Resamples audio to **16 kHz** when necessary.
4. Splits it into fixed windows of `64600` samples, which is about **4.04 seconds** at 16 kHz.
5. Repeat-pads only the final short window.
6. Averages the per-window probabilities and returns either `REAL` or `AI_SPOOF`.

V3 is the current baseline. A future V4 can be introduced as a backend/model update while keeping the frontend's `/detect` request and response contract, so the user interface does not need a redesign. The current detector intentionally validates and accepts only the V3 checkpoint until such an update is made.

## Git workflow for beginners

Use `main` as the stable branch and make new work on a descriptive branch.

```bash
git checkout main
git pull origin main
git checkout -b feature/improve-upload-ui

# edit files, then inspect what changed
git status
git add frontend/src/App.jsx frontend/src/styles.css
git commit -m "Improve audio upload UI"
git push -u origin feature/improve-upload-ui
```

Then open a pull request on GitHub from your feature branch into `main`. A pull request lets teammates review the changes before they are merged.

Before starting new work, update your branch:

```bash
git checkout main
git pull origin main
```

Never commit `.venv/`, `frontend/node_modules/`, or `frontend/dist/`. They are machine-generated or local dependency folders, can be recreated from the dependency manifests, and would make the repository unnecessarily large.

## Troubleshooting

### Virtual environment will not activate

In Git Bash, use `source .venv/Scripts/activate`. In PowerShell, use `.\.venv\Scripts\Activate.ps1`. If PowerShell blocks scripts, use Git Bash or follow your organization's PowerShell execution-policy guidance.

### Git Bash paths look different from Windows paths

Use `/` in Git Bash commands, such as `demo/demo_real.wav`. Use `\` in typical PowerShell paths, such as `demo\demo_real.wav`.

### FastAPI does not start

Check that the virtual environment is active and dependencies are installed:

```bash
python -m pip install -r requirements.txt
python -m uvicorn src.api:app --reload
```

Also verify that `models/AASIST-L-SIH-v3.pth` exists. The detector raises an error at startup if the V3 checkpoint is missing.

### Frontend does not start

Run the commands from the `frontend/` directory:

```bash
npm install
npm run dev
```

If `npm` is not recognized, install Node.js and reopen Git Bash.

### Browser shows a CORS or network error

Start the backend first and keep it running at `http://127.0.0.1:8000`. The API currently allows frontend requests only from the default Vite origins `http://localhost:5173` and `http://127.0.0.1:5173`. If you intentionally run the frontend on another origin, add that exact origin to the CORS allow-list in `src/api.py`.

### Microphone does not work

Use a modern browser, allow microphone access when prompted, and check that the correct microphone is enabled in operating-system/browser settings. The app shows a permission error if the browser refuses access.

### The backend is unavailable while using the frontend

The frontend catches request failures and shows an error message. Start or restart the backend, then try a new upload or microphone session.

### CUDA is unavailable

This is expected on machines without a usable CUDA-enabled PyTorch installation. `VoiceDetector` selects CUDA when `torch.cuda.is_available()` is true; otherwise it falls back to CPU automatically. CPU inference can be slower.

### Model checkpoint missing

Confirm the file exists at `models/AASIST-L-SIH-v3.pth`. Do not substitute a different model filename: the current `VoiceDetector` is deliberately restricted to the V3 checkpoint.

## Limitations

- This is a research/SIH prototype, not a production-ready fraud decision system.
- It will not catch every AI-generated or cloned voice.
- Real-time mode is microphone chunk analysis, not direct cellular phone-call interception.
- There is no speaker verification yet.
- There is no risk engine or prevention/alert system yet.
- There is no WebSocket implementation currently; communication uses HTTP file uploads.
- Results should not be used as the only basis for a security, identity, or financial decision.

## Future roadmap

- Add and evaluate better Indian-English and deepfake training data, such as IndieFake if it is obtained.
- Evaluate a possible V4 model while keeping a stable frontend/API contract.
- Improve streaming behavior and reduce end-to-end latency.
- Run stronger, documented evaluation across relevant real and spoofed audio.
- Add speaker verification where appropriate.
- Explore risk scoring, prevention flows, and alerts.
- Explore carefully scoped call or VoIP integration.

## Credits

The AASIST architecture is based on the original [clovaai/aasist project](https://github.com/clovaai/aasist). Please refer to that repository for the original project, research context, and license information.
