# ToneMatch AI

AI-powered guitar tone matching system. Upload a reference track and your DI recording — the system automatically finds the best virtual rig (pedal + amp + cabinet) from 259 NAM models and optimizes post-processing to match the target tone.

## How It Works

The system runs a three-stage pipeline:

1. **Smart Sommelier (Reference Analysis)** — analyzes the reference track to determine genre, gain level, and tonal characteristics, then narrows down the search space.
2. **Fast Grid Search** — evaluates combinations of FX pedals (NAM), amplifiers (NAM), and cabinets (IR) to find the TOP-3 rigs closest to the reference.
3. **Deep Post-FX Optimization** — fine-tunes 7 post-processing parameters (Pre-EQ, Reverb, Delay, Final EQ) using a "Sighted" optimizer that minimizes a 4-component error vector (harmonic loss, envelope loss, spectral shape loss, brightness loss).

## Quick Start

### Requirements

- Python 3.9+
- PyTorch 2.0+
- See `requirements.txt` for full list

### Installation

```bash
git clone https://github.com/<your-username>/toneMatchAi.git
cd toneMatchAi
pip install -r requirements.txt
```

### Usage

**Option 1: Command Line — Universal Matching (recommended)**

Place your files in the project root:
- `my_guitar.wav` (or `.mp3`) — your DI recording
- `reference.wav` (or `.mp3`) — the target tone

```bash
python run_universal_match.py
```

This will search all 259 NAM models for the best rig and optimize post-FX automatically.

**Option 2: Command Line — Fixed Rig**

```bash
python run_final_tune.py
```

Uses a fixed equipment chain (DS1 -> 5150 BlockLetter -> BlendOfAll IR) and only optimizes post-FX parameters.

**Option 3: Streamlit Web UI**

```bash
streamlit run src/app.py
```

Upload files via the browser, listen to results, and download the processed audio.

## Project Structure

```
toneMatchAi/
├── src/
│   ├── app.py                  # Streamlit web interface
│   ├── core/
│   │   ├── analysis.py         # Spectral analysis & loss functions
│   │   ├── data_generator.py   # Training data generation
│   │   ├── ddsp_processor.py   # Differentiable DSP processing
│   │   ├── io.py               # Audio I/O (load/save wav/mp3)
│   │   ├── loss.py             # Loss function components
│   │   ├── matching.py         # Match EQ filter creation
│   │   ├── nam_processor.py    # Neural Amp Modeler integration
│   │   ├── optimizer.py        # Core optimizer (Grid Search + Post-FX)
│   │   └── processor.py        # Audio processing chain
│   └── utils/
│       └── helpers.py          # Utility functions
├── assets/
│   ├── nam_models/             # 259 NAM models (pedals & amps)
│   └── impulse_responses/      # Cabinet IR files
├── run_universal_match.py      # Entry point: full universal matching
├── run_final_tune.py           # Entry point: fixed rig + post-FX tuning
├── requirements.txt            # Python dependencies
└── .gitignore
```

## Output

After running, the system produces:
- A `.wav` file with the matched tone
- A detailed report with the discovered rig and optimized parameters

## Technology Stack

- **NAM (Neural Amp Modeler)** — neural network guitar amp/pedal emulation
- **Pedalboard** — real-time audio effects processing (Spotify)
- **PyTorch** — differentiable DSP and neural network training
- **Librosa / SciPy** — spectral analysis and optimization
- **Streamlit** — web UI




















