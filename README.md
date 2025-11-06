# bertolingo

Self-study toy project.

My goal here is to understand the Transformer architecture (both encoder and decoder), to achieve English -> Italian translation.

## Project Structure

The project is organized into a modular folder structure:

```
bertolingo/
├── src/                    # Source code package
│   ├── __init__.py
│   ├── config.py          # Configuration parameters
│   ├── dataset.py          # Dataset loading and processing
│   ├── models.py           # Model definitions
│   ├── train.py            # Training and evaluation
│   └── inference.py        # Translation/inference
├── scripts/                # Utility scripts
│   └── setup_and_run.sh   # Setup script for remote server
├── docs/                   # Documentation
│   └── DEPLOY.md           # Deployment guide
├── main.py                 # Main entry point
├── requirements.txt        # Python dependencies
├── pyproject.toml          # Project configuration
└── README.md               # This file
```

## Installation

### Using uv (Recommended)

First, install [uv](https://github.com/astral.sh/uv):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then install dependencies:

```bash
# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate  # On Windows
uv pip install -r requirements.txt
```

The `pyproject.toml` file is included for project metadata and tool configuration (like ruff), but since this is a script-based project, we use `uv pip install` with `requirements.txt` for dependency management.

### Using pip (Alternative)

```bash
pip install -r requirements.txt
```

## Usage

### Training

Train the model from scratch:

```bash
# Using uv (no need to activate venv)
uv run python main.py --mode train --plot

# Or with activated venv
python main.py --mode train --plot

# Specify custom plot directory
python main.py --mode train --plot --plot-dir my_plots
```

This will:
- Load the Helsinki-NLP/opus-100 dataset (en-it)
- Create a character-level tokenizer
- Train the model for the configured number of epochs
- Save the model to `bertolingo_model.pt`
- Save training curves plot to `plots/` directory (or custom directory if specified)

### Translation

Translate text using a trained model:

```bash
uv run python main.py --mode translate --checkpoint bertolingo_model.pt --text "Hello, how are you?"
```

### Evaluation

Evaluate a trained model on the validation set:

```bash
uv run python main.py --mode eval --checkpoint bertolingo_model.pt
```

## Configuration

Edit `config.py` to adjust:
- Model architecture (embedding dimensions, number of heads, blocks)
- Training hyperparameters (learning rate, epochs, batch size)
- Data settings (sample size, context window)

## Code Quality

This project uses [ruff](https://github.com/astral-sh/ruff) for linting and code formatting.

### Check for issues

```bash
# Check all files
uv run ruff check .

# Check specific file
uv run ruff check main.py
```

### Fix issues automatically

```bash
# Fix all auto-fixable issues
uv run ruff check --fix .

# Show what would be fixed without making changes
uv run ruff check --diff .
```

### Format code (if using ruff format)

```bash
# Format all files
uv run ruff format .

# Check formatting without making changes
uv run ruff format --check .
```

Ruff configuration is in `pyproject.toml` under `[tool.ruff]`.

## Remote GPU Deployment

To deploy and run training on a remote GPU server, see [docs/DEPLOY.md](docs/DEPLOY.md) for detailed instructions.

Quick start:
```bash
# 1. Push code to Git repository (if not already done)
git push origin main

# 2. SSH into remote server
ssh root@213.192.2.85 -p 40059 -i private_key.pem

# 3. Clone repository
git clone <your-repo-url>
cd bertolingo

# 4. Run setup
chmod +x scripts/setup_and_run.sh
./scripts/setup_and_run.sh

# 5. Activate and train
source .venv/bin/activate
python main.py --mode train --plot
```

## Study resources

- karpathy tutorial on nanogpt
- gpt 5 / sonnet 4.5 
- random blog / videos on specific topics
