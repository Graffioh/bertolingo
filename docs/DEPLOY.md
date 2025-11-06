# Deployment Guide for Remote GPU Server

## Quick Start (Recommended)

### Step 1: Push code to Git repository

If you haven't already, push your code to GitHub/GitLab:

```bash
# From your local bertolingo directory
git remote add origin <your-repo-url>  # if not already added
git push -u origin main
```

### Step 2: Clone and setup on remote server

```bash
# SSH into the remote server
ssh root@213.192.2.85 -p 40059 -i private_key.pem

# Clone the repository
git clone <your-repo-url>
cd bertolingo

# Run setup script
chmod +x scripts/setup_and_run.sh
./scripts/setup_and_run.sh

# Activate virtual environment and run training
source .venv/bin/activate
python main.py --mode train --plot
```

## Manual Setup (Alternative)

If you prefer to set up manually:

```bash
# SSH into the server
ssh root@213.192.2.85 -p 40059 -i private_key.pem

# Clone repository
git clone <your-repo-url>
cd bertolingo

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.cargo/bin:$PATH"

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt

# Run training
python main.py --mode train --plot
```

## Using tmux for Long-Running Training

Since training can take a long time, use `tmux` to keep it running:

```bash
# SSH into server
ssh root@213.192.2.85 -p 40059 -i private_key.pem

# Start a tmux session
tmux new -s training

# Navigate to project and activate environment
cd bertolingo
source .venv/bin/activate

# Run training
python main.py --mode train --plot

# Detach: Ctrl+B, then D
# Reattach: tmux attach -t training
```

## Monitoring Training

### Check GPU usage:
```bash
watch -n 1 nvidia-smi
```

### Check if training is running:
```bash
ps aux | grep python
```

### View recent output:
```bash
# If using tmux
tmux attach -t training

# Or check logs if you redirect output
tail -f training.log
```

## Updating Code

To pull the latest changes:

```bash
# SSH into server
ssh root@213.192.2.85 -p 40059 -i private_key.pem

# Navigate to project
cd bertolingo

# Pull latest changes
git pull

# If dependencies changed, update them
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Transferring Model Checkpoints Back

After training completes, download the model:

```bash
# From your local machine
scp -P 40059 -i private_key.pem \
  root@213.192.2.85:/root/bertolingo/bertolingo_model.pt \
  ./bertolingo_model.pt
```
