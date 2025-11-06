"""
Configuration file for the translation model
"""

# Model hyperparameters
d_embd = 64
n_heads = 8
head_size = d_embd // n_heads
n_encoder_blocks = 4
n_decoder_blocks = 4
dropout = 0.4

# Training hyperparameters
lr = 3e-4
adam_weight_decay = 0.01
num_epochs = 150
batch_size = 32

# Data configuration
context_window = 128
sample_size = 1000  # Number of training samples to use (for experimentation)

# Training configuration
print_every = 50  # Print loss every N batches
eval_every = 500  # Evaluate on validation set every N batches

# Dataset configuration
dataset_name = "Helsinki-NLP/opus-100"
dataset_config = "en-it"

