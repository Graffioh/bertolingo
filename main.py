"""
Main entry point for training and using the translation model
"""

import argparse

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from config import (
    adam_weight_decay,
    batch_size,
    context_window,
    d_embd,
    dataset_config,
    dataset_name,
    dropout,
    eval_every,
    lr,
    n_decoder_blocks,
    n_encoder_blocks,
    n_heads,
    num_epochs,
    print_every,
    sample_size,
)
from dataset import create_dataloaders, create_tokenizer, load_translation_dataset
from inference import translate
from models import Seq2SeqModel
from train import evaluate_model, train_model


def main():
    parser = argparse.ArgumentParser(description="Train or use translation model")
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "translate", "eval"],
        help="Mode: train, translate, or eval",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint for inference/evaluation",
    )
    parser.add_argument(
        "--text", type=str, default=None, help="Text to translate (for translate mode)"
    )
    parser.add_argument(
        "--save-model",
        type=str,
        default="bertolingo_model.pt",
        help="Path to save the trained model",
    )
    parser.add_argument("--plot", action="store_true", help="Plot training curves after training")
    args = parser.parse_args()

    # Load dataset and create tokenizer
    print("Loading dataset...")
    x_train, y_train, x_val, y_val = load_translation_dataset(dataset_name, dataset_config)

    print("Creating tokenizer...")
    stoi, itos, vocab_size = create_tokenizer(x_train, y_train)
    print(f"Vocabulary size: {vocab_size}")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.mode == "train":
        # Create data loaders
        print("Creating data loaders...")
        train_loader, val_loader = create_dataloaders(
            x_train,
            y_train,
            x_val,
            y_val,
            stoi,
            batch_size=batch_size,
            sample_size=sample_size,
            max_len=context_window,
            num_workers=0,
        )

        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")

        # Initialize model
        print("Initializing model...")
        model = Seq2SeqModel(
            vocab_size=vocab_size,
            d_embd=d_embd,
            n_heads=n_heads,
            dropout=dropout,
            n_encoder_blocks=n_encoder_blocks,
            n_decoder_blocks=n_decoder_blocks,
            context_window=context_window,
            padding_idx=stoi["<PAD>"],
        )

        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")

        model = model.to(device)

        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss(ignore_index=stoi["<PAD>"], reduction="mean")
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=adam_weight_decay)

        # Train model
        train_losses, val_losses = train_model(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            device,
            stoi,
            num_epochs=num_epochs,
            print_every=print_every,
            eval_every=eval_every,
        )

        # Plot training curves
        if args.plot:
            plt.figure(figsize=(10, 6))
            epochs = list(range(1, len(train_losses) + 1))

            plt.plot(epochs, train_losses, label="Train Loss", marker="o")

            if len(val_losses) > 0:
                plt.plot(epochs[: len(val_losses)], val_losses, label="Val Loss", marker="s")

            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Training and Validation Loss")
            plt.legend()
            plt.grid(True)
            plt.show()

            print(f"\nFinal Training Loss: {train_losses[-1]:.4f}")
            if len(val_losses) > 0:
                print(f"Final Validation Loss: {val_losses[-1]:.4f}")

        # Save model
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_losses": train_losses,
            "val_losses": val_losses,
            "epoch": num_epochs,
            "vocab_size": vocab_size,
            "d_embd": d_embd,
            "n_heads": n_heads,
            "n_encoder_blocks": n_encoder_blocks,
            "n_decoder_blocks": n_decoder_blocks,
            "dropout": dropout,
            "context_window": context_window,
            "stoi": stoi,
            "itos": itos,
        }

        torch.save(checkpoint, args.save_model)
        print(f"\nModel saved to '{args.save_model}'")

    elif args.mode == "translate":
        if args.checkpoint is None:
            print("Error: --checkpoint required for translate mode")
            return

        if args.text is None:
            print("Error: --text required for translate mode")
            return

        # Load model
        print(f"Loading model from {args.checkpoint}...")
        checkpoint = torch.load(args.checkpoint, map_location=device)

        # Reconstruct model
        model = Seq2SeqModel(
            vocab_size=checkpoint["vocab_size"],
            d_embd=checkpoint["d_embd"],
            n_heads=checkpoint["n_heads"],
            dropout=checkpoint["dropout"],
            n_encoder_blocks=checkpoint["n_encoder_blocks"],
            n_decoder_blocks=checkpoint["n_decoder_blocks"],
            context_window=checkpoint["context_window"],
            padding_idx=checkpoint["stoi"]["<PAD>"],
        )

        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)
        model.eval()

        # Translate
        stoi = checkpoint["stoi"]
        itos = checkpoint["itos"]
        translated = translate(model, args.text, stoi, itos, device)

        print(f"\nEnglish: {args.text}")
        print(f"Italian: {translated}")

    elif args.mode == "eval":
        if args.checkpoint is None:
            print("Error: --checkpoint required for eval mode")
            return

        # Load model
        print(f"Loading model from {args.checkpoint}...")
        checkpoint = torch.load(args.checkpoint, map_location=device)

        # Reconstruct model
        model = Seq2SeqModel(
            vocab_size=checkpoint["vocab_size"],
            d_embd=checkpoint["d_embd"],
            n_heads=checkpoint["n_heads"],
            dropout=checkpoint["dropout"],
            n_encoder_blocks=checkpoint["n_encoder_blocks"],
            n_decoder_blocks=checkpoint["n_decoder_blocks"],
            context_window=checkpoint["context_window"],
            padding_idx=checkpoint["stoi"]["<PAD>"],
        )

        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)

        # Create data loaders
        print("Creating data loaders...")
        _, val_loader = create_dataloaders(
            x_train,
            y_train,
            x_val,
            y_val,
            checkpoint["stoi"],
            batch_size=batch_size,
            sample_size=None,  # Use all validation data
            max_len=context_window,
            num_workers=0,
        )

        # Evaluate
        criterion = nn.CrossEntropyLoss(ignore_index=checkpoint["stoi"]["<PAD>"], reduction="mean")
        val_loss = evaluate_model(model, val_loader, criterion, device, checkpoint["stoi"])

        print(f"\nValidation Loss: {val_loss:.4f}")


if __name__ == "__main__":
    main()
