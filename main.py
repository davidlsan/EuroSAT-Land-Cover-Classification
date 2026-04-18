import argparse
import os
from torch import nn
import torch
from train import build_dataloaders, build_model, evaluate, get_device, train_one_epoch


def main():

    os.makedirs("weights", exist_ok=True)
    
    # change 32 -> 16 for lower end computers
    train_loader, val_loader, num_classes = build_dataloaders(32, 2)

    # Parse CLI args to support the --pretrained flag
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained weights, omit the flag for random init")
    args = parser.parse_args()

    mode = "imagenet_init" if args.pretained else "from_scratch"
    ckpt_path = f"weights/best_{mode}.pt"

    device = get_device()
    model = build_model(num_classes, device, pretrained=args.pretrained)
    print(f"Training using {mode}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    best_val_acc = 0.0
    print(f"Training on {device}")

    # Train for 15 epochs (turns)
    for epoch in range (1, 16): 
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch} | "
            f"Train Loss: {train_loss} | "
            f"Val Loss: {val_loss} | "
            f"Val Acc: {val_acc}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "val_acc": best_val_acc,
                    "epoch": epoch,
                    "pretrained": args.pretrained,
                    "init": mode,
                }, ckpt_path)
            print (f"Saved new best model with val_acc {best_val_acc}")


if __name__ == "__main__":
    main()
