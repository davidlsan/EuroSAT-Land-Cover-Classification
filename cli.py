import argparse

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--modality", choices=["rgb", "msi"], default="rgb")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()