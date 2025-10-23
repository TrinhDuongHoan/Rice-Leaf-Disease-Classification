import torch
import argparse
from torch.utils.data import DataLoader
from pathlib import Path
from utils.data.loader import build_loader  
from utils.data.dataset import PaddyDataset 
from training.train_model import train_model


train_csv = "../src/utils/data/Paddy_Dataset/splits/train.csv"
val_csv   = "../src/utils/data/Paddy_Dataset/splits/val.csv"


def main(args):
    train_csv = Path(args.train_csv)
    val_csv = Path(args.val_csv)

    assert train_csv.exists(), f"Không tìm thấy {train_csv}"
    assert val_csv.exists(), f"Không tìm thấy {val_csv}"

    train_loader, val_loader, label2id = build_loader(
        train_csv=str(train_csv),
        val_csv=str(val_csv),
        batch_size=4,
        num_workers=2,
        image_size=224
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, history, best_epochs = train_model("mobilenetv3_eca", train_loader, val_loader, device)

    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model")

    parser.add_argument("--train_csv", type=str, default=str(train_csv), help="Path to train CSV file")
    parser.add_argument("--val_csv", type=str, default=str(val_csv), help="Path to validation CSV file")
    args = parser.parse_args()
    main(args)
