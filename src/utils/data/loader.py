from torch.utils.data import DataLoader
from torchvision import transforms
from utils.data.dataset import PaddyDataset
import pandas as pd

def build_loader(train_csv, val_csv, batch_size=8, num_workers=2, image_size=224):
    tfm = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])

    train_df = pd.read_csv(train_csv)
    val_df   = pd.read_csv(val_csv)

    label2id = {lbl: i for i, lbl in enumerate(sorted(train_df["label"].unique()))}

    train_ds = PaddyDataset(train_df, tfm, label2id)
    val_ds   = PaddyDataset(val_df, tfm, label2id)

    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_ld   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_ld, val_ld, label2id
