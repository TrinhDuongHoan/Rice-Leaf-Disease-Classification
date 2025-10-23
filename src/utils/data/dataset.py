from torch.utils.data import Dataset
from PIL import Image

class PaddyDataset(Dataset):
    def __init__(self, df, transforms=None, label2id=None):
        self.df = df.reset_index(drop=True)
        self.tfm = transforms
        self.label2id = label2id

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row["image_path"]
        label_name = row.get("label", None)

        label = self.label2id[label_name] if self.label2id and label_name in self.label2id else -1

        img = Image.open(image_path).convert("RGB")
        if self.tfm:
            img = self.tfm(img)

        return img, label
