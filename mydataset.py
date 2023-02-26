import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
        ])

class MyDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path, encoding="shift-jis")
        self.img_paths = df["path"].tolist()
        self.meta = df["info1"].tolist()
        self.y = df["label"].tolist()

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert("RGB")
        img = transform(img)
        _meta = self.meta[idx]
        label = self.y[idx]
        return img, _meta, label

if __name__ == "__main__":
    ds = MyDataset("data.csv")
