import torch
from mydataset import MyDataset
from torch.utils.data import DataLoader

mydataset = MyDataset("data.csv")
train_dataloader = DataLoader(mydataset, batch_size=3, shuffle=True)


from model import MyModel

model = MyModel()
for x, meta, y in train_dataloader:
    meta = meta.unsqueeze(1).float()
    y = y.type(torch.float64)
    output = model(x, meta)
    print(output)