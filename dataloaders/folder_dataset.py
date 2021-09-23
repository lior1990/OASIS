import os

from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader


class FolderDataset(Dataset):
    def __init__(self, path, transforms, data_rep=1000):
        self.transforms = transforms
        self.path = path
        self.files = os.listdir(path)
        self.n = len(self.files)
        self.data_rep = data_rep

    def __getitem__(self, index):
        return self.transforms(default_loader(os.path.join(self.path, self.files[index % self.n])))

    def __len__(self):
        return self.n * self.data_rep
