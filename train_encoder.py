import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from dataloaders import dataloaders
from dataloaders.folder_dataset import FolderDataset
import config

from models import models
from models.quantizer import VectorQuantizer
from models.unet.unet import UNet

from models.losses import VGGLoss
from utils import utils

opt = config.read_arguments(train=False, train_encoder=True)

#--- create dataloader ---#
_, _ = dataloaders.get_dataloaders(opt)  # do this because opt is modified in-place

transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])
dataset = FolderDataset(opt.path, transforms, data_rep=1)
dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        drop_last=False
    )

#--- create models ---#
model = models.OASIS_model(opt)
model = models.put_on_multi_gpus(model, opt)
model.eval()

vector_quantizer = VectorQuantizer(opt.semantic_nc)

#--- create utils ---#
image_saver = utils.results_saver(opt)


n_classes = opt.semantic_nc
encoder = UNet(3, n_classes)
if opt.optimizer == "adam":
    optimizer = torch.optim.Adam(encoder.parameters(), lr=opt.lr, betas=(0.0, 0.9))
elif opt.optimizer == "sgd":
    optimizer = torch.optim.SGD(encoder.parameters(), lr=opt.lr)
else:
    raise NotImplementedError(opt.optimizer)

if opt.loss == "l1":
    criterion = torch.nn.L1Loss()
elif opt.loss == "l2":
    criterion = torch.nn.MSELoss()
elif opt.loss == "vgg":
    criterion = VGGLoss(opt.gpu_ids)
else:
    raise NotImplementedError(opt.loss)

n_epochs = opt.niter

if opt.gpu_ids != "-1":
    encoder.cuda()

for epoch in range(n_epochs):
    for i, img in enumerate(dataloader):
        if opt.gpu_ids != "-1":
            img = img.cuda()

        label_tensor = encoder(img)
        label_tensor_one_hot, _ = vector_quantizer(label_tensor)

        generated = model(None, label_tensor_one_hot, "eval", None)

        loss = criterion(generated, img)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch}, Loss: {loss.item()}")

encoder.eval()

for i, img in enumerate(dataloader):
    with torch.no_grad():
        if opt.gpu_ids != "-1":
            img = img.cuda()

        label_tensor = encoder(img)
        label_tensor_one_hot, min_encoding_indices = vector_quantizer(label_tensor)

        generated = model(None, label_tensor_one_hot, "eval", None)

        for b in range(generated.shape[0]):
            print(f'process image {i} {b}')
            print(f"labels for image {i}_{b} are: {torch.unique(min_encoding_indices)}")
            image_saver([label_tensor_one_hot], [generated], [f"{i}_{b}.png"])
