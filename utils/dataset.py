import torch
import glob
import os

from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import Dataset
from PIL import Image


class load_dataset(Dataset):
    def __init__(self, img_dir, img_transform, mask_dir, mask_transform):
        self.img_dir = img_dir
        self.img_fnames = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
        self.img_transform = img_transform

        self.mask_dir = mask_dir
        self.mask_fnames = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
        self.mask_transform = mask_transform

    def __getitem__(self, i: int):
        fpath = self.img_fnames[i]
        img = Image.open(fpath).convert("RGB")
        state = torch.get_rng_state()
        if self.img_transform is not None:
            img = self.img_transform(img)

        mpath = self.mask_fnames[i]
        mask = Image.open(mpath).convert("1")
        torch.set_rng_state(state)
        if self.mask_transform is not None:
            mask = self.mask_transform(mask)

        return img, mask

    def __len__(self):
        return len(self.img_fnames)


def get_data_loaders(bs_train, bs_test, img_factor=1, dataset="CFD"):
    # Define standard variables
    current_path = os.path.abspath(os.getcwd())
    image = "/CFD/cfd_image/"
    gt = "/CFD/seg_gt/"
    ratio = [70, 48]

    if dataset == "CRACK500":
        image = "/CRACK500/crack_image/"
        gt = "/CRACK500/seg_gt/"
        ratio = [2021, 1347]

    tf_compos, tf_compos_gt = get_transforms(img_factor)
    if dataset == "CFD":
        dataset = load_dataset(current_path + image, tf_compos, current_path + gt, tf_compos_gt)
    else:
        dataset = load_dataset(current_path + image, tf_compos, current_path + gt, tf_compos_gt)

    print(f"dataset len: {len(dataset)}")
    train_data, test_data = random_split(dataset, ratio, generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_data, batch_size=bs_train)
    test_loader = DataLoader(test_data, batch_size=bs_test)

    print(
        f"dataset: {len(dataset)}, objects_train: {len(train_data)}, BS_train: {bs_train}, dataloader len: {len(train_loader)}")
    print(f"objects_test: {len(test_data)}, BS_test: {bs_test}, dataloader len: {len(test_loader)}")

    return dataset, train_loader, test_loader


def get_transforms(img_factor):
    resize_image = tuple(int(img_factor * x) for x in (320, 480))
    crop_image = resize_image[0]

    shared_transforms = [
        transforms.RandomCrop(crop_image),
        transforms.Pad(200, padding_mode='reflect'),
        transforms.RandomRotation((0, 360)),
        transforms.CenterCrop(crop_image),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
    ]
    # Bilinear interpolation since value can be [-255, +255]
    tf_compos = transforms.Compose([
        transforms.Resize(resize_image, interpolation=InterpolationMode.BILINEAR),
        *shared_transforms,
        transforms.ColorJitter(brightness=0.05, contrast=0.05),
        transforms.ToTensor()
    ])
    # NN interpolation since value can only be [0 or 1],
    tf_compos_gt = transforms.Compose([
        transforms.Resize(resize_image, interpolation=InterpolationMode.NEAREST),
        *shared_transforms,
        transforms.ToTensor()
    ])
    return tf_compos, tf_compos_gt
