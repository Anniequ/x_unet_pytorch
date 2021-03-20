# _*_ coding: utf-8 _*_
# @author: anniequ
# @file: dataset.py
# @time: 2021/1/17 15:54
# @Software: PyCharm
import torch.utils.data as data
import PIL.Image as Image
import os


def make_dataset(root):
    imgs = []
    n = len(os.listdir(root)) // 2  # 数据集中有img and mask 两种图像
    for i in range(n):
        img = os.path.join(root, "%03d.png" % i)
        mask = os.path.join(root, "%03d_mask.png" % i)
        imgs.append((img, mask))

    return imgs


class LiverDataset(data.Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        imgs = make_dataset(root)
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, item):
        x_path, y_path = self.imgs[item]
        origin_x = Image.open(x_path)
        origin_y = Image.open(y_path)
        if self.transform is not None:
            img_x = self.transform(origin_x)
        else:
            img_x = origin_x
        if self.target_transform is not None:
            img_y = self.target_transform(origin_y)
        else:
            img_y = origin_y

        return img_x, img_y

    def __len__(self):
        return len(self.imgs)
