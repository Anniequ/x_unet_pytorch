# _*_ coding: utf-8 _*_
# @author: anniequ
# @file: main.py
# @time: 2021/1/17 16:08
# @Software: PyCharm
import numpy as np
import torch
import argparse
import PIL.Image as Image
from torch.utils.data import DataLoader
from torch import autograd, optim
from torchvision.transforms import transforms
from unet import Unet
from dataset import LiverDataset
from mIou import *
from datetime import datetime
from time import strftime, localtime
import os
import cv2

# 是否使用cuda
device = torch.device("cpu") # cuda


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def res_record(content):
    with open('./results/result.txt', 'a') as f:
        f.write(content)


def train_model(model, criterion, optimizer, dataload, num_epochs=20):
    print("Start training at ", strftime("%Y-%m-%d %H:%M:%S", localtime()))
    for epoch in range(num_epochs):
        prev_time = datetime.now()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        dt_size = len(dataload.dataset)
        epoch_loss = 0
        step = 0
        for x, y in dataload:
            # print(x.size(), y.size())
            step += 1
            inputs = x.to(device)
            labels = y.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            if (step % 10) == 0:
                print("%d/%d, train_loss:%0.3f" % (step, (dt_size - 1) // dataload.batch_size + 1, loss.item()))
        # print the results of the current training
        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = 'Time:{:.0f}:{:.0f}:{:.0f}'.format(h, m, s)
        epoch_str = "epoch {} loss:{:.4f} ".format(epoch, epoch_loss / 400)
        print(epoch_str + time_str)
        res_record("Time:" + strftime("%Y-%m-%d %H:%M:%S  ", localtime()))
        res_record(epoch_str + '\n')
    print("End training at ", strftime("%Y-%m-%d %H:%M:%S", localtime()))
    # 记录数据
    torch.save(model.state_dict(),
                       './results/weights{}_{}_{}.pth'.format(localtime().tm_mday, localtime().tm_hour,
                                                              localtime().tm_sec))
    return model


# 训练模型
def train():
    model = Unet(3, 1).to(device)
    model.load_state_dict(torch.load(r"./results/weights4_13_40.pth"))
    batch_size = 5
    criterion = torch.nn.BCELoss()
    optimizer = optim.Adam(model.parameters())
    liver_dataset = LiverDataset(r"D:\project\data_sets\data_sci\train", transform=x_transforms,
                                 target_transform=y_transforms)
    dataloaders = DataLoader(liver_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    train_model(model, criterion, optimizer, dataloaders)


def get_data(i):
    import dataset
    imgs = dataset.make_dataset(r"D:\project\data_sets\data_sci\val")
    imgx = []
    imgy = []
    for img in imgs:
        imgx.append(img[0])
        imgy.append(img[1])
    return imgx[i], imgy[i]


# 显示模型的输出结果
def test():
    model = Unet(3, 1).to(device)  # unet输入是三通道，输出是一通道，因为不算上背景只有肝脏一个类别
    weight_pre = r"./results/weights4_18_35.pth"
    model.load_state_dict(torch.load(weight_pre))  # 载入训练好的模型
    liver_dataset = LiverDataset(r"D:\project\data_sets\data_sci\val", transform=x_transforms,
                                 target_transform=y_transforms)
    dataloaders = DataLoader(liver_dataset, batch_size=1)
    model.eval()
    import matplotlib.pyplot as plt
    plt.ion()  # 开启动态模式

    with torch.no_grad():
        i = 0  # 验证集中第i张图
        miou_total = 0
        num = len(dataloaders)  # 验证集图片的总数
        for x, _ in dataloaders:
            x = x.to(device)
            y = model(x)

            img_y = torch.squeeze(y).cpu().numpy()  # 输入损失函数之前要把预测图变成numpy格式，且为了跟训练图对应，要额外加多一维表示batchsize
            mask = get_data(i)[1]  # 得到当前mask的路径
            miou_total += get_iou(mask, img_y)  # 获取当前预测图的miou，并加到总miou中
            plt.subplot(121)
            plt.imshow(Image.open(get_data(i)[0]))
            plt.subplot(122)
            plt.imshow(img_y)
            plt.pause(0.01)
            if i < num: i += 1  # 处理验证集下一张图
        plt.show()
        print('Miou=%f' % (miou_total / 10))
        res_record("weights4_13_40.pth Miou=%f \n" % (miou_total / 10))


if __name__ == "__main__":
    x_transforms = transforms.Compose([
        transforms.ToTensor(),  # -> [0,1]
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # ->[-1,1]
    ])

    # mask只需要转换为tensor
    y_transforms = transforms.ToTensor()
    # 参数解析器,用来解析从终端读取的命令
    parse = argparse.ArgumentParser()
    # parse = argparse.ArgumentParser()
    parse.add_argument("--action", type=str, help="train or test", default="train")
    parse.add_argument("--batch_size", type=int, default=1)
    parse.add_argument("--ckp", type=str, help="the path of model weight file")
    args = parse.parse_args()

    # train
    # train()

    args.ckp = r"./results/weights3_21_8.pth"
    test()
