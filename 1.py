import torch
from torchvision import transforms
from torch import optim
from dataset import LiverDataset
from unet import Unet
from torch.utils.data import DataLoader
from PIL import Image
from mIou import get_iou
from datetime import datetime
from time import strftime, localtime

device = torch.device("cuda")


def res_record(content):
    with open('./results/result.txt', 'a') as f:
        f.write(content)


def train():
    x_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    y_transforms = transforms.ToTensor()

    model = Unet(3, 1).to(device)
    model.load_state_dict(torch.load(r"./results/weights.pth"))
    batch_size = 1
    num_epochs = 2
    criterion = torch.nn.BCELoss()
    optimizer = optim.Adam(model.parameters())
    liver_dataset = LiverDataset(r'D:\project\data_sets\liver\train', transform=x_transforms,
                                 target_transform=y_transforms)
    data_loaders = DataLoader(liver_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    print("Start training at ", strftime("%Y-%m-%d %H:%M:%S", localtime()))
    for epoch in range(num_epochs):
        prev_time = datetime.now()
        print('Epoch{}/{}'.format(epoch, num_epochs))
        print('-' * 10)
        dt_size = len(data_loaders.dataset)
        epoch_loss = 0
        step = 0
        for x, y in data_loaders:
            step += 1
            inputs = x.to(device)
            labels = y.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            if (step % 10) == 0:
                print("%d/%d, train_loss:%0.3f" % (step, (dt_size - 1) // data_loaders.batch_size + 1, loss.item()))
        # print the results of the current training
        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = 'Time:{:.0f}:{:.0f}:{:.0f}'.format(h, m, s)
        epoch_str = "epoch {} loss:{:.4f} ".format(epoch, epoch_loss/400)
        print(epoch_str + time_str)
        res_record("Time:" + strftime("%Y-%m-%d %H:%M:%S  ", localtime()))
        res_record(epoch_str + '\n')
    print("End training at ", strftime("%Y-%m-%d %H:%M:%S", localtime()))
    # 记录数据
    torch.save(model.state_dict(),
               './results/weights{}_{}_{}.pth'.format(localtime().tm_mday, localtime().tm_hour, localtime().tm_sec))


def get_data(i):
    import dataset
    imgs = dataset.make_dataset(r"D:\project\data_sets\liver\val")
    imgx = []
    imgy = []
    for img in imgs:
        imgx.append(img[0])
        imgy.append(img[1])
    return imgx[i], imgy[i]


def test():
    model = Unet(3, 1).to(device)
    weight_pre = r"./results/weights18_14_41.pth"
    model.load_state_dict(torch.load(weight_pre))
    x_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    y_transforms = transforms.ToTensor()
    liver_dataset = LiverDataset(r"D:\project\data_sets\liver\val", transform=x_transforms,
                                 target_transform=y_transforms)
    dataloaders = DataLoader(liver_dataset, batch_size=1)
    model.eval()
    import matplotlib.pyplot as plt
    plt.ion()  # 开启动态模式

    with torch.no_grad():
        i = 0
        miou_total = 0
        num = len(dataloaders)
        for x, y in dataloaders:
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
        print('Miou=%f' % (miou_total / 20))
        res_record("weights18_14_41.pth Miou=%f \n"% (miou_total / 20))


if __name__ == '__main__':
    # train()
    test()
