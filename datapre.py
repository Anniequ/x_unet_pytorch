# _*_ coding: utf-8 _*_
# @author: anniequ
# @file: changename.py
# @time: 2020/12/29 16:48
# @Software: PyCharm
import os
import shutil

PATH_x = "D:\project\python\\x_unet_pytorch\data_sci\\val"  # path为X射线图片存放的路径
PAth_json = "D:\project\python\\x_unet\data\\train\imagejson"  # path为json文件存放的路径

GT_from_PATH = "D:\project\python\\x_unet\data\\train\labelset"  # ground truth file 存放的路径
GT_to_PATH = "D:\project\python\\x_unet\data\\train\label"  # ground truth 将存放的路径


def changeXtoNum(path, isLabel=False):
    """
    将path路径下的X射线图片名更改为number.png或者number_label.png
    """
    fileList = os.listdir(path)
    n = 0
    for i in fileList:
        oldName = path + os.sep + fileList[n]  # os.sep 添加系统分隔符
        if not isLabel:
            newName = path + os.sep + str(n).zfill(3) + '.png'
        else:
            a, b, c = i.split('_')
            newName = path + os.sep + a + '_mask.png'
        if oldName == newName:
            n += 1
            continue
        os.rename(oldName, newName)
        print(oldName, '====>', newName)
        n += 1


def copy_file(from_dir, to_dir):
    # 将from_dir中的文件copy到to_dir
    filepath_list = os.listdir(from_dir)
    if not os.path.isdir(to_dir):
        os.mkdir(to_dir)
    # 1
    # name_list = os.listdir(from_dir)

    # # 2
    # sample = random.sample(pathDir, 2)
    # print(sample)

    # 3
    for file_path in filepath_list:
        name = "{}_label.png".format(file_path)
        gt_file_path = os.path.join(from_dir, file_path, name)
        try:
            # print(name)
            if not os.path.isfile(os.path.join(gt_file_path)):
                print("{} is not existed".format(os.path.join(gt_file_path)))
            shutil.copy(gt_file_path, os.path.join(to_dir, name))
            # print("{} has copied to {}".format(os.path.join(from_dir, name), os.path.join(to_dir, name)))
        except:
            # print("failed to move {}".format(from_dir + name))
            pass
        # shutil.copyfile(fileDir+name, tarDir+name)
    print("{} has copied to {}".format(from_dir, to_dir))


def moveLabel(path_j, path_label, num):
    while num:
        old_path = path_j + os.sep + str(num - 1) + "_json\label.png"
        new_name = path_j + os.sep + str(num - 1) + "_json\\" + str(num - 1) + "_label.png"
        os.rename(old_path, new_name)
        shutil.copy(new_name, path_label)
        print("copy " + str(num - 1) + "_label.png ====> dir label")
        num -= 1


# changeXtoNum(PATH_x, False)
# copy_file(GT_from_PATH, GT_to_PATH)
# changeXtoNum(GT_to_PATH, True)
# toDataset(PAth_json)
# moveLabel(PAth_json, PATH_label, 57)
# copy_file(GT_from_PATH, GT_to_PATH)
