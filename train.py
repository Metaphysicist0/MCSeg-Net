"""
训练器模块
"""
import math
import os
import logging
from PIL import Image
import torchvision
from egeunet import EGEUNet
import TDUNet
import unet
from ResUNetPP import ResUNetPP
import segmentation_loss
from torchvision import datasets, models, transforms
import torch
import dataset
import torch.nn as nn
import MCSegNet
import matplotlib.pyplot as plt
from segmentation_loss import SoftDiceLoss, FocalLoss
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch.autograd import Variable
import warnings
warnings.filterwarnings('ignore')

train_loss = []
dice_loss = []

loader = transforms.Compose([transforms.ToTensor()])
unloader = transforms.ToPILImage()


# 训练器
class Trainer:

    def __init__(self, path, model, model_copy, img_save_path):
        self.path = path
        self.model = model
        self.model_copy = model_copy
        self.img_save_path = img_save_path
        # 使用的设备
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # 网络

        lr = 1e-4
        nums_layer = 5
        load_from_check_point = False  # set false to train from the scratch; otherwise set iter num to resume training
        self.net = MCSegNet.TResUnet().to(self.device)
        # self.net = EGEUNet().to(self.device)
        # self.net = TransResUNet.TResUnet().to(self.device)
        # 优化器
        self.opt = torch.optim.Adam(self.net.parameters())

        # 这里直接使用结合Sigmoid的二分类交叉熵来训练
        # 可以尝试改其他损失，DiceLoss、FocalLoss
        self.loss_func = nn.BCEWithLogitsLoss()
        self.dice_func = segmentation_loss.SoftDiceLoss()

        # 设备好，batch_size和num_workers可以给大点
        self.loader = DataLoader(dataset.Datasets(path), batch_size=16, shuffle=True, num_workers=8)

        # 判断是否存在模型
        if os.path.exists(self.model):
            self.net.load_state_dict(torch.load(model), False)
            print(f"Loaded{model}!")

        else:
            print("No Param!")
        os.makedirs(img_save_path, exist_ok=True)

    # 训练

    def train(self, stop_value):
        epoch = 1

        while True:
            for inputs, labels in tqdm(self.loader, desc=f"Epoch {epoch}/{stop_value}",
                                       ascii=True, total=len(self.loader)):
                # 图片和分割标签

                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # 输出生成的图像
                out = self.net(inputs)
                loss = self.loss_func(out, labels)
                diceloss = self.dice_func(out, labels)

                train_loss.append(loss.item())
                dice_loss.append(diceloss.item())

                # 后向
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                # 输入的图像，取第一张
                x = inputs[0]
                # 生成的图像，取第一张
                x_ = out[0]
                # 标签的图像，取第一张
                y = labels[0]
                # 三张图，从第0轴拼接起来，再保存
                img = torch.stack([x, y, x_], 0)
                # img = torch.stack([x_], 0)
                # img = torch.stack([x, x_], 0)
                save_image(img.cpu(), os.path.join(self.img_save_path, f"{epoch}.png"))
                # print("image save successfully !")

            print(f"\nEpoch: {epoch}/{stop_value}, Loss: {loss},\n Diceloss:{diceloss}")
            torch.save(self.net.state_dict(), self.model)
            # print("model is saved !")

            # 备份
            if epoch % 5 == 0:
                torch.save(self.net.state_dict(), "model.pth")
                print("model_copy is saved !")
            if epoch > stop_value:
                break
            epoch += 1

        with open("./train_loss.txt", 'w') as train_los:
            train_los.write(str(train_loss))

        with open("./diceloss.txt", 'w') as dice_los:
            dice_los.write(str(dice_loss))


if __name__ == '__main__':
    # 路径改一下
    t = Trainer(r"DRIVE/training", r'./model.pth', r'./model_{}_{}.pth', img_save_path=r'./train_img')
    t.train(500)
