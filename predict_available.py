import os
import cv2
import torchvision
import torch
import ResUNetPP
import TransResUNetPP.TransResUNet
import numpy as np


def resize(img, size):
    # 图片的宽高
    h, w = img.shape[0:2]
    # 需要的尺寸
    _w = _h = size
    # 不改变图像的宽高比例
    scale = min(_h / h, _w / w)
    h = int(h * scale)
    w = int(w * scale)
    # 缩放图像
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
    # 上下左右分别要扩展的像素数
    top = (_h - h) // 2
    left = (_w - w) // 2
    bottom = _h - h - top
    right = _w - w - left
    # 生成一个新的填充过的图像，这里用纯黑色进行填充(0,0,0)
    new_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return new_img


def predict(img_path, mode="bgr"):
    '''
    预测
    :param img_path: 待预测图片的路径
    :param mode: 颜色通道顺序，可选bgr或者rgb，通常cv2读取是bgr，但是从dataset.py里面看到，是有转成rgb训练的；默认bgr
    '''
    img_o = cv2.imread(img_path)

    img_o = resize(img_o, 256)
    trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    img_o = trans(img_o).reshape(-1, 3, 256, 256)

    nums_layer = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = ResUNetPP.ResUNetPP(5).to(device)
    # net = TransResUNetPP.TransResUNet.TResUnet().to(device)
    # 读取保存的模型参数
    net.load_state_dict(torch.load("./model.pth"))

    pred = net(img_o.to(device)).cpu().detach()
    pred = pred.reshape(3, 256, 256)
    pred = pred.permute(1, 2, 0).numpy()

    if mode == "bgr":
        pred = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)
    elif mode == "rgb":
        pass

    target_img = cv2.resize(pred, (224, 224))
    cat_images = np.concatenate(
        [target_img * 255], axis=1
    )
    ###########################
    # 注释掉可以不展示预测结果
    cv2.imshow("prediction", pred)
    cv2.imwrite(r"G:\BCI_Competition\TransResUNetPP\submission\1.png", cat_images)
    cv2.waitKey(0)
    cv2.destroyWindow("prediction")
    ###########################

    return pred


if __name__ == '__main__':
    pred = predict(img_path=r"G:\BCI_Competition\TransResUNetPP\DRIVE\test\1.jpg")