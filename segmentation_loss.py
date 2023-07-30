import torch.nn as nn
import torch


class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True, sigmoid=False):
        super(SoftDiceLoss, self).__init__()
        self.sigmoid = sigmoid

    def forward(self, logits, targets):
        num = targets.size(0)
        smooth = 1

        if self.sigmoid == True:
            logits = torch.sigmoid(logits)
        m1 = logits.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)

        # dice = 2. * (intersection.sum(1) + smooth) / ((m1**2).sum(1) + (m2**2).sum(1) + smooth)
        # loss = 1 - dice.sum() / num
        dice = 2. * (intersection.sum() + smooth) / ((m1 ** 2).sum() + (m2 ** 2).sum() + smooth)
        loss = 1 - dice.mean()
        return loss


class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma, sigmoid=False):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.sigmoid = sigmoid

    def forward(self, logit, label):
        if self.sigmoid == True:
            logit = torch.sigmoid(logit)
        batch = label.size(0)
        logit = logit.view(batch, -1)
        label = label.view(batch, -1)

        image_size = 256 * 256 * 3
        # positive_sample = self.alpha*(1-logit)**self.gamma*label*torch.log(logit)
        # negative_sample = (1-self.alpha)*logit**self.gamma*(1-label)*torch.log(1-logit)
        positive_sample = (1 - logit) ** self.gamma * label * torch.log(logit)
        negative_sample = logit ** self.gamma * (1 - label) * torch.log(1 - logit)
        # focal_loss = -(positive_sample+negative_sample).sum()/batch/image_size
        focal_loss = -(positive_sample + negative_sample).mean()
        return focal_loss