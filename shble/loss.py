import torch
from torch import nn
import torch.nn.functional as F


class WeightedCrossEntropyLoss(nn.Module):

    def __init__(self, alpha=None, size_average=True):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average


    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=-1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = logpt.data.exp()

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data).to(input.get_device())
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * at

        loss = -1 * (1 - pt) * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


# https://github.com/clcarwin/focal_loss_pytorch/blob/master/focalloss.py
class FocalLoss(nn.Module):

    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average


    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input, dim=-1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = logpt.data.exp()

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data).to(input.get_device())
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * at

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()


class MFELoss(nn.Module):

    def __init__(self, args, others_idx):
        super(MFELoss, self).__init__()
        self.others_idx = others_idx
        self.softmax = nn.Softmax(dim=1)


    def forward(self, preds, target):
        batch_size, _ = preds.size()

        preds = self.softmax(preds)

        fpe = 0
        fne = 0
        fpe_num = 0
        fne_num = 0
        for i in range(batch_size):
            if target[i] == self.others_idx:
                fne += 1/2 * ((preds[i].sum()-preds[i][self.others_idx])**2
                             + (preds[i][self.others_idx]-1)**2)
                fne_num += 1
            else:
                fpe += 1/2 * ((-preds[i][self.others_idx])**2
                             + (preds[i][self.others_idx])**2)
                fpe_num += 1
        fpe = fpe / fpe_num
        fne = fne / fne_num

        loss = fpe + fne
        return loss


class MSFELoss(nn.Module):

    def __init__(self, args, others_idx):
        super(MSFELoss, self).__init__()
        self.others_idx = others_idx
        self.softmax = nn.Softmax(dim=1)

    def forward(self, preds, target):
        batch_size, _ = preds.size()

        preds = self.softmax(preds)

        fpe = 0
        fne = 0
        fpe_num = 0
        fne_num = 0
        for i in range(batch_size):
            if target[i] == self.others_idx:
                fne += 1 / 2 * ((preds[i].sum() - preds[i][self.others_idx]) ** 2
                                + (preds[i][self.others_idx] - 1) ** 2)
                fne_num += 1
            else:
                fpe += 1 / 2 * ((-preds[i][self.others_idx]) ** 2
                                + (preds[i][self.others_idx]) ** 2)
                fpe_num += 1
        fpe = fpe / fpe_num
        fne = fne / fne_num

        loss = fpe**2 + fne**2
        return loss


class AMFELoss(nn.Module):

    def __init__(self, args, others_idx):
        super(AMFELoss, self).__init__()
        self.alpha = args.mfe_alpha
        self.others_idx = others_idx
        self.softmax = nn.Softmax(dim=1)


    def forward(self, preds, target):
        batch_size, _ = preds.size()

        preds = self.softmax(preds)

        fpe = 0
        fne = 0
        fpe_num = 0
        fne_num = 0
        for i in range(batch_size):
            if target[i] == self.others_idx:
                fne += 1/2 * ((preds[i].sum()-preds[i][self.others_idx])**2
                             + (preds[i][self.others_idx]-1)**2)
                fne_num += 1
            else:
                fpe += 1/2 * ((-preds[i][self.others_idx])**2
                             + (preds[i][self.others_idx])**2)
                fpe_num += 1
        fpe = fpe / fpe_num
        fne = fne / fne_num

        loss = (1-self.alpha) * fpe + self.alpha * fne
        return loss


class AMSFELoss(nn.Module):

    def __init__(self, args, others_idx):
        super(AMSFELoss, self).__init__()
        self.alpha = args.mfe_alpha
        self.others_idx = others_idx
        self.softmax = nn.Softmax(dim=1)


    def forward(self, preds, target):
        batch_size, _ = preds.size()

        preds = self.softmax(preds)

        fpe = 0
        fne = 0
        fpe_num = 0
        fne_num = 0
        for i in range(batch_size):
            if target[i] == self.others_idx:
                fne += 1/2 * ((preds[i].sum()-preds[i][self.others_idx])**2
                             + (preds[i][self.others_idx]-1)**2)
                fne_num += 1
            else:
                fpe += 1/2 * ((-preds[i][self.others_idx])**2
                             + (preds[i][self.others_idx])**2)
                fpe_num += 1
        fpe = fpe / fpe_num
        fne = fne / fne_num

        loss = (1-self.alpha)**2 * fpe**2 + self.alpha**2 * fne**2
        return loss


class ModifiedMFELoss(nn.Module):

    def __init__(self, args, data):
        super(ModifiedMFELoss, self).__init__()
        self.alpha = args.mfe_alpha
        self.others_idx = data.LABEL.vocab.stoi['others']
        self.happy_idx = data.LABEL.vocab.stoi['happy']
        self.sad_idx = data.LABEL.vocab.stoi['sad']
        self.angry_idx = data.LABEL.vocab.stoi['angry']
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, outputs, target):
        batch_size, _ = outputs.size()

        preds = self.softmax(outputs)

        fpe = 0
        fne = 0
        fpe_num = 0
        fne_num = 0
        for i in range(batch_size):
            if target[i] == self.others_idx:
                fne += (preds[i][self.others_idx] - 1) ** 2
                fpe += (preds[i][self.happy_idx]) ** 2
                fpe += (preds[i][self.sad_idx]) ** 2
                fpe += (preds[i][self.angry_idx]) ** 2
                fne_num += 1
                fpe_num += 3
            elif target[i] == self.happy_idx:
                fpe += (preds[i][self.sad_idx]) ** 2
                fpe += (preds[i][self.angry_idx]) ** 2
                fpe_num += 2
            elif target[i] == self.sad_idx:
                fpe += (preds[i][self.happy_idx]) ** 2
                fpe += (preds[i][self.angry_idx]) ** 2
                fpe_num += 2
            else:
                fpe += (preds[i][self.sad_idx]) ** 2
                fpe += (preds[i][self.happy_idx]) ** 2
                fpe_num += 2
        fpe = fpe / fpe_num
        fne = fne / fne_num

        loss = (self.alpha / 3.0) * fpe + self.alpha * fne
        return loss