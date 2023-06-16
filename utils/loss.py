import torch
import torch.nn as nn

class SegmentationLosses(object):
    def __init__(self, weight=None, size_average=True, batch_average=True, ignore_index=255, cuda=False):
        self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = size_average
        self.batch_average = batch_average
        self.cuda = cuda

    def build_loss(self, mode='focal'):
        """Choices: ['ce' or 'focal' or 'mixed']"""
        if mode == 'ce':
            return self.CrossEntropyLoss
        elif mode == 'focal':
            return self.FocalLoss
        elif mode == 'mixed':
            return self.MixedLoss
        else:
            raise NotImplementedError

    def CrossEntropyLoss(self, logit, target):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        loss = criterion(logit, target.long())

        if self.batch_average:
            loss = loss.mean()

        return loss

    def FocalLoss(self, logit, target, gamma=2, alpha=0.5):
        n, c, h, w = logit.size()
        smooth = 1.
        logit = logit.long()/logit.max()
        target = target.long()/target.max()

        criterion_focal = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                            size_average=self.size_average)
        if self.cuda:
            criterion_focal = criterion_focal.cuda()

        logpt_focal = -criterion_focal(logit, target.long())
        pt_focal = torch.exp(logpt_focal)
        if alpha is not None:
            logpt_focal *= alpha
        focal_loss = -((1 - pt_focal) ** gamma) * logpt_focal

        if self.batch_average:
            focal_loss = focal_loss.mean()

        return focal_loss
    
    def DiceLoss(self, logit, target):
        n, c, h, w = logit.size()
        smooth = 1.
        
        logit = torch.sigmoid(logit).view(n, -1)
        target = target.view(n, -1)
        
        intersection = (logit * target).sum(1)
        dice_score = (2. * intersection + smooth) / (logit.sum(1) + target.sum(1) + smooth)
        dice_loss = 1. - dice_score

        if self.batch_average:
            dice_loss = dice_loss.mean()

        return dice_loss

    
    def MixedLoss(self, logit, target, gamma=2, alpha=0.5, a=0.1):
        n, c, h, w = logit.size()
        smooth = 1.
        logit = logit.long()/logit.max()
        target = target.long()/target.max()

        # Compute Focal Loss
        criterion_focal = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                            size_average=self.size_average)
        if self.cuda:
            criterion_focal = criterion_focal.cuda()

        logpt_focal = -criterion_focal(logit, target.long())
        pt_focal = torch.exp(logpt_focal)
        if alpha is not None:
            logpt_focal *= alpha
        focal_loss = -((1 - pt_focal) ** gamma) * logpt_focal

        # Convert logit to one-hot representation
        logit = logit.argmax(dim=1)

        # Compute Dice Coefficient (1 - Dice Loss)
        logit_dice = torch.sigmoid(logit).view(n, -1)
        target_dice = target.view(n, -1)
        
        intersection = (logit_dice * target_dice).sum(1)
        dice_coeff = (2. * intersection + smooth) / (logit_dice.sum(1) + target_dice.sum(1) + smooth)
        
        # Compute mixed loss
        mixed_loss = (1 - a) * focal_loss - a * torch.log(1 - dice_coeff)

        if self.batch_average:
            mixed_loss = mixed_loss.mean()

        return mixed_loss



if __name__ == "__main__":
    loss = SegmentationLosses(cuda=False)
    a = torch.rand(1, 3, 7, 7)
    b = torch.rand(1, 7, 7)
    print(loss.CrossEntropyLoss(a, b).item())
    print(loss.FocalLoss(a, b, gamma=0, alpha=None).item())
    print(loss.FocalLoss(a, b, gamma=2, alpha=0.5).item())
    print(loss.MixedLoss(a, b).item())




