from torch import nn
import torch.nn.functional as F


class HiCMseLoss(nn.Module):
    def __init__(self,alpha=1):
        super().__init__()
        self.alpha=alpha
    def forward(self,preds,targets):
        if targets.shape[-1]==1:
            return 0.5*F.mse_loss(preds[:,:,:,2:],targets)
        else:
            return F.mse_loss(preds,targets)


class EpiMseLoss(nn.Module):
    def __init__(self,alpha=1):
        super().__init__()
        self.alpha=alpha
    def forward(self,pred,target,lmask):
        sum_loss = 0
        for i in range(pred.shape[0]):
            sum_loss += F.mse_loss(pred[i, :, lmask[i] > 0], target[i, :, :], reduction='sum')
        return self.alpha*sum_loss/(pred.shape[0]*pred.shape[1])