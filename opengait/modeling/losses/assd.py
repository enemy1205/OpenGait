import torch.nn as nn
import torch
from .base import BaseLoss


class AssociatedLoss(BaseLoss):
    def __init__(self, loss_term_weight=1.0, assdloss_type="L1"):
        super(AssociatedLoss, self).__init__(loss_term_weight)
        self.loss_type = assdloss_type
        if assdloss_type == "L1":
            self.criterion = nn.L1Loss()
        elif assdloss_type == "L2":
            self.criterion = nn.MSELoss()
        else:
            self.criterion = nn.CosineEmbeddingLoss()

    def forward(self, logits, labels):
        bs = logits.shape[0]
        labels = labels.float()
        logits = logits.float()

        if self.loss_type == "Cosine":
            target = torch.ones(bs).cuda()
            assd_loss = self.criterion(logits.view(bs, -1), labels.view(bs, -1), target)
        else:
            assd_loss = self.criterion(logits, labels)
        self.info.update({"loss": assd_loss.detach().clone()})
        return assd_loss, self.info
