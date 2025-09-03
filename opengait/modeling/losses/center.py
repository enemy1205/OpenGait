import torch
import torch.nn.functional as F

from .base import BaseLoss, gather_and_scale_wrapper


class CenterLoss(BaseLoss):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=10, feat_dim=2, loss_term_weight=1.0):
        super(CenterLoss, self).__init__(loss_term_weight)
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.centers = torch.nn.Parameter(
            torch.randn(self.num_classes, self.feat_dim).cuda()
        )

    @gather_and_scale_wrapper
    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = embeddings.size(0)
        distmat = (
            torch.pow(embeddings, 2)
            .sum(dim=1, keepdim=True)
            .expand(batch_size, self.num_classes)
            + torch.pow(self.centers, 2)
            .sum(dim=1, keepdim=True)
            .expand(self.num_classes, batch_size)
            .t()
        )
        distmat.addmm_(1, -2, embeddings, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e12).sum() / batch_size

        return loss
