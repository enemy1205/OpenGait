import torch.nn as nn
from .base import BaseLoss


class GanLoss(BaseLoss):
    def __init__(
        self,
        loss_term_weight=1.0,
        imgloss_type="L1",
        adversarial_weight=0.1,
        img_weight=0.1,
    ):
        super(GanLoss, self).__init__(loss_term_weight)
        if imgloss_type == "L1":
            self.criterion = nn.L1Loss()
        else:
            self.criterion = nn.MSELoss()
        self.adversarial_weight = adversarial_weight
        self.img_weight = img_weight

    def forward(self, pred_silt_video, gt_silt_video, gen_vid_feat):
        pred_silt_video = pred_silt_video.float()
        gt_silt_video = gt_silt_video.float()
        gen_vid_feat = gen_vid_feat.float()

        gan_loss = (-gen_vid_feat).mean()

        valid_loss = self.criterion(pred_silt_video, gt_silt_video)

        gen_loss = self.adversarial_weight * gan_loss + self.img_weight * valid_loss
        self.info.update({"gen_loss": gen_loss.detach().clone()})
        return gen_loss, self.info

    # def forward(self, pred_silt_video, gt_silt_video):
    #     pred_silt_video = pred_silt_video.float()
    #     gt_silt_video = gt_silt_video.float()

    #     valid_loss = self.criterion(pred_silt_video, gt_silt_video)

    #     gen_loss = valid_loss
    #     self.info.update({
    #         'gen_loss':gen_loss.detach().clone()
    #     })
    #     return gen_loss,self.info
