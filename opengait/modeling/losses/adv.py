import torch.nn as nn
from .base import BaseLoss
import torch
from torch.autograd import Variable
import torch.autograd as autograd

class AdversarialLoss(BaseLoss):
    def __init__(self, loss_term_weight=1.0,wgan_weight = 0.1):

        super(AdversarialLoss, self).__init__(loss_term_weight)

        self.criterion = nn.ReLU()
        
        self.wgan_weight = wgan_weight

    # def forward(self, logits, labels , wgan_loss):
    #     """
    #         logits: fake_img
    #         labels: real_img
    #     """
    #     labels = labels.float()
    #     logits = logits.float()
    #     dis_loss = (self.criterion(1 + logits).mean()+self.criterion(1 - labels).mean())/2
    #     interpolates,d_interpolates = wgan_loss
    #     fake =Variable(torch.ones(d_interpolates.size()).cuda(),requires_grad=False)
    #     gradients = autograd.grad(
    #     outputs=d_interpolates,
    #     inputs=interpolates,
    #     grad_outputs=fake,
    #     create_graph=True,
    #     retain_graph=True,
    #     only_inputs=True,
    #     )[0]
    #     gradients = gradients.view(gradients.size(0), -1)
    #     gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    #     dis_loss += self.wgan_weight*gradient_penalty
    #     self.info.update({
    #         'dis_loss':dis_loss.detach().clone()
    #     })
    #     return dis_loss,self.info
    
    def forward(self, logits, labels):
        """
            logits: fake_img
            labels: real_img
        """
        labels = labels.float()
        logits = logits.float()
        dis_loss = (self.criterion(1 + logits).mean()+self.criterion(1 - labels).mean())/2
        self.info.update({
            'dis_loss':dis_loss.detach().clone()
        })
        return dis_loss,self.info