import torch
import torch.nn.functional as F
from torch.autograd import Variable
from .base import BaseLoss
from math import exp
import numpy as np

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM_Dissimilarity(BaseLoss):
    def __init__(self,loss_term_weight=1.0, window_size = 3, size_average = True):
        super(SSIM_Dissimilarity, self).__init__(loss_term_weight)
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window.type_as(img1)
            img2 = img2.type_as(img1)
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            img2 = img2.type_as(img1)
            self.window = window
            self.channel = channel

        ssim_loss = (1-_ssim(img1, img2, window, self.window_size, channel, self.size_average))/2
        
        self.info.update({
            'ssim_loss':ssim_loss.detach().clone()
        })
        
        return ssim_loss ,self.info

def ssim(img1, img2, window_size = 5, size_average = True):
    (_, channel, _, _) = img1.shape
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    ssim_d = (1-_ssim(img1, img2, window, window_size, channel, size_average))/2
    return ssim_d


def psnr(img1, img2):
    mse = np.mean((img1-img2)**2)
    if mse == 0:
        return float('inf')
    else:
        return 20*np.log10(255/np.sqrt(mse))
    
    

if __name__ == '__main__':
    pred = torch.ones(10,1,64,44).cuda().half()
    gt = torch.ones(10,1,64,44).cuda().half()
    res = ssim(pred,gt)
    print(res)