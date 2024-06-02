import torch.optim as optim
import os.path as osp
from ..base_model import BaseModel
from .deepgaitv2 import DeepGaitV2_NO_Base
from ..backbones.sttn import InpaintGenerator,Discriminator
from utils import get_valid_args, get_attr_from,mkdir
import numpy as np
import torch
    
class STTN_E2E_CONCAT(BaseModel):
    
    def build_network(self, model_cfg):
        self.netGait = DeepGaitV2_NO_Base(model_cfg['Gait'])
        self.netGen = InpaintGenerator(model_cfg['Gen'])
        self.netDis = Discriminator(model_cfg['Dis'],use_sigmoid=True)
        self.dis_lr = model_cfg['lr_D']

    def finetune_parameters(self):
        dis_tune_params = list()
        others_params = list()
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if 'netDis' in name:
                dis_tune_params.append(p)
            else:
                others_params.append(p)
        return [{'params': dis_tune_params, 'lr': self.dis_lr} ,{'params': others_params}]

    def get_optimizer(self, optimizer_cfg):
        self.msg_mgr.log_info(optimizer_cfg)
        # 利用for等循环生成数组，将导致优化器参数无法定制化
        rec_cfg = optimizer_cfg[0]
        optimizer = get_attr_from([optim], rec_cfg['solver'])
        valid_arg = get_valid_args(optimizer, rec_cfg, ['solver'])
        rec_optimizer = optimizer(self.finetune_parameters(), **valid_arg)
        gait_cfg = optimizer_cfg[1]
        optimizer = get_attr_from([optim], gait_cfg['solver'])
        valid_arg = get_valid_args(optimizer, gait_cfg, ['solver'])
        gait_optimizer = optimizer(filter(lambda p: p.requires_grad, self.netGait.parameters()), **valid_arg)
        return [rec_optimizer,gait_optimizer]
    
    def get_scheduler(self, scheduler_cfg):
        self.msg_mgr.log_info(scheduler_cfg)
        Scheduler = get_attr_from(
            [optim.lr_scheduler], scheduler_cfg['scheduler'])
        valid_arg = get_valid_args(Scheduler, scheduler_cfg, ['scheduler'])
        scheduler = [Scheduler(opt, **valid_arg) for opt in self.optimizer] 
        return scheduler

    def save_ckpt(self, iteration):
        if torch.distributed.get_rank() == 0:
            mkdir(osp.join(self.save_path, "checkpoints/"))
            save_name = self.engine_cfg['save_name']
            checkpoint = {
                'model': self.state_dict(),
                'optimizer_rec': self.optimizer[0].state_dict(),
                'optimizer_gait': self.optimizer[1].state_dict(),
                'scheduler_rec': self.scheduler[0].state_dict(),
                'scheduler_gait': self.scheduler[1].state_dict(),
                'iteration': iteration}
            torch.save(checkpoint,
                       osp.join(self.save_path, 'checkpoints/{}-{:0>5}.pt'.format(save_name, iteration)))

    def train_step(self, loss_sum) -> bool:
        """Conduct loss_sum.backward(), self.optimizer.step() and self.scheduler.step().

        Args:
            loss_sum:The loss of the current batch.
        Returns:
            bool: True if the training is finished, False otherwise.
        """
        for i,optimizer in enumerate(self.optimizer):
            optimizer.zero_grad()
        if loss_sum <= 1e-9:
            self.msg_mgr.log_warning(
                "Find the loss sum less than 1e-9 but the training process will continue!")

        if self.engine_cfg['enable_float16']:
            self.Scaler.scale(loss_sum).backward()
            for optimizer in self.optimizer:
                self.Scaler.step(optimizer)
            scale = self.Scaler.get_scale()
            self.Scaler.update()
            # Warning caused by optimizer skip when NaN
            # https://discuss.pytorch.org/t/optimizer-step-before-lr-scheduler-step-error-using-gradscaler/92930/5
            if scale != self.Scaler.get_scale():
                self.msg_mgr.log_debug("Training step skip. Expected the former scale equals to the present, got {} and {}".format(
                    scale, self.Scaler.get_scale()))
                return False
        else:
            loss_sum.backward()
            for optimizer in self.optimizer:
                optimizer.step()

        self.iteration += 1
        for scheduler in self.scheduler:
            scheduler.step()
        return True

    def forward(self, inputs):
        ipts, labs, _, _, seqL = inputs
        # ipts[0 or 1] : [b,t,h,w]
        gt_sils = ipts[0].unsqueeze(2)
        occ_sils = ipts[1].unsqueeze(2)
        b, t, c, h, w = occ_sils.size()
        # NetG input : [b,x,c,h,w]
        # recovered_sils : [b,t,1,h,w]
        # enc_feat : [b,128,t,h/4,w/4]
        recovered_sils,enc_feat = self.netGen(occ_sils)
        
        recovered_sils = recovered_sils.view(b*t, c, h, w)
        gt_sils = gt_sils.view(b*t, c, h, w)
        
        real_sils_embs = self.netDis(gt_sils)
        fake_sils_embs = self.netDis(recovered_sils.detach())
                
        gen_vid_feat = self.netDis(recovered_sils)
        
        retval = {
            'training_feat': {
                'adv': {'logits': fake_sils_embs, 'labels': real_sils_embs},
                'gan': {'pred_silt_video':recovered_sils,'gt_silt_video':gt_sils,'gen_vid_feat':gen_vid_feat}
            },
            'visual_summary': {
                'image/gt_sils': gt_sils, 'image/occ_sils': occ_sils.view(b*t, c, h, w), "image/rec_sils": recovered_sils.view(b*t, c, h, w)            
            },
            'inference_feat': {
                'gt': gt_sils.view(b*t, c, h, w),
                'pred': recovered_sils.view(b*t, c, h, w)
            }
        }
        retval_gait = self.netGait([[recovered_sils.view(b,t, h, w)], labs, None, None, seqL])
        retval['training_feat']['triplet'] = retval_gait['training_feat']['triplet']
        retval['training_feat']['softmax'] = retval_gait['training_feat']['softmax']
        retval['inference_feat']['embeddings'] = retval_gait['inference_feat']['embeddings']
        
        return retval