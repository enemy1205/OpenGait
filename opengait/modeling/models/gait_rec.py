import torch.optim as optim
import os.path as osp
from ..base_model import BaseModel
from ..backbones.sttn import Discriminator
# from .gaitset import GaitSet_Nobase
from ..backbones.simvp import Encoder,Mid_Xnet,Decoder
from utils import get_valid_args, get_attr_from
import numpy as np
import torch

class SimVP_Rec(BaseModel):
    def build_network(self, model_cfg):
        T, C, H, W = model_cfg['shape_in']
        hid_S = model_cfg['hid_S']
        N_S = model_cfg['N_S']
        hid_T = model_cfg['hid_T']
        N_T = model_cfg['N_T']
        incep_ker = model_cfg['incep_ker']
        groups = model_cfg['groups']
        # self.netGait = GaitSet_Nobase(model_cfg['GaitSet'])
        self.enc = Encoder(C, hid_S, N_S)
        self.hid = Mid_Xnet(T*hid_S, hid_T, N_T, incep_ker, groups)
        self.dec = Decoder(hid_S, C, N_S)
        self.dis = Discriminator(model_cfg['Dis'],use_sigmoid=True)
        self.dis_lr = model_cfg['lr_D']
        self._relu = torch.nn.ReLU1(True)
        
    def finetune_parameters(self):
        fine_tune_params = list()
        others_params = list()
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if 'dis' in name:
                fine_tune_params.append(p)
            else:
                others_params.append(p)
        return [{'params': fine_tune_params, 'lr': self.dis_lr}, {'params': others_params}]

    def get_optimizer(self, optimizer_cfg):
        self.msg_mgr.log_info(optimizer_cfg)
        optimizer = get_attr_from([optim], optimizer_cfg['solver'])
        valid_arg = get_valid_args(optimizer, optimizer_cfg, ['solver'])
        optimizer = optimizer(self.finetune_parameters(), **valid_arg)
        return optimizer

    # def get_scheduler(self, scheduler_cfg):
    #     self.msg_mgr.log_info(scheduler_cfg)
    #     Scheduler = get_attr_from(
    #         [optim.lr_scheduler], scheduler_cfg['scheduler'])
    #     valid_arg = get_valid_args(Scheduler, scheduler_cfg, ['scheduler'])
    #     valid_arg['steps_per_epoch'] = len(self.train_loader)
    #     valid_arg['epochs'] = self.engine_cfg['total_iter']
    #     scheduler = Scheduler(self.optimizer, **valid_arg)
    #     return scheduler

    def forward(self, inputs):
        ipts, labs, _, _, seqL = inputs
        # ipts[0 or 1] : [b,t,h,w]
        gt_sils = ipts[0].unsqueeze(2)
        occ_sils = ipts[1].unsqueeze(2)
        b, t, c, h, w = occ_sils.shape
        x = occ_sils.view(b*t, c, h, w)

        embed, skip = self.enc(x)
        _, C_, H_, W_ = embed.shape

        z = embed.view(b, t, C_, H_, W_)
        hid = self.hid(z)
        hid = hid.reshape(b*t, C_, H_, W_)

        pred_y  = self.dec(hid, skip)
        pred_y = self._relu(pred_y)
        # pred_y  = pred_y .reshape(b, t, c, h, w)
        
        recovered_sils = pred_y.view(b*t, c, h, w)
        gt_sils = gt_sils.view(b*t, c, h, w)   
        
        gen_vid_feat = self.dis(recovered_sils)
        
        real_sils_embs = self.dis(gt_sils)
        fake_sils_embs = self.dis(recovered_sils.detach())
        
        
        retval = {
            'training_feat': {
                'adv': {'logits': fake_sils_embs, 'labels': real_sils_embs},
                'gan': {'pred_silt_video':pred_y,'gt_silt_video':gt_sils,'gen_vid_feat':gen_vid_feat},
            },
            'visual_summary': {
                'image/gt_sils': gt_sils.view(b*t, c, h, w), 'image/occ_sils': occ_sils.view(b*t, c, h, w), "image/rec_sils": pred_y.view(b*t, c, h, w)            
            },
            'inference_feat': {
                'gt': gt_sils.view(b*t, c, h, w),
                'pred': pred_y.view(b*t, c, h, w)
            }
        }
        
        # retval_gait = self.netGait([[recovered_sils.view(b,t, h, w)], labs, None, None, seqL])
        # real_gait = self.netGait([[gt_sils.view(b,t, h, w)], labs, None, None, seqL])
        # retval['training_feat']['triplet'] = retval_gait['training_feat']['triplet']
        # retval['inference_feat']['embeddings'] = retval_gait['inference_feat']['embeddings']
        return retval
    
    
    def eval_forward(self,inputs):
        ipts, labs, _, _, seqL = inputs
        # ipts[0 or 1] : [b,t,h,w]
        gt_sils = ipts[0].unsqueeze(2)
        occ_sils = ipts[1].unsqueeze(2)
        b, t, c, h, w = occ_sils.shape
        x = occ_sils.view(b*t, c, h, w)
        
    
    def inference(self, rank):
        from tqdm import tqdm
        from utils import NoOp
        from utils import Odict, ts2np, ddp_all_gather
        from torch.cuda.amp import autocast
        """Inference all the test data.

        Args:
            rank: the rank of the current process.Transform
        Returns:
            Odict: contains the inference results.
        """
        total_size = len(self.test_loader)
        if rank == 0:
            pbar = tqdm(total=total_size, desc='Transforming')
        else:
            pbar = NoOp()
        batch_size = self.test_loader.batch_sampler.batch_size
        rest_size = total_size
        info_dict = Odict()
        for inputs in self.test_loader:
            ipts = self.inputs_pretreament(inputs)
            with autocast(enabled=self.engine_cfg['enable_float16']):
                retval = self.forward(ipts)
                inference_feat = retval['inference_feat']
                for k, v in inference_feat.items():
                    inference_feat[k] = ddp_all_gather(v, requires_grad=False)
                del retval
            for k, v in inference_feat.items():
                inference_feat[k] = ts2np(v)
            info_dict.append(inference_feat)
            rest_size -= batch_size
            if rest_size >= 0:
                update_size = batch_size
            else:
                update_size = total_size % batch_size
            pbar.update(update_size)
        pbar.close()
        for k, v in info_dict.items():
            v = np.concatenate(v)[:total_size]
            info_dict[k] = v
        return info_dict