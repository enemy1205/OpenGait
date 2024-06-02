import torch.optim as optim
from ..base_model import BaseModel
from ..backbones.sttn_swingait import InpaintGenerator,Discriminator
from utils import get_valid_args, get_attr_from


class STTN_Rec(BaseModel):
    
    def build_network(self, model_cfg):
        self.netGen = InpaintGenerator(model_cfg['Gen'])
        self.netDis = Discriminator(model_cfg['Dis'],use_sigmoid=True)
        # self._relu = torch.nn.ReLU1(True)
        self.dis_lr = model_cfg['lr_D']


    def finetune_parameters(self):
        dis_tune_params = list()
        others_params = list()
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if 'dis' in name:
                dis_tune_params.append(p)
            else:
                others_params.append(p)
        return [{'params': dis_tune_params, 'lr': self.dis_lr} ,{'params': others_params}]

    def get_optimizer(self, optimizer_cfg):
        self.msg_mgr.log_info(optimizer_cfg)
        optimizer = get_attr_from([optim], optimizer_cfg['solver'])
        valid_arg = get_valid_args(optimizer, optimizer_cfg, ['solver'])
        optimizer = optimizer(self.finetune_parameters(), **valid_arg)
        return optimizer


    def forward(self, inputs):
        ipts, labs, _, _, seqL = inputs
        # ipts[0 or 1] : [b,t,h,w]
        gt_sils = ipts[0].unsqueeze(1)
        occ_sils = ipts[1]
        # occ_sils = ipts[1].unsqueeze(2)
        # occ_sils = ipts[1].unsqueeze(2)
        b, c,t, h, w = gt_sils.size()
        # NetG input : [b,x,c,h,w]
        # recovered_sils : [b,c,t,h,w]
        recovered_sils = self.netGen(occ_sils)
        
        real_sils_embs = self.netDis(gt_sils)
        fake_sils_embs = self.netDis(recovered_sils.detach())
                
        gen_vid_feat = self.netDis(recovered_sils)
        
        retval = {
            'training_feat': {
                'adv': {'logits': fake_sils_embs, 'labels': real_sils_embs},
                'gan': {'pred_silt_video':recovered_sils.transpose(1, 2),'gt_silt_video':gt_sils.transpose(1, 2),'gen_vid_feat':gen_vid_feat}
            },
            'visual_summary': {
                'image/gt_sils': gt_sils.transpose(1, 2).view(b*t, c, h, w), 'image/occ_sils': occ_sils.view(b*t, c, h, w), "image/rec_sils": recovered_sils.transpose(1, 2).view(b*t, c, h, w)            
            },
            'inference_feat': {
                'gt': gt_sils.transpose(1, 2).view(b*t, c, h, w),
                'pred': recovered_sils
            }
        }

        return retval
    
    