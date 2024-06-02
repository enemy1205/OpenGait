import torch.optim as optim
from kornia import morphology as morph
from ..base_model import BaseModel
# from .gaitset import GaitSet_Nobase
from .deepgaitv2 import DeepGaitV2_NO_Base
from ..backbones.sttn import InpaintGenerator,Discriminator
from utils import get_valid_args, get_attr_from
import torch

    
class STTN_E2E_EDGE(BaseModel):
    
    def build_network(self, model_cfg):
        self.netGait = DeepGaitV2_NO_Base(model_cfg['Gait'])
        # self.netGait = GaitSet_Nobase(model_cfg['Gait'])
        self.netGen = InpaintGenerator(model_cfg['Gen'])
        self.netDis = Discriminator(model_cfg['Dis'],use_sigmoid=True)
        self.dis_lr = model_cfg['lr_D']
        self.gait_lr = model_cfg['lr_gait']
        self.kernel = torch.ones(
            (model_cfg['kernel_size'], model_cfg['kernel_size']))

    def finetune_parameters(self):
        dis_tune_params = list()
        gait_tune_params = list()
        others_params = list()
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if 'dis' in name:
                dis_tune_params.append(p)
            elif 'netGait' in name:
                gait_tune_params.append(p)
            else:
                others_params.append(p)
        return [{'params': dis_tune_params, 'lr': self.dis_lr},{'params':gait_tune_params,'lr':self.gait_lr} ,{'params': others_params}]

    def get_optimizer(self, optimizer_cfg):
        self.msg_mgr.log_info(optimizer_cfg)
        optimizer = get_attr_from([optim], optimizer_cfg['solver'])
        valid_arg = get_valid_args(optimizer, optimizer_cfg, ['solver'])
        optimizer = optimizer(self.finetune_parameters(), **valid_arg)
        return optimizer

    def preprocess(self, sils):
        dilated_mask = (morph.dilation(sils, self.kernel.to(sils.device)).detach()
                        ) > 0.5  # Dilation
        eroded_mask = (morph.erosion(sils, self.kernel.to(sils.device)).detach()
                       ) > 0.5   # Erosion
        edge_mask = dilated_mask ^ eroded_mask
        # return edge_mask, eroded_mask
        return edge_mask

    def forward(self, inputs):
        ipts, labs, _, _, seqL = inputs
        # ipts[0 or 1] : [b,t,h,w]
        gt_sils = ipts[0].unsqueeze(2)
        occ_sils = ipts[1].unsqueeze(2)
        b, t, c, h, w = occ_sils.size()
        # NetG input : [b,t,c,h,w]
        edge_mask = self.preprocess(ipts[0]).view(b*t, c, h, w)
        recovered_sils = self.netGen(occ_sils)
        recovered_sils = edge_mask*recovered_sils + recovered_sils
        
        recovered_sils = recovered_sils.view(b*t, c, h, w)
        gt_sils = gt_sils.view(b*t, c, h, w)
        
        real_sils_embs = self.netDis(gt_sils)
        fake_sils_embs = self.netDis(recovered_sils.detach())
                
        gen_vid_feat = self.netDis(recovered_sils)
        
        retval = {
            'training_feat': {
                'adv': {'logits': fake_sils_embs, 'labels': real_sils_embs},
                'ssim': {'img1': recovered_sils , 'img2':gt_sils},
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