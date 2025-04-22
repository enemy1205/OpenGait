from einops import rearrange
from ..base_model import BaseModel
from ..backbones.fuse_former import TransformerBlock,SoftSplit,AddPosEmb,SoftComp,Encoder
from ..modules import SetBlockWrapper, HorizontalPoolingPyramid, PackSequenceWrapper, SeparateFCs, SeparateBNNecks, conv1x1, conv3x3, BasicBlock2D, BasicBlockP3D, BasicBlock3D
import torch
import torch.nn as nn

class CatFusion(nn.Module): 
    def __init__(self, in_channels=64):
        super(CatFusion, self).__init__()
        self.conv = SetBlockWrapper(
            nn.Sequential(
                conv1x1(in_channels * 2, 2*in_channels), 
            )
        )

    def forward(self, sil_feat, map_feat): 
        '''
            sil_feat: [n, c, s, h, w]
            map_feat: [n, c, s, h, w]
        '''
        feats = torch.cat([sil_feat, map_feat],1)
        retun = self.conv(feats)
        return retun
    

class OriginFormerGaitConcat_CenterLoss(BaseModel):
    # 
    def build_network(self, model_cfg):
        self.encoder = Encoder()
        channel = model_cfg['ss_channel']
        hidden = model_cfg['hidden']
        stack_num = model_cfg['stack_num']
        num_head = model_cfg['num_head']
        kernel_size = (model_cfg['kernel_size'][0],model_cfg['kernel_size'][1])
        padding = (model_cfg['padding'][0],model_cfg['padding'][1])
        stride = (model_cfg['stride'][0],model_cfg['stride'][1])
        output_size = (model_cfg['output_size'][0],model_cfg['output_size'][1])
        
        blocks = []
        dropout = model_cfg['dropout'] 
        t2t_params = {'kernel_size': kernel_size, 'stride': stride, 'padding': padding, 'output_size': output_size}
        n_vecs = 1
        for i, d in enumerate(kernel_size):
            n_vecs *= int((output_size[i] + 2 * padding[i] - (d - 1) - 1) / stride[i] + 1)
        for _ in range(stack_num):
            blocks.append(TransformerBlock(hidden=hidden, num_head=num_head, dropout=dropout, n_vecs=n_vecs,
                                           t2t_params=t2t_params))
        self.transformer = nn.Sequential(*blocks)
        self.ss = SoftSplit(channel // 2, hidden, kernel_size, stride, padding, dropout=dropout)

        self.add_pos_emb = AddPosEmb(n_vecs, hidden)
        self.sc = SoftComp(channel // 2, hidden, output_size, kernel_size, stride, padding)
        self.fusion = CatFusion(channel//2)
        self.FCs = SeparateFCs(16, channel, channel // 2)
        self.BNNecks = SeparateBNNecks(16, channel // 2, class_num=model_cfg['SeparateBNNecks']['class_num'])

        self.TP = PackSequenceWrapper(torch.max)
        self.HPP = HorizontalPoolingPyramid(bin_num=[16])
    
    def forward(self, inputs):
        # 不使用 encoder fea_map
        ipts, labs, typs, vies, seqL = inputs

        sils = ipts[0].unsqueeze(2)
        
        n, s, c, h, w = sils.size()
        
        assert sils.size(-1) in [44, 88]

        del ipts        
        
        enc_feat = self.encoder(sils.view(n*s,c,h,w))      # [b*s,256,h/4,w/4]
                
        trans_feat = self.ss(enc_feat, n) #[b,1050,512]
        trans_feat = self.add_pos_emb(trans_feat) #[b,1050,512]
        trans_feat = self.transformer(trans_feat) #[b,1050,512]
        trans_feat = self.sc(trans_feat, s) #[t*b,256,h/4,w/4]
        
        output_size = enc_feat.size()
        
        enc_feat = enc_feat.reshape(n, s, *output_size[1:]).transpose(1, 2).contiguous()
        trans_feat = trans_feat.reshape(n, s, *output_size[1:]).transpose(1, 2).contiguous()
        
        enc_feat = self.fusion(trans_feat,enc_feat)
        # Temporal Pooling, TP
        outs_2 = self.TP(enc_feat, seqL, options={"dim": 2})[0]  # [n, c, h, w]
        # Horizontal Pooling Matching, HPM
        feat_2 = self.HPP(outs_2)  # [n, c, p]
        
        # embed_1 = self.FCs(torch.add(feat_1,feat_2))  # [n, c, p]
        embed_1 = self.FCs(feat_2)  # [n, c, p]
        embed_2, logits = self.BNNecks(embed_1)  # [n, c, p] , [n, class_num, p]

        embed = embed_1

        retval = {
            'training_feat': {
                'center': {'embeddings': embed_1, 'labels': labs},
                'softmax': {'logits': logits, 'labels': labs}
            },
            'visual_summary': {
                'image/sils': rearrange(sils, 'n s c h w -> (n s) c h w'),
            },
            'inference_feat': {
                'embeddings': embed
            }
        }

        return retval