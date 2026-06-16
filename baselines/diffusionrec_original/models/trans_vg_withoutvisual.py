from pickle import FALSE
from numpy import dtype
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import random
import math

from detectron2.modeling.poolers import ROIPooler

from torch.nn.parameter import Parameter
#from pytorch_pretrained_bert.modeling import BertModel
import torchvision
#from transformers import BertModel
from .visual_model.detr import build_detr
from .language_model.bert import build_bert
#from .language_model.language_diffusion import WrappedModel
from .vl_transformer import build_vl_transformer
from.multi_head.head import DynamicHead
from utils.box_utils import bbox_iou
from utils.box_utils import xywh2xyxy
from utils.box_utils import xyxy2xywh
from einops.einops import rearrange,repeat
#from .language_model import gaussian_diffusion as gd
#from .language_model.gaussian_diffusion import SpacedDiffusion, space_timesteps
 
from icecream import ic
import numpy as np

_DEFAULT_SCALE_CLAMP = math.log(100000.0 / 16)



# def box_cxcywh_to_xyxy(x):
#     x_c, y_c, w, h = x.unbind(-1)
#     b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
#          (x_c + 0.5 * w), (y_c + 0.5 * h)]
#     return torch.stack(b, dim=-1)

def extract(a, t, x_shape):
    """extract the appropriate  t  index for a batch of indices"""
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class MuModule:
    pass
class TransVG(nn.Module):
    def __init__(self, args):
        super(TransVG, self).__init__()
        hidden_dim = args.vl_hidden_dim
        divisor = 16 if args.dilation else 32
        self.num_visu_token = int((args.imsize / divisor) ** 2)
        self.num_text_token = args.max_query_len

        self.visumodel = build_detr(args)
        self.textmodel = build_bert(args)

        num_total = self.num_visu_token + self.num_text_token + 1
        self.vl_pos_embed = nn.Embedding(num_total, hidden_dim)
        self.reg_token = nn.Embedding(1, hidden_dim)

        self.visu_proj = nn.Linear(self.visumodel.num_channels, hidden_dim)
        self.text_proj = nn.Linear(self.textmodel.num_channels, hidden_dim)
        self.vl_transformer = build_vl_transformer(args)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        


    def forward(self, img_data, text_data):
        
        bs = img_data.tensors.shape[0]
        # visual backbone
        visu_mask, visu_src = self.visumodel(img_data)
        visu_src = self.visu_proj(visu_src) # (N*B)xC
   
        # language bert
        
        text_fea = self.textmodel(text_data)
        text_src, text_mask = text_fea.decompose()
        assert text_mask is not None
        text_src = self.text_proj(text_src)
        # permute BxLenxC to LenxBxC
        text_src = text_src.permute(1, 0, 2)
        text_mask = text_mask.flatten(1)

        # target regression token
        tgt_src = self.reg_token.weight.unsqueeze(1).repeat(1, bs, 1)
        tgt_mask = torch.zeros((bs, 1)).to(tgt_src.device).to(torch.bool)
        
        vl_src = torch.cat([tgt_src, text_src, visu_src], dim=0)
        vl_mask = torch.cat([tgt_mask, text_mask, visu_mask], dim=1)
        vl_pos = self.vl_pos_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        
        vg_hs = self.vl_transformer(vl_src, vl_mask, vl_pos) # (1+L+N)xBxC
        vg_hs = vg_hs[0]
        
        pred_box = self.bbox_embed(vg_hs).sigmoid()

        return pred_box

class TransVGSwin(nn.Module):
    def __init__(self, args):
        super(TransVGSwin, self).__init__()
        hidden_dim = args.vl_hidden_dim
        divisor = 32
        self.num_visu_token = int((args.imsize / divisor) ** 2) + int((args.imsize / (2*divisor)) ** 2)
        self.num_text_token = args.max_query_len
        from models.QRNet import QRNet
        self.visumodel = QRNet(args)
        self.textmodel = build_bert(args)
        #pdb.set_trace()
        num_total = self.num_visu_token + self.num_text_token + 1
        self.vl_pos_embed = nn.Embedding(num_total, hidden_dim)
        self.scale_embed = nn.Embedding(5,self.visumodel.num_channels)
        self.reg_token = nn.Embedding(1, hidden_dim)
        #pdb.set_trace()
        self.visu_proj = nn.Linear(self.visumodel.num_channels, hidden_dim)
        self.text_proj = nn.Linear(self.textmodel.num_channels, hidden_dim)
        #self.vl_transformer = build_vl_transformer(args)
        #self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        #self.bbox_embed_c = MLP(hidden_dim * 2, hidden_dim, 4, 3)


        #diffusion process
        #diffusion language
        #self.timestep_map = []
        # self.use_timesteps = 2000
        # self.rescale_timesteps = True
        # '''
        # for i in range(0, self.use_timesteps):
        #     self.timestep_map.append(i)
        # '''
        # noise_schedule = 'sqrt'
        # timestep_respacing = [self.use_timesteps]
        # language_betas = gd.get_named_beta_schedule(noise_schedule, self.use_timesteps)
        # kwargs = {'betas':language_betas, 'rescale_timesteps':timestep_respacing, 'predict_xstart':True,
        #           'learn_sigmas': False,'sigma_small':False, 'use_kl':False, 'rescale_learned_sigmas': False}
        # diffusion_step = space_timesteps(self.use_timesteps, timestep_respacing)
        # self.diffusion = SpacedDiffusion(diffusion_step,  **kwargs)
        # self.schedule_sampler = WrappedModel(self.use_timesteps)

        #diffsuion vision
        #build dynamic head
        self.head = DynamicHead()
        # ROI_pooler_resolution = 7
        # ROI_pooler_scales = tuple([0.25, 0.125, 0.0625, 0.03125])
        # ROI_sampling_ratio = 2
        # ROI_pooler_type = 'ROIAlignV2'
        # self.pooler = self._init_box_pooler(ROI_pooler_resolution, ROI_pooler_scales, ROI_sampling_ratio, ROI_pooler_type)
        #init boxes
        self.Threshold = 0.7 
        self.num_timesteps = 1000
        self.num_proposals = 500
        self.ddim_sampling_eta = 1.0
    
        betas = cosine_beta_schedule(self.num_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))
        self.scale = 2.0
        self.sampling_timesteps = 30
        #ddim parameter
        # self.self_condition = False
        # self.sampling_timesteps = 30
        # betas = cosine_beta_schedule(self.num_timesteps)
        # alphas = 1. - betas
        # self.alphas_cumprod = torch.cumprod(alphas, dim=0).cuda()
        # self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod).cuda()
        # self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod).cuda()
        # self.scale = 2.0
        # self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod)
        # self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod - 1)
        

        self.query = Parameter(torch.FloatTensor(1, self.textmodel.num_channels))
        self.reset_parameters()
        #visual language matching
        self.language_proj = nn.Linear(self.textmodel.num_channels, hidden_dim)
        self.l1 = nn.Linear(self.textmodel.num_channels, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, 1)
        self.Tanh = nn.Tanh()


    @staticmethod
    def _init_box_pooler(pooler_resolution, pooler_scales, sampling_ratio, pooler_type):
        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        return box_pooler

    
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.query.size(1))
        self.query.data.uniform_(-stdv, stdv)
    

    def predict_noise_from_start(self, x_t, t, x0):
        #pdb.set_trace()
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        #pdb.set_trace()
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise


    def prepare_diffusion_concat(self, gt_boxes, device):
        """
        :param gt_boxes: (cx, cy, w, h), normalized
        :param num_proposals:
        """
        all_diffused_boxes = []
        all_noises = []
        ts = []
        #t= torch.tensor([1000], device=device).long()
        for i in range(0, gt_boxes.size(0)):
            i_box = gt_boxes[i].unsqueeze(0)
            #pdb.set_trace()
            t = torch.randint(100, self.num_timesteps, (1,), device=device).long()
            #print(t)
            noise = torch.randn(self.num_proposals, 4, device=device)

            num_gt = i_box.shape[0]
            if not num_gt:  # generate fake gt boxes if empty gt boxes
                i_box = torch.as_tensor([[0.5, 0.5, 1., 1.]], dtype=torch.float, device=device)
                num_gt = 1

            if num_gt < self.num_proposals:
                box_placeholder = torch.randn(self.num_proposals - num_gt, 4,
                                            device=device) / 6. + 0.5  # 3sigma = 1/2 --> sigma: 1/6
                box_placeholder[:, 2:] = torch.clip(box_placeholder[:, 2:], min=1e-4)
                x_start = torch.cat((i_box, box_placeholder), dim=0)
            elif num_gt > self.num_proposals:
                select_mask = [True] * self.num_proposals + [False] * (num_gt - self.num_proposals)
                random.shuffle(select_mask)
                x_start = i_box[select_mask]
            else:
                x_start = i_box

            x_start = (x_start * 2. - 1.) * 2.0
            # noise sample
            x = self.q_sample(x_start=x_start, t=t, noise=noise)
            x = torch.clamp(x, min=-1 * 2.0, max=2.0)
            x = ((x / 2.0) + 1) / 2.
            #diff_boxes = box_cxcywh_to_xyxy(
            # )
            diff_boxes = xywh2xyxy(x)
            all_diffused_boxes.append(diff_boxes)
            all_noises.append(noise)
            ts.append(t)

        return torch.stack(all_diffused_boxes)
        #return torch.stack(all_diffused_boxes), torch.stack(all_noises), torch.stack(ts)
     
    #def box_feature_selction(self, ex_bboxes, bboxes, features, gt_boxes, Threshold):
    #def box_feature_selction(self, ex_bboxes, bboxes, features, sentence_features, Threshold, next_boxes = None):
    # def box_feature_selction(self, bboxes, features, sentence_features, Threshold, next_boxes = None):
    #     #pdb.set_trace()
    #     num = bboxes.size(1)
    #     sentence_features_expand = sentence_features.unsqueeze(1).expand(sentence_features.size(0), num, sentence_features.size(1))
    #     v_l_scores = self.l2(self.Tanh(self.l1(sentence_features_expand) + features)).squeeze(2)
    #     #v_l_weight = F.softmax(v_l_score, dim = 1)
    #     #v_l_weight_sum = torch.sum(v_l_weight, dim=1).unsqueeze(1).expand(v_l_weight.size(0), v_l_weight.size(1))
    #     #v_l_weight[v_l_weight != 0] = v_l_weight[v_l_weight_sum != 0] / v_l_weight_sum[v_l_weight_sum != 0]
    #     selected_num = int(num * Threshold)
    #     '''
    #     iou_scores = []
    #     for i in range(0, bboxes.size(1)):
    #         iou_scores.append(bbox_iou(bboxes[:,i,:], gt_boxes))
    #     '''
    #     #pdb.set_trace()
    #     #iou_scores = torch.stack(iou_scores).transpose(0, 1)
    #     #pdb.set_trace()
    #     if next_boxes == None: 
    #         final_selected_pred_boxes = []
    #         final_selected_boxes_features = []
    #         #final_selected_ex_boxes = []
    #         final_computed_scores = []
    #         #pdb.set_trace()
    #         #for i in range(0, iou_scores.size(0)):
    #         for i in range(0, v_l_scores.size(0)):            
    #             j_selected_pred_boxes = []
    #             j_selected_box_features = []
    #             #j_selected_ex_boxes = []
    #             j_computed_scores = []
    #             #score, idx = torch.sort(iou_scores[i], descending=True)
    #             score, idx = torch.sort(v_l_scores[i], descending=True)
    #             #print(score[0:10])
    #             #pdb.set_trace()
    #             for j in range(0, selected_num):             
    #                 j_selected_pred_boxes.append(bboxes[i][idx[j]])
    #                 j_selected_box_features.append(features[i][idx[j]])
    #                 #j_selected_ex_boxes.append(ex_bboxes[i][idx[j]])
    #                 j_computed_scores.append(score[idx[j]])
                         
    #             final_selected_pred_boxes.append(torch.stack(j_selected_pred_boxes))
    #             final_selected_boxes_features.append(torch.stack(j_selected_box_features))
    #             #final_selected_ex_boxes.append(torch.stack(j_selected_ex_boxes))
    #             final_computed_scores.append(torch.stack(j_computed_scores))
    #         #pdb.set_trace()     
    #         final_selected_pred_boxes = torch.stack(final_selected_pred_boxes)
    #         final_selected_boxes_features = torch.stack(final_selected_boxes_features)
    #         #final_selected_ex_boxes = torch.stack(final_selected_ex_boxes)
    #         final_computed_scores = torch.stack(final_computed_scores)
    #         return final_selected_pred_boxes, final_selected_boxes_features, final_computed_scores
    #         #return final_selected_ex_boxes, final_selected_pred_boxes, final_selected_boxes_features, final_computed_scores
    #     else:
    #         #pdb.set_trace()
    #         final_selected_pred_boxes = []
    #         final_selected_boxes_features = []
    #         final_selected_ex_boxes = []
    #         final_selected_next_boxes = []
    #         for i in range(0, v_l_scores.size(0)):            
    #             j_selected_pred_boxes = []
    #             j_selected_box_features = []
    #             j_selected_ex_boxes = []
    #             j_selected_next_boxes = []
    #             #score, idx = torch.sort(iou_scores[i], descending=True)
    #             score, idx = torch.sort(v_l_scores[i], descending=True)
    #             #print(score[0:10])
    #             #pdb.set_trace()
    #             for j in range(0, selected_num):             
    #                 j_selected_pred_boxes.append(bboxes[i][idx[j]])
    #                 j_selected_box_features.append(features[i][idx[j]])
    #                 j_selected_ex_boxes.append(ex_bboxes[i][idx[j]])
    #                 j_selected_next_boxes.append(ex_bboxes[i][idx[j]])
            
    #             final_selected_pred_boxes.append(torch.stack(j_selected_pred_boxes))
    #             final_selected_boxes_features.append(torch.stack(j_selected_box_features))
    #             final_selected_ex_boxes.append(torch.stack(j_selected_ex_boxes))
    #             final_selected_next_boxes.append(torch.stack(j_selected_next_boxes))
    #         #pdb.set_trace()     
    #         final_selected_pred_boxes = torch.stack(final_selected_pred_boxes)
    #         final_selected_boxes_features = torch.stack(final_selected_boxes_features)
    #         final_selected_ex_boxes = torch.stack(final_selected_ex_boxes)
    #         final_selected_next_boxes = torch.stack(final_selected_next_boxes)
    #         return final_selected_ex_boxes, final_selected_pred_boxes, final_selected_next_boxes, final_selected_boxes_features


    def box_feature_selction(self, bboxes, features, sentence_features, Threshold, next_boxes = None):
        #pdb.set_trace()
        num = bboxes.size(1)
        sentence_features_expand = sentence_features.unsqueeze(1).expand(sentence_features.size(0), num, sentence_features.size(1))
        v_l_scores = self.l2(self.Tanh(self.l1(sentence_features_expand) + features)).squeeze(2)
        #v_l_weight = F.softmax(v_l_score, dim = 1)
        #v_l_weight_sum = torch.sum(v_l_weight, dim=1).unsqueeze(1).expand(v_l_weight.size(0), v_l_weight.size(1))
        #v_l_weight[v_l_weight != 0] = v_l_weight[v_l_weight_sum != 0] / v_l_weight_sum[v_l_weight_sum != 0]
        selected_num = int(num * Threshold)
        final_selected_pred_boxes = []
        final_selected_boxes_features = []
        final_computed_scores = []
        for i in range(0, v_l_scores.size(0)):            
            j_selected_pred_boxes = []
            j_selected_box_features = []
            j_computed_scores = []
            score, idx = torch.sort(v_l_scores[i], descending=True)
            for j in range(0, selected_num):             
                j_selected_pred_boxes.append(bboxes[i][idx[j]])
                j_selected_box_features.append(features[i][idx[j]])
                j_computed_scores.append(score[idx[j]])
                        
            final_selected_pred_boxes.append(torch.stack(j_selected_pred_boxes))
            final_selected_boxes_features.append(torch.stack(j_selected_box_features))
            final_computed_scores.append(torch.stack(j_computed_scores))
        #pdb.set_trace()     
        final_selected_pred_boxes = torch.stack(final_selected_pred_boxes)
        final_selected_boxes_features = torch.stack(final_selected_boxes_features)
        final_computed_scores = torch.stack(final_computed_scores)
        return final_selected_pred_boxes, final_selected_boxes_features, final_computed_scores
           
 

    #ddim module
    def model_predictions(self, backbone_feats, images_whwh, x, vg_hs, t, x_self_cond=None, clip_x_start=False):  
        x_boxes = torch.clamp(x, min=-1 * self.scale, max=self.scale)
        x_boxes = ((x_boxes / self.scale) + 1) / 2
        x_boxes = xywh2xyxy(x_boxes)     
        x_boxes = x_boxes * images_whwh[:, None, :]
        pred_bboxes, bboxes_features = self.head(backbone_feats, vg_hs, x_boxes, t, None)
        x_start = pred_bboxes[pred_bboxes.size(0)- 1] / images_whwh[:, None, :]
        x_start = xyxy2xywh(x_start)
        x_start = (x_start * 2 - 1.) * self.scale
        x_start = torch.clamp(x_start, min=-1 * self.scale, max=self.scale)
        pred_noise = self.predict_noise_from_start(x, t, x_start)

        return pred_noise, x_start, pred_bboxes[pred_bboxes.size(0)- 1], bboxes_features[bboxes_features.size(0)- 1]

    def ddim_sample_vision(self, backbone_feats, images_whwh, vg_hs, sentence_fea, device, clip_denoised=True, do_postprocess=True):
        batch = images_whwh.shape[0]
        shape = (batch, self.num_proposals, 4)
        total_timesteps, sampling_timesteps, eta = self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        
        img = torch.randn(shape, device=device)
        #ensemble_score, ensemble_label, ensemble_coord = [], [], []  #maybe need to delete
        x_start = None
        count = 0
        final_predicted_boexs = []
        final_boxes_features = []
        #diffused_boxes = []
        #diffused_boxes.append()
        for time, time_next in time_pairs:
            #pdb.set_trace()
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None
            pred_box_noise, pred_next_boxes, pred_boxes, pred_features = self.model_predictions(backbone_feats, images_whwh, 
                                                                        img, vg_hs, time_cond,
                                                                        self_cond, clip_x_start=clip_denoised)
            
            #pdb.set_trace()
            box_noise, selected_pred_boxes, next_boxes, selected_boxes_features = self.box_feature_selction(pred_box_noise, pred_boxes,
                                                                                pred_features, 
                                                                                sentence_fea, self.Threshold, pred_next_boxes)
            #pdb.set_trace()
            pred_noise, x_start = box_noise, next_boxes
            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            noise = torch.randn_like(x_start)
            #pdb.set_trace()
            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

            final_predicted_boexs.append(selected_pred_boxes)
            final_boxes_features.append(selected_boxes_features)
            count = count + 1
            if count == 5:
                break

        return final_predicted_boexs, final_boxes_features


    def forward(self, img_data, text_data, word_selection, 
                text_info_data, gt_boxes, device, extra:dict):
        
        bs = img_data.tensors.shape[0]
        training_states=extra.get('training_states',None)
        # language bert
        #text_fea, text_fea_s, sentence_fea, sentence_fea_s = self.textmodel(text_info_data, text_info_data_s, text_data, text_data_s, device)
        #pdb.set_trace()
        text_fea, sentence_fea, valid_sentence_fea = self.textmodel(text_info_data, text_data, word_selection, device)      
        #pdb.set_trace()   
        text_src_orig, text_mask = text_fea.decompose()
        #valid word selection
       # word_selection_expand = word_selection.unsqueeze(2).expand(text_src.size(0), text_src.size(1), text_src.size(2))
        #valid_words = text_src * word_selection_expand
        #vaild_info = torch.bmm(self.w1.unsqueeze(1), valid_words).squeeze(1)     
        #original
        assert text_mask is not None
        text_cls = text_src_orig[:,0]
        #pdb.set_trace()
        #text_src = self.text_proj(text_src_orig)
        #pdb.set_trace()
        text_src = self.text_proj(sentence_fea)
        # visual backbone
        #x,out_mask, features_list = self.visumodel(img_data.tensors,img_data.mask,text_cls,extra=extra)
        features_list = self.visumodel(img_data.tensors,img_data.mask,text_cls,extra=extra)
        '''
        x,out_mask, features_list = self.visumodel(img_data.tensors,img_data.mask,text_cls,extra=extra)
        visu_mask=torch.cat(out_mask[-2:],dim=1)
        visu_src=torch.cat(x[-2:],dim=0)
        visu_scale=torch.cat([
            repeat(self.scale_embed.weight[-2],'D -> L B D',B=bs,L=x[-2].shape[0]),
            repeat(self.scale_embed.weight[-1],'D -> L B D',B=bs,L=x[-1].shape[0])
        ],dim=0)
        visu_src = self.visu_proj(visu_src+visu_scale) # (N*B)xC

        # permute BxLenxC to LenxBxC
        text_src = text_src.permute(1, 0, 2)
        text_mask = text_mask.flatten(1)

        # target regression token
        tgt_src = self.reg_token.weight.unsqueeze(1).repeat(1, bs, 1)
        tgt_mask = torch.zeros((bs, 1)).to(tgt_src.device).to(torch.bool)
        
        vl_src = torch.cat([tgt_src, text_src, visu_src], dim=0)
        vl_mask = torch.cat([tgt_mask, text_mask, visu_mask], dim=1)
        vl_pos = self.vl_pos_embed.weight.unsqueeze(1).repeat(1, bs, 1)

        vg_hs = self.vl_transformer(vl_src, vl_mask, vl_pos) # (1+L+N)xBxC
        vg_hs = vg_hs[0]
        '''
        #pseudo diffusion REC
        images_whwh = torch.ones((bs, 4)).to(device) * 640
        if training_states != None:    
            x_boxes = self.prepare_diffusion_concat(gt_boxes, device)
            x_boxes = x_boxes * images_whwh[:, None, :]
            process_boxes = []
            #time_steps = 5
            total_timesteps, sampling_timesteps, eta = self.num_timesteps, 1000, self.ddim_sampling_eta
            times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
            times = list(reversed(times.int().tolist()))
            time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
            count = 0
            process_boxes.append(x_boxes)
            saved_boxes = []
            saved_features = []
            computed_scores = []
            for time, time_next in time_pairs:
                #pdb.set_trace()
                alpha = self.alphas_cumprod[time]
                alpha_next = self.alphas_cumprod[time_next]
                sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
                c = (1 - alpha_next - sigma ** 2).sqrt()
                noise = torch.randn_like(sentence_fea)
                sentence_fea = sentence_fea * alpha_next.sqrt() + c * self.query + sigma * noise
                #sentence_fea = sentence_fea + self.query


                time_cond = torch.full((x_boxes.size(0),), time, device=device, dtype=torch.long)
                #pred_bboxes, bboxes_features = self.head(features_list, vg_hs, process_boxes[count], time_cond, None)
                pred_bboxes, bboxes_features = self.head(features_list, text_src, process_boxes[count], time_cond, None)
                selected_boxes, selected_features, scores = self.box_feature_selction(pred_bboxes[pred_bboxes.size(0)-1],
                                                                               bboxes_features[bboxes_features.size(0)-1], 
                                                                               sentence_fea, self.Threshold)
                
                saved_boxes.append(selected_boxes)
                process_boxes.append(selected_boxes)
                saved_features.append(selected_features)
                computed_scores.append(scores)
                count = count + 1
                if count == 5:
                    break
            
            norm_pred_boxes = saved_boxes[len(saved_boxes) - 1] / images_whwh[:, None, :]
            #language visual matching
             #pdb.set_trace()
             #proj_diffused_sentences = self.language_proj(diffused_sentences)
             #proj_sentences = F.normalize(proj_diffused_sentences, p=2, dim=1)
            '''
            proj_sentences_expand = proj_sentences.unsqueeze(1).expand(proj_sentences.size(0), saved_features[len(saved_features) - 1].size(1),
                                                                        proj_sentences.size(1))
            '''
            #visual_emb_normalized = F.normalize(saved_features[len(saved_features) - 1], p=2, dim=2)
            #vl_similarity = F.cosine_similarity(proj_sentences_expand, visual_emb_normalized, dim=2)
            #pdb.set_trace()
            return xyxy2xywh(norm_pred_boxes).sigmoid(), computed_scores[len(computed_scores) - 1], sentence_fea, valid_sentence_fea
        else:
            #pdb.set_trace()
            shape = (bs, self.num_proposals, 4)
            img = torch.randn(shape, device=device)
            x_boxes = torch.clamp(img, min=-1 * self.scale, max=self.scale)
            x_boxes = ((x_boxes / self.scale) + 1) / 2
            x_boxes = xywh2xyxy(x_boxes)     
            x_boxes = x_boxes * images_whwh[:, None, :]
            saved_boxes = []
            saved_features = []
            computed_scores = []
            process_boxes = []
            process_boxes.append(x_boxes)
            count = 0 
            total_timesteps, sampling_timesteps, eta = self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta
            times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
            times = list(reversed(times.int().tolist()))
            time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
            for time, time_next in time_pairs:
                #pdb.set_trace()
                alpha = self.alphas_cumprod[time]
                alpha_next = self.alphas_cumprod[time_next]
                sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
                c = (1 - alpha_next - sigma ** 2).sqrt()
                noise = torch.randn_like(sentence_fea)
                sentence_fea = sentence_fea * alpha_next.sqrt() + c * self.query + sigma * noise
                time_cond = torch.full((bs,), time, device=device, dtype=torch.long)
                pred_bboxes, bboxes_features = self.head(features_list, vg_hs, process_boxes[count], time_cond, None)
                selected_boxes, selected_features, scores = self.box_feature_selction(pred_bboxes[pred_bboxes.size(0)-1],
                                                                               bboxes_features[bboxes_features.size(0)-1], 
                                                                               sentence_fea, self.Threshold)

                saved_boxes.append(selected_boxes)
                process_boxes.append(selected_boxes)
                saved_features.append(selected_features)
                computed_scores.append(scores)
                count = count + 1
                if count == 5:
                    break

                norm_pred_boxes = saved_boxes[len(saved_boxes) - 1] / images_whwh[:, None, :]
            #pdb.set_trace()
            return xyxy2xywh(norm_pred_boxes).sigmoid()




                                                

        # #diffusion REC     
        # images_whwh = torch.ones((bs, 4)).to(device) * 640
        # x_boxes, noise, t = self.prepare_diffusion_concat(gt_boxes, device)
        # noise_generated_boxes = []
        # noise_generated_boxes.append(x_boxes.clone())
        # #training
        # if training_states != None:
        #     #vision process 
        #     x_boxes = x_boxes * images_whwh[:, None, :]
        #     #gt_boxes_xyxy = xywh2xyxy(gt_boxes.unsqueeze(1))
        #     #pdb.set_trace()
        #     #gt_boxes_xyxy = gt_boxes_xyxy * images_whwh[:, None, :]
        #     #t = t.squeeze(-1)
        #     #time_sample = []
        #     x_start = None
        #     #eta = self.num_timesteps
        #     '''
        #     for i in range(0, len(t)):
        #         #pdb.set_trace()
        #         times = torch.linspace(-1, t[i] - 1, steps=self.sampling_timesteps + 1)
        #         time_sample.append(torch.Tensor(list(reversed(times.int().tolist()))))
        #     schedule = torch.stack(time_sample).type(torch.long).to(device)
        #     '''
        #     saved_boxes = []
        #     process_boxes = []
        #     saved_features = []
        #     process_boxes.append(x_boxes)
        #     computed_scores = []
        #     #saved_boxes.append(x_boxes)
        #     #saved_features.append(None)
            
        #     total_timesteps, sampling_timesteps, eta = self.num_timesteps, 1000, self.ddim_sampling_eta
        #     times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        #     times = list(reversed(times.int().tolist()))
        #     time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        #     count = 0
        #     for time, time_next in time_pairs:
        #         #print(time, time_next)
        #         time_cond = torch.full((x_boxes.size(0),), time, device=device, dtype=torch.long)      
        #         pred_bboxes, bboxes_features = self.head(features_list, vg_hs, process_boxes[count], time_cond, None)
        #         '''
        #         x_ex, selected_boxes, selected_features = self.box_feature_selction(noise_generated_boxes[count], pred_bboxes[pred_bboxes.size(0)-1],
        #                                                                       bboxes_features[bboxes_features.size(0)-1], 
        #                                                                       gt_boxes_xyxy.squeeze(1), self.Threshold)
        #         '''
        #         x_ex, selected_boxes, selected_features, scores = self.box_feature_selction(noise_generated_boxes[count], pred_bboxes[pred_bboxes.size(0)-1],
        #                                                                       bboxes_features[bboxes_features.size(0)-1], 
        #                                                                       sentence_fea, self.Threshold)
                
        #         #print(selected_boxes.size())
        #         x_start = selected_boxes / images_whwh[:, None, :]
        #         x_start = xyxy2xywh(x_start)
        #         x_ex = xyxy2xywh(x_ex)
        #         x_start = (x_start * 2 - 1.) * self.scale
        #         x_start = torch.clamp(x_start, min=-1 * self.scale, max=self.scale)
        #         pred_noise = self.predict_noise_from_start(x_ex, time_cond, x_start)
        #         noise_generated_boxes.append(x_ex)
        #         alpha = self.alphas_cumprod[time]
        #         alpha_next = self.alphas_cumprod[time_next]
        #         sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
        #         c = (1 - alpha_next - sigma ** 2).sqrt()
        #         noise = torch.randn_like(x_start)
        #         next_box_norm_xywh = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise
        #         next_box_norm_xyxy = xywh2xyxy(next_box_norm_xywh)
        #         process_boxes.append(next_box_norm_xyxy * images_whwh[:, None, :])
        #         saved_boxes.append(selected_boxes)
        #         saved_features.append(selected_features)
        #         computed_scores.append(scores)
        #         count = count + 1
        #         if count == 5:
        #             break
            
        #     #language_process
        #     new_bs, loss_weights = self.schedule_sampler(bs)
        #     language_loss = self.diffusion.training_losses_seq2seq(self.textmodel, new_bs, text_src_orig)
        #     language_loss = (language_loss * loss_weights).mean()
        #     #init_words = self.textmodel.language_trans(text_src_orig, new_bs)
        #     #diffused_sentences = torch.bmm(self.w1.unsqueeze(1), diffused_words).squeeze(1)
        #     norm_pred_boxes = saved_boxes[len(saved_boxes) - 1] / images_whwh[:, None, :]
        #     #language visual matching
        #     #pdb.set_trace()
        #     #proj_diffused_sentences = self.language_proj(diffused_sentences)
        #     #proj_sentences = F.normalize(proj_diffused_sentences, p=2, dim=1)
        #     '''
        #     proj_sentences_expand = proj_sentences.unsqueeze(1).expand(proj_sentences.size(0), saved_features[len(saved_features) - 1].size(1),
        #                                                                proj_sentences.size(1))
        #     '''
        #     #visual_emb_normalized = F.normalize(saved_features[len(saved_features) - 1], p=2, dim=2)
        #     #vl_similarity = F.cosine_similarity(proj_sentences_expand, visual_emb_normalized, dim=2)
        #     #pdb.set_trace()
        #     return xyxy2xywh(norm_pred_boxes).sigmoid(), computed_scores[len(computed_scores) - 1], language_loss




        #     # for time in range(0, schedule.size(1)):
        #     #     saved_boxes.append(selected_boxes)
        #     #     saved_features.append(selected_features)     
        # else:
        #     #pdb.set_trace()
        #     diffused_boxes, diffused_boxes_features = self.ddim_sample_vision(features_list, images_whwh, vg_hs, sentence_fea, device)
        #     #diffusion_language
        #     text_x_start = text_src_orig



        #     new_bs, loss_weights = self.schedule_sampler(bs)
        #     language_loss = self.diffusion.training_losses_seq2seq(self.textmodel, new_bs, text_src_orig)
        #     language_loss = (language_loss * loss_weights).mean()

        #     diffused_sentences = torch.bmm(self.w1.unsqueeze(1), diffused_words).squeeze(1)
        #     norm_pred_boxes = diffused_boxes[len(diffused_boxes) - 1] / images_whwh[:, None, :]
        #     #language visual matching
        #     #pdb.set_trace()
        #     proj_diffused_sentences = self.language_proj(diffused_sentences)
        #     proj_sentences = F.normalize(proj_diffused_sentences, p=2, dim=1)
        #     proj_sentences_expand = proj_sentences.unsqueeze(1).expand(proj_sentences.size(0),
        #                                                                diffused_boxes_features[len(diffused_boxes_features) - 1].size(1),
        #                                                                proj_sentences.size(1))
        #     visual_emb_normalized = F.normalize(diffused_boxes_features[len(diffused_boxes_features) - 1], p=2, dim=2)
        #     vl_similarity = F.cosine_similarity(proj_sentences_expand, visual_emb_normalized, dim=2)
        #     return xyxy2xywh(norm_pred_boxes).sigmoid(), vl_similarity, language_loss


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
