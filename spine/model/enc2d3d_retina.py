import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import timm
from spine.model.blocks import MyUnetDecoder3d, pvtv2_encode
from spine.model.loss import F_zxy_loss, F_grade_loss, F_focal_heatmap_loss, F_JS_heatmap_loss
from spine.model.heatmap import heatmap_to_coord, heatmap_to_grade
from .utils import _sigmoid

class Enc2d3dRetina(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        # self.output_type = ['infer', 'loss']
        self.register_buffer('D', torch.tensor(0))
        self.register_buffer('mean', torch.tensor(0.5))
        self.register_buffer('std', torch.tensor(0.5))

        self.arch = 'resnet18' #'pvt_v2_b0'

        encoder_dim = {
            'pvt_v2_b0': [32, 64, 160, 256],
            'pvt_v2_b1': [64, 128, 320, 512],
            'pvt_v2_b2': [64, 128, 320, 512],
            'pvt_v2_b4': [64, 128, 320, 512],
            'resnet18':  [64, 128, 256, 512],
        }.get(self.arch, [768])

        decoder_dim = [256, 128, 64]
        #decoder_dim = \
              #[ 256, 128, 64]
              # [256, 128, 128]
              #[256, 128, 64]

        self.encoder = timm.create_model(
            model_name=self.arch, pretrained=pretrained, in_chans=3, num_classes=0, global_pool=''
        )
        self.decoder = MyUnetDecoder3d(
            in_channel=encoder_dim[-1],
            skip_channel=encoder_dim[:-1][::-1],
            out_channel=decoder_dim,
        )

        num_point = 5*5
        num_grade = 3
        self.background = nn.Conv3d(decoder_dim[-1], 25, kernel_size=1)
        self.grade_classifier = nn.Linear(decoder_dim[-1], 3)


    def forward(self, batch, output_types=('infer', 'loss')):
        device = self.D.device
        image = batch['image']
        D = batch['D']
        num_image = len(D)

        B, H, W = image.shape
        image = image.reshape(B, 1, H, W)

        # x = image.float() / 255
        x = image.float()
        x = (x - self.mean) / self.std
        x = x.expand(-1, 3, -1, -1)

        #---
        if 'pvt_v2' in self.arch:
            encode = pvtv2_encode(x, self.encoder)
        else:
            encode = self.encoder.forward_intermediates(x, intermediates_only=True)

        #for(i, e) in enumerate(encode):
        #    print(f'encode {i} {e.shape}')

        encode = [ torch.split(e, D.tolist(), 0) for e in encode ]
        # for b in encode:
        #     for i, e in enumerate(b):
        #         print(f'encode {i} {e.shape}')
        #heatmap   = []
        points_heatmap = []
        coords = []
        obj_probs = []
        grades = []
        for i in range(num_image):
            e = [ encode[s][i].transpose(1,0).unsqueeze(0) for s in range(len(encode)) ]
            l, _ = self.decoder(
                feature=e[-1], skip=e[:-1][::-1]
            )

            num_point, num_grade = 5*5,3
            #all = self.heatmap(l).squeeze(0)
            #_, d, h, w = all.shape
            # print(l.shape)
            #all = all.reshape(num_point,num_grade,d,h,w)
            #all = all.flatten(1).softmax(-1).reshape(num_point,num_grade, d, h, w)
            #heatmap.append(all)
            # num_point, 1, d, h, w
            points_i = _sigmoid(self.background(l)).squeeze(0).unsqueeze(1)
            # winner takes all
            img_coords_max, img_coords_i = points_i.flatten(1).max(1)
            img_coords_i = torch.stack(torch.unravel_index(img_coords_i, points_i.shape[-3:]), axis=1)
            # take last 3 dims of l indexed by img_coords_i
            features = l[..., img_coords_i[:, 0], img_coords_i[:, 1], img_coords_i[:,2]]
            features = features.squeeze(0).permute(1, 0)
            img_grades = self.grade_classifier(features).squeeze(0)
            grades.append(img_grades)
            obj_probs.append(img_coords_max)
            coords.append(img_coords_i)
            points_heatmap.append(points_i)

        coords = torch.stack(coords, axis=0)
        obj_probs = torch.stack(obj_probs, axis=0)
        grade = torch.stack(grades, axis=0)
        # print(f"grade shape {grade.shape}")
        # print(f" coords shape {coords.shape}")  
        output = {}
        if 'loss' in output_types:
            bd, _, bh, bw = batch['heatmap'].shape
            hmap = batch['heatmap'].reshape(bd, 25, 3, bh, bw)
            batch_points_heatmap = hmap.max(2, keepdim=False).values

            gt = torch.split(batch['heatmap'], D.tolist(), 0)
            obj_det_mask = []
            for i,gt_i in enumerate(gt):
                gt_i = gt_i.permute(1,0,2,3)
                gt_i = gt_i.reshape(25, 3, *gt_i.shape[-3:])
                coords_i = coords[i]
                gt_obj_point = gt_i[torch.arange(coords_i.shape[0]), :, coords_i[:, 0], coords_i[:, 1], coords_i[:,2]].max(1).values
                obj_det_mask.append(gt_obj_point>0.2)
            obj_det_mask = torch.stack(obj_det_mask, axis=0)

            output['heatmap_loss'] = F_focal_heatmap_loss(points_heatmap, batch_points_heatmap , D,
                                                          batch['coords_mask'], batch['grade'])
            output['grade_loss'] = F_grade_loss(grade,  batch['grade'].to(device), obj_det_mask)
            output['zxy_loss'] = torch.tensor(0.0)
            # output['heatmap_loss'] = F_JS_heatmap_loss(heatmap, batch['heatmap'], D)
            # output['zxy_loss'] = F_zxy_loss(coords, batch['coords'], batch['coords_mask'], heatmap)
            #output['grade_loss'] = F_grade_loss(grade,  batch['grade'].to(device))

            # if False: #turn on dynamic matching in later stage of training
            #     index, valid = do_dynamic_match_truth_05(xy, batch['xy'].to(device))
            #     truth = batch['grade'].to(device)
            #     truth_matched = []
            #     for i in range(num_image):a
            #         truth_matched.append(truth[i][index[i]])
            #     truth_matched = torch.stack(truth_matched)
            #     output['grade_loss'] = F_grade_loss(grade[valid],  truth_matched[valid])
            # else:
            #     output['grade_loss'] = F_grade_loss(grade,  batch['grade'].to(device))

            output['loss'] = output['heatmap_loss'] + output['grade_loss'] + output['zxy_loss']

        if 'infer' in output_types:
            output['heatmap'] = points_heatmap
            output['coords'] = coords
            output['grade'] = grade

        return output