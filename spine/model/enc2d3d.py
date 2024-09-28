import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import timm
from spine.model.blocks import MyUnetDecoder3d, pvtv2_encode
from spine.model.loss import F_zxy_loss, F_grade_loss, F_focal_heatmap_loss
from spine.model.heatmap import heatmap_to_coord, heatmap_to_grade


class Enc2d3d(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        # self.output_type = ['infer', 'loss']
        self.register_buffer('D', torch.tensor(0))
        self.register_buffer('mean', torch.tensor(0.5))
        self.register_buffer('std', torch.tensor(0.5))

        arch = 'pvt_v2_b1'

        encoder_dim = {
            'pvt_v2_b1': [64, 128, 320, 512],
            'pvt_v2_b2': [64, 128, 320, 512],
            'pvt_v2_b4': [64, 128, 320, 512],
        }.get(arch, [768])

        decoder_dim = \
              [256, 128, 128]
              #[256, 128, 64]

        self.encoder = timm.create_model(
            model_name=arch, pretrained=pretrained, in_chans=3, num_classes=0, global_pool=''
        )
        self.decoder = MyUnetDecoder3d(
            in_channel=encoder_dim[-1],
            skip_channel=encoder_dim[:-1:][::-1],
            out_channel=decoder_dim,
        )

        num_point = 5*5
        num_grade = 3
        self.heatmap = nn.Conv3d(decoder_dim[-1], num_grade*num_point, kernel_size=1)


    def forward(self, batch, output_types=('infer', 'loss')):
        device = self.D.device
        image = batch['image'].to(device)
        D = batch['D']
        num_image = len(D)

        B, H, W = image.shape
        image = image.reshape(B, 1, H, W)

        # x = image.float() / 255
        x = image.float()
        x = (x - self.mean) / self.std
        x = x.expand(-1, 3, -1, -1)

        #---
        encode = pvtv2_encode(x, self.encoder)
        encode = [ torch.split(e, D.tolist(), 0) for e in encode ]
        # for b in encode:
        #     for i, e in enumerate(b):
        #         print(f'encode {i} {e.shape}')

        heatmap   = []
        for i in range(num_image):
            e = [ encode[s][i].transpose(1,0).unsqueeze(0) for s in range(4) ]
            l, _ = self.decoder(
                feature=e[-1], skip=e[:-1][::-1]
            )

            num_point, num_grade = 5*5,3
            all = self.heatmap(l).squeeze(0)
            _, d, h, w = all.shape
            all = all.reshape(num_point,num_grade,d,h,w)
            all = all.flatten(1).softmax(-1).reshape(num_point,num_grade, d, h, w)
            heatmap.append(all)

        xy, z = heatmap_to_coord(heatmap)
        grade = heatmap_to_grade(heatmap)
        # print('heatmap[0]',heatmap[0].shape)
        # print('xy',xy.shape)
        # print('z',z.shape)
        # print('grade',grade.shape)
        # for h in heatmap:
        #    print(h.shape)

        # points, grades, d,h,w -> d, points, grades, h, w
        # heatmap = torch.cat([all.permute(2,0,1,3,4) for all in heatmap])
        # heatmap = torch.concatenate(heatmap)
        # print("heatmap shape ",heatmap.shape)
        #print('heatmap',heatmap.shape)

        output = {}
        if 'loss' in output_types:
            output['heatmap_loss'] = F_focal_heatmap_loss(heatmap, batch['heatmap'].to(device), D)
            output['zxy_loss'] = 0
            #output['zxy_loss'] = F_zxy_loss(z, xy, batch['z'].to(device), batch['xy'].to(device))
            #output['grade_loss'] = F_grade_loss(grade,  batch['grade'].to(device))

            if False: #turn on dynamic matching in later stage of training
                index, valid = do_dynamic_match_truth_05(xy, batch['xy'].to(device))
                truth = batch['grade'].to(device)
                truth_matched = []
                for i in range(num_image):
                    truth_matched.append(truth[i][index[i]])
                truth_matched = torch.stack(truth_matched)
                output['grade_loss'] = F_grade_loss(grade[valid],  truth_matched[valid])
            else:
                output['grade_loss'] = F_grade_loss(grade,  batch['grade'].to(device))

            output['loss'] = output['heatmap_loss'] + output['grade_loss'] # + output['zxy_loss']

        if 'infer' in output_types:
            output['heatmap'] = heatmap
            output['z'] = z
            output['xy'] = xy
            output['grade'] = grade

        return output