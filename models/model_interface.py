import inspect
import logging
import cv2
import torch
import importlib
import numpy as np
from torch import Tensor
from torch.nn import functional as F
import torch.optim.lr_scheduler as lrs
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
import utils


# import models
from models.SuperPoint import SuperPointNet
from models.alike import ALNet
from models.d2net import D2Net
from models.Xfeat import XFeat
from models.sfd2 import ResSegNetV2
from models.YOLOPoint import YOLOPointv52,superFeatModelv2_1,superFeatModelv2
# import tasks
from tasks.repeatability import repeatability, plot_repeatability
# from models.d2net import process_multiscale

from export_model import export_model

class MInterface(pl.LightningModule):
    def __init__(self, params) -> None:
        super().__init__()
        self.params = params
        self.matcher = None
        # model choice

        dummy = torch.zeros(1, 3, 512, 512)
        print(params['model_type'])
        if params['model_type'] == 'SuperPoint':
            self.model = SuperPointNet()
            self.model.load_state_dict(torch.load(params['SuperPoint_params']['weight']))
            self.model.eval()
            if params['tooonx'] == True:
                torch.onnx.export(self.model, (dummy,), "/media/wrm/ubuntu_relative/course/cource1/tast_1/superpoint.onnx",
                                  opset_version=16,
                                  input_names=['input'], output_names=['heatmap', 'desc'],
                                  dynamic_axes={'input': {0: 'batch_size'}, 'heatmap': {0: 'batch_size'},
                                                'desc': {0: 'batch_size'}})

                print("Done.!")
                # export_model(self.model, '/media/wrm/ubuntu_relative/course/cource1/tast_1/superpoint')
                # raise Exception("success to convert to onnx")


        elif params['model_type'] == 'ALNet':
            self.model = ALNet(c1=32, c2=64, c3=128, c4=128, dim=128, single_head=False)
            self.model.load_state_dict(torch.load(params['alike_params']['weight']))
            self.model.eval()
            if params['tooonx'] == True:
                torch.onnx.export(self.model, (dummy,), "/media/wrm/ubuntu_relative/course/cource1/tast_1/ALNet.onnx",
                                  opset_version=16,
                                  input_names=['input'], output_names=['heatmap', 'desc'],
                                  dynamic_axes={'input': {0: 'batch_size'}, 'heatmap': {0: 'batch_size'},
                                                'desc': {0: 'batch_size'}})

                print("Done.!")
                # export_model(self.model, '/media/wrm/ubuntu_relative/course/cource1/tast_1/ALNet')



        elif params['model_type'] == 'D2Net':
            self.model = D2Net(model_file=params['d2net_params']['weight']).to("cuda")
            self.model.eval()
            if params['tooonx'] == True:
                torch.onnx.export(self.model, (dummy.to('cuda'),), "/media/wrm/ubuntu_relative/course/cource1/tast_1/D2Net.onnx",
                                  opset_version=16,
                                  input_names=['input'], output_names=['heatmap', 'desc'],
                                  dynamic_axes={'input': {0: 'batch_size'}, 'heatmap': {0: 'batch_size'},
                                                'desc': {0: 'batch_size'}})

                print("Done.!")
                # export_model(self.model, '/media/wrm/ubuntu_relative/course/cource1/tast_1/D2Net')


        elif params['model_type'] == 'xfeat':
            self.model = XFeat()
            self.model.load_state_dict(torch.load(params['xfeat_params']['weight']))
            self.model.eval()
            if params['tooonx'] == True:
                torch.onnx.export(self.model, (dummy,), "/media/wrm/ubuntu_relative/course/cource1/tast_1/xfeat.onnx",
                                  opset_version=16,
                                  input_names=['input'], output_names=['heatmap', 'desc'],
                                  dynamic_axes={'input': {0: 'batch_size'}, 'heatmap': {0: 'batch_size'},
                                                'desc': {0: 'batch_size'}})

                print("Done.!")
                # export_model(self.model, '/media/wrm/ubuntu_relative/course/cource1/tast_1/xfeat')

        elif params['model_type'] == 'superFeatModelv2':
            self.model = superFeatModelv2()
            self.model.load_state_dict(torch.load(params['xfeat_params']['weight']))
            self.model.eval()
            # if params['tooonx'] == True:
            #     torch.onnx.export(self.model, (dummy,), "/media/wrm/ubuntu_relative/course/cource1/tast_1/xfeat.onnx",
            #                       opset_version=16,
            #                       input_names=['input'], output_names=['heatmap', 'desc'],
            #                       dynamic_axes={'input': {0: 'batch_size'}, 'heatmap': {0: 'batch_size'},
            #                                     'desc': {0: 'batch_size'}})
            #
            #     print("Done.!")
                # export_model(self.model, '/media/wrm/ubuntu_relative/course/cource1/tast_1/xfeat')

        elif params['model_type'] == 'ResSegNetV2':
            self.model = ResSegNetV2()
            self.model.load_state_dict(torch.load(params['ResSeg_params']['weight']),strict=False)
            self.model.eval()
            # if params['tooonx'] == True:
            #     torch.onnx.export(self.model, (dummy.to('cuda'),), "/media/wrm/ubuntu_relative/course/cource1/tast_1/D2Net.onnx",
            #                       opset_version=11,
            #                       input_names=['input'], output_names=['heatmap', 'desc'],
            #                       dynamic_axes={'input': {0: 'batch_size'}, 'heatmap': {0: 'batch_size'},
            #                                     'desc': {0: 'batch_size'}})

        elif params['model_type'] == 'YOLOPointv52':
            self.model = YOLOPointv52()
            self.model.load_state_dict(torch.load(params['Yolonet_params']['weight']),strict=False)
            self.model.eval()




        else:
            raise NotImplementedError

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print(f"Total parameters: {total_params}")
        print(f"Trainable parameters: {trainable_params}")

        self.num_feat = None
        self.repeatability = None
        self.rep_mean_err = None
        self.Match_Score0 = None
        self.Match_Score1 = None
        self.accuracy = None
        self.matching_score = None
        self.track_error = None
        self.last_batch = None
        self.fundamental_error = None
        self.fundamental_radio = None
        self.fundamental_num = None
        self.r_est = None
        self.t_est = None




    def on_test_start(self) -> None:
        self.num_feat = []
        self.repeatability = []
        self.rep_mean_err = []
        self.Match_Score0 = []
        self.Match_Score1 = []
        self.accuracy = []
        self.matching_score = []
        self.track_error = []
        self.fundamental_error = []
        self.fundamental_radio = []
        self.fundamental_num = []
        self.r_est = [np.eye(3)]
        self.t_est = [np.zeros([3, 1])]

    def on_test_end(self) -> None:
        self.num_feat = np.mean(self.num_feat)
        self.accuracy = np.mean(self.accuracy)
        self.matching_score = np.mean(self.matching_score)
        print('task: ', self.params['task_type'])
        if self.params['task_type'] == 'repeatability':
            rep = torch.as_tensor(self.repeatability).cpu().numpy()
            plot_repeatability(rep, self.params['repeatability_params']['save_path'])
            rep = np.mean(rep)

            ms = torch.as_tensor(self.matching_score).cpu().numpy()
            ms = np.mean(ms)

            error = torch.as_tensor(self.rep_mean_err).cpu().numpy()
            error = error[~np.isnan(error)]
            plot_repeatability(error, self.params['repeatability_params']['save_path'].replace('.png', '_error.png'))
            error = np.mean(error)

            # Score0 = torch.as_tensor(self.Match_Score0).cpu().numpy()
            # Score1 = torch.as_tensor(self.Match_Score1).cpu().numpy()
            # Score0 = Score0[~np.isnan(Score0)]
            # Score1 = Score1[~np.isnan(Score1)]
            # plot_repeatability(Score0, self.params['repeatability_params']['save_path'].replace('.png', '_Match_Score0.png'))
            # plot_repeatability(Score1, self.params['repeatability_params']['save_path'].replace('.png', '_Match_Score1.png'))
            # Score0 = np.mean(Score0)
            # Score1 = np.mean(Score1)

            print('repeatability', rep, ' rep_mean_err', error, ' matching_score', ms)

    def test_step(self, batch: Tensor, batch_idx: int) -> STEP_OUTPUT:

        warp01_params = {}
        warp10_params = {}
        if 'warp01_params' in batch:
            for k, v in batch['warp01_params'].items():
                warp01_params[k] = v[0]
            for k, v in batch['warp10_params'].items():
                warp10_params[k] = v[0]

        # pairs dataset
        score_map_0 = None
        score_map_1 = None
        desc_map_0 = None
        desc_map_1 = None

        # image pair dataset
        if batch['dataset'][0] == 'HPatches' or \
           batch['dataset'][0] == 'megaDepth' or \
           batch['dataset'][0] == 'image_pair':

            # if self.params['model_type'] == 'D2Net':
            #     result0 = process_multiscale(batch['image0'],self.model)
            #     result1 = process_multiscale(batch['image1'], self.model)
            #     score_map_0 = torch.from_numpy(result0[0])
            #     score_map_1 = torch.from_numpy(result1[0])
            #     print(score_map_0.shape)
            #
            #     # score_map_0 = result0[0].detach()
            #     # score_map_1 = result1[0].detach()
            #     if result0[2] is not None:
            #         # desc_map_0 = result0[2].detach()
            #         # desc_map_1 = result1[2].detach()
            #         desc_map_0 = torch.from_numpy(result0[2])
            #         desc_map_1 = torch.from_numpy(result1[2])
            # else :
            result0 = self.model(batch['image0'])
            result1 = self.model(batch['image1'])
            score_map_0 = result0[0].detach()
            score_map_1 = result1[0].detach()
            if result0[1] is not None:
                desc_map_0 = result0[1].detach()
                desc_map_1 = result1[1].detach()

        # sequence dataset
        last_img = None

        if batch['dataset'][0] == 'Kitti' or batch['dataset'][0] == 'Euroc' or batch['dataset'][0] == 'TartanAir':
            if self.last_batch is None:
                self.last_batch = batch
            result0 = self.model(self.last_batch['image0'])
            result1 = self.model(batch['image0'])
            score_map_0 = result0[0].detach()
            score_map_1 = result1[0].detach()
            if result0[1] is not None:
                desc_map_0 = result0[1].detach()
                desc_map_1 = result1[1].detach()
            last_img = self.last_batch['image0']
            self.last_batch = batch
        # task
        result = None
        if self.params['task_type'] == 'repeatability':
            result = repeatability(batch_idx, batch['image0'], score_map_0,
                                   batch['image1'], score_map_1,
                                   desc_map_0, desc_map_1,
                                   warp01_params, warp10_params, self.params)
            self.num_feat.append(result['num_feat'])
            self.repeatability.append(result['repeatability'])
            self.rep_mean_err.append(result['mean_error'])
            self.matching_score.append(result['matching_score'])
            # self.Match_Score0.append(result['Match_Score0'])
            # self.Match_Score1.append(result['Match_Score1'])
        return result
