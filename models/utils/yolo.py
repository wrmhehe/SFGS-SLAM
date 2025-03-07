# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
YOLO-specific modules

Usage:
    $ python path/to/models/yolo.py --cfg yolov5s.yaml
"""

import argparse
import sys
from copy import deepcopy
from pathlib import Path
import torch
import torch.nn as nn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
# ROOT = ROOT.relative_to(Path.cwd())  # relative

from models.utils.common import *
# from models.experimental import *
# from utils.autoanchor import check_anchor_order
from models.utils.general_yolo import LOGGER, check_version, check_yaml, make_divisible, print_args
# from utils.plots import feature_visualization
# from utils.torch_utils import (copy_attr, fuse_conv_and_bn, initialize_weights, model_info, scale_img, select_device,
#                                time_sync)
import contextlib
try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None


# class Detect(nn.Module):
#     stride = None  # strides computed during build
#
#     def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
#         super().__init__()
#         self.nc = nc  # number of classes
#         self.no = nc + 5  # number of outputs per anchor
#         self.nl = len(anchors)  # number of detection layers
#         self.na = len(anchors[0]) // 2  # number of anchors
#         self.grid = [torch.zeros(1)] * self.nl  # init grid
#         self.anchor_grid = [torch.zeros(1)] * self.nl  # init anchor grid
#         self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
#         self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
#         self.inplace = inplace  # use in-place ops (e.g. slice assignment)
#
#
#     def forward(self, x):
#         z = []  # inference output
#         # for item in x:
#         #     print(item.shape)
#         # torch.Size([2, 128, 80, 80])
#         # torch.Size([2, 256, 40, 40])
#         # torch.Size([2, 256, 20, 20])
#
#         for i in range(self.nl):
#             x[i] = self.m[i](x[i])  # conv
#             bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
#             x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
#
#             if not self.training:  # inference
#                 if self.grid[i].shape[2:4] != x[i].shape[2:4]:
#                     self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)
#
#                 y = x[i].sigmoid()
#                 if self.inplace:
#                     y[..., 0:2] = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
#                     y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
#                 else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
#                     xy = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
#                     wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
#                     y = torch.cat((xy, wh, y[..., 4:]), -1)
#                 z.append(y.view(bs, -1, self.no))
#
#
#
#         # print(self.stride)
#         # for item in x:
#         #     print(item.shape)
#         # if not self.training:
#         #     print(torch.cat(z, 1).shape)
#
#         # torch.Size([4, 3, 20, 20, 85])
#         # torch.Size([4, 3, 80, 80, 85])
#         # torch.Size([4, 3, 40, 40, 85])
#         # torch.Size([4, 3, 20, 20, 85])
#
#
#         # torch.Size([4, 3, 80, 80, 85])
#         # torch.Size([4, 3, 40, 40, 85])                                                                                                                                                         | 93/1250 [00:08<01:19, 14.56it/s]
#         # torch.Size([4, 3, 20, 20, 85])
#         # torch.Size([4, 25200, 85])
#         # torch.Size([4, 3, 80, 80, 85])
#         # torch.Size([4, 3, 40, 40, 85])
#         # torch.Size([4, 3, 20, 20, 85])
#         # torch.Size([4, 25200, 85])
#
#         # if not self.training:
#         #     print(torch.cat(z, 1).shape)
#         # torch.Size([2, 25200, 85])
#         # torch.Size([2, 25200, 85])
#         return x if self.training else (torch.cat(z, 1), x)
#
#     def _make_grid(self, nx=20, ny=20, i=0):
#         d = self.anchors[i].device
#         if check_version(torch.__version__, '1.10.0'):  # torch>=1.10.0 meshgrid workaround for torch>=0.7 compatibility
#             yv, xv = torch.meshgrid([torch.arange(ny).to(d), torch.arange(nx).to(d)], indexing='ij')
#         else:
#             yv, xv = torch.meshgrid([torch.arange(ny).to(d), torch.arange(nx).to(d)])
#         grid = torch.stack((xv, yv), 2).expand((1, self.na, ny, nx, 2)).float()
#         print(self.anchors[i])
#         print(self.stride[i])
#         anchor_grid = (self.anchors[i].clone() * self.stride[i]) \
#             .view((1, self.na, 1, 1, 2)).expand((1, self.na, ny, nx, 2)).float()
#         return grid, anchor_grid
#
#     def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
#         # https://arxiv.org/abs/1708.02002 section 3.3
#         # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
#         m = self.model[-1]  # Detect() module
#         for mi, s in zip(m.m, m.stride):  # from
#             b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
#             b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
#             b.data[:, 5:] += math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # cls
#             mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)








# TODO: adapt to YOLOPoint Framework

class Detect(nn.Module):
    stride = None  # strides computed during build

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
        # super().__init__()
        # self.nc = nc  # number of classes
        # self.no = nc + 5  # number of outputs per anchor
        # self.nl = len(anchors)  # number of detection layers
        # self.na = len(anchors[0]) // 2  # number of anchors
        # self.grid = [torch.zeros(1)] * self.nl  # init grid
        # self.anchor_grid = [torch.zeros(1)] * self.nl  # init anchor grid
        # self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        # self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        # self.inplace = inplace  # use in-place ops (e.g. slice assignment)


        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.empty(0) for _ in range(self.nl)]  # init grid
        self.anchor_grid = [torch.empty(0) for _ in range(self.nl)]  # init anchor grid
        self.register_buffer("anchors", torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.inplace = inplace  # use inplace ops (e.g. slice assignment)

    def forward(self, x):
        z = []  # inference output
        # for item in x:
        #     print(item.shape)
        # torch.Size([2, 128, 80, 80])
        # torch.Size([2, 256, 40, 40])
        # torch.Size([2, 256, 20, 20])

        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                # if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                #     self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)
                #
                # y = x[i].sigmoid()
                # if self.inplace:
                #     y[..., 0:2] = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                #     y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                # else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                #     xy = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                #     wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                #     y = torch.cat((xy, wh, y[..., 4:]), -1)
                # z.append(y.view(bs, -1, self.no))

        # print(self.no,self.na)
        # for item in x:
        #     print(item.shape)
        # 117 3
        # torch.Size([2, 3, 80, 80, 117])
        # torch.Size([2, 3, 40, 40, 117])
        # torch.Size([2, 3, 20, 20, 117])

                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)
                if isinstance(self, Segment):  # (boxes + masks)
                    xy, wh, conf, mask = x[i].split((2, 2, self.nc + 1, self.no - self.nc - 5), 4)
                    # print(xy.shape,self.grid[i].shape,self.stride[i])
                    # torch.Size([2, 3, 80, 80, 2]) torch.Size([1, 3, 80, 80, 2]) tensor(8., device='cuda:0')
                    xy = (xy.sigmoid() * 2 + self.grid[i]) * self.stride[i]  # xy
                    wh = (wh.sigmoid() * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, conf.sigmoid(), mask), 4)
                else:  # Detect (boxes only)
                    xy, wh, conf = x[i].sigmoid().split((2, 2, self.nc + 1), 4)
                    xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
                    wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, self.na * nx * ny, self.no))
        # print(self.stride)

        return x if self.training else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0, torch_1_10=check_version(torch.__version__, "1.10.0")):
        """Generates a mesh grid for anchor boxes with optional compatibility for torch versions < 1.10."""
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2  # grid shape
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        yv, xv = torch.meshgrid(y, x, indexing="ij") if torch_1_10 else torch.meshgrid(y, x)  # torch>=0.7 compatibility
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5

        # print(self.anchors.shape)
        # print(self.stride.shape)
        # torch.Size([3, 3, 2])
        # torch.Size([3])                                                                                                                                                                                  | 0/625 [00:00<?, ?it/s]
        # torch.Size([3, 3, 2])
        # torch.Size([3])

        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        return grid, anchor_grid

    # def forward(self, x):
    #     z = []  # inference output
    #     for i in range(self.nl):
    #         x[i] = self.m[i](x[i])  # conv
    #         bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
    #         x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
    #
    #         if not self.training:  # inference
    #             if self.grid[i].shape[2:4] != x[i].shape[2:4]:
    #                 self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)
    #
    #             y = x[i].sigmoid()
    #             if self.inplace:
    #                 y[..., 0:2] = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
    #                 y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
    #             else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
    #                 xy = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
    #                 wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
    #                 y = torch.cat((xy, wh, y[..., 4:]), -1)
    #             z.append(y.view(bs, -1, self.no))
    #
    #     return x if self.training else (torch.cat(z, 1), x)

    # def _make_grid(self, nx=20, ny=20, i=0):
    #     d = self.anchors[i].device
    #     if check_version(torch.__version__, '1.10.0'):  # torch>=1.10.0 meshgrid workaround for torch>=0.7 compatibility
    #         yv, xv = torch.meshgrid([torch.arange(ny).to(d), torch.arange(nx).to(d)], indexing='ij')
    #     else:
    #         yv, xv = torch.meshgrid([torch.arange(ny).to(d), torch.arange(nx).to(d)])
    #     grid = torch.stack((xv, yv), 2).expand((1, self.na, ny, nx, 2)).float()
    #     anchor_grid = (self.anchors[i].clone() * self.stride[i]) \
    #         .view((1, self.na, 1, 1, 2)).expand((1, self.na, ny, nx, 2)).float()
    #     print(self.anchors.shape,self.stride.shape)
    #     return grid, anchor_grid

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)


class Segment(Detect):
    """YOLOv5 Segment head for segmentation models, extending Detect with mask and prototype layers."""

    def __init__(self, nc=80, anchors=(), nm=32, npr=256, ch=(), inplace=True):
        """Initializes YOLOv5 Segment head with options for mask count, protos, and channel adjustments."""
        super().__init__(nc, anchors, ch, inplace)
        self.nm = nm  # number of masks
        self.npr = npr  # number of protos
        self.no = 5 + nc + self.nm  # number of outputs per anchor
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.proto = Proto(ch[0], self.npr, self.nm)  # protos
        self.detect = Detect.forward

    def forward(self, x):
        """Processes input through the network, returning detections and prototypes; adjusts output based on
        training/export mode.
        """
        p = self.proto(x[0])
        x = self.detect(self, x)

        # if not self.training:
        #     for item in x:
        #         print(item.shape)
        #         torch.Size([2, 25200, 117])


        # print(p.shape)

        # torch.Size([2, 3, 80, 80, 117])
        # torch.Size([2, 3, 40, 40, 117])
        # torch.Size([2, 3, 20, 20, 117])
        # torch.Size([2, 32, 160, 160])
        return (x, p) #if self.training else (x[0], p, x[1])  #(x[0], p) if self.export else (x[0], p, x[1])





class Detectv8(nn.Module):
    """YOLOv8 Detect head for detection models."""
    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, ch=()):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], self.nc)  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        shape = x[0].shape  # BCHW
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:
            return x
        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.export and self.format in ('saved_model', 'pb', 'tflite', 'edgetpu', 'tfjs'):  # avoid TF FlexSplitV ops
            box = x_cat[:, :self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4:]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)

class BaseModel(nn.Module):
    # YOLOv5 base model
    def forward(self, x, profile=False, visualize=False):
        return self._forward_once(x, profile, visualize)  # single-scale inference, train

    def _forward_once(self, x, profile=False, visualize=False):
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
            # if visualize:
            #     feature_visualization(x, m.type, m.i, save_dir=visualize)
        return x

    def _profile_one_layer(self, m, x, dt):
        c = m == self.model[-1]  # is final layer, copy input as inplace fix
        o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPs
        # t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        # dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
        LOGGER.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    # def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
    #     LOGGER.info('Fusing layers... ')
    #     for m in self.model.modules():
    #         if isinstance(m, (Conv, DWConv)) and hasattr(m, 'bn'):
    #             m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
    #             delattr(m, 'bn')  # remove batchnorm
    #             m.forward = m.forward_fuse  # update forward
    #     # self.info()
    #     return self

    # def info(self, verbose=False, img_size=640):  # print model information
    #     model_info(self, verbose, img_size)

    def _apply(self, fn):
        # Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self

class DetectionModel(BaseModel):
    # YOLOv5 detection model
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None):  # model, input channels, number of classes
        super().__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg, encoding='ascii', errors='ignore') as f:
                self.yaml = yaml.safe_load(f)  # model dict

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        if anchors:
            LOGGER.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # override yaml value
        # self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        self.inplace = self.yaml.get('inplace', True)

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            s = 256  # 2x min stride
            m.inplace = self.inplace
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.empty(1, ch, s, s))])  # forward
            # check_anchor_order(m)  # must be in pixel-space (not grid-space)
            m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride
            self._initialize_biases()  # only run once

        # Init weights, biases
        # initialize_weights(self)
        self.info()
        LOGGER.info('')

    def forward(self, x, augment=False, profile=False, visualize=False):
        if augment:
            return self._forward_augment(x)  # augmented inference, None
        return self._forward_once(x, profile, visualize)  # single-scale inference, train


    # def _forward_augment(self, x):
    #     img_size = x.shape[-2:]  # height, width
    #     s = [1, 0.83, 0.67]  # scales
    #     f = [None, 3, None]  # flips (2-ud, 3-lr)
    #     y = []  # outputs
    #     for si, fi in zip(s, f):
    #         xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
    #         yi = self._forward_once(xi)[0]  # forward
    #         # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
    #         yi = self._descale_pred(yi, fi, si, img_size)
    #         y.append(yi)
    #     y = self._clip_augmented(y)  # clip augmented tails
    #     return torch.cat(y, 1), None  # augmented inference, train

    def _descale_pred(self, p, flips, scale, img_size):
        # de-scale predictions following augmented inference (inverse operation)
        if self.inplace:
            p[..., :4] /= scale  # de-scale
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr
        else:
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale  # de-scale
            if flips == 2:
                y = img_size[0] - y  # de-flip ud
            elif flips == 3:
                x = img_size[1] - x  # de-flip lr
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p

    def _clip_augmented(self, y):
        # Clip YOLOv5 augmented inference tails
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4 ** x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[1] // g) * sum(4 ** x for x in range(e))  # indices
        y[0] = y[0][:, :-i]  # large
        i = (y[-1].shape[1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][:, i:]  # small
        return y

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1).detach()  # conv.bias(255) to (3,85)
            b[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b[:, 5:] += math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)




def parse_model(d, ch):  # model_dict, input_channels(3)
    LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            with contextlib.suppress(NameError):
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings

        n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in (Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, Focus,
                 BottleneckCSP, C3, C3TR, C3SPP, C3Ghost, nn.ConvTranspose2d):
            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]
            if m in [BottleneckCSP, C3, C3TR, C3Ghost]:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m is Detect:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        LOGGER.info(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)

