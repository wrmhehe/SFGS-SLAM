from models.utils.common import Conv, C3, SPPF, C2f
from models.utils.yolo import Detect, Segment
from torch import nn
import torch
import math
from models.utils.utils import load_model
from models.utils.torch_utils_yolo import fuse_conv_and_bn
from models.utils.general_yolo import make_divisible, LOGGER
from copy import deepcopy
import torch.nn.functional as F

anchors_default = [
    [10, 13, 16, 30, 33, 23],
    [30, 61, 62, 45, 59, 119],
    [116, 90, 156, 198, 373, 326]
    ]

class Model(nn.Module):
    def __init__(self, names=(), model_name='YOLOPoint', version=None, inp_ch=3, anchors=None):
        # model, input channels, number of classes
        super().__init__()
        def _check_anchor_order(m):
            # Check anchor order against stride order for YOLOv5 Detect() module m, and correct if necessary
            a = m.anchors.prod(-1).view(-1)  # anchor area
            da = a[-1] - a[0]  # delta a
            ds = m.stride[-1] - m.stride[0]  # delta s
            if da.sign() != ds.sign():  # same order
                LOGGER.info('Reversing anchor order')
                m.anchors[:] = m.anchors.flip(0)

        anchors = anchors or anchors_default

        # Define model
        nc = len(names) if hasattr(names, '__len__') and len(names) > 0 else 1 # 1 is dummy
        version = version.lower() if isinstance(version , str) else version

        if version == 'n':
            dm, wm = 0.33, 0.25
        elif version == 's':
            dm, wm = 0.33, 0.5
        elif version == 'm':
            dm, wm = 0.67, 0.75
        elif version == 'l':
            dm, wm = 1., 1.
        elif version == 'x':
            dm, wm = 1.33, 1.25
        elif version is None:
            dm, wm = None, None
        else:
            raise Exception(f'Version {version} is not a valid input. Choose one of n, s, m, l, x.')

        self.model = load_model(meta_model=False,
                                width_multiple=wm,
                                depth_multiple=dm,
                                inp_ch=inp_ch,
                                nc=nc,
                                anchors=anchors,
                                model_name=model_name)

        # self.model = YOLOPointMFull(inp_ch=inp_ch, nc=nc, anchors=anchors)

        if hasattr(self.model, 'Detect'):
            # Build strides, anchors
            m = self.model.Detect
            s = 256  # 2x min stride

            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, inp_ch, s, s))['objects']])  # forward
            # print('aaaaaaaaaaaaaaaaaaaajmjjjjjjjjjjjjjjjczxnjjasndajanwjndjwadwa')
            m.anchors /= m.stride.view(-1, 1, 1)
            _check_anchor_order(m)
            self._initialize_biases()

    def forward(self, x):
        return self.model(x)

    def _apply(self, fn):
        # Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers
        self = super()._apply(fn)
        if hasattr(self.model, 'Detect'):
            m = self.model.Detect  # Detect()
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        for m in self.model.modules():
            if isinstance(m, Conv) and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
        return self

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model.Detect
        for mi, s in zip(m.m, m.stride):
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def load_state_dict(self, target_state_dict, strict=True, verbose=False):
        # loads the state_dict; reinitializes Detect layer if class number has changed
        key = 'model.Detect.m.0.bias' if 'model.Detect.m.0.bias' in target_state_dict else 'Detect.m.0.bias'
        if key in target_state_dict:
            target_detect_shape = target_state_dict[key].shape
            current_detect_shape = self.state_dict()[key].shape
            if target_detect_shape == current_detect_shape:
                super().load_state_dict(target_state_dict, strict)
            else:
                if verbose:
                    LOGGER.info("Number of classes have changed. Reinitializing Detect layer.\n")
                self.load_partial_state_dict(target_state_dict, strict, verbose)
        else:   # no object inference
            try:
                self.model.load_state_dict(target_state_dict, strict=strict)
            except RuntimeError as e:
                super().load_state_dict(target_state_dict, strict=strict)
                # print(e)
                # self.load_partial_state_dict(target_state_dict, strict=strict)

    def load_partial_state_dict(self, target_state_dict, strict=True, verbose=False):
        # loads all params up to Detect
        if verbose:
            LOGGER.info('Loading state_dict')
        current_state_dict = self.state_dict()
        state_dict_new = deepcopy(current_state_dict)
        for keys in zip(current_state_dict, target_state_dict):
            layer_this = '.'.join(keys[0].split('.')[-2:])
            layer_new = '.'.join(keys[1].split('.')[-2:])
            if layer_this == layer_new and current_state_dict[keys[0]].shape == target_state_dict[keys[1]].shape:
                if verbose:
                    LOGGER.info(f"{keys[0]} {' ' * (50 - len(keys[0]))} {keys[1]}")
                state_dict_new[keys[1]] = target_state_dict[keys[0]]
        super().load_state_dict(state_dict_new, strict)

    def freeze_layers(self, to_freeze, verbose=True):
        if verbose:
            LOGGER.info("Freezing weights...")
        for i, (name, param) in enumerate(self.named_parameters()):
            freeze = '--> freeze' if (i in to_freeze and hasattr(param, 'requires_grad')) else ''
            if verbose:
                LOGGER.info(f"{i} {name} {' ' * (45 - len(name) - len(str(i)))} {freeze}")
            if freeze:
                param.requires_grad = False


class YOLOPoint(nn.Module):
    def __init__(self, width_multiple=1., depth_multiple=1., inp_ch=3, nc=80, anchors=None):
        super(YOLOPoint, self).__init__()

        c1, c2, c3, c4, c5 = [make_divisible(2**k*width_multiple, 8) for k in range(6, 11)]   # 64, 128, 256, 512, 1024
        n1, n2, n3 = [max(round(k*depth_multiple), 1) for k in (3, 6, 9)] # 3, 6, 9

        # CSPNet shared Backbone
        self.Conv1 = Conv(inp_ch, c1, 6, 2, 2)   # ch_in, ch_out, kernel, stride, padding, groups
        self.Conv2 = Conv(c1, c2, 3, 2)
        self.Bottleneck1 = C3(c2, c2, n1)    # ch_in, ch_out, number
        self.Conv3 = Conv(c2, c3, 3, 2)
        self.Bottleneck2 = C3(c3, c3, n2)

        # YOLO exclusive Backbone
        self.Conv4 = Conv(c3, c4, 3, 2)
        self.Bottleneck3 = C3(c4, c4, n3)
        self.Conv5 = Conv(c4, c5, 3, 2)
        self.Bottleneck4 = C3(c5, c5, n1)
        self.SPPooling = SPPF(c5, c5, 5)   # 768 channels out

        # Object Detector Head
        self.Conv6 = Conv(c5, c4, 1, 1, 0)
        # ups, cat
        self.Bottleneck5 = C3(c5, c4, n1)  # apparently C3 can also reduce width
        self.Conv7 = Conv(c4, c3, 1, 1, 0)
        # ups, cat
        self.Bottleneck6 = C3(c4, c3, n1)  # --> detect
        self.Conv8 = Conv(c3, c3, 3, 2, 1)
        # cat
        self.Bottleneck7 = C3(c4, c4, n1)  # --> detect
        self.Conv9 = Conv(c4, c4, 3, 2, 1)
        # cat
        self.Bottleneck8 = C3(c5, c5, n1)  # --> detect
        self.Detect = Detect(nc, anchors=anchors, ch=(c3, c4, c5))

        # Detector Head
        self.BottleneckDet = C3(c3, c3, n1)
        self.ConvDet = nn.Conv2d(c3, 65, 1, 1, 0, bias=False)
        # self.ConvDet = Conv(c2, 65, 1, 1, 0, act=False)

        # Descriptor Head
        self.ConvDescB = Conv(c3, c2, 3, 2, 1)
        self.ConvDescA = Conv(c2, c2, 3, 2, 1)
        self.ups = torch.nn.Upsample(scale_factor=(2, 2), mode='nearest')
        # self.Cat = Concat
        self.BottleneckDesc = C3(c3, c3, n1)
        self.ConvDesc = nn.Conv2d(c3, c3, 3, 1, 1, bias=False)
        # self.ConvDesc = Conv(c3, c3, 3, 1, 1, act=False)

    def forward(self, x):
        # Shared Encoder
        x = self.Conv1(x)
        x = self.Conv2(x)
        xa = self.Bottleneck1(x)
        x = self.Conv3(xa)

        # Detector Head
        semi = self.BottleneckDet(x)
        semi = self.ConvDet(semi)

        # desc and YOLO Encoder
        xb = self.Bottleneck2(x)

        # Descriptor Head
        descA = self.ConvDescA(xa)
        descB = self.ConvDescB(xb)
        descB = self.ups(descB)
        desc = torch.cat((descA, descB), dim=1)
        desc = self.BottleneckDesc(desc)
        desc = self.ConvDesc(desc)
        dn = torch.norm(desc, p=2, dim=1)  # Compute the norm.
        desc = desc.div(torch.unsqueeze(dn, 1))  # Divide by norm to normalize.

        # YOLO Exclusive Encoder
        x = self.Conv4(xb)
        xc = self.Bottleneck3(x)
        x = self.Conv5(xc)
        x = self.Bottleneck4(x)
        x = self.SPPooling(x)

        # Object Detector Head
        xd = self.Conv6(x)
        x = self.ups(xd)
        x = torch.cat((x, xc), dim=1)
        x = self.Bottleneck5(x)
        xe = self.Conv7(x)
        x = self.ups(xe)
        x = torch.cat((x, xb), dim=1)
        xf = self.Bottleneck6(x)
        x = self.Conv8(xf)
        x = torch.cat((x, xe), dim=1)
        xg = self.Bottleneck7(x)
        x = self.Conv9(xg)
        x = torch.cat((x, xd), dim=1)
        x = self.Bottleneck8(x)
        x = self.Detect([xf, xg, x])

        return {'semi': semi, 'desc': desc, 'objects': x}

class YOLOPointv52(nn.Module):
    # removed Conv6 and reduced width at that stage (heads now exactly like v8)
    def __init__(self, width_multiple=1., depth_multiple=1., inp_ch=3, nc=80, anchors=None):
        super(YOLOPointv52, self).__init__()

        c1, c2, c3, c4, c5 = [make_divisible(2**k*width_multiple, 8) for k in range(6, 11)]   # 64, 128, 256, 512, 1024
        n1, n2, n3 = [max(round(k*depth_multiple), 1) for k in (3, 6, 9)] # 3, 6, 9

        # CSPNet shared Backbone
        self.Conv1 = Conv(inp_ch, c1, 6, 2, 2)   # ch_in, ch_out, kernel, stride, padding, groups
        self.Conv2 = Conv(c1, c2, 3, 2)
        self.Bottleneck1 = C2f(c2, c2, n1)    # ch_in, ch_out, number
        self.Conv3 = Conv(c2, c3, 3, 2)
        self.Bottleneck2 = C2f(c3, c3, n2)

        # YOLO exclusive Backbone
        self.Conv4 = Conv(c3, c4, 3, 2)
        self.Bottleneck3 = C2f(c4, c4, n3)
        self.Conv5 = Conv(c4, c4, 3, 2)
        self.Bottleneck4 = C2f(c4, c4, n1)
        self.SPPooling = SPPF(c4, c4, 5)

        # Object Detector Head
        # ups, cat
        self.Bottleneck5 = C2f(c5, c4, n1)
        # self.Conv7 = Conv(c4, c3, 1, 1, 0)
        # ups, cat
        self.Bottleneck6 = C2f(c4+c3, c3, n1)  # --> detect
        self.Conv8 = Conv(c3, c3, 3, 2, 1)
        # cat
        self.Bottleneck7 = C2f(c4+c3, c4, n1)  # --> detect
        self.Conv9 = Conv(c4, c4, 3, 2, 1)
        # cat
        self.Bottleneck8 = C2f(c5, c4, n1)  # --> detect
        # self.Detect = Detect(nc, anchors=anchors, ch=(c3, c4, c4))

        # Detector Head
        self.BottleneckDet = C2f(c3, 65, n1)

        # Descriptor Head
        self.ConvDescB = Conv(c3, c2, 3, 2, 1)
        self.MaxPool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.ups = torch.nn.Upsample(scale_factor=(2, 2), mode='nearest')
        # self.Cat = Concat
        self.BottleneckDesc = C2f(c3, c3, n1)

    def forward(self, x):
        B, C, H, W = x.shape
        Hc = int(H / 8)
        Wc = int(W / 8)
        # Shared Encoder
        x = self.Conv1(x)
        x = self.Conv2(x)
        xa = self.Bottleneck1(x)
        x = self.Conv3(xa)

        # Detector Head
        semi = self.BottleneckDet(x)
        # semi = self.ConvDet(semi)

        # desc and YOLO Encoder
        xb = self.Bottleneck2(x)

        # Descriptor Head
        descA = self.MaxPool(xa)
        descB = self.ConvDescB(xb)
        descB = self.ups(descB)
        desc = torch.cat((descA, descB), dim=1)
        desc = self.BottleneckDesc(desc)
        # desc = self.ConvDesc(desc)
        dn = torch.norm(desc, p=2, dim=1)  # Compute the norm.
        desc = desc.div(torch.unsqueeze(dn, 1))  # Divide by norm to normalize.

        # YOLO Exclusive Encoder
        x = self.Conv4(xb)
        xc = self.Bottleneck3(x)
        x = self.Conv5(xc)
        x = self.Bottleneck4(x)
        xd = self.SPPooling(x)

        # Object Detector Head
        # xd = self.Conv6(x)
        x = self.ups(xd)
        x = torch.cat((x, xc), dim=1)
        xe = self.Bottleneck5(x)
        # xe = self.Conv7(x)
        x = self.ups(xe)
        x = torch.cat((x, xb), dim=1)
        xf = self.Bottleneck6(x)    # --> Detect
        x = self.Conv8(xf)
        x = torch.cat((x, xe), dim=1)
        xg = self.Bottleneck7(x)
        x = self.Conv9(xg)
        x = torch.cat((x, xd), dim=1)
        x = self.Bottleneck8(x)
        # for item in x:
        #     print(item.shape)
        #     torch.Size([256, 20, 20])
        # torch.Size([256, 20, 20])
        # torch.Size([256, 20, 20])
        # torch.Size([256, 20, 20])
        # x = self.Detect([xf, xg, x])

        # print('semi:', semi.shape, 'desc:', desc.shape, 'objects:',len(x),',', x[0].shape,x[1].shape,x[2].shape)
        # semi: torch.Size([4, 65, 80, 80]) desc: torch.Size([4, 128, 80, 80]) objects: 3
        # semi: torch.Size([4, 65, 80, 80]) desc: torch.Size([4, 128, 80, 80])
        # objects: 3 x[0] torch.Size([4, 3, 80, 80, 85])
        # x[1] torch.Size([8, 3, 40, 40, 85])  x[2] torch.Size([8, 3, 20, 20, 85])

        # return {'semi': semi, 'desc': desc, 'objects': x}


        dense = torch.exp(semi)
        dense /= (torch.sum(dense, dim=1) + 0.00001)
        nodust = dense[:, :-1, :, :]
        nodust = nodust.permute(0,2,3,1)
        heatmap = nodust.reshape(Hc, Wc, 8, 8)
        heatmap = heatmap.permute(0, 2, 1, 3)
        heatmap = heatmap.reshape(1,1,H,W)

        # nodust = dense[:, :-1, :, :].permute(0, 2, 3, 1)
        # heatmap = torch.reshape(nodust, [B, Hc, Wc, 8, 8])
        # heatmap = heatmap.permute(0, 1, 3, 2, 4)
        # heatmap = torch.reshape(heatmap, [B, 1, Hc * 8, Wc * 8])
        # print(heatmap.shape, desc.shape,semi.shape)
        # torch.Size([1, 1, 512, 512]) torch.Size([1, 256, 64, 64]) torch.Size([1, 65, 64, 64])

        return heatmap, desc

class BasicLayer(nn.Module):
    """Basic Convolutional Layer: Conv2d -> BatchNorm -> ReLU"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, stride=stride, dilation=dilation,
                      bias=bias),
            nn.BatchNorm2d(out_channels, affine=False),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.layer(x)

class superFeatModelv2(nn.Module):
    def __init__(self):
        super(superFeatModelv2, self).__init__()

        self.norm = nn.InstanceNorm2d(1)

        # self.head = C3(c1=1,c2=1, n=3)

        # self.skip1 = nn.Sequential(	 nn.AvgPool2d(4, stride = 4),
		# 	  						 nn.Conv2d (1, 24, 1, stride = 1, padding=0) )

        # self.block1 = nn.Sequential(
		# 								BasicLayer( 1,  4, stride=1),
		# 								BasicLayer( 4,  8, stride=2),
		# 								BasicLayer( 8,  8, stride=1),
		# 								BasicLayer( 8, 24, stride=2),
		# 							)
        self.Conv1 = Conv(1, 8, 3, 2)
        self.Bottleneck1 = C2f(8, 8, 3)
        # self.block2 = nn.Sequential(
		# 								BasicLayer(24, 24, stride=1),
		# 								BasicLayer(24, 24, stride=1),
		# 							 )
        self.Conv2 = Conv(8, 24, 3, 2)
        self.Bottleneck2 = C2f(24, 24, 6)
        # self.block3 = nn.Sequential(
		# 								BasicLayer(24, 64, stride=2),
		# 								BasicLayer(64, 64, stride=1),
		# 								BasicLayer(64, 64, 1, padding=0),
		# 							 )
        self.Conv3 = Conv(24, 64, 3, 2)
        self.Bottleneck3 = C2f(64, 64, 6)
        # self.block4 = nn.Sequential(
		# 								BasicLayer(64, 64, stride=2),
		# 								BasicLayer(64, 64, stride=1),
		# 								BasicLayer(64, 64, stride=1),
		# 							 )
        self.Conv4 = Conv(64, 128, 3, 2)
        self.Bottleneck4 = C2f(128, 128, 6)
        # self.block5 = nn.Sequential(
		# 								BasicLayer( 64, 128, stride=2),
		# 								BasicLayer(128, 128, stride=1),
		# 								BasicLayer(128, 128, stride=1),
		# 								BasicLayer(128,  64, 1, padding=0),
		# 							 )
        self.Conv5 = Conv(128, 256, 3, 2)
        self.Bottleneck5 = C2f(256, 256, 6)
        self.down = nn.Conv2d(256, 128, 1)
        self.up = nn.Conv2d(64, 128, 1)

        self.block_fusion = nn.Sequential(
            BasicLayer(128, 128, stride=1),
            BasicLayer(128, 128, stride=1),
            nn.Conv2d(128, 128, 1, padding=0)
        )

        # self.block_fusion =  nn.Sequential(
		# 								BasicLayer(64, 64, stride=1),
		# 								BasicLayer(64, 64, stride=1),
		# 								nn.Conv2d (64, 64, 1, padding=0)
		# 							 )

        # self.C3 = C3(c1=24, c2=24, n=6)





        self.heatmap_head = nn.Sequential(
										BasicLayer(128, 128, 1, padding=0),
										BasicLayer(128, 128, 1, padding=0),
										nn.Conv2d (128, 1, 1),
										nn.Sigmoid()
									)


        self.keypoint_head = nn.Sequential(
										BasicLayer(64, 64, 1, padding=0),
										BasicLayer(64, 64, 1, padding=0),
										BasicLayer(64, 64, 1, padding=0),
										nn.Conv2d (64, 65, 1),
									)


        self.fine_matcher =  nn.Sequential(
											nn.Linear(256, 512),
											nn.BatchNorm1d(512, affine=False),
									  		nn.ReLU(inplace = True),
											nn.Linear(512, 512),
											nn.BatchNorm1d(512, affine=False),
									  		nn.ReLU(inplace = True),
											nn.Linear(512, 512),
											nn.BatchNorm1d(512, affine=False),
									  		nn.ReLU(inplace = True),
											nn.Linear(512, 512),
											nn.BatchNorm1d(512, affine=False),
									  		nn.ReLU(inplace = True),
											nn.Linear(512, 128),
										)




    def _unfold2d(self, x, ws = 2):
        """
            Unfolds tensor in 2D with desired ws (window size) and concat the channels
        """
        B, C, H, W = x.shape
        x = x.unfold(2,  ws , ws).unfold(3, ws,ws)                             \
            .reshape(B, C, H//ws, W//ws, ws**2)
        return x.permute(0, 1, 4, 2, 3).reshape(B, -1, H//ws, W//ws)


    def forward(self, x):
        """
            input:
                x -> torch.Tensor(B, C, H, W) grayscale or rgb images
            return:
                feats     ->  torch.Tensor(B, 64, H/8, W/8) dense local features
                keypoints ->  torch.Tensor(B, 65, H/8, W/8) keypoint logit map
                heatmap   ->  torch.Tensor(B,  1, H/8, W/8) reliability map

        """
        #dont backprop through normalization
        with torch.no_grad():
            x = x.mean(dim=1, keepdim = True)
            x = self.norm(x)

        x1 = self.Conv1(x)
        x1 = self.Bottleneck1(x1)
        x2 = self.Conv2(x1)
        x2 = self.Bottleneck2(x2)
        x3 = self.Conv3(x2)
        x3 = self.Bottleneck3(x3)
        x4 = self.Conv4(x3)
        x4 = self.Bottleneck4(x4)
        x5 = self.Conv5(x4)
        x5 = self.Bottleneck5(x5)
        x3 = self.up(x3)
        x5 = self.down(x5)
        # print(x1.shape,x2.shape,x3.shape,x4.shape,x5.shape)
        # torch.Size([8, 8, 320, 320]) torch.Size([8, 24, 160, 160])
        # torch.Size([8, 64, 80, 80]) torch.Size([8, 128, 40, 40])
        # torch.Size([8, 256, 20, 20])
        # x1 shape: torch.Size([8, 24, 160, 160])
        # x2 shape: torch.Size([8, 24, 160, 160])
        # x3 shape: torch.Size([8, 64, 80, 80])
        # x4 shape: torch.Size([8, 64, 40, 40])
        # x5 shape: torch.Size([8, 64, 20, 20])

        # x4 shape: torch.Size([8, 64, 80, 80])
        # x5 shape: torch.Size([8, 64, 80, 80])



        # x = self.head(x)
        # #main backbone
        # x1 = self.block1(x)
        # x2 = self.block2(x1 + self.skip1(x))
        # x2 = self.C3(x2)
        # x3 = self.block3(x2)
        # x4 = self.block4(x3)
        # x5 = self.block5(x4)

        #pyramid fusion
        # print(x3.shape, x5.shape, x4.shape)
        x4 = F.interpolate(x4, (x3.shape[-2], x3.shape[-1]), mode='bilinear')
        x5 = F.interpolate(x5, (x3.shape[-2], x3.shape[-1]), mode='bilinear')
        # print(x3.shape, x5.shape,x4.shape)
        # torch.Size([8, 128, 80, 80]) torch.Size([8, 128, 20, 20]) torch.Size([8, 128, 40, 40])                                                                                                        | 0/7392 [00:00<?, ?it/s]
        # torch.Size([8, 128, 40, 40]) torch.Size([8, 128, 40, 40]) torch.Size([8, 128, 40, 40])
        feats = self.block_fusion( x3 + x4 + x5 )

        #heads
        heatmap = self.heatmap_head(feats) # Reliability map
        keypoints = self.keypoint_head(self._unfold2d(x, ws=8)) #Keypoint map logits
        # print('keypoints shape:', keypoints.shape,'feats shape:', feats.shape, heatmap.shape)
        heatmap = self.get_kpts_heatmap(keypoints)
        # return keypoints, feats, #heatmap
        return heatmap,feats
        # return keypoints, feats
        # keypoints shape: torch.Size([8, 65, 80, 80]) feats shape: torch.Size([8, 128, 80, 80])
        # return {'semi': keypoints, 'desc': feats}

    def get_kpts_heatmap(self, kpts, softmax_temp=1.0):
        scores = F.softmax(kpts * softmax_temp, 1)[:, :64]
        B, _, H, W = scores.shape
        heatmap = scores.permute(0, 2, 3, 1).reshape(B, H, W, 8, 8)
        heatmap = heatmap.permute(0, 1, 3, 2, 4).reshape(B, 1, H * 8, W * 8)
        return heatmap

class superFeatModelv2_1(nn.Module):
    def __init__(self):
        super(superFeatModelv2_1, self).__init__()

        self.norm = nn.InstanceNorm2d(1)

        self.Conv1 = Conv(1, 8, 3, 2)
        self.Bottleneck1 = C2f(8, 8, 3)

        self.Conv2 = Conv(8, 24, 3, 2)
        self.Bottleneck2 = C2f(24, 24, 6)

        self.Conv3 = Conv(24, 64, 3, 2)
        self.Bottleneck3 = C2f(64, 64, 6)

        self.Conv4 = Conv(64, 128, 3, 2)
        self.Bottleneck4 = C2f(128, 64, 6)

        # self.Conv5 = Conv(128, 256, 3, 2)
        # self.Bottleneck5 = C2f(256, 256, 6)
        # self.down = nn.Conv2d(256, 128, 1)
        # self.up = nn.Conv2d(64, 128, 1)

        self.block_fusion = nn.Sequential(
            BasicLayer(64, 64, stride=1),
            BasicLayer(64, 64, stride=1),
            nn.Conv2d(64, 64, 1, padding=0)
        )



        self.heatmap_head = nn.Sequential(
										BasicLayer(64, 64, 1, padding=0),
										BasicLayer(64, 64, 1, padding=0),
										nn.Conv2d (64, 1, 1),
										nn.Sigmoid()
									)


        self.keypoint_head = nn.Sequential(
										BasicLayer(64, 64, 1, padding=0),
										BasicLayer(64, 64, 1, padding=0),
										BasicLayer(64, 64, 1, padding=0),
										nn.Conv2d (64, 65, 1),
									)


        self.fine_matcher =  nn.Sequential(
											nn.Linear(128, 256),
											nn.BatchNorm1d(256, affine=False),
									  		nn.ReLU(inplace = True),
											nn.Linear(256, 256),
											nn.BatchNorm1d(256, affine=False),
									  		nn.ReLU(inplace = True),
											nn.Linear(256, 256),
											nn.BatchNorm1d(256, affine=False),
									  		nn.ReLU(inplace = True),
											nn.Linear(256, 256),
											nn.BatchNorm1d(256, affine=False),
									  		nn.ReLU(inplace = True),
											nn.Linear(256, 64),
										)




    def _unfold2d(self, x, ws = 2):
        """
            Unfolds tensor in 2D with desired ws (window size) and concat the channels
        """
        B, C, H, W = x.shape
        x = x.unfold(2,  ws , ws).unfold(3, ws,ws)                             \
            .reshape(B, C, H//ws, W//ws, ws**2)
        return x.permute(0, 1, 4, 2, 3).reshape(B, -1, H//ws, W//ws)


    def forward(self, x):
        """
            input:
                x -> torch.Tensor(B, C, H, W) grayscale or rgb images
            return:
                feats     ->  torch.Tensor(B, 64, H/8, W/8) dense local features
                keypoints ->  torch.Tensor(B, 65, H/8, W/8) keypoint logit map
                heatmap   ->  torch.Tensor(B,  1, H/8, W/8) reliability map

        """
        #dont backprop through normalization
        with torch.no_grad():
            x = x.mean(dim=1, keepdim = True)
            x = self.norm(x)

        x1 = self.Conv1(x)
        x1 = self.Bottleneck1(x1)
        x2 = self.Conv2(x1)
        x2 = self.Bottleneck2(x2)
        x3 = self.Conv3(x2)
        x3 = self.Bottleneck3(x3)
        x4 = self.Conv4(x3)
        x4 = self.Bottleneck4(x4)
        # x5 = self.Conv5(x4)
        # x5 = self.Bottleneck5(x5)
        # x3 = self.up(x3)
        # x5 = self.down(x5)

        # print(x1.shape,x2.shape,x3.shape,x4.shape,x5.shape)
        # torch.Size([8, 8, 320, 320]) torch.Size([8, 24, 160, 160])
        # torch.Size([8, 64, 80, 80]) torch.Size([8, 128, 40, 40])
        # torch.Size([8, 256, 20, 20])
        # x1 shape: torch.Size([8, 24, 160, 160])
        # x2 shape: torch.Size([8, 24, 160, 160])
        # x3 shape: torch.Size([8, 64, 80, 80])
        # x4 shape: torch.Size([8, 64, 40, 40])
        # x5 shape: torch.Size([8, 64, 20, 20])

        # x4 shape: torch.Size([8, 64, 80, 80])
        # x5 shape: torch.Size([8, 64, 80, 80])



        # x = self.head(x)
        # #main backbone
        # x1 = self.block1(x)
        # x2 = self.block2(x1 + self.skip1(x))
        # x2 = self.C3(x2)
        # x3 = self.block3(x2)
        # x4 = self.block4(x3)
        # x5 = self.block5(x4)

        #pyramid fusion
        # print(x3.shape, x5.shape, x4.shape)
        x4 = F.interpolate(x4, (x3.shape[-2], x3.shape[-1]), mode='bilinear')
        # x5 = F.interpolate(x5, (x3.shape[-2], x3.shape[-1]), mode='bilinear')
        # print(x3.shape, x5.shape,x4.shape)
        # torch.Size([8, 128, 80, 80]) torch.Size([8, 128, 20, 20]) torch.Size([8, 128, 40, 40])                                                                                                        | 0/7392 [00:00<?, ?it/s]
        # torch.Size([8, 128, 40, 40]) torch.Size([8, 128, 40, 40]) torch.Size([8, 128, 40, 40])
        feats = self.block_fusion( x3 + x4 )

        # #heads
        # heatmap = self.heatmap_head(feats) # Reliability map
        # keypoints = self.keypoint_head(self._unfold2d(x, ws=8)) #Keypoint map logits
        # # print('keypoints shape:', keypoints.shape,'feats shape:', feats.shape, heatmap.shape)
        # return feats, keypoints, heatmap
        #
        # # return keypoints, feats
        # # keypoints shape: torch.Size([8, 65, 80, 80]) feats shape: torch.Size([8, 128, 80, 80])
        # # return {'semi': keypoints, 'desc': feats}

        #heads

        keypoints = self.keypoint_head(self._unfold2d(x, ws=8)) #Keypoint map logits
        # print('keypoints shape:', keypoints.shape,'feats shape:', feats.shape, heatmap.shape)
        heatmap = self.get_kpts_heatmap(keypoints)
        # return keypoints, feats, #heatmap
        return heatmap,feats
        # return keypoints, feats
        # keypoints shape: torch.Size([8, 65, 80, 80]) feats shape: torch.Size([8, 128, 80, 80])
        # return {'semi': keypoints, 'desc': feats}

    def get_kpts_heatmap(self, kpts, softmax_temp=1.0):
        scores = F.softmax(kpts * softmax_temp, 1)[:, :64]
        B, _, H, W = scores.shape
        heatmap = scores.permute(0, 2, 3, 1).reshape(B, H, W, 8, 8)
        heatmap = heatmap.permute(0, 1, 3, 2, 4).reshape(B, 1, H * 8, W * 8)
        return heatmap
















class YOLOSegPointv52(nn.Module):
    # removed Conv6 and reduced width at that stage (heads now exactly like v8)
    def __init__(self, width_multiple=1., depth_multiple=1., inp_ch=3, nc=80, anchors=None):
        super(YOLOSegPointv52, self).__init__()

        c1, c2, c3, c4, c5 = [make_divisible(2**k*width_multiple, 8) for k in range(6, 11)]   # 64, 128, 256, 512, 1024
        n1, n2, n3 = [max(round(k*depth_multiple), 1) for k in (3, 6, 9)] # 3, 6, 9

        # CSPNet shared Backbone
        self.Conv1 = Conv(inp_ch, c1, 6, 2, 2)   # ch_in, ch_out, kernel, stride, padding, groups
        self.Conv2 = Conv(c1, c2, 3, 2)
        self.Bottleneck1 = C2f(c2, c2, n1)    # ch_in, ch_out, number
        self.Conv3 = Conv(c2, c3, 3, 2)
        self.Bottleneck2 = C2f(c3, c3, n2)

        # YOLO exclusive Backbone
        self.Conv4 = Conv(c3, c4, 3, 2)
        self.Bottleneck3 = C2f(c4, c4, n3)
        self.Conv5 = Conv(c4, c4, 3, 2)
        self.Bottleneck4 = C2f(c4, c4, n1)
        self.SPPooling = SPPF(c4, c4, 5)

        # Object Detector Head
        # ups, cat
        self.Bottleneck5 = C2f(c5, c4, n1)
        # self.Conv7 = Conv(c4, c3, 1, 1, 0)
        # ups, cat
        self.Bottleneck6 = C2f(c4+c3, c3, n1)  # --> detect
        self.Conv8 = Conv(c3, c3, 3, 2, 1)
        # cat
        self.Bottleneck7 = C2f(c4+c3, c4, n1)  # --> detect
        self.Conv9 = Conv(c4, c4, 3, 2, 1)
        # cat
        self.Bottleneck8 = C2f(c5, c4, n1)  # --> detect
        self.Segment = Segment(nc, anchors=anchors, nm=32, npr=256, ch=(c3, c4, c4))

        # Detector Head
        self.BottleneckDet = C2f(c3, 65, n1)

        # Descriptor Head
        self.ConvDescB = Conv(c3, c2, 3, 2, 1)
        self.MaxPool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.ups = torch.nn.Upsample(scale_factor=(2, 2), mode='nearest')
        # self.Cat = Concat
        self.BottleneckDesc = C2f(c3, c3, n1)

    def forward(self, x):
        B, C, H, W = x.shape
        Hc = int(H / 8)
        Wc = int(W / 8)
        # Shared Encoder
        x = self.Conv1(x)
        x = self.Conv2(x)
        xa = self.Bottleneck1(x)
        x = self.Conv3(xa)

        # Detector Head
        semi = self.BottleneckDet(x)
        # semi = self.ConvDet(semi)

        # desc and YOLO Encoder
        xb = self.Bottleneck2(x)

        # Descriptor Head
        descA = self.MaxPool(xa)
        descB = self.ConvDescB(xb)
        descB = self.ups(descB)
        desc = torch.cat((descA, descB), dim=1)
        desc = self.BottleneckDesc(desc)
        # desc = self.ConvDesc(desc)
        dn = torch.norm(desc, p=2, dim=1)  # Compute the norm.
        desc = desc.div(torch.unsqueeze(dn, 1))  # Divide by norm to normalize.

        # YOLO Exclusive Encoder
        x = self.Conv4(xb)
        xc = self.Bottleneck3(x)
        x = self.Conv5(xc)
        x = self.Bottleneck4(x)
        xd = self.SPPooling(x)

        # Object Detector Head
        # xd = self.Conv6(x)
        x = self.ups(xd)
        x = torch.cat((x, xc), dim=1)
        xe = self.Bottleneck5(x)
        # xe = self.Conv7(x)
        x = self.ups(xe)
        x = torch.cat((x, xb), dim=1)
        xf = self.Bottleneck6(x)    # --> Detect
        x = self.Conv8(xf)
        x = torch.cat((x, xe), dim=1)
        xg = self.Bottleneck7(x)
        x = self.Conv9(xg)
        x = torch.cat((x, xd), dim=1)
        x = self.Bottleneck8(x)
        x = self.Segment([xf, xg, x])

        # print('semi:', semi.shape, 'desc:', desc.shape, 'objects:',len(x),',', x[0].shape,x[1].shape,x[2].shape)
        # semi: torch.Size([4, 65, 80, 80]) desc: torch.Size([4, 128, 80, 80]) objects: 3
        # semi: torch.Size([4, 65, 80, 80]) desc: torch.Size([4, 128, 80, 80])
        # objects: 3 x[0] torch.Size([4, 3, 80, 80, 85])
        # x[1] torch.Size([8, 3, 40, 40, 85])  x[2] torch.Size([8, 3, 20, 20, 85])

        #return {'semi': semi, 'desc': desc, 'objects': x}


                # addition semi to score map
        dense = torch.softmax(semi, dim=1)
        nodust = dense[:, :-1, :, :].permute(0, 2, 3, 1)
        heatmap = torch.reshape(nodust, [B, Hc, Wc, 8, 8])
        heatmap = heatmap.permute(0, 1, 3, 2, 4)
        heatmap = torch.reshape(heatmap, [B, 1, Hc * 8, Wc * 8])
        return semi, desc


class YOLOPointDet(nn.Module):
    def __init__(self, width_multiple=1., depth_multiple=1., inp_ch=3, nc=80, anchors=None):
        super(YOLOPointDet, self).__init__()

        c1, c2, c3, c4, c5 = [make_divisible(2**k*width_multiple, 8) for k in range(6, 11)]   # 64, 128, 256, 512, 1024
        n1, n2, n3 = [max(round(k*depth_multiple), 1) for k in (3, 6, 9)] # 3, 6, 9

        # CSPNet shared Backbone
        self.Conv1 = Conv(inp_ch, c1, 6, 2, 2)   # ch_in, ch_out, kernel, stride, padding, groups
        self.Conv2 = Conv(c1, c2, 3, 2)
        self.Bottleneck1 = C3(c2, c2, n1)    # ch_in, ch_out, number
        self.Conv3 = Conv(c2, c3, 3, 2)
        self.Bottleneck2 = C3(c3, c3, n2)

        # Detector Head
        self.BottleneckDet = C3(c3, c3, n1) # was c3 c3
        self.ConvDet = nn.Conv2d(c3, 65, 1, 1, 0, bias=False)
        # self.ConvDet = Conv(c2, 65, 1, 1, 0, act=False)

        # Descriptor Head
        self.ConvDescB = Conv(c3, c2, 3, 2, 1)
        self.ConvDescA = Conv(c2, c2, 3, 2, 1)
        self.ups = torch.nn.Upsample(scale_factor=(2, 2), mode='nearest')
        # self.Cat = Concat
        self.BottleneckDesc = C3(c3, c3, n1)
        self.ConvDesc = nn.Conv2d(c3, c3, 3, 1, 1, bias=False)
        # self.ConvDesc = Conv(c3, c3, 3, 1, 1, act=False)

    def forward(self, x):
        # Shared Encoder
        x = self.Conv1(x)
        x = self.Conv2(x)
        xa = self.Bottleneck1(x)
        x = self.Conv3(xa)

        # Detector Head
        semi = self.BottleneckDet(x)
        semi = self.ConvDet(semi)

        # desc and YOLO Encoder
        x = self.Bottleneck2(x)

        # Descriptor Head
        descA = self.ConvDescA(xa)
        descB = self.ConvDescB(x)
        descB = self.ups(descB)
        desc = torch.cat((descA, descB), dim=1)
        desc = self.BottleneckDesc(desc)
        desc = self.ConvDesc(desc)
        dn = torch.norm(desc, p=2, dim=1)  # Compute the norm.
        desc = desc.div(torch.unsqueeze(dn, 1))  # Divide by norm to normalize.

        return {'semi': semi, 'desc': desc}

class SuperPointNet(torch.nn.Module):
    """ Pytorch definition of SuperPoint Network. """

    def __init__(self, width_multiple=None, depth_multiple=None, inp_ch=1, nc=None, anchors=None):
        super(SuperPointNet, self).__init__()
        self.relu = torch.nn.ReLU(inplace=True)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256
        # Shared Encoder.
        self.conv1a = torch.nn.Conv2d(inp_ch, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = torch.nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = torch.nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = torch.nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = torch.nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = torch.nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = torch.nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = torch.nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)
        # Detector Head.
        self.convPa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convPb = torch.nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)
        # Descriptor Head.
        self.convDa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = torch.nn.Conv2d(c5, d1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """ Forward pass that jointly computes unprocessed point and descriptor
        tensors.
        Input
          x: Image pytorch tensor shaped N x 1 x H x W.
        Output
          semi: Output point pytorch tensor shaped N x 65 x H/8 x W/8.
          desc: Output descriptor pytorch tensor shaped N x 256 x H/8 x W/8.
        """
        # Shared Encoder.
        x = self.relu(self.conv1a(x))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        # Detector Head.
        cPa = self.relu(self.convPa(x))
        semi = self.convPb(cPa)
        # Descriptor Head.
        cDa = self.relu(self.convDa(x))
        desc = self.convDb(cDa)
        dn = torch.norm(desc, p=2, dim=1)  # Compute the norm.
        desc = desc.div(torch.unsqueeze(dn, 1))  # Divide by norm to normalize.
        return {'semi': semi, 'desc': desc}

class YOLOv8Point(nn.Module):
    def __init__(self, width_multiple=1., depth_multiple=1., inp_ch=3, nc=80, anchors=None):
        super(YOLOv8Point, self).__init__()

        c1, c2, c3, c4, c5 = [make_divisible(2**k*width_multiple, 8) for k in range(6, 11)]   # 64, 128, 256, 512, 1024
        n1, n2, n3 = [max(round(k*depth_multiple), 1) for k in (3, 6, 9)] # 3, 6, 9
        r=1

        self.Conv0 = Conv(inp_ch, c1, 3, 2, 1)  # ch_in, ch_out, kernel, stride, padding, groups
        self.Conv1 = Conv(c1, c2, 3, 2, 1)
        self.Bottleneck2 = C2f(c2, c2, n1, shortcut=True)    # ch_in, ch_out, number, shortcut, groups, expansion
        self.Conv3 = Conv(c2, c3, 3, 2, 1)
        self.Bottleneck4 = C2f(c3, c3, n2, shortcut=True)
        self.Conv5 = Conv(c3, c4, 3, 2, 1)
        self.Bottleneck6 = C2f(c4, c4, n2, shortcut=True)
        self.Conv7 = Conv(c4, c4*r, 3, 2, 1)
        self.Bottleneck8 = C2f(c4*r, c4*r, n1, shortcut=True)
        self.SPPooling9 = SPPF(c4*r, c4*r, 5)   # 768 channels out

        # Object Detector Head
        # ups, cat
        self.Bottleneck12 = C2f(c4*(1+r), c4, n1)
        # ups, cat
        self.Bottleneck15 = C2f(c3+c4, c3, n1)  # --> detect
        self.Conv16 = Conv(c3, c3, 3, 2, 1)
        # ups, cat
        self.Bottleneck18 = C2f(c3+c4, c4, n1)  # --> detect
        self.Conv19 = Conv(c4, c4, 3, 2, 1)
        # cat
        self.Bottleneck21 = C2f(c4*(1+r), c4*r, n1)  # --> detect
        self.Detect = Detect(nc, anchors=anchors, ch=(c3, c4, c4*r))

        # YP Neck
        self.Conv22 = Conv(c4, c3, 3, 1, 1)
        # ups, cat
        self.Conv25 = Conv(c4, c3, 3, 1, 1)
        self.Pool26 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        # cat

        # Detector Head
        self.BottleneckDet = C2f(c3+c2, 65, n1)

        # Descriptor Head
        self.BottleneckDesc = C2f(c3+c2, c3, n1)
        self.ConvDesc = nn.Conv2d(c3, c3, 3, 1, 1, bias=False)

        self.Ups = torch.nn.Upsample(scale_factor=(2, 2), mode='nearest')


    def forward(self, x):
        # Backbone
        x = self.Conv0(x)
        x = self.Conv1(x)
        xa = self.Bottleneck2(x)
        x = self.Conv3(xa)
        xb = self.Bottleneck4(x)
        x = self.Conv5(x)
        xc = self.Bottleneck6(x)
        x = self.Conv7(xc)
        x = self.Bottleneck8(x)
        xd = self.SPPooling9(x)

        # YOLO heads
        x = self.Ups(xd)
        x = torch.cat((x, xc), dim=1)
        xe = self.Bottleneck12(x)
        x = self.Ups(xe)
        x = torch.cat((x, xb), dim=1)
        xf = self.Bottleneck15(x)    # --> Detect
        x = self.Conv16(xf)
        x = torch.cat((x, xe), dim=1)
        xg = self.Bottleneck18(x)   # --> Detect
        x = self.Conv19(xg)
        x = torch.cat((x, xd), dim=1)
        x = self.Bottleneck21(x)    # --> Detect
        x = self.Detect([xf, xg, x])

        # YOLOPoint Heads
        xc = self.Conv22(xc)
        xc = self.Ups(xc)
        xc = torch.cat((xc, xb), dim=1)
        xc = self.Conv25(xc)
        xa = self.Pool26(xa)
        xa = torch.cat((xa, xc), dim=1)

        # Detector Head
        semi = self.BottleneckDet(xa)

        # Descriptor Head
        desc = self.BottleneckDesc(xa)
        desc = self.ConvDesc(desc)
        desc = torch.norm(desc, p=2, dim=1)  # Compute the norm.
        desc = desc.div(torch.unsqueeze(desc, 1))  # Divide by norm to normalize

        return {'semi': semi, 'desc': desc, 'objects': x}
