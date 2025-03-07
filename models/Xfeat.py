import torch
import torch.nn as nn
import torch.nn.functional as F


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


class XFeat(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.InstanceNorm2d(1)

        self.skip1 = nn.Sequential(nn.AvgPool2d(4, stride=4),
                                   nn.Conv2d(1, 24, 1, stride=1, padding=0))

        self.block1 = nn.Sequential(
            BasicLayer(1, 4, stride=1),
            BasicLayer(4, 8, stride=2),
            BasicLayer(8, 8, stride=1),
            BasicLayer(8, 24, stride=2),
        )

        self.block2 = nn.Sequential(
            BasicLayer(24, 24, stride=1),
            BasicLayer(24, 24, stride=1),
        )

        self.block3 = nn.Sequential(
            BasicLayer(24, 64, stride=2),
            BasicLayer(64, 64, stride=1),
            BasicLayer(64, 64, 1, padding=0),
        )
        self.block4 = nn.Sequential(
            BasicLayer(64, 64, stride=2),
            BasicLayer(64, 64, stride=1),
            BasicLayer(64, 64, stride=1),
        )

        self.block5 = nn.Sequential(
            BasicLayer(64, 128, stride=2),
            BasicLayer(128, 128, stride=1),
            BasicLayer(128, 128, stride=1),
            BasicLayer(128, 64, 1, padding=0),
        )

        self.block_fusion = nn.Sequential(
            BasicLayer(64, 64, stride=1),
            BasicLayer(64, 64, stride=1),
            nn.Conv2d(64, 64, 1, padding=0)
        )

        self.heatmap_head = nn.Sequential(
            BasicLayer(64, 64, 1, padding=0),
            BasicLayer(64, 64, 1, padding=0),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )

        self.keypoint_head = nn.Sequential(
            BasicLayer(64, 64, 1, padding=0),
            BasicLayer(64, 64, 1, padding=0),
            BasicLayer(64, 64, 1, padding=0),
            nn.Conv2d(64, 65, 1),
        )

        self.fine_matcher = nn.Sequential(
            nn.Linear(128, 512),
            nn.BatchNorm1d(512, affine=False),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512, affine=False),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512, affine=False),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512, affine=False),
            nn.ReLU(inplace=True),
            nn.Linear(512, 64),
        )

    def _unfold2d(self, x, ws=2):
        """
            Unfolds tensor in 2D with desired ws (window size) and concat the channels
        """
        B, C, H, W = x.shape
        x = x.unfold(2, ws, ws).unfold(3, ws, ws) \
            .reshape(B, C, H // ws, W // ws, ws ** 2)
        return x.permute(0, 1, 4, 2, 3).reshape(B, -1, H // ws, W // ws)

    def get_kpts_heatmap(self, kpts, softmax_temp=1.0):
        scores = F.softmax(kpts * softmax_temp, 1)[:, :64]
        B, _, H, W = scores.shape
        heatmap = scores.permute(0, 2, 3, 1).reshape(B, H, W, 8, 8)
        heatmap = heatmap.permute(0, 1, 3, 2, 4).reshape(B, 1, H * 8, W * 8)
        return heatmap

    def forward(self, x):
        """
            input:
                x -> torch.Tensor(B, C, H, W) grayscale or rgb images
            return:
                feats     ->  torch.Tensor(B, 64, H/8, W/8) dense local features
                keypoints ->  torch.Tensor(B, 65, H/8, W/8) keypoint logit map
                heatmap   ->  torch.Tensor(B,  1, H/8, W/8) reliability map
        """
        with torch.no_grad():
            x = x.mean(dim=1, keepdim=True)
            x = self.norm(x)

        # main backbone
        x1 = self.block1(x)
        x2 = self.block2(x1 + self.skip1(x))
        x3 = self.block3(x2)
        x4 = self.block4(x3)
        x5 = self.block5(x4)

        # pyramid fusion
        x4 = F.interpolate(x4, (x3.shape[-2], x3.shape[-1]), mode='bilinear')
        x5 = F.interpolate(x5, (x3.shape[-2], x3.shape[-1]), mode='bilinear')
        feats = self.block_fusion(x3 + x4 + x5)
        feats = F.normalize(feats, dim=1)

        key_points = self.keypoint_head(self._unfold2d(x, ws=8))  # Keypoint map logits
        heatmap = self.get_kpts_heatmap(key_points)
        # print(heatmap.shape, feats.shape, key_points.shape)
        #torch.Size([1, 1, 512, 512]) torch.Size([1, 64, 64, 64]) torch.Size([1, 65, 64, 64])

        return heatmap, feats


if __name__ == '__main__':
    from thop import profile
    net = XFeat()
    weight = torch.load('../weights/xfeat.pt')
    net.load_state_dict(weight)
    net.eval()
    image = torch.randn(1, 3, 512, 512)


    flops, params = profile(net, inputs=(image,))
    print('{:<30}  {:<8} GFLops'.format('Computational complexity: ', flops / 1e9))
    print('{:<30}  {:<8} KB'.format('Number of parameters: ', params / 1e3))












# """
# 	"XFeat: Accelerated Features for Lightweight Image Matching, CVPR 2024."
# 	https://www.verlab.dcc.ufmg.br/descriptors/xfeat_cvpr24/
# """
#
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import time
#
# class BasicLayer(nn.Module):
# 	"""
# 	  Basic Convolutional Layer: Conv2d -> BatchNorm -> ReLU
# 	"""
# 	def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False):
# 		super().__init__()
# 		self.layer = nn.Sequential(
# 									  nn.Conv2d( in_channels, out_channels, kernel_size, padding = padding, stride=stride, dilation=dilation, bias = bias),
# 									  nn.BatchNorm2d(out_channels, affine=False),
# 									  nn.ReLU(inplace = True),
# 									)
#
# 	def forward(self, x):
# 	  return self.layer(x)
#
# class XFeat(nn.Module):
# 	"""
# 	   Implementation of architecture described in
# 	   "XFeat: Accelerated Features for Lightweight Image Matching, CVPR 2024."
# 	"""
#
# 	def __init__(self):
# 		super().__init__()
# 		self.norm = nn.InstanceNorm2d(1)
#
#
# 		########### ⬇️ CNN Backbone & Heads ⬇️ ###########
#
# 		self.skip1 = nn.Sequential(	 nn.AvgPool2d(4, stride = 4),
# 			  						 nn.Conv2d (1, 24, 1, stride = 1, padding=0) )
#
# 		self.block1 = nn.Sequential(
# 										BasicLayer( 1,  4, stride=1),
# 										BasicLayer( 4,  8, stride=2),
# 										BasicLayer( 8,  8, stride=1),
# 										BasicLayer( 8, 24, stride=2),
# 									)
#
# 		self.block2 = nn.Sequential(
# 										BasicLayer(24, 24, stride=1),
# 										BasicLayer(24, 24, stride=1),
# 									 )
#
# 		self.block3 = nn.Sequential(
# 										BasicLayer(24, 64, stride=2),
# 										BasicLayer(64, 64, stride=1),
# 										BasicLayer(64, 64, 1, padding=0),
# 									 )
# 		self.block4 = nn.Sequential(
# 										BasicLayer(64, 64, stride=2),
# 										BasicLayer(64, 64, stride=1),
# 										BasicLayer(64, 64, stride=1),
# 									 )
#
# 		self.block5 = nn.Sequential(
# 										BasicLayer( 64, 128, stride=2),
# 										BasicLayer(128, 128, stride=1),
# 										BasicLayer(128, 128, stride=1),
# 										BasicLayer(128,  64, 1, padding=0),
# 									 )
#
# 		self.block_fusion =  nn.Sequential(
# 										BasicLayer(64, 64, stride=1),
# 										BasicLayer(64, 64, stride=1),
# 										nn.Conv2d (64, 64, 1, padding=0)
# 									 )
#
# 		self.heatmap_head = nn.Sequential(
# 										BasicLayer(64, 64, 1, padding=0),
# 										BasicLayer(64, 64, 1, padding=0),
# 										nn.Conv2d (64, 1, 1),
# 										nn.Sigmoid()
# 									)
#
#
# 		self.keypoint_head = nn.Sequential(
# 										BasicLayer(64, 64, 1, padding=0),
# 										BasicLayer(64, 64, 1, padding=0),
# 										BasicLayer(64, 64, 1, padding=0),
# 										nn.Conv2d (64, 65, 1),
# 									)
#
#
#   		########### ⬇️ Fine Matcher MLP ⬇️ ###########
#
# 		self.fine_matcher =  nn.Sequential(
# 											nn.Linear(128, 512),
# 											nn.BatchNorm1d(512, affine=False),
# 									  		nn.ReLU(inplace = True),
# 											nn.Linear(512, 512),
# 											nn.BatchNorm1d(512, affine=False),
# 									  		nn.ReLU(inplace = True),
# 											nn.Linear(512, 512),
# 											nn.BatchNorm1d(512, affine=False),
# 									  		nn.ReLU(inplace = True),
# 											nn.Linear(512, 512),
# 											nn.BatchNorm1d(512, affine=False),
# 									  		nn.ReLU(inplace = True),
# 											nn.Linear(512, 64),
# 										)
#
# 	def _unfold2d(self, x, ws = 2):
# 		"""
# 			Unfolds tensor in 2D with desired ws (window size) and concat the channels
# 		"""
# 		B, C, H, W = x.shape
# 		x = x.unfold(2,  ws , ws).unfold(3, ws,ws)                             \
# 			.reshape(B, C, H//ws, W//ws, ws**2)
# 		return x.permute(0, 1, 4, 2, 3).reshape(B, -1, H//ws, W//ws)
#
#
# 	def forward(self, x):
# 		"""
# 			input:
# 				x -> torch.Tensor(B, C, H, W) grayscale or rgb images
# 			return:
# 				feats     ->  torch.Tensor(B, 64, H/8, W/8) dense local features
# 				keypoints ->  torch.Tensor(B, 65, H/8, W/8) keypoint logit map
# 				heatmap   ->  torch.Tensor(B,  1, H/8, W/8) reliability map
#
# 		"""
# 		#dont backprop through normalization
# 		with torch.no_grad():
# 			x = x.mean(dim=1, keepdim = True)
# 			x = self.norm(x)
#
# 		#main backbone
# 		x1 = self.block1(x)
# 		x2 = self.block2(x1 + self.skip1(x))
# 		x3 = self.block3(x2)
# 		x4 = self.block4(x3)
# 		x5 = self.block5(x4)
#
# 		#pyramid fusion
# 		x4 = F.interpolate(x4, (x3.shape[-2], x3.shape[-1]), mode='bilinear')
# 		x5 = F.interpolate(x5, (x3.shape[-2], x3.shape[-1]), mode='bilinear')
# 		feats = self.block_fusion( x3 + x4 + x5 )
#
# 		#heads
# 		heatmap = self.heatmap_head(feats) # Reliability map
# 		keypoints = self.keypoint_head(self._unfold2d(x, ws=8)) #Keypoint map logits
#
# 		scores = F.softmax(keypoints * 1.0, 1)[:, :64]
# 		B, _, H, W = scores.shape
# 		heatmap = scores.permute(0, 2, 3, 1).reshape(B, H, W, 8, 8)
# 		heatmap = heatmap.permute(0, 1, 3, 2, 4).reshape(B, 1, H * 8, W * 8)
#
# 		#print(heatmap.shape)
# 		# semi = semi.squeeze(0)
# 		# dense = torch.exp(heatmap) / (torch.sum(torch.exp(heatmap), axis=0) + 0.0001)
# 		# nodust = dense[:-1, :, :].permute(1, 2, 0)
# 		# B, C, H, W = x.shape
# 		# Hc = int(H / 8)
# 		# Wc = int(W / 8)
# 		# heatmap = feats.reshape(Hc, Wc, 8, 8).permute(0, 2, 1, 3).reshape(1, 1, Hc * 8, Wc * 8)
#
#
# 		return heatmap, keypoints #, feats
#
#
# 	def NMS(self, x, threshold=0.05, kernel_size=5):
# 		B, _, H, W = x.shape
# 		pad = kernel_size // 2
# 		local_max = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=pad)(x)
# 		pos = (x == local_max) & (x > threshold)
# 		pos_batched = [k.nonzero()[..., 1:].flip(-1) for k in pos]
#
# 		pad_val = max([len(x) for x in pos_batched])
# 		pos = torch.zeros((B, pad_val, 2), dtype=torch.long, device=x.device)
#
# 		# Pad kpts and build (B, N, 2) tensor
# 		for b in range(len(pos_batched)):
# 			pos[b, :len(pos_batched[b]), :] = pos_batched[b]
#
# 		return pos