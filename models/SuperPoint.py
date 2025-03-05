import torch
import numpy as np


class SuperPointNet(torch.nn.Module):
    """ Pytorch definition of SuperPoint Network. """

    def __init__(self):
        super(SuperPointNet, self).__init__()
        # TODO 添加SP的网络结构
        self.relu = torch.nn.ReLU(inplace=True)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256
        # Shared Encoder.
        self.conv1a = torch.nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
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
            B, C, H, W = x.shape
            # torch.Size([1, 3, 512, 512])
            # print(x.shape)
            Hc = int(H / 8)
            Wc = int(W / 8)
            x = torch.sum(x, dim=1, keepdim=True) # 把输入的图像转换为灰度图
            # Shared Encoder.
            # TODO 添加SP的网络结构
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
            desc = desc.div(torch.unsqueeze(dn, 1))  # Normalize.

            # addition semi to score map
            dense = torch.softmax(semi, dim=1)
            nodust = dense[:, :-1, :, :].permute(0, 2, 3, 1)
            heatmap = torch.reshape(nodust, [B, Hc, Wc, 8, 8])
            heatmap = heatmap.permute(0, 1, 3, 2, 4)
            heatmap = torch.reshape(heatmap, [B, 1, Hc * 8, Wc * 8])

            # print(heatmap.shape,desc.shape,semi.shape)
            #torch.Size([1, 1, 512, 512]) torch.Size([1, 256, 64, 64])
            return heatmap, desc














            # semi = semi.squeeze(0)
            # dense = torch.exp(semi)/(torch.sum(torch.exp(semi), axis = 0)+0.0001)
            # nodust = dense[:-1, :, :].permute(1, 2, 0)
            # # Reshape to get full resolution heatmap.
            # Hc = int(H / 8)
            # Wc = int(W / 8)
            #
            # heatmap = nodust.reshape(Hc, Wc, 8, 8).permute(0, 2, 1, 3).reshape(1,1,Hc*8, Wc*8)
            # # print(heatmap.shape)
            # return heatmap, desc
            # # return semi, desc


            # # Descriptor Head
            # cDa = self.relu(self.convDa(x))
            # desc = self.convDb(cDa)
            # dn = torch.norm(desc, p=2, dim=1, keepdim=True)  # Compute the norm and keep dimensions
            # desc = desc / (dn + 1e-4)  # Normalize
            #
            # semi = semi.squeeze(0)
            # dense = torch.exp(semi) / (torch.sum(torch.exp(semi), axis=0, keepdim=True) + 1e-4)
            # nodust = dense[:-1, :, :].permute(1, 2, 0)
            #
            # # Reshape to get full resolution heatmap
            # heatmap = nodust.reshape(Hc, Wc, 8, 8).permute(0, 2, 1, 3).reshape(1, 1, Hc * 8, Wc * 8)
            # print(heatmap.shape)
            # return heatmap, desc

# class SuperPoint(object):
#     def __init__(self, weights_path):
#         self.net = SuperPointNet()
#         self.net.load_state_dict(torch.load(weights_path, map_location='cpu'))
#         self.net.eval()
#
#     def __call__(self, img):
#         """
#         :param img: [B, 1, H, W] H, W % 64 == 0 in range [0,1].
#         :return:  score map        [B, 1, H, W] in range [0,1].
#                   local desc map 0 [B, 3, H, W]
#                   local desc map 1 [B, 16, H/8, W/8] normalized
#                   desc map         [B, 32, H/64, W/64]
#         """
#         B, C, H, W = img.shape
#         assert H % 64 == 0 and W % 64 == 0
#         if C == 3:
#             img = torch.mean(img, dim=1, keepdim=True)
#         with torch.no_grad():
#             semi, desc = self.net(img)
#
#
#         semi = semi.data.cpu().numpy().squeeze()
#         dense = np.exp(semi) / (np.sum(np.exp(semi), axis=0) + 0.0001)
#         nodust = dense[:-1, :, :].transpose(1, 2, 0)
#         Hc = int(H / 8)
#         Wc = int(W / 8)
#         heatmap = np.reshape(nodust, [Hc, Wc, 8, 8])
#         heatmap = np.transpose(heatmap, [0, 2, 1, 3])
#         heatmap = np.reshape(heatmap, [Hc * 8, Wc * 8])
#         desc_map_0 = torch.cat((img, torch.cat((img, img), dim=1)), dim=1)
#         desc_map_0 = desc_map_0 / (torch.norm(desc_map_0, p=2, dim=1, keepdim=True) + 0.0001)
#         desc_map_1 = desc[:, 0:4, :, :]
#         desc_map_1 = desc_map_1 / (torch.norm(desc_map_1, p=2, dim=1, keepdim=True) + 0.0001)
#         Hc = int(H / 64)
#         Wc = int(W / 64)
#         desc_map_2 = torch.zeros(1, 256, Hc, Wc)
#         for i in range(Hc):
#             for j in range(Wc):
#                 desc_map_2[:, :, i, j] = torch.mean(desc[:, :256, i * 8:(i + 1) * 8, j * 8:(j + 1) * 8], dim=(2, 3))
#         desc_map_2 = desc_map_2 / (torch.norm(desc_map_2, p=2, dim=1, keepdim=True) + 0.0001)
#         return torch.from_numpy(heatmap).unsqueeze(0).unsqueeze(0), desc_map_0, desc_map_1, desc_map_2
#
#
if __name__ == '__main__':
    from thop import profile
    import numpy as np
    import cv2
    import time
    net = SuperPointNet()
    net.load_state_dict(torch.load('../weights/superpoint_v1.pth', map_location='cpu'))
    net.eval()
    # image = torch.tensor(np.random.random((1, 1, 512, 512)), dtype=torch.float32)
    image_path = "/media/wrm/S7/Run_W/dataset/kitti360/KITTI-360/data_2d_raw/2013_05_28_drive_0007_sync/image_00/data_rect/0000000000.png"
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # 读取为灰度图
    image = cv2.resize(image, (512, 512))  # 调整大小（如果需要）
    image = image.astype(np.float32) / 255.0  # 归一化到 [0, 1]
    image = torch.tensor(image).unsqueeze(0).unsqueeze(0)  # (H, W) → (1, H, W) → (1, 1, H, W)


    flops, params = profile(net, inputs=(image,))
    print('{:<30}  {:<8} GFLops'.format('Computational complexity: ', flops / 1e9))
    print('{:<30}  {:<8} KB'.format('Number of parameters: ', params / 1e3))

    start = time.time()
    heatmap, desc = net(image)
    endl = time.time()
    print(endl - start, 's')
    print(heatmap.shape, desc.shape)

    # 将灰度图转换为 BGR 彩色图用于显示
    img_disp = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    # 绘制特征点，圆点半径和颜色可根据需要调整
    for pt in keypoints:
        y, x = pt  # 注意：np.argwhere 返回的顺序为 (row, col)
        cv2.circle(img_disp, (int(x), int(y)), radius=2, color=(0, 0, 255), thickness=-1)

    # 显示结果
    cv2.imshow("Feature Points", img_disp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()