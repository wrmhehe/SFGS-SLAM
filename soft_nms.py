import torch
from torch import nn

# coordinates system
#  ------------------------------>  [ x: range=-1.0~1.0; w: range=0~W ]
#  | -----------------------------
#  | |                           |
#  | |                           |
#  | |                           |
#  | |         image             |
#  | |                           |
#  | |                           |
#  | |                           |
#  | |---------------------------|
#  v
# [ y: range=-1.0~1.0; h: range=0~H ]

def simple_nms(scores, nms_radius: int):
    """ Fast Non-maximum suppression to remove nearby points """
    assert (nms_radius >= 0)

    def max_pool(x):
        return torch.nn.functional.max_pool2d(
            x, kernel_size=nms_radius * 2 + 1, stride=1, padding=nms_radius)

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)

    for _ in range(2):
        supp_mask = max_pool(max_mask.float()) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool(supp_scores)
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, scores, zeros)


def detect_keypoints(scores_map, normalized_coordinates=True):

    # parameters
    radius = 2
    top_k = -1
    scores_th = 0.1
    n_limit = 200000
    temperature = 0.1
    kernel_size = 2 * radius + 1
    unfold = nn.Unfold(kernel_size=kernel_size, padding=radius)
    # local xy grid
    x = torch.linspace(-radius, radius, kernel_size)
    # (kernel_size*kernel_size) x 2 : (w,h)
    hw_grid = torch.stack(torch.meshgrid([x, x])).view(2, -1).t()[:, [1, 0]]

    b, c, h, w = scores_map.shape
    scores_nograd = scores_map.detach()
    nms_scores = simple_nms(scores_nograd, 2)

    # remove border
    nms_scores[:, :, :radius + 1, :] = 0
    nms_scores[:, :, :, :radius + 1] = 0
    nms_scores[:, :, h - radius:, :] = 0
    nms_scores[:, :, :, w - radius:] = 0

    # detect keypoints without grad
    if top_k > 0:
        topk = torch.topk(nms_scores.view(b, -1), top_k)
        indices_keypoints = topk.indices  # B x top_k
    else:
        if scores_th > 0:
            masks = nms_scores > scores_th
            if masks.sum() == 0:
                th = scores_nograd.reshape(b, -1).mean(dim=1)  # th = self.scores_th
                masks = nms_scores > th.reshape(b, 1, 1, 1)
        else:
            th = scores_nograd.reshape(b, -1).mean(dim=1)  # th = self.scores_th
            masks = nms_scores > th.reshape(b, 1, 1, 1)
        masks = masks.reshape(b, -1)

        indices_keypoints = []  # list, B x (any size)
        scores_view = scores_nograd.reshape(b, -1)
        for mask, scores in zip(masks, scores_view):
            indices = mask.nonzero(as_tuple=False)[:, 0]
            if len(indices) > n_limit:
                kpts_sc = scores[indices]
                sort_idx = kpts_sc.sort(descending=True)[1]
                sel_idx = sort_idx[:n_limit]
                indices = indices[sel_idx]
            indices_keypoints.append(indices)

    # detect soft keypoints with grad backpropagation
    patches = unfold(scores_map)  # B x (kernel**2) x (H*W)
    hw_grid = hw_grid.to(patches)  # to device
    keypoints = []
    scoredispersitys = []
    kptscores = []
    for b_idx in range(b):
        patch = patches[b_idx].t()  # (H*W) x (kernel**2)
        indices_kpt = indices_keypoints[b_idx]  # one dimension vector, say its size is M
        patch_scores = patch[indices_kpt]  # M x (kernel**2)

        # max is detached to prevent undesired backprop loops in the graph
        # @TODO 从patch_scores中找到最大值的位置， 保存到 xy_residual 中
        # 1. 计算patch_scores中的最大值
        max_v = patch_scores.max(dim=1).values.detach()[:,None]
        # 2. 计算patch_scores中的softmax值
        x_exp = ((patch_scores - max_v)/temperature).exp()
        # 3. 按照公式计算坐标偏移
        # \frac{ \sum{(i,j) \times \exp(x/T)} }{ \sum{\exp(x/T)} }
        xy_residual = x_exp @ hw_grid/x_exp.sum(dim=1)[:,None]


        # compute result keypoints
        # 计算结果为 NMS结果的坐标 + 偏移
        keypoints_xy_nms = torch.stack([indices_kpt % w, indices_kpt // w], dim=1)  # Mx2
        keypoints_xy = keypoints_xy_nms + xy_residual

        hw_grid_dist2 = torch.norm((hw_grid[None, :, :] - xy_residual[:, None, :]) / radius,
                                   dim=-1) ** 2
        scoredispersity = (x_exp * hw_grid_dist2).sum(dim=1) / x_exp.sum(dim=1)

        if normalized_coordinates:
            keypoints_xy = keypoints_xy / keypoints_xy.new_tensor([w - 1, h - 1]) * 2 - 1  # (w,h) -> (-1~1,-1~1)

        kptscore = torch.nn.functional.grid_sample(scores_map[b_idx].unsqueeze(0), keypoints_xy.view(1, 1, -1, 2),
                                                       mode='bilinear', align_corners=True)[0, 0, 0, :]  # CxN

        keypoints.append(keypoints_xy)
        scoredispersitys.append(scoredispersity)
        kptscores.append(kptscore)

    return keypoints, scoredispersitys, kptscores


if __name__ == '__main__':
    # test detect_keypoints
    # scores_map = torch.randn(1, 1, 10, 10, requires_grad=True)
    # 主动设置一些值，判断是否能提取局部极大值
    scores_map = torch.tensor([[[[0.1, 0.2, 0.3, 0.4, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
                                 [0.2, 0.3, 0.4, 0.5, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
                                 [0.3, 0.4, 0.5, 0.6, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2],
                                 [0.4, 0.5, 0.6, 0.7, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3],
                                 [0.5, 0.6, 0.7, 0.8, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
                                 [0.4, 0.5, 0.6, 0.7, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3],
                                 [0.3, 0.4, 0.5, 0.6, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2],
                                 [0.2, 0.3, 0.4, 0.5, 0.6, 0.5, 0.4, 0.9, 0.2, 0.1],
                                 [0.1, 0.2, 0.3, 0.4, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
                                 [0.0, 0.1, 0.2, 0.3, 0.4, 0.3, 0.2, 0.1, 0.0, 0.0]]]], requires_grad=True)

    keypoints, scoredispersitys, kptscores = detect_keypoints(scores_map, normalized_coordinates=False)
    print(keypoints[0].requires_grad) # 检测是否有梯度
    print(keypoints[0])
    # 参考结果
    #  True
    #  tensor([[4.0000, 4.0000],
    #         [5.5481, 4.0000],
    #         [4.0000, 5.5481],
    #         [6.6358, 6.6358]])


