import torch
import numpy as np
import cv2
from utils.projection import warp
from utils.extracter import detection
from utils.visualization import plot_kps_error, write_txt, plot_matches


def mutual_argmax(value, mask=None, as_tuple=True):
    """
    Args:
        value: MxN
        mask:  MxN

    Returns:

    """
    value = value - value.min()  # convert to non-negative tensor
    if mask is not None:
        value = value * mask

    max0 = value.max(dim=1, keepdim=True)  # the col index the max value in each row
    max1 = value.max(dim=0, keepdim=True)
    # print('max0', max0)

    valid_max0 = value == max0[0]
    valid_max1 = value == max1[0]
    # print('valid_max0', valid_max0)


    mutual = valid_max0 * valid_max1
    # print('mutual', mutual)
    if mask is not None:
        mutual = mutual * mask

    return mutual.nonzero(as_tuple=as_tuple)


def mutual_argmin(value, mask=None):
    return mutual_argmax(-value, mask)


def compute_keypoints_distance(kpts0, kpts1, p=2):
    """
    Args:
        kpts0: torch.tensor [M,2]
        kpts1: torch.tensor [N,2]
        p: (int, float, inf, -inf, 'fro', 'nuc', optional): the order of norm

    Returns:
        dist, torch.tensor [N,M]
    """
    dist = kpts0[:, None, :] - kpts1[None, :, :]  # [M,N,2]
    dist = torch.norm(dist, p=p, dim=2)  # [M,N]
    return dist

# def val_key_points(kps0, kps1, desc0, desc1, warp01, warp10, th: int = 3):
#     num_feat = min(kps0.shape[0], kps1.shape[0])
#
#     # ==================================== covisible keypoints
#     # 调用warp函数，将图像0中的特征点投影到图像1中
#     kps0_cov, kps01_cov, _, _ = warp(kps0, warp01)
#     kps1_cov, kps10_cov, _, _ = warp(kps1, warp10)
#     num_cov_feat = (len(kps0_cov) + len(kps1_cov)) / 2  # number of covisible keypoints
#     if kps0_cov.shape[0] == 0 or kps1_cov.shape[0] == 0:
#         return {
#             'num_feat': 0,
#             'repeatability': 0,
#             'mean_error': 0,
#             'errors': None,
#         }
#     # ==================================== get gt matching keypoints
#     # 计算图像0中特征和图像1投影到图像0中的特征的距离
#     dist01 = compute_keypoints_distance(kps0_cov, kps10_cov)
#     # 计算图像1中特征和图像0投影到图像1中的特征的距离
#     dist10 = compute_keypoints_distance(kps1_cov, kps01_cov)
#     # TODO: 重复率计算
#     # 1. 对于图像0中的特征p,首先投影到图像1中，得到和其他所有特征的距禼 d01
#     # 2. 图像1中所有特征又投影回到图像0中，计算所有特征和p的距离 d10
#     # 3. 因此，图像0中特征p的距离为，d10 和 d01 的平均值
#     # 4. 正确调用 mutual_argmin 函数，得到最小距离的索引
#     # 5. 得到最小距离
#     # 6. 注意距离是[0-1]之间的值，需要乘以 warp01['resize'] 进行缩放
#     # 7. 计算重复率，即小于阈值的特征点的数量
#
#     dist = (dist01+dist10.t())/2
#     if 'resize' in warp01:
#         dist = dist * warp01['resize']
#     else:
#         dist = dist * warp01['width']
#     dist_mask = dist < th
#     gt_num = dist_mask.sum().cpu()
#     mean_error = dist[dist_mask].cpu().numpy().min()
#     errors = torch.min(dist, axis = 1)[0]
#
#
#     matches_est = mutual_argmax(desc0 @ desc1.t())
#
#     mkpts0, mkpts1 = kps0[matches_est[0]], kps1[matches_est[1]]
#
#     # 匹配得分（Match Score) = 正确匹配的特征 / 所有能够被匹配的特征数量
#     # print(mkpts0.shape,mkpts1)
#
#
#
#     # 返回的一些关联变量
#     # gt_num 正确关联点的数量
#     # mean_error 正确关联点的平均距离
#     # errors 图像0 中每一个特征的对应的最小距离
#
#     return {
#         'num_feat': num_feat,
#         'repeatability': gt_num / num_feat, # 重复率
#         'mean_error': mean_error,
#         'errors': errors,
#         'Match_Score0':  len(matches_est[0])/num_feat,
#         'Match_Score1':  len(matches_est[1])/num_feat,
#     }
def val_key_points(kps0, kps1, desc0, desc1, warp01, warp10, th: int = 3):
    # print('kps0',kps0.shape)
    num_feat = min(kps0.shape[0], kps1.shape[0])
    # print('num_feat',num_feat)
    # print('desc1', desc1.shape)
    # match
    matches_est = mutual_argmax(desc0.t() @ desc1)
    # print('descs_matrix', (desc0.t() @ desc1))

    mkpts0, mkpts1 = kps0[matches_est[0]], kps1[matches_est[1]]
    # print('matches_est', matches_est[0].shape, matches_est[1].shape)
    # print('mkpts0', mkpts0.shape)
    # print('warp01', warp01)

    # kps0 torch.Size([500, 3])
    # num_feat 500
    # desc1 torch.Size([128, 500])
    # descs_matrix tensor([[0.4931, 0.4998, 0.6273,  ..., 0.2888, 0.3098, 0.4671],
    #         [0.5260, 0.4570, 0.4427,  ..., 0.3211, 0.3948, 0.3662],
    #         [0.5054, 0.6129, 0.6105,  ..., 0.4712, 0.5549, 0.6358],
    #         ...,
    #         [0.3082, 0.5706, 0.4637,  ..., 0.5227, 0.3251, 0.4641],
    #         [0.4190, 0.6856, 0.5675,  ..., 0.5795, 0.4895, 0.7351],
    #         [0.3350, 0.6227, 0.4262,  ..., 0.4250, 0.4591, 0.5272]],
    #        device='cuda:0')
    # matches_est torch.Size([355]) torch.Size([355])
    # mkpts0 torch.Size([355, 3])
    # warp01 {'mode': 'homo', 'width': tensor(800, device='cuda:0'), 'height': tensor(600, device='cuda:0'), 'homography_matrix': tensor([[1., 0., 0.],
    #         [0., 1., 0.],
    #         [0., 0., 1.]], device='cuda:0'), 'resize': tensor(512, device='cuda:0')}
    # mkpts0 torch.Size([355, 2])
    # mkpts01 torch.Size([355, 2])
    # ids0 torch.Size([355])
    # dist torch.Size([355])
    # num_inlier torch.Size([])
    # kps0_cov torch.Size([500, 2])
    # kps01_cov torch.Size([500, 2])
    # dist01 torch.Size([500, 500])
    # dist10 torch.Size([500, 500])

    mkpts0, mkpts01, ids0, _ = warp(mkpts0, warp01)
    # print('mkpts0', mkpts0.shape)
    # print('mkpts01', mkpts01.shape)
    # print('ids0', ids0.shape)

    dist = torch.sqrt(((mkpts01 - mkpts1[ids0, :2]) ** 2).sum(axis=1))
    # print('dist', dist.shape)

    if dist.shape[0] == 0:
        dist = dist.new_tensor([float('inf')])
    if 'resize' in warp01:
        dist = dist * warp01['resize']
    else:
        dist = dist * warp01['width']
    num_inlier = (dist <= th).sum().cpu()
    # print('num_inlier', num_inlier.shape)

    # ==================================== covisible keypoints
    kps0_cov, kps01_cov, _, _ = warp(kps0, warp01)
    kps1_cov, kps10_cov, _, _ = warp(kps1, warp10)

    # print('kps0_cov', kps0_cov.shape)
    # print('kps01_cov', kps01_cov.shape)

    num_cov_feat = (len(kps0_cov) + len(kps1_cov)) / 2  # number of covisible keypoints
    if kps0_cov.shape[0] == 0 or kps1_cov.shape[0] == 0:
        return {
            'num_feat': 0,
            'repeatability': 0,
            'matching_score': 0,
            'mean_error': 0,
            'errors': None,
            'matches_est': None,
        }
    # ==================================== get gt matching keypoints
    dist01 = compute_keypoints_distance(kps0_cov, kps10_cov)
    dist10 = compute_keypoints_distance(kps1_cov, kps01_cov)
    # print('dist01', dist01.shape)
    # print('dist10', dist10.shape)

    dist_mutual = (dist01 + dist10.t()) / 2.
    imutual = torch.arange(min(dist_mutual.shape), device=dist_mutual.device)
    dist_mutual[imutual, imutual] = 99999  # mask out diagonal
    mutual_min_indices = mutual_argmin(dist_mutual)
    dist = dist_mutual[mutual_min_indices]
    if 'resize' in warp01:
        dist = dist * warp01['resize']
        dist_mutual = dist_mutual * warp10['resize']
    else:
        dist = dist * warp01['width']
        dist_mutual = dist_mutual * warp10['width']
    gt_num = (dist <= th).sum().cpu()  # number of gt matching keypoints
    error = dist[dist <= th].cpu().numpy()
    mean_error = error.mean()
    errors = torch.min(dist_mutual, dim=1)[0]

    return {
        'num_feat': num_feat,
        'repeatability': gt_num / num_feat,
        'matching_score': num_inlier / max(num_cov_feat, 1),
        'mean_error': mean_error,
        'errors': errors,
        'matches_est': matches_est,
    }


# def repeatability(idx, img_0, score_map_0, desc_map_0, img_1, score_map_1, desc_map_1, warp01, warp10, params):
#     """
#     Args:
#         idx: int
#         img_0: torch.tensor [H,W,3]
#         score_map_0: torch.tensor [H,W]
#         img_1: torch.tensor [H,W,3]
#         score_map_1: torch.tensor [H,W]
#         warp01: dict
#         warp10: dict
#         params: int
#     Returns:
#         dict
#     """
#     # print(score_map_0.shape,desc_map_0.shape)
#     # 1. detection
#     kps0 = detection(score_map_0, params['extractor_params'])
#     kps1 = detection(score_map_1, params['extractor_params'])
#
#     # print(kps0,kps1)
#
#
#     kps0_ = kps0[:, :2] * 2 - 1
#     kps1_ = kps1[:, :2] * 2 - 1
#     desc0 = torch.nn.functional.grid_sample(desc_map_0, kps0_.unsqueeze(0).unsqueeze(0),
#                                             mode='bilinear', align_corners=True)[0, :, 0, :]  # CxN
#     desc0 = torch.nn.functional.normalize(desc0, p=2, dim=0)
#     desc1 = torch.nn.functional.grid_sample(desc_map_1, kps1_.unsqueeze(0).unsqueeze(0),
#                                             mode='bilinear', align_corners=True)[0, :, 0, :]  # CxN
#     desc1 = torch.nn.functional.normalize(desc1, p=2, dim=0)
#
#     # 2. validation
#     result = val_key_points(kps0, kps1, desc0, desc1, warp01, warp10, th=params['repeatability_params']['th'])
#
#
#
#     # 2. validation
#     result = val_key_points(kps0, kps1, desc0, desc1, warp01, warp10, th=params['repeatability_params']['th'])
#     # 3. save image
#     show = plot_kps_error(img_0, kps0, result['errors'], params['repeatability_params']['image'])
#     root = params['repeatability_params']['output']
#     cv2.imwrite(root + str(idx) + '_repeatability_0.png', show)
#     show = plot_kps_error(img_1, kps1, None, params['repeatability_params']['image'])
#     cv2.imwrite(root + str(idx) + '_repeatability_1.png', show)
#     return result

def repeatability(idx, img_0, score_map_0, img_1, score_map_1, desc_map_0, desc_map_1, warp01, warp10, params):
    """
    Args:
        idx: int
        img_0: torch.tensor [H,W,3]
        score_map_0: torch.tensor [H,W]
        img_1: torch.tensor [H,W,3]
        score_map_1: torch.tensor [H,W]
        warp01: dict
        warp10: dict
        params: int
    Returns:
        dict
    """

    # 1. detection
    kps0 = detection(score_map_0, params['extractor_params'])
    kps1 = detection(score_map_1, params['extractor_params'])

    kps0_ = kps0[:, :2] * 2 - 1
    kps1_ = kps1[:, :2] * 2 - 1
    desc0 = torch.nn.functional.grid_sample(desc_map_0, kps0_.unsqueeze(0).unsqueeze(0),
                                            mode='bilinear', align_corners=True)[0, :, 0, :]  # CxN
    desc0 = torch.nn.functional.normalize(desc0, p=2, dim=0)
    desc1 = torch.nn.functional.grid_sample(desc_map_1, kps1_.unsqueeze(0).unsqueeze(0),
                                            mode='bilinear', align_corners=True)[0, :, 0, :]  # CxN
    desc1 = torch.nn.functional.normalize(desc1, p=2, dim=0)

    # 2. validation
    result = val_key_points(kps0, kps1, desc0, desc1, warp01, warp10, th=params['repeatability_params']['th'])
    # 3. save image
    show = plot_kps_error(img_0, kps0, result['errors'], params['repeatability_params']['image'])
    root = params['repeatability_params']['output']
    cv2.imwrite(root + str(idx) + '_repeatability_0.png', show)
    show = plot_kps_error(img_1, kps1, None, params['repeatability_params']['image'])
    cv2.imwrite(root + str(idx) + '_repeatability_1.png', show)
    # add
    _, _, h, w = img_0.shape
    kps0_hw = kps0[result['matches_est'][0], 0:2] * torch.tensor([w - 1, h - 1]).to(kps0.device)
    kps1_hw = kps1[result['matches_est'][1], 0:2] * torch.tensor([w - 1, h - 1]).to(kps0.device)
    show = plot_matches(img_0, img_1, kps0_hw, kps1_hw)
    cv2.imwrite(root + str(idx) + '_matches.png', show)
    return result


def plot_repeatability(repeatability, save_path):
    import matplotlib.pyplot as plt
    plt.plot(repeatability)
    plt.savefig(save_path)
    plt.close()
    write_txt(save_path.replace('.png', '.txt'), repeatability)
