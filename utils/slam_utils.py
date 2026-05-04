from time import sleep
import os
import torch
import torch.nn.functional as F
import cv2
from torchvision.transforms import GaussianBlur
from torchmetrics.functional import structural_similarity_index_measure as ssim
import warnings
from datetime import datetime
from utils.polar_loss import AzimuthLoss
from utils.normal_loss import NormalConsistencyLoss
from gaussian_splatting.utils.loss_utils import l1_loss, ssim
import numpy as np

# 可视化用于debug
import torchvision.utils as vutils
import time
import matplotlib.pyplot as plt

# sigma越小边缘权重衰减得更快
def create_center_weight_mask(shape, sigma=0.2):
    """Create a weight mask that gives higher weights to the center region.

    Args:
        shape: Tuple of (H, W) or (B, C, H, W)
        sigma: Controls how quickly the weight drops off from the center

    Returns:
        torch.Tensor: Weight mask with the same spatial dimensions as input
    """
    H, W = shape

    center_y = (H - 1) / 2
    center_x = (W - 1) / 2

    # Create coordinate grids
    y_coords = torch.arange(H, dtype=torch.float32)
    x_coords = torch.arange(W, dtype=torch.float32)
    y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')

    # Calculate squared distances from center
    squared_dist = (y_grid - center_y) ** 2 + (x_grid - center_x) ** 2
    # Normalize distances
    max_dist = (center_y ** 2 + center_x ** 2)
    normalized_dist = squared_dist / max_dist

    # Create gaussian weight mask
    weight_mask = torch.exp(-normalized_dist / (2 * sigma ** 2))

    # Ensure minimum weight is 0.1 (10% of maximum weight)
    weight_mask = 0.1 + 0.8 * weight_mask

    if len(shape) == 4:
        weight_mask = weight_mask.view(1, 1, H, W).expand(shape[0], shape[1], -1, -1)

    return weight_mask

# dolp的值越大，权重越大，且权重范围在[0, 1]之间波动
def create_polar_weight_mask(gt_dolp, weight_dolp_mask):
    """Create a weight mask based on the DoLP values.

当gt_dolp < 0.2时，返回值为1（torch.ones_like(gt_dolp)）
当gt_dolp >= 0.2时，返回值为1 + 2 * gt_dolp
保留了原来的weight_dolp_mask参数功能，用于在统一权重（1）和自定义权重方案之间进行插值
使用示例：

当weight_dolp_mask = 1时，完全使用新的掩膜计算方案
当weight_dolp_mask = 0时，所有区域权重都为1
当weight_dolp_mask在0到1之间时，会在统一权重和新的掩膜方案之间进行线性插值

    Args:
        gt_dolp: Ground truth DoLP tensor
        weight_dolp_mask: Weight factor for DoLP loss (0 to 1)
            Controls how much of the custom weighting scheme is applied:
            - when weight_dolp_mask = 0: all regions have weight 1
            - when weight_dolp_mask = 1: full custom weighting is applied
            - intermediate values blend between uniform and custom weighting

    Returns:
        torch.Tensor: Weight mask with the same shape as gt_dolp
    """
    # Create base mask where:
    # - regions with DoLP < 0.2 have weight 1
    # - regions with DoLP >= 0.2 have weight (1 + 2*DoLP)
    base_mask = torch.where(gt_dolp < 0.2,
                            torch.ones_like(gt_dolp),
                            1 + 2 * gt_dolp)

    # Blend between uniform weighting (weight_dolp_mask=0)
    # and custom weighting (weight_dolp_mask=1)
    return (1 - weight_dolp_mask) + weight_dolp_mask * base_mask


# dolp的值越大，权重越小，且权重范围只在[0.5, 1]之间波动
def create_inverse_polar_weight_mask(gt_dolp, weight_dolp_mask):
    """Create an inverse weight mask based on the DoLP values.
    Weight will be higher for low DoLP values and lower for high DoLP values.
    Weights are constrained to be between 0.5 and 1.0

    Args:
        gt_dolp: Ground truth DoLP tensor (values between 0 and 1)
        weight_dolp_mask: Weight factor for DoLP loss (0 to 1)
            - when weight_dolp_mask = 0: all regions have weight 1
            - when weight_dolp_mask = 1: weights range from 0.5 to 1 based on inverse of DoLP values

    Returns:
        torch.Tensor: Weight mask with the same shape as gt_dolp
    """
    # Create inverse weights (1 - gt_dolp) and scale them to [0.5, 1] range
    inverse_weights = 1 - gt_dolp  # Now high DoLP has low weight
    scaled_weights = 0.5 + 0.5 * inverse_weights  # Scale to [0.5, 1] range

    # Interpolate between uniform weighting (1.0) and scaled weights based on weight_dolp_mask
    return 1.0 * (1 - weight_dolp_mask) + scaled_weights * weight_dolp_mask

# Define Laplacian kernel
laplacian_kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32, device="cuda").view(1, 1, 3, 3)


def laplacian_filter(image, kernel_size=3, sigma=0.5):
    # Add batch and channel dimensions
    image = image.unsqueeze(0)

    # Apply Gaussian filter using torchvision
    gaussian_blur = GaussianBlur(kernel_size=kernel_size, sigma=sigma)
    image = gaussian_blur(image)

    # Apply Laplacian filter
    edges = F.conv2d(image, laplacian_kernel)
    return edges.squeeze()

def image_gradient(image):
    # Compute image gradient using Scharr Filter
    c = image.shape[0]
    conv_y = torch.tensor(
        [[3, 0, -3], [10, 0, -10], [3, 0, -3]], dtype=torch.float32, device="cuda"
    )
    conv_x = torch.tensor(
        [[3, 10, 3], [0, 0, 0], [-3, -10, -3]], dtype=torch.float32, device="cuda"
    )
    normalizer = 1.0 / torch.abs(conv_y).sum()
    p_img = torch.nn.functional.pad(image, (1, 1, 1, 1), mode="reflect")[None]
    img_grad_v = normalizer * torch.nn.functional.conv2d(
        p_img, conv_x.view(1, 1, 3, 3).repeat(c, 1, 1, 1), groups=c
    )
    img_grad_h = normalizer * torch.nn.functional.conv2d(
        p_img, conv_y.view(1, 1, 3, 3).repeat(c, 1, 1, 1), groups=c
    )
    return img_grad_v[0], img_grad_h[0]


def image_gradient_mask(image, eps=0.01):
    # Compute image gradient mask
    c = image.shape[0]
    conv_y = torch.ones((1, 1, 3, 3), dtype=torch.float32, device="cuda")
    conv_x = torch.ones((1, 1, 3, 3), dtype=torch.float32, device="cuda")
    p_img = torch.nn.functional.pad(image, (1, 1, 1, 1), mode="reflect")[None]
    p_img = torch.abs(p_img) > eps
    img_grad_v = torch.nn.functional.conv2d(
        p_img.float(), conv_x.repeat(c, 1, 1, 1), groups=c
    )
    img_grad_h = torch.nn.functional.conv2d(
        p_img.float(), conv_y.repeat(c, 1, 1, 1), groups=c
    )

    return img_grad_v[0] == torch.sum(conv_x), img_grad_h[0] == torch.sum(conv_y)


def depth_reg(depth, gt_image, huber_eps=0.1, mask=None):
    mask_v, mask_h = image_gradient_mask(depth)
    gray_grad_v, gray_grad_h = image_gradient(gt_image.mean(dim=0, keepdim=True))
    depth_grad_v, depth_grad_h = image_gradient(depth)
    gray_grad_v, gray_grad_h = gray_grad_v[mask_v], gray_grad_h[mask_h]
    depth_grad_v, depth_grad_h = depth_grad_v[mask_v], depth_grad_h[mask_h]

    w_h = torch.exp(-10 * gray_grad_h**2)
    w_v = torch.exp(-10 * gray_grad_v**2)
    err = (w_h * torch.abs(depth_grad_h)).mean() + (
        w_v * torch.abs(depth_grad_v)
    ).mean()
    return err


# def get_loss_tracking(config, image, depth, opacity, viewpoint, initialization=False):
#     image_ab = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b
#     if config["Training"]["monocular"]:
#         return get_loss_tracking_rgb(config, image_ab, depth, opacity, viewpoint)
#     return get_loss_tracking_rgbd(config, image_ab, depth, opacity, viewpoint)
#
#
# def get_loss_tracking_rgb(config, image, depth, opacity, viewpoint):
#     weight_edge_loss = config["Training"]["weight_edge_loss"] if "weight_edge_loss" in config["Training"] else 0.5
#     gt_image = viewpoint.original_image.cuda()
#     l1 = opacity * torch.abs(image * viewpoint.rgb_pixel_mask -
#                              gt_image * viewpoint.rgb_pixel_mask)
#     # if config["Training"].get("use_edge_loss", False):
#     #     edges_image = laplacian_filter(image.mean(dim=0, keepdim=True) * viewpoint.rgb_pixel_mask)
#     #     edges_gt_image = laplacian_filter(gt_image.mean(dim=0, keepdim=True) * viewpoint.rgb_pixel_mask)
#     #     edge_loss = ssim_edge_loss(edges_image, edges_gt_image)
#     #     return l1.mean() + weight_edge_loss * edge_loss
#     return l1.mean()
#
#
# def get_loss_tracking_rgbd(
#     config, image, depth, opacity, viewpoint, initialization=False
# ):
#     alpha = config["Training"]["alpha"] if "alpha" in config["Training"] else 0.95
#     depth_pixel_mask = (viewpoint.gt_depth > 0.01).view(*depth.shape)
#     opacity_mask = (opacity > 0.95).view(*depth.shape)
#
#     l1_rgb = get_loss_tracking_rgb(config, image, depth, opacity, viewpoint)
#     depth_mask = depth_pixel_mask * opacity_mask
#     l1_depth = torch.abs(depth * depth_mask - viewpoint.gt_depth * depth_mask)
#     return alpha * l1_rgb + (1 - alpha) * l1_depth.mean()
#
#
# def get_loss_mapping(config, image, depth, viewpoint, opacity, initialization=False):
#     if initialization:
#         image_ab = image
#     else:
#         image_ab = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b
#     if config["Training"]["monocular"]:
#         return get_loss_mapping_rgb(config, image_ab, depth, viewpoint)
#     if config["Training"].get("use_edge_loss", False):  # 有一个默认值
#         return get_loss_mapping_rgbd_edge(config, image_ab, depth, viewpoint)
#     return get_loss_mapping_rgbd(config, image_ab, depth, viewpoint)
#
#
# def get_loss_mapping_rgb(config, image, depth, viewpoint):
#     gt_image = viewpoint.original_image.cuda()
#     l1_rgb = torch.abs(image * viewpoint.rgb_pixel_mask_mapping - gt_image * viewpoint.rgb_pixel_mask_mapping)
#
#     return l1_rgb.mean()
#
#
# def get_loss_mapping_rgbd(config, image, depth, viewpoint, initialization=False):
#     alpha = config["Training"]["alpha"] if "alpha" in config["Training"] else 0.95
#     gt_image = viewpoint.original_image.cuda()
#
#     rgb_pixel_mask = viewpoint.rgb_pixel_mask_mapping
#     depth_pixel_mask = (viewpoint.gt_depth > 0.01).view(*depth.shape)
#
#     l1_rgb = torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)
#     l1_depth = torch.abs(depth * depth_pixel_mask - viewpoint.gt_depth * depth_pixel_mask)
#
#     return alpha * l1_rgb.mean() + (1 - alpha) * l1_depth.mean()

# 下面都是引入了normal的loss计算，代码能自动判断输入的类型，如果是False或者True则当作initialization处理，否则当作normal处理

# def get_normal_loss(pred_normal, gt_normal, mask):
#     # 使用cosine similarity直接计算差异时，可能会忽略法线方向的问题（因为n和 - n的余弦相似度是 - 1）
#     # 简单的L1loss可能不够平滑
#     # 1. 确保法线是单位向量
#     pred_normal = F.normalize(pred_normal, dim=0)  # 在通道维度上归一化
#     gt_normal = F.normalize(gt_normal, dim=0)
#
#     # 2. 计算角度差异（弧度）
#     # 将normal转换为 [3, N] 形状，其中N = H*W
#     H, W = pred_normal.shape[1:]
#     pred_normal = pred_normal.view(3, -1)  # [3, H*W]
#     gt_normal = gt_normal.view(3, -1)  # [3, H*W]
#
#     # 计算每个位置的点积
#     dot_product = torch.sum(pred_normal * gt_normal, dim=0)  # [H*W]
#     dot_product = torch.clamp(dot_product, -1.0, 1.0)
#
#     # 计算角度误差
#     angular_error = torch.acos(dot_product)  # [H*W]
#     reversed_angular_error = torch.acos(-dot_product)
#     angular_error = torch.minimum(angular_error, reversed_angular_error)
#
#     # 重塑回原始空间维度
#     angular_error = angular_error.view(H, W)  # [H, W]
#
#     # 应用mask
#     mask = mask.squeeze(0)  # [H, W]
#     masked_loss = angular_error[mask].mean()
#
#     return masked_loss

# 上面的法线loss开销可能比较大，下面的更加简单
def get_normal_loss(pred_normal, gt_normal, mask):
    """
    改进的法线损失计算
    Args:
        pred_normal: [3, H, W]
        gt_normal: [3, H, W]
        mask: [1, H, W]
    """
    # 1. L2 归一化
    pred_normal = F.normalize(pred_normal, dim=0, eps=1e-8)
    gt_normal = F.normalize(gt_normal, dim=0, eps=1e-8)

    if not torch.any(mask):
        return torch.tensor(0.0, device=pred_normal.device)

    # 2. 方案一：使用 1 - |cos| 作为损失
    dot_product = torch.sum(pred_normal * gt_normal, dim=0)
    loss_cos = 1 - torch.abs(dot_product)

    # 3. 方案二：使用 L2 距离的平方（等价于 2(1-|cos|)）
    # loss_l2 = torch.sum((pred_normal - gt_normal) ** 2, dim=0)
    # loss_l2_reversed = torch.sum((pred_normal + gt_normal) ** 2, dim=0)
    # loss_l2 = torch.minimum(loss_l2, loss_l2_reversed) / 4

    # 4. 应用 mask
    mask = mask.squeeze(0)
    masked_loss = loss_cos[mask].mean()

    return masked_loss

def get_multiscale_normal_loss(pred_normal, gt_normal, mask, scales=[1, 2, 4]):
    total_loss = 0
    for scale in scales:
        # 下采样法线和mask
        if scale > 1:
            pred_scaled = F.avg_pool2d(pred_normal, scale)
            gt_scaled = F.avg_pool2d(gt_normal, scale)
            mask_scaled = F.avg_pool2d(mask.float(), scale) > 0.5
        else:
            pred_scaled, gt_scaled, mask_scaled = pred_normal, gt_normal, mask

        total_loss += get_normal_loss(pred_scaled, gt_scaled, mask_scaled)
    return total_loss / len(scales)


def get_depth_discontinuity_mask(depth, threshold=0.1):
    """
    计算深度不连续区域的mask
    输入depth可以是 [H, W, 1] 或 [1, H, W]
    """
    if depth.size(-1) == 1:  # [H, W, 1]格式
        depth = depth.squeeze(-1)
    elif depth.size(0) == 1:  # [1, H, W]格式
        depth = depth.squeeze(0)

    # 计算水平和垂直方向的深度梯度
    depth_dx = torch.abs(depth[:, 1:] - depth[:, :-1])
    depth_dy = torch.abs(depth[1:, :] - depth[:-1, :])

    # 填充到原始尺寸
    discontinuity_mask_x = F.pad(depth_dx > threshold, (0, 1), 'replicate')
    discontinuity_mask_y = F.pad(depth_dy > threshold, (0, 0, 0, 1), 'replicate')

    return ~(discontinuity_mask_x | discontinuity_mask_y)  # 返回连续区域的mask

def get_loss_tracking(config, image, depth, opacity, viewpoint, normal):
    image_ab = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b
    if config["Dataset"]["sensor_type"] == 'monocular':
        return get_loss_tracking_rgb(config, image_ab, depth, opacity, viewpoint, normal=normal)
    return get_loss_tracking_rgbd(config, image_ab, depth, opacity, viewpoint, normal=normal)

def get_loss_tracking_rgb(config, image, depth, opacity, viewpoint, **kwargs):
    weight_normal = config["Training"].get("weight_normal_loss", 0.5)
    weight_dolp_mask = config["Training"].get("weight_dolp_mask", 0.8)
    use_normal_loss = config.get("use_normal_loss", False)
    use_dolp_mask = config.get("use_dolp_mask", False)
    normal = kwargs.get('normal', None)

    gt_image = viewpoint.original_image.cuda()
    # 排除边缘和纯黑背景的梯度
    l1 = opacity * torch.abs(image * viewpoint.rgb_pixel_mask - gt_image * viewpoint.rgb_pixel_mask)
    # l1 = opacity * torch.abs(image - gt_image)
    # if config.get("use_edge_loss", False):
    #     # 用于绿线标记，现在没什么用了
    #     # l1 loss without mask
    #     l1 = opacity * torch.abs(image - gt_image)
    #     # # 创建边缘掩码（G通道为1的区域）
    #     edge_mask = (gt_image[1, :, :] == 1).float()
    #     # 增加边缘区域的L1权重
    #     l1 = l1 * (1 + 9 * edge_mask)

    combined_loss = l1
    # save_weights_once(viewpoint.rgb_pixel_mask)
    return combined_loss.mean()

def get_loss_tracking_rgb_azimuth(config, image, depth, opacity, viewpoint, **kwargs):
    weight_normal = config["Training"].get("weight_normal_loss", 0.5)
    weight_dolp_mask = config["Training"].get("weight_dolp_mask", 0.8)
    use_normal_loss = config.get("use_normal_loss", False)
    use_dolp_mask = config.get("use_dolp_mask", False)
    normal = kwargs.get('normal', None)

    gt_image = viewpoint.original_image.cuda()
    l1 = opacity * torch.abs(image * viewpoint.rgb_pixel_mask - gt_image * viewpoint.rgb_pixel_mask)

    # 去掉了mask
    # Create edge_mask where the G channel of gt_image is 1
    edge_mask = (gt_image[1, :, :] == 1).float()

    # l1 loss without mask
    l1 = opacity * torch.abs(image - gt_image)

    # Increase l1 weight for regions marked by edge_mask
    l1 = l1 * (1 + 9 * edge_mask)

    combined_loss = l1



    # Handle polar loss and normal loss with their combinations
    if use_dolp_mask and hasattr(viewpoint, 'gt_dolp'):
        gt_dolp = viewpoint.gt_dolp.cuda()
        if not (0 <= gt_dolp.min() <= gt_dolp.max() <= 1):
            warnings.warn("gt_dolp values should be in range [0, 1]")
        polar_weights = create_polar_weight_mask(gt_dolp, weight_dolp_mask)

        # Apply polar weights to combined loss
        combined_loss = l1 * polar_weights

        # Handle normal loss with polar weights if both are enabled
        if use_normal_loss:  # 由于没有gt_normal,这里是aolp得到的方位角的loss

            normal = normal.permute(1, 2, 0)  # Change shape from [3, 512, 612] to [512, 612, 3]

            # Convert normal to azimuth angles
            normal_x = normal[..., 0] * 2 - 1  # Using ... to handle any number of leading dimensions
            normal_y = normal[..., 1] * 2 - 1
            normal_azimuth = torch.atan2(normal_y, normal_x)

            # Ensure positive angles (equivalent to modulo operation)
            normal_azimuth = normal_azimuth + 2 * torch.pi
            normal_azimuth = torch.fmod(normal_azimuth, 2 * torch.pi)

            gt_aolp = viewpoint.gt_aolp.cuda()
            # Convert AOLP to radians (assuming it's normalized to [0,1])
            aolp_angle = gt_aolp * torch.pi
            # Add batch dimension if not present
            if normal_azimuth.dim() == 2:
                normal_azimuth = normal_azimuth.unsqueeze(0)
            # Initialize loss function
            loss_fn = AzimuthLoss().cuda()
            # Compute loss
            azimuth_loss = loss_fn(normal_azimuth, aolp_angle, gt_dolp)  # todo:加入normal_pixel_mask
            # Apply polar weights to normal loss as well
            azimuth_loss = azimuth_loss * polar_weights
            combined_loss = combined_loss + weight_normal * azimuth_loss

    # Handle normal loss without polar weights if polar loss is not used
    elif use_normal_loss:
        gt_dolp = viewpoint.gt_dolp.cuda()
        if not (0 <= gt_dolp.min() <= gt_dolp.max() <= 1):
            warnings.warn("gt_dolp values should be in range [0, 1]")
        normal = normal.permute(1, 2, 0)  # Change shape from [3, 512, 612] to [512, 612, 3]
        # Convert normal to azimuth angles
        normal_x = normal[..., 0] * 2 - 1  # Using ... to handle any number of leading dimensions
        normal_y = normal[..., 1] * 2 - 1
        normal_azimuth = torch.atan2(normal_y, normal_x)
        # Ensure positive angles (equivalent to modulo operation)
        normal_azimuth = normal_azimuth + 2 * torch.pi
        normal_azimuth = torch.fmod(normal_azimuth, 2 * torch.pi)

        gt_aolp = viewpoint.gt_aolp.cuda()
        # Convert AOLP to radians (assuming it's normalized to [0,1])
        aolp_angle = gt_aolp * torch.pi
        # Add batch dimension if not present
        if normal_azimuth.dim() == 2:
            normal_azimuth = normal_azimuth.unsqueeze(0)
        # Initialize loss function
        loss_fn = AzimuthLoss().cuda()
        # Compute loss
        azimuth_loss = loss_fn(normal_azimuth, aolp_angle, gt_dolp)  # todo:加入normal_pixel_mask

        combined_loss = l1 + weight_normal * azimuth_loss

    return combined_loss.mean()


def get_loss_tracking_rgbd(config, image, depth, opacity, viewpoint, **kwargs):
    # Get configuration parameters
    alpha = config["Training"].get("alpha", 0.95)
    weight_normal = config["Training"].get("weight_normal_loss", 0.5)
    weight_dolp_mask = config["Training"].get("weight_dolp_mask", 0.8)
    use_normal_loss = config.get("use_normal_loss", False)
    use_dolp_mask = config.get("use_dolp_mask", False)
    normal = kwargs.get('normal', None)
    center_focus = config.get("center_focus", False)

    # Create masks
    depth_pixel_mask = (viewpoint.gt_depth > 0.01).view(*depth.shape)
    opacity_mask = (opacity > 0.95).view(*depth.shape)
    depth_mask = depth_pixel_mask * opacity_mask

    # Calculate base losses
    l1_rgb = get_loss_tracking_rgb(config, image, depth, opacity, viewpoint)
    l1_depth = torch.abs(depth * depth_mask - viewpoint.gt_depth * depth_mask)

    # Initialize combined loss
    combined_loss = alpha * l1_rgb + (1 - alpha) * l1_depth

    # Handle polar loss and normal loss with their combinations
    if use_dolp_mask and hasattr(viewpoint, 'gt_dolp'):
        gt_dolp = viewpoint.gt_dolp.cuda()
        if not (0 <= gt_dolp.min() <= gt_dolp.max() <= 1):
            warnings.warn("gt_dolp values should be in range [0, 1]")

        # Create inverse polar weights (higher weights for low DoLP values)
        polar_weights = create_inverse_polar_weight_mask(gt_dolp, weight_dolp_mask)

        # Apply polar weights to combined loss
        combined_loss = combined_loss * polar_weights

        # Handle normal loss with polar weights if both are enabled
        if use_normal_loss:
            gt_normal = viewpoint.gt_normal.cuda()
            normal_loss = get_normal_loss(normal, gt_normal, depth_mask)
            # Apply polar weights to normal loss as well
            normal_loss = normal_loss * polar_weights
            combined_loss = combined_loss + weight_normal * normal_loss

            # Log losses
            # timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            # with open('loss_tracking.txt', 'a') as f:
            #     f.write(f"[{timestamp}] RGB: {l1_rgb.mean().item():.6f} "
            #             f"Depth: {l1_depth.mean().item():.6f} "
            #             f"Normal: {normal_loss.item():.6f}\n")

    # Handle normal loss without polar weights if polar loss is not used
    elif use_normal_loss:
        gt_normal = viewpoint.gt_normal.cuda()
        normal_loss = get_normal_loss(normal, gt_normal, depth_mask)
        combined_loss = combined_loss + weight_normal * normal_loss

        # Log losses
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        # with open('loss_tracking.txt', 'a') as f:
        #     f.write(f"[{timestamp}] User: sunday6666 - RGB: {l1_rgb.mean().item():.6f} "
        #             f"Depth: {l1_depth.mean().item():.6f} "
        #             f"Normal: {normal_loss.item():.6f}\n")

    return combined_loss.mean()



def get_loss_mapping(config, image, depth, depth_normal, viewpoint, **kwargs):
    initialization = kwargs.get('initialization', False)
    normal = kwargs.get('normal', None)

    if initialization:
        image_ab = image
    else:
        image_ab = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b

    # if config["Training"]["monocular"]:
    if config["Dataset"]["sensor_type"] == 'monocular':
        # return get_loss_mapping_rgb(config, image_ab, depth, viewpoint, normal=normal)
        return get_loss_mapping_rgb(config, image_ab, depth, depth_normal, viewpoint)
    if config.get("use_edge_loss", False):
        return get_loss_mapping_rgbd_edge(config, image_ab, depth, depth_normal, viewpoint, normal=normal)
    return get_loss_mapping_rgbd(config, image_ab, depth, viewpoint, normal=normal)


def compute_flat_loss(scaling, lambda_flatten=1.0, lambda_circular=1.0):
    """计算扁平圆盘形状的loss，同时鼓励在一个方向上扁平，其他方向形成圆形

    Args:
        scaling: [N, 3] 每个高斯的三个方向的缩放参数
        lambda_flatten: 扁平化约束的权重
        lambda_circular: 圆形化约束的权重

    Returns:
        flat_loss: 扁平化损失
        circular_loss: 圆形化损失
        combined_loss: 总损失
    """
    # 找出最小缩放值及其索引
    min_scale, min_indices = torch.min(scaling, dim=1)
    min_scale = torch.clamp(min_scale, 0, 30)

    # 计算扁平化损失
    flatten_loss = torch.abs(min_scale).mean()

    # 为每个高斯找出不是最小的另外两个缩放值
    batch_indices = torch.arange(scaling.shape[0], device=scaling.device)
    mask = torch.ones_like(scaling, dtype=torch.bool)
    mask[batch_indices, min_indices] = False
    other_scales = scaling[mask].reshape(scaling.shape[0], 2)  # [N, 2]

    # 计算另外两个维度的圆形化损失（使用L1距离）
    circular_loss = torch.abs(other_scales[:, 0] - other_scales[:, 1]).mean()

    # 结合两个损失
    combined_loss = lambda_flatten * flatten_loss + lambda_circular * circular_loss

    return combined_loss

def get_loss_mapping_rgb(config, image, depth, depth_normal, viewpoint, **kwargs):
    """
    计算RGB图像和法线约束的损失函数
    Args:
        config: 配置参数
        image: 预测的RGB图像
        depth: 深度图（本函数未使用）
        viewpoint: 视点信息，包含原始图像等
        **kwargs: 额外参数，包含法线信息
    Returns:
        final_loss: 最终的总损失
    """
    center_focus = config.get("center_focus", False)
    gt_image = viewpoint.original_image.cuda()

    # 获取像素掩码
    rgb_pixel_mask = viewpoint.rgb_pixel_mask_mapping
    # save_weights_once(viewpoint.rgb_pixel_mask_mapping)
    # 计算L1损失
    l1_rgb = torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)
    # l1_rgb = torch.abs(image - gt_image)
    if config.get("use_edge_loss", False):
        # 如果有绿线的话
        # # 创建边缘掩码（G通道为1的区域）
        edge_mask = (gt_image[1, :, :] == 1).float()
        # 增加边缘区域的L1权重
        l1_rgb = l1_rgb * (1 + 9 * edge_mask)

    combined_loss = l1_rgb

    # 处理中心聚焦
    if center_focus:
        # 创建中心权重掩码并移动到正确的设备
        center_weights = create_center_weight_mask(rgb_pixel_mask.shape[-2:]).to(rgb_pixel_mask.device)
        # 将中心权重应用到最终的组合损失
        final_loss = (combined_loss * center_weights).mean()
    else:
        final_loss = combined_loss.mean()

    return final_loss


# 使用全局变量跟踪是否已保存图像
_weights_saved = False


def save_weights_once(weights, filename="mask.png"):
    """
    仅保存一次权重图到当前目录
    """
    global _weights_saved

    # 如果已经保存过，直接返回
    if _weights_saved:
        return

    # 确保权重是CPU上的numpy数组
    if torch.is_tensor(weights):
        weights = weights.detach().cpu().numpy()

    # 如果权重是3D张量，取第一个通道
    if len(weights.shape) == 3:
        weights = weights[0]

    # 归一化到0-255范围
    weights_normalized = (weights * 255).astype(np.uint8)

    # 保存图像
    cv2.imwrite(filename, weights_normalized)
    _weights_saved = True
    print(f"Saved weight visualization to: {filename}")

def get_loss_mapping_rgb_normal(config, image, depth, depth_normal, viewpoint, **kwargs):
    """
    计算RGB图像和法线约束的损失函数
    Args:
        config: 配置参数
        image: 预测的RGB图像
        depth: 深度图
        viewpoint: 视点信息，包含原始图像等
        **kwargs: 额外参数，包含法线信息
    Returns:
        final_loss: 最终的总损失
    """
    weight_normal = config["Training"].get("weight_normal_loss", 0.5)
    weight_dolp_mask = config["Training"].get("weight_dolp_mask", 0.8)
    use_normal_loss = config.get("use_normal_loss", False)
    use_azimuth_loss = config.get("use_azimuth_loss", False)
    use_dolp_mask = config.get("use_dolp_mask", False)
    center_focus = config.get("center_focus", False)
    normal = kwargs.get('normal', None)
    gt_image = viewpoint.original_image.cuda()
    gt_aolp = viewpoint.gt_aolp.cuda()
    # 获取像素掩码
    rgb_pixel_mask = viewpoint.rgb_pixel_mask_mapping
    normal_pixel_mask = rgb_pixel_mask

    image = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b

    # 计算L1损失
    l1_rgb = torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)
    # save_weights_once(rgb_pixel_mask)

    # # 创建边缘掩码（G通道为1的区域）
    # edge_mask = (gt_image[1, :, :] == 1).float()
    # # 增加边缘区域的L1权重
    # l1_rgb = l1_rgb * (1 + 9 * edge_mask)

    combined_loss = l1_rgb
    gt_dolp = viewpoint.gt_dolp.cuda()
    if not (0 <= gt_dolp.min() <= gt_dolp.max() <= 1):
        warnings.warn("gt_dolp values should be in range [0, 1]")

    # Handle polar loss and normal loss with their combinations
    if use_dolp_mask and hasattr(viewpoint, 'gt_dolp'):

        polar_weights = create_polar_weight_mask(gt_dolp, weight_dolp_mask)

        # Apply polar weights to combined loss
        combined_loss = l1_rgb * polar_weights
        # 处理法线损失，先不考虑dolp_mask
        if use_normal_loss:
            # 初始化NormalConsistencyLoss
            normal_loss_fn = NormalConsistencyLoss().cuda()

            # 调用提前聚类好的计算法线损失
            gt_seg = viewpoint.gt_seg.cuda() if (hasattr(viewpoint, "gt_seg") and viewpoint.gt_seg is not None) else None
            # if gt_seg is None:
            #     print("该帧无gt_seg")
            total_normal_loss = normal_loss_fn(depth_normal, normal, gt_seg, None, gt_dolp)
            # 将法线损失与RGB损失组合
            combined_loss = l1_rgb + weight_normal * total_normal_loss * polar_weights
    elif use_normal_loss:
        # 初始化NormalConsistencyLoss
        normal_loss_fn = NormalConsistencyLoss().cuda()
        # 调用提前聚类好的计算法线损失
        gt_seg = viewpoint.gt_seg.cuda() if (hasattr(viewpoint, "gt_seg") and viewpoint.gt_seg is not None) else None
        # if gt_seg is None:
        #     print("该帧无gt_seg")
        total_normal_loss = normal_loss_fn(depth_normal, normal, gt_seg, None, gt_dolp)

        # 将法线损失与RGB损失组合
        combined_loss = l1_rgb + weight_normal * total_normal_loss
    # 处理中心聚焦
    if center_focus:
        # 创建中心权重掩码并移动到正确的设备
        center_weights = create_center_weight_mask(rgb_pixel_mask.shape[-2:]).to(rgb_pixel_mask.device)
        # 将中心权重应用到最终的组合损失
        final_loss = (combined_loss * center_weights).mean()
        final_weights = center_weights + polar_weights
        # 保存最终的组合权重图（只会保存一次）
        # save_weights_once(final_weights)
        print("中心权重还是很奇怪，先不用")
    else:
        final_loss = combined_loss.mean()

    return final_loss


def get_loss_mapping_rgb_normal_azimuth(config, image, depth, depth_normal, viewpoint, **kwargs):
    """
    计算RGB图像和法线约束的损失函数
    Args:
        config: 配置参数
        image: 预测的RGB图像
        depth: 深度图
        viewpoint: 视点信息，包含原始图像等
        **kwargs: 额外参数，包含法线信息
    Returns:
        final_loss: 最终的总损失
    """
    weight_normal = config["Training"].get("weight_normal_loss", 0.5)
    weight_dolp_mask = config["Training"].get("weight_dolp_mask", 0.8)
    use_normal_loss = config.get("use_normal_loss", False)
    use_dolp_mask = config.get("use_dolp_mask", False)
    center_focus = config.get("center_focus", False)
    normal = kwargs.get('normal', None)
    gt_image = viewpoint.original_image.cuda()
    gt_aolp = viewpoint.gt_aolp.cuda()
    # 获取像素掩码
    rgb_pixel_mask = viewpoint.rgb_pixel_mask_mapping
    normal_pixel_mask = rgb_pixel_mask

    image = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b

    # 计算L1损失
    l1_rgb = torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)
    # save_weights_once(rgb_pixel_mask)

    # # 创建边缘掩码（G通道为1的区域）
    # edge_mask = (gt_image[1, :, :] == 1).float()
    # # 增加边缘区域的L1权重
    # l1_rgb = l1_rgb * (1 + 9 * edge_mask)

    combined_loss = l1_rgb
    gt_dolp = viewpoint.gt_dolp.cuda()
    if not (0 <= gt_dolp.min() <= gt_dolp.max() <= 1):
        warnings.warn("gt_dolp values should be in range [0, 1]")
    # Handle polar loss and normal loss with their combinations
    if use_dolp_mask and hasattr(viewpoint, 'gt_dolp'):

        polar_weights = create_polar_weight_mask(gt_dolp, weight_dolp_mask)
        # Apply polar weights to combined loss
        combined_loss = l1_rgb * polar_weights
        # 处理法线损失，先不考虑dolp_mask
        if use_normal_loss:
            # 初始化NormalConsistencyLoss
            normal_loss_fn = NormalConsistencyLoss().cuda()

            # 调用提前聚类好的计算法线损失
            gt_seg = viewpoint.gt_seg.cuda() if (hasattr(viewpoint, "gt_seg") and viewpoint.gt_seg is not None) else None
            # if gt_seg is None:
            #     print("该帧无gt_seg")
            total_normal_loss = normal_loss_fn(depth_normal, normal, gt_seg, gt_aolp, gt_dolp)

            # 将法线损失与RGB损失组合
            combined_loss = l1_rgb + weight_normal * total_normal_loss * polar_weights
    elif use_normal_loss:
        # 初始化NormalConsistencyLoss
        normal_loss_fn = NormalConsistencyLoss().cuda()
        # 调用提前聚类好的计算法线损失
        gt_seg = viewpoint.gt_seg.cuda() if (hasattr(viewpoint, "gt_seg") and viewpoint.gt_seg is not None) else None
        # if gt_seg is None:
        #     print("该帧无gt_seg")
        total_normal_loss = normal_loss_fn(depth_normal, normal, gt_seg, gt_aolp, gt_dolp)

        # 将法线损失与RGB损失组合
        combined_loss = l1_rgb + weight_normal * total_normal_loss
    # 处理中心聚焦
    if center_focus:
        # 创建中心权重掩码并移动到正确的设备
        center_weights = create_center_weight_mask(rgb_pixel_mask.shape[-2:]).to(rgb_pixel_mask.device)
        # 将中心权重应用到最终的组合损失
        final_loss = (combined_loss * center_weights).mean()
        final_weights = center_weights + polar_weights
        # 保存最终的组合权重图（只会保存一次）
        # save_weights_once(final_weights)
        print("中心权重还是很奇怪，先不用")
    else:
        final_loss = combined_loss.mean()

    return final_loss

# 这个先弃用，因为没有考虑分割的情况，是逐像素去歧义的
# def get_loss_mapping_rgb_azimuth(config, depth_normal, normal, normal_pixel_mask, viewpoint):
#     """
#     Calculate the azimuth loss between predicted normal map and ground truth AOLP,
#     excluding regions where gt_aolp is 0.0.
#
#     Args:
#         config: Configuration object
#         depth_normal: Depth normal (not used in this function)
#         normal: Predicted normal map tensor of shape [3, H, W]
#         normal_pixel_mask: Binary mask tensor of shape [H, W]
#         viewpoint: Object containing ground truth AOLP
#
#     Returns:
#         azimuth_loss: Scalar tensor containing the masked cosine similarity loss
#     """
#     # Ensure all inputs are on GPU and in correct format
#     normal = normal.permute(1, 2, 0)  # Change shape from [3, H, W] to [H, W, 3]
#
#     # Extract x and y components and normalize to [-1, 1]
#     # 确认normal是在[-1, 1]范围内
#     normal_x = normal[..., 0]  # 直接使用，不需要 * 2 - 1
#     normal_y = normal[..., 1]  # 直接使用，不需要 * 2 - 1
#
#     # Calculate azimuth angle using atan2
#     normal_azimuth = torch.atan2(normal_y, normal_x)
#
#     # Simplified conversion to range [0, π]
#     normal_azimuth = (normal_azimuth + 2 * torch.pi) % (2 * torch.pi)
#     # normal_azimuth = torch.where(normal_azimuth > torch.pi, 2 * torch.pi - normal_azimuth, normal_azimuth)
#
#     # 不能使用下面的方式，下面转换后的值不一样！
#     # This avoids the two-step conversion and conditional
#     # normal_azimuth = torch.abs(torch.remainder(normal_azimuth + torch.pi, torch.pi))
#
#     # Ensure ground truth AOLP is on GPU
#     gt_aolp = viewpoint.gt_aolp.cuda()
#
#     # Create mask for valid gt_aolp values (excluding 0.0)
#     valid_gt_mask = (gt_aolp > 0.0) & (gt_aolp <= torch.pi)
#
#     # Combine all masks
#     normal_pixel_mask = normal_pixel_mask.cuda()
#
#     combined_mask = (normal_pixel_mask & valid_gt_mask)
#
#     # Value range check for gt_aolp
#     if torch.any(gt_aolp < 0) or torch.any(gt_aolp > torch.pi):
#         print("Warning: gt_aolp contains values outside [0, π] range")
#
#     # 调用可视化函数
#     # visualize_and_save_comparison(gt_aolp, normal_azimuth, combined_mask)
#
#     # 计算绝对角度差，0和180就是差很大
#     absolute_diff = torch.abs(normal_azimuth - gt_aolp)
#
#     # Calculate masked loss (1 - cos_similarity to convert similarity to loss)
#     masked_loss = (1 - absolute_diff) * combined_mask.float()
#
#     # Average over valid pixels
#     num_valid_pixels = torch.sum(combined_mask) + 1e-8  # avoid division by zero
#     azimuth_loss = torch.sum(masked_loss) / num_valid_pixels
#
#     # Optional: Log statistics for debugging
#     # with torch.no_grad():
#     #     total_pixels = float(normal_pixel_mask.numel())
#     #     valid_pixels = float(torch.sum(combined_mask).item())
#     #     if valid_pixels > 0:  # Only calculate statistics if there are valid pixels
#     #         mean_pred_azimuth = torch.sum(normal_azimuth * combined_mask) / valid_pixels
#     #         mean_gt_aolp = torch.sum(gt_aolp * combined_mask) / valid_pixels
#     #         print(f"Valid pixels: {valid_pixels / total_pixels * 100:.2f}% "
#     #               f"Mean pred azimuth: {mean_pred_azimuth.item():.3f}, "
#     #               f"Mean gt_aolp: {mean_gt_aolp.item():.3f}")
#
#     return azimuth_loss


def get_loss_mapping_rgbd(config, image, depth, viewpoint, **kwargs):
    alpha = config["Training"].get("alpha", 0.95)
    weight_normal = config["Training"].get("weight_normal_loss", 0.5)
    weight_dolp_mask = config["Training"].get("weight_dolp_mask", 0.8)
    use_normal_loss = config.get("use_normal_loss", False)
    use_dolp_mask = config.get("use_dolp_mask", False)
    center_focus = config.get("center_focus", False)
    normal = kwargs.get('normal', None)
    gt_image = viewpoint.original_image.cuda()

    rgb_pixel_mask = viewpoint.rgb_pixel_mask_mapping
    depth_pixel_mask = (viewpoint.gt_depth > 0.01).view(*depth.shape)
    normal_pixel_mask = rgb_pixel_mask & depth_pixel_mask

    # Calculate base L1 losses
    l1_rgb = torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)
    l1_depth = torch.abs(depth * depth_pixel_mask - viewpoint.gt_depth * depth_pixel_mask)

    # 考虑深度不连续区域
    # if use_normal_loss:
    #     continuity_mask = get_depth_discontinuity_mask(depth)   # todo 报错，先不弄
    #     normal_pixel_mask = normal_pixel_mask & continuity_mask

    # Initialize combined loss
    combined_loss = alpha * l1_rgb + (1 - alpha) * l1_depth

    # Handle polar loss and normal loss with their combinations
    if use_dolp_mask and hasattr(viewpoint, 'gt_dolp'):
        gt_dolp = viewpoint.gt_dolp.cuda()
        if not (0 <= gt_dolp.min() <= gt_dolp.max() <= 1):
            warnings.warn("gt_dolp values should be in range [0, 1]")
        polar_weights = create_polar_weight_mask(gt_dolp, weight_dolp_mask)

        # Apply polar weights to combined loss
        combined_loss = combined_loss * polar_weights

        # Handle normal loss with polar weights if both are enabled
        if use_normal_loss:
            gt_normal = viewpoint.gt_normal.cuda()
            normal_loss = get_normal_loss(normal, gt_normal, normal_pixel_mask)
            # Apply polar weights to normal loss as well
            normal_loss = normal_loss * polar_weights
            combined_loss = combined_loss + weight_normal * normal_loss

    elif use_normal_loss:
        gt_normal = viewpoint.gt_normal.cuda()
        # 使用改进的法线loss
        # normal_loss = get_multiscale_normal_loss(normal, gt_normal, normal_pixel_mask)    # 先试试普通的吧
        # 当深度误差 l1_depth 较大时，说明深度预测不准确，此时应该降低法线损失的权重
        # exp(-5x) 是一个从1迅速衰减到0的函数
        # -5是一个经验值，控制衰减的速度
        normal_loss = get_normal_loss(normal, gt_normal, normal_pixel_mask)
        # 自适应权重
        # adaptive_weight = weight_normal * torch.exp(-5 * l1_depth.mean())
        adaptive_weight = weight_normal
        combined_loss = combined_loss + adaptive_weight * normal_loss

    if center_focus:
        # Create center weights tensor and move to correct device
        center_weights = create_center_weight_mask(depth.shape[-2:]).to(depth.device)
        # Apply center weighting to final combined loss
        final_loss = (combined_loss * center_weights).mean()
    else:
        final_loss = combined_loss.mean()

    # Write losses to file if normal loss is used
    if use_normal_loss:
        # Get current timestamp
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        # todo:这个写入还有报错
        # with open('loss.txt', 'a') as f:
        #     f.write(f"[{timestamp}] {alpha * l1_rgb.mean().item()} {(1 - alpha) * l1_depth.mean().item()} {weight_normal * normal_loss.item()}\n")

    return final_loss
# # Define Sobel kernels
# sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device="cuda").view(1, 1, 3, 3)
# sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device="cuda").view(1, 1, 3, 3)
# def sobel_filter(image):
#     # Add batch and channel dimensions
#     image = image.unsqueeze(0)
#     # Apply Sobel filters
#     grad_x = F.conv2d(image, sobel_x)
#     grad_y = F.conv2d(image, sobel_y)
#     # Compute gradient magnitude and add a small number to avoid zero gradients
#     grad_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2) + 1e-6
#     return grad_magnitude.squeeze()
#
#
# # Define Scharr kernels
# scharr_x = torch.tensor([[3, 0, -3], [10, 0, -10], [3, 0, -3]], dtype=torch.float32, device="cuda").view(1, 1, 3, 3)
# scharr_y = torch.tensor([[3, 10, 3], [0, 0, 0], [-3, -10, -3]], dtype=torch.float32, device="cuda").view(1, 1, 3, 3)
# def scharr_filter(image):
#     # Add batch and channel dimensions
#     image = image.unsqueeze(0)
#     # Apply Scharr filters
#     grad_x = F.conv2d(image, scharr_x)
#     grad_y = F.conv2d(image, scharr_y)
#     # Compute gradient magnitude and add a small number to avoid zero gradients
#     grad_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2) + 1e-6
#     return grad_magnitude.squeeze()


def binary_cross_entropy_edge_loss(edges_image, edges_gt_image):    # 用于稀疏的概念边缘图的loss计算
    # Flatten the tensors
    edges_image = edges_image.view(-1)
    edges_gt_image = edges_gt_image.view(-1)

    # Compute binary cross entropy loss
    edge_loss = F.binary_cross_entropy_with_logits(edges_image, edges_gt_image)
    return edge_loss


def ssim_edge_loss(edges_image, edges_gt_image):
    # Ensure the tensors have at least 3 dimensions
    if edges_image.dim() == 2:
        edges_image = edges_image.unsqueeze(0).unsqueeze(0)
    elif edges_image.dim() == 3:
        edges_image = edges_image.unsqueeze(0)

    if edges_gt_image.dim() == 2:
        edges_gt_image = edges_gt_image.unsqueeze(0).unsqueeze(0)
    elif edges_gt_image.dim() == 3:
        edges_gt_image = edges_gt_image.unsqueeze(0)

    # Ensure the input image size is large enough
    min_size = 11  # This is an example value, adjust as needed
    if edges_image.size(-1) < min_size or edges_image.size(-2) < min_size:
        raise ValueError("Input image size is too small for SSIM calculation")

    # Adjust image size
    edges_image = edges_image[:, :, :min_size, :min_size]
    edges_gt_image = edges_gt_image[:, :, :min_size, :min_size]

    # Compute SSIM loss
    edge_loss = 1 - ssim(edges_image, edges_gt_image)
    return edge_loss

def get_loss_mapping_rgbd_edge(config, image, depth, viewpoint, initialization=False):
    # print("Using edge loss")
    alpha = config["Training"]["alpha"] if "alpha" in config["Training"] else 0.95
    weight_edge_loss = config["Training"]["weight_edge_loss"] if "weight_edge_loss" in config["Training"] else 0.5
    gt_image = viewpoint.original_image.cuda()

    rgb_pixel_mask = viewpoint.rgb_pixel_mask_mapping
    depth_pixel_mask = (viewpoint.gt_depth > 0.01).view(*depth.shape)

    l1_rgb = torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)
    l1_depth = torch.abs(depth * depth_pixel_mask - viewpoint.gt_depth * depth_pixel_mask)

    # Check for NaN in edges
    if torch.isnan(image).any():
        print("NaN detected in image")

    # Detect edges using Sobel operator
    edges_image = laplacian_filter(image.mean(dim=0, keepdim=True) * rgb_pixel_mask)
    edges_gt_image = laplacian_filter(gt_image.mean(dim=0, keepdim=True) * rgb_pixel_mask)

    # Compute edge loss
    edge_loss = ssim_edge_loss(edges_image, edges_gt_image)
    print("引入边缘损失目前没有用，需改进")
    # Save edges as images，debug用的
    # cv2.imwrite('edges_image.png', edges_image.detach().cpu().numpy() * 255)
    # cv2.imwrite('edges_gt_image.png', edges_gt_image.detach().cpu().numpy() * 255)
    # sleep(1)
    # image_test = image * rgb_pixel_mask
    # print(f"Shape: {image_test.shape}")
    # print(f"Size: {image_test.size()}")
    # print(f"Dtype: {image_test.dtype}")
    # print(f"Max value: {image_test.max().item()}")
    # print(f"Min value: {image_test.min().item()}")

    # Print the losses
    # print(f"L1 RGB Loss: {l1_rgb.mean().item()}")
    # print(f"L1 Depth Loss: {l1_depth.mean().item()}")
    # print(f"Edge Loss: {edge_loss.item()}")
    #     # Save edges as images，debug用的
    #     cv2.imwrite('image_test.png', image_test.detach().cpu().numpy() * 255)
    #     cv2.imwrite('edges_image.png', edges_image.detach().cpu().numpy() * 255)
    #     cv2.imwrite('edges_gt_image.png', edges_gt_image.detach().cpu().numpy() * 255)


    total_loss = alpha * l1_rgb.mean() + (1 - alpha) * l1_depth.mean() + weight_edge_loss * edge_loss

    # Check if edge_loss is NaN
    # if torch.isnan(edge_loss):
    #     print("Edge loss is NaN, ignoring edge loss.")
    #     total_loss = alpha * l1_rgb.mean() + (1 - alpha) * l1_depth.mean()
    # else:
    #     total_loss = alpha * l1_rgb.mean() + (1 - alpha) * l1_depth.mean() + edge_loss
    return total_loss

def get_median_depth(depth, opacity=None, mask=None, return_std=False):
    depth = depth.detach().clone()
    opacity = opacity.detach()
    valid = depth > 0
    if opacity is not None:
        valid = torch.logical_and(valid, opacity > 0.95)
    if mask is not None:
        valid = torch.logical_and(valid, mask)
    valid_depth = depth[valid]
    if return_std:
        return valid_depth.median(), valid_depth.std(), valid
    return valid_depth.median()


def visualize_and_save_comparison(gt_aolp, normal_azimuth, combined_mask, save_dir='debug_visualizations'):
    """
    将方位角数据进行可视化并保存

    Args:
        gt_aolp (torch.Tensor): 真实方位角数据 [H, W]
        normal_azimuth (torch.Tensor): 预测的方位角数据 [H, W]
        combined_mask (torch.Tensor): 掩码 [H, W]
        save_dir (str): 保存目录
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # 确保数据在CPU上并转换为numpy数组
    gt_aolp = gt_aolp.detach().cpu().numpy()
    normal_azimuth = normal_azimuth.detach().cpu().numpy()
    combined_mask = combined_mask.detach().cpu().numpy()
    print(f"gt_aolp shape: {gt_aolp.shape}")
    print(f"normal_azimuth shape: {normal_azimuth.shape}")
    print(f"combined_mask shape: {combined_mask.shape}")

    # 归一化到0-1范围
    gt_aolp_norm = np.clip(gt_aolp / np.pi, 0, 1)
    normal_azimuth_norm = np.clip(normal_azimuth / np.pi, 0, 1)

    # 创建colormap对象
    cmap = plt.get_cmap('plasma')

    # 手动应用colormap (先squeeze去除维度为1的维度, 再用 [..., :3] 正确切片)
    colored_gt = cmap(gt_aolp_norm.squeeze())[..., :3]
    colored_pred = cmap(normal_azimuth_norm.squeeze())[..., :3]

    # 扩展mask维度以匹配RGB通道 (先squeeze去除维度为1的维度)
    mask_3d = np.expand_dims(combined_mask.squeeze(), axis=-1).repeat(3, axis=-1) # 修改这里，先 squeeze combined_mask

    print(f"colored_gt shape after colormap: {colored_gt.shape}") # 打印 colormap 后的形状
    print(f"mask_3d shape: {mask_3d.shape}") # 打印 mask_3d 的形状


    # 应用mask（将无效区域设为黑色）
    colored_gt = colored_gt * mask_3d
    colored_pred = colored_pred * mask_3d

    # 计算差异图（归一化后的差异）
    diff = np.abs(gt_aolp_norm - normal_azimuth_norm)
    colored_diff = cmap(diff.squeeze())[..., :3] * mask_3d # 修改这里，squeeze diff


    print(f"colored_gt shape before permute: {colored_gt.shape}")
    print(f"colored_gt ndim before permute: {colored_gt.ndim}")
    # 转换为PyTorch张量并调整通道顺序 [H, W, 3] -> [3, H, W]
    colored_gt = torch.from_numpy(colored_gt).float().permute(2, 0, 1)
    colored_pred = torch.from_numpy(colored_pred).float().permute(2, 0, 1)
    colored_diff = torch.from_numpy(colored_diff).float().permute(2, 0, 1)

    # 保存图像
    vutils.save_image(
        colored_gt,
        os.path.join(save_dir, f'colored_masked_gt_aolp_{timestamp}.png'),
        normalize=False
    )
    vutils.save_image(
        colored_pred,
        os.path.join(save_dir, f'colored_masked_normal_azimuth_{timestamp}.png'),
        normalize=False
    )
    vutils.save_image(
        colored_diff,
        os.path.join(save_dir, f'colored_difference_{timestamp}.png'),
        normalize=False
    )

    # 保存一个组合图像，便于对比
    combined_img = torch.cat([colored_gt, colored_pred, colored_diff], dim=2)  # 水平拼接
    vutils.save_image(
        combined_img,
        os.path.join(save_dir, f'combined_comparison_{timestamp}.png'),
        normalize=False
    )

    print(f"\nDebug visualizations saved to '{save_dir}' folder:")
    print(f"1. Ground Truth AOLP (colored): colored_masked_gt_aolp_{timestamp}.png")
    print(f"2. Predicted Normal Azimuth (colored): colored_masked_normal_azimuth_{timestamp}.png")
    print(f"3. Absolute Difference (colored): colored_difference_{timestamp}.png")
    print(f"4. Combined Comparison: combined_comparison_{timestamp}.png")
    print("\nPausing for 5 seconds...")
    time.sleep(5)
