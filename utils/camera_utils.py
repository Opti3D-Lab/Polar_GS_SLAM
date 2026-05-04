import torch
from torch import nn

from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2
from utils.slam_utils import image_gradient, image_gradient_mask
import torch.nn.functional as F

class Camera(nn.Module):
    def __init__(
        self,
        uid,
        color,
        depth,
        normal,     # 可能是gt，也可能是SLAM输出
        gt_aolp,    # 这两个只能是gt，不会是SLAM输出，所以直接加gt前缀
        gt_dolp,
        gt_seg,
        gt_T,
        projection_matrix,
        fx,
        fy,
        cx,
        cy,
        fovx,
        fovy,
        image_height,
        image_width,
        device="cuda:0",
    ):
        super(Camera, self).__init__()
        self.uid = uid
        self.device = device

        self.T = torch.eye(4, device=device).to(torch.float32)
        self.T_gt = gt_T.to(device=device).to(torch.float32).clone()
        
        self.original_image = color
        self.depth = depth
        self.normal = normal
        self.gt_aolp = gt_aolp
        self.gt_dolp = gt_dolp
        self.gt_seg = gt_seg
        self.grad_mask = None

        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.FoVx = fovx
        self.FoVy = fovy
        self.image_height = image_height
        self.image_width = image_width

        self.cam_rot_delta = nn.Parameter(
            torch.zeros(3, requires_grad=True, device=device)
        )
        self.cam_trans_delta = nn.Parameter(
            torch.zeros(3, requires_grad=True, device=device)
        )

        self.exposure_a = nn.Parameter(
            torch.tensor([0.0], requires_grad=True, device=device)
        )
        self.exposure_b = nn.Parameter(
            torch.tensor([0.0], requires_grad=True, device=device)
        )

        self.projection_matrix = projection_matrix.to(device=device)
        
        



    # 被前端调用，输入的dataset是一个list，里面是一个个元组，元组里面是gt_color, gt_depth, gt_normal, gt_pose, gt_aolp, gt_dolp
    @staticmethod
    def init_from_dataset(dataset, idx, projection_matrix):
        # gt_color, gt_depth, gt_normal, gt_pose, gt_aolp, gt_dolp= dataset[idx]
        # 有可能没有aolp和dolp
        data = dataset[idx]
        gt_color, gt_depth, gt_normal, gt_pose = data[:4]
        gt_aolp = gt_dolp = gt_seg = None
        if len(data) > 4:
            gt_aolp, gt_dolp = data[4], data[5]
        if len(data) > 5:
            gt_seg = data[6]

        return Camera(  # 调用上面的init函数
            idx,
            gt_color,
            gt_depth,
            gt_normal,
            gt_aolp,
            gt_dolp,
            gt_seg,
            gt_pose,
            projection_matrix,
            dataset.fx,
            dataset.fy,
            dataset.cx,
            dataset.cy,
            dataset.fovx,
            dataset.fovy,
            dataset.height,
            dataset.width,
            device=dataset.device,
        )

    @staticmethod
    def init_from_gui(uid, T, FoVx, FoVy, fx, fy, cx, cy, H, W):
        projection_matrix = getProjectionMatrix2(
            znear=0.01, zfar=100.0, fx=fx, fy=fy, cx=cx, cy=cy, W=W, H=H
        ).transpose(0, 1)
        return Camera(
            uid, None, None, None, None, None, None, T, projection_matrix, fx, fy, cx, cy, FoVx, FoVy, H, W
        ) # 多了个normal,所以要多一个None

    @property
    def world_view_transform(self):
        return self.T.transpose(0, 1).to(device=self.device)

    @property
    def full_proj_transform(self):
        return (
            self.world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)

    @property
    def camera_center(self):
        return self.world_view_transform #TODO: Need to invert for high order SHs by inverse_t(self.world_view_transform).
        
    def compute_grad_mask(self, config):
        # 梯度是从rgb中计算的，滤去变化较大的边缘区域
        edge_threshold = config["Training"]["edge_threshold"]

        gray_img = self.original_image.mean(dim=0, keepdim=True)
        gray_grad_v, gray_grad_h = image_gradient(gray_img)
        mask_v, mask_h = image_gradient_mask(gray_img)
        gray_grad_v = gray_grad_v * mask_v
        gray_grad_h = gray_grad_h * mask_h
        img_grad_intensity = torch.sqrt(gray_grad_v**2 + gray_grad_h**2)
        
        if config["Dataset"]["type"] == "replica":
            size = 32
            multiplier = edge_threshold
            _, h, w = self.original_image.shape
            I = img_grad_intensity.unsqueeze(0)
            I_unf = F.unfold(I, size, stride=size)
            median_patch, _ = torch.median(I_unf, dim=1,keepdim=True)
            mask = (I_unf > (median_patch * multiplier)).float()
            I_f = F.fold(mask, I.shape[-2:],size,stride=size).squeeze(0)
            self.grad_mask = I_f
        else:
            median_img_grad_intensity = img_grad_intensity.median()
            self.grad_mask = (
                img_grad_intensity > median_img_grad_intensity * edge_threshold
            )

        gt_image = self.original_image.cuda()
        _, h, w = self.original_image.cuda().shape
        mask_shape = (1, h, w)
        rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]
        rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*mask_shape)
        self.rgb_pixel_mask = rgb_pixel_mask * self.grad_mask   # 追踪时排除边缘
        self.rgb_pixel_mask_mapping = rgb_pixel_mask    # 二者都是01二值mask，用来排除纯黑区域的
        
        if self.depth is not None:
            self.gt_depth = torch.from_numpy(self.depth).to(
            dtype=torch.float32, device=self.device
        )[None] # None增加一个维度，变成1xHxW

        if self.gt_aolp is not None:
            self.gt_aolp = torch.from_numpy(self.gt_aolp).to(
            dtype=torch.float32, device=self.device
        )[None]

        if self.gt_dolp is not None:
            self.gt_dolp = torch.from_numpy(self.gt_dolp).to(
            dtype=torch.float32, device=self.device
        )[None]

        if self.gt_seg is not None:
            self.gt_seg = torch.from_numpy(self.gt_seg).to(
            dtype=torch.float32, device=self.device
        )[None]

        if self.normal is not None:
            self.gt_normal = torch.from_numpy(self.normal).permute(2, 0, 1).to(
            dtype=torch.float32, device=self.device)    # 转换通道顺序从HWC到CHW



        
    
    def clean(self):
        self.original_image = None
        self.depth = None
        self.normal = None
        self.grad_mask = None

        self.cam_rot_delta = None
        self.cam_trans_delta = None

        self.exposure_a = None
        self.exposure_b = None
        
        self.rgb_pixel_mask = None
        self.rgb_pixel_mask_mapping = None
        self.gt_depth = None
        self.gt_normal = None
        self.gt_aolp = None
        self.gt_dolp = None
        self.gt_seg = None

class CameraMsg():
    def __init__(self, Camera):
        self.uid = Camera.uid
        self.T = Camera.T
        self.T_gt = Camera.T_gt