import time

import numpy as np
import torch
import torch.multiprocessing as mp
import math
from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2
from gui import gui_utils
from utils.camera_utils import Camera, CameraMsg
from utils.eval_utils import eval_ate, save_gaussians
from utils.logging_utils import Log
from utils.multiprocessing_utils import clone_obj
from utils.pose_utils import update_pose
from utils.slam_utils import get_loss_tracking, get_median_depth
from utils.save_render import save_visualizations   # 自己新建的文件
# 主要功能：
# 添加关键帧：根据一定的条件判断并添加新的关键帧。
# 跟踪：根据当前帧和前一帧的位姿进行跟踪，并优化相机参数
# 窗口管理：管理当前窗口中的关键帧，确保窗口大小在设定范围内。

def generate_ellipsoid_depth(height=512, width=612):
    """
    生成椭球体深度图的PyTorch张量实现

    Args:
        height (int): 图像高度，默认512
        width (int): 图像宽度，默认612
        device (str): 使用的设备，默认'cuda'

    Returns:
        torch.Tensor: 形状为(1, height, width)的深度图张量，单位为毫米
    """
    # 将计算直接使用cpu
    device = torch.device('cpu')

    # 创建坐标网格（使用PyTorch）
    y = torch.arange(height, device=device).view(-1, 1).expand(-1, width)
    x = torch.arange(width, device=device).view(1, -1).expand(height, -1)

    # 设置椭球参数
    center_x = 300
    center_y = 200
    radius_x = 250
    radius_y = 300

    # 计算每个点到球心的归一化距离（椭圆方程）
    dist_from_center = torch.sqrt(
        ((x - center_x) / radius_x) ** 2 +
        ((y - center_y) / radius_y) ** 2
    )

    # 生成带噪声的背景深度
    background_depth = 2000 * torch.ones(1, height, width, device=device)
    background_depth += torch.randn(1, height, width, device=device) * 300

    # 创建mask
    mask = dist_from_center <= 1.0

    # 计算椭球深度值
    z_values = torch.sqrt(1 - dist_from_center[mask] ** 2) * radius_y

    # 创建最终深度图
    depth_mm = background_depth.clone()

    # 在mask区域应用椭球深度
    depth_mm = depth_mm.view(height, width)
    depth_mm[mask] = 2000 - z_values * 1.8

    # 添加噪声
    noise = torch.randn_like(depth_mm, device=device) * 100
    depth_mm = depth_mm + noise

    # 裁剪深度值到合理范围
    depth_mm = torch.clamp(depth_mm, 1100, 2900)

    # 重塑为(1, height, width)格式
    depth_mm = depth_mm.view(1, height, width)

    return depth_mm / 1000

class FrontEnd(mp.Process):
    def __init__(self, config):
        super().__init__()
        self.device = "cuda:0"  # 设置设备为第二个 GPU
        self.config = config
        self.background = None
        self.pipeline_params = None
        self.frontend_queue = None
        self.backend_queue = None
        self.q_main2vis = None
        self.q_vis2main = None

        self.initialized = False
        self.kf_indices = []
        self.monocular = config["Training"]["monocular"]
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.current_window = []

        self.reset = True
        self.requested_init = False
        self.requested_keyframe = 0
        self.use_every_n_frames = 1

        self.gaussians = None
        self.cameras = dict()
        self.pause = False
        self.total_frames = 0

    def set_hyperparams(self):  # 设置超参数
        self.save_dir = self.config["Results"]["save_dir"]
        self.save_results = self.config["Results"]["save_results"]
        self.save_trj = self.config["Results"]["save_trj"]
        self.save_trj_kf_intv = self.config["Results"]["save_trj_kf_intv"]

        self.tracking_itr_num = self.config["Training"]["tracking_itr_num"]
        self.kf_interval = self.config["Training"]["kf_interval"]
        self.window_size = self.config["Training"]["window_size"]
        self.single_thread = (
            self.config["Dataset"]["single_thread"]
            if "single_thread" in self.config["Dataset"]
            else False
        )

        self.use_gui = self.config["Results"]["use_gui"]
        self.constant_velocity_warmup = 200  # TODO: fix hardcoding

    def add_new_keyframe(self, cur_frame_idx, depth=None, opacity=None, init=False):
        rgb_boundary_threshold = self.config["Training"]["rgb_boundary_threshold"]  # 从配置中获取 RGB 边界阈值
        self.kf_indices.append(cur_frame_idx)
        viewpoint = self.cameras[cur_frame_idx]
        gt_img = viewpoint.original_image.cuda()
        valid_rgb = (gt_img.sum(dim=0) > rgb_boundary_threshold)[None]  # 过滤掉那些 RGB 值总和低于该阈值的像素
        if self.monocular:
            if depth is None:
                # 生成一个初始深度图，并添加一些随机噪声
                initial_depth = 2 * torch.ones(1, gt_img.shape[1], gt_img.shape[2])
                initial_depth += torch.randn_like(initial_depth) * 0.3

                # 生成适配firebox的椭球
                # initial_depth = generate_ellipsoid_depth(gt_img.shape[1], gt_img.shape[2])  # 默认612*512
                # print(f"initial_depth mean: {initial_depth.mean().item()}, device: {initial_depth.device}")
                # print(f"initial_depth的形状为：{initial_depth.shape}")
            else:
                # 已经在之前生成了深度图了，这里只是处理一下，过滤掉异常值，并添加一些随机噪声
                depth = depth.detach().clone()
                opacity = opacity.detach()
                use_inv_depth = False  # 逆深度即深度的倒数
                if use_inv_depth:
                    inv_depth = 1.0 / depth
                    inv_median_depth, inv_std, valid_mask = get_median_depth(
                        inv_depth, opacity, mask=valid_rgb, return_std=True
                    )
                    # 用中值和标准差过滤掉异常的逆深度值，并用中值替换这些异常值。
                    invalid_depth_mask = torch.logical_or(
                        inv_depth > inv_median_depth + inv_std,
                        inv_depth < inv_median_depth - inv_std,
                    )
                    invalid_depth_mask = torch.logical_or(
                        invalid_depth_mask, ~valid_mask
                    )
                    inv_depth[invalid_depth_mask] = inv_median_depth
                    # 为深度图添加一些随机噪声
                    inv_initial_depth = inv_depth + torch.randn_like(
                        inv_depth
                    ) * torch.where(invalid_depth_mask, inv_std * 0.5, inv_std * 0.2)
                    initial_depth = 1.0 / inv_initial_depth
                else:
                    median_depth, std, valid_mask = get_median_depth(
                        depth, opacity, mask=valid_rgb, return_std=True
                    )
                    invalid_depth_mask = torch.logical_or(
                        depth > median_depth + std, depth < median_depth - std
                    )
                    invalid_depth_mask = torch.logical_or(
                        invalid_depth_mask, ~valid_mask
                    )
                    depth[invalid_depth_mask] = median_depth
                    initial_depth = depth + torch.randn_like(depth) * torch.where(
                        invalid_depth_mask, std * 0.5, std * 0.2
                    )

                initial_depth[~valid_rgb] = 0  # Ignore the invalid rgb pixels
            return initial_depth.cpu().numpy()[0]
        # use the observed depth
        # 提供了深度图的话直接使用
        initial_depth = torch.from_numpy(viewpoint.depth).unsqueeze(0)
        initial_depth[~valid_rgb.cpu()] = 0  # Ignore the invalid rgb pixels
        return initial_depth[0].numpy()

    def initialize(self, cur_frame_idx, viewpoint):
        self.initialized = not self.monocular
        self.kf_indices = []
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.current_window = []
        # remove everything from the queues
        while not self.backend_queue.empty():
            self.backend_queue.get()

        # Initialise the frame at the ground truth pose
        viewpoint.T = viewpoint.T_gt

        self.kf_indices = []
        depth_map = self.add_new_keyframe(cur_frame_idx, init=True)
        self.request_init(cur_frame_idx, viewpoint, depth_map)
        self.reset = False

    def tracking(self, cur_frame_idx, viewpoint):

        if self.initialized and cur_frame_idx > self.constant_velocity_warmup and self.monocular:
            prev_prev = self.cameras[cur_frame_idx - self.use_every_n_frames - 1]
            prev = self.cameras[cur_frame_idx - self.use_every_n_frames]

            pose_prev_prev = prev_prev.T
            pose_prev = prev.T
            velocity = pose_prev @ torch.linalg.inv(pose_prev_prev)
            pose_new = velocity @ pose_prev
            viewpoint.T = pose_new
        else:
            prev = self.cameras[cur_frame_idx - self.use_every_n_frames]
            viewpoint.T = prev.T

        opt_params = []
        opt_params.append(
            {
                "params": [viewpoint.cam_rot_delta],
                "lr": self.config["Training"]["lr"]["cam_rot_delta"],
                "name": "rot_{}".format(viewpoint.uid),
            }
        )
        opt_params.append(
            {
                "params": [viewpoint.cam_trans_delta],
                "lr": self.config["Training"]["lr"]["cam_trans_delta"],
                "name": "trans_{}".format(viewpoint.uid),
            }
        )
        opt_params.append(
            {
                "params": [viewpoint.exposure_a],
                "lr": 0.01,
                "name": "exposure_a_{}".format(viewpoint.uid),
            }
        )
        opt_params.append(
            {
                "params": [viewpoint.exposure_b],
                "lr": 0.01,
                "name": "exposure_b_{}".format(viewpoint.uid),
            }
        )

        pose_optimizer = torch.optim.Adam(opt_params)
        for tracking_itr in range(self.tracking_itr_num):
            render_pkg = render(
                viewpoint, self.gaussians, self.pipeline_params, self.background
            )
            image, depth, opacity, normal = (
                render_pkg["render"],
                render_pkg["depth"],
                render_pkg["opacity"],
                render_pkg["normal"],
            )
            pose_optimizer.zero_grad()

            loss_tracking = get_loss_tracking(
                self.config, image, depth, opacity, viewpoint, normal
            )
            loss_tracking.backward()

            with torch.no_grad():
                pose_optimizer.step()
                converged = update_pose(viewpoint)

            if tracking_itr % 50 == 0:  # 每 50 次迭代将当前帧的渲染结果发送到可视化队列
                self.q_main2vis.put(
                    gui_utils.GaussianPacket(
                        current_frame=CameraMsg(viewpoint),
                        gtcolor=viewpoint.original_image,
                        gtdepth=viewpoint.depth
                        if not self.monocular
                        else np.zeros((viewpoint.image_height, viewpoint.image_width)),
                    )
                )
            if converged:   # 如果收敛则停止迭代
                break

        self.median_depth = get_median_depth(depth, opacity)
        # 保存图像
        # save_visualizations(image, depth, normal, cur_frame_idx, save_raw_normal=True, depth_scale=2000)
        return render_pkg

    # def is_keyframe(
    #         self,
    #         cur_frame_idx,
    #         last_keyframe_idx,
    #         cur_frame_visibility_filter,
    #         occ_aware_visibility,
    # ):
    #     kf_translation = self.config["Training"]["kf_translation"]
    #     kf_min_translation = self.config["Training"]["kf_min_translation"]
    #     kf_overlap = self.config["Training"]["kf_overlap"]
    #
    #     # 计算几何位移
    #     curr_frame = self.cameras[cur_frame_idx]
    #     last_kf = self.cameras[last_keyframe_idx]
    #     pose_CW = curr_frame.T
    #     last_kf_CW = last_kf.T
    #     last_kf_WC = torch.linalg.inv(last_kf_CW)
    #     dist = torch.norm((pose_CW @ last_kf_WC)[0:3, 3])
    #
    #     dist_check = dist > kf_translation * self.median_depth
    #     dist_check2 = dist > kf_min_translation * self.median_depth
    #
    #     # 计算高斯交并比
    #     union = torch.logical_or(
    #         cur_frame_visibility_filter, occ_aware_visibility[last_keyframe_idx]
    #     ).count_nonzero()
    #     intersection = torch.logical_and(
    #         cur_frame_visibility_filter, occ_aware_visibility[last_keyframe_idx]
    #     ).count_nonzero()
    #     point_ratio_2 = intersection / union
    #     return (point_ratio_2 < kf_overlap and dist_check2) or dist_check

    # 加入基于可见区域的偏振差异
    def is_keyframe(self, cur_frame_idx, last_keyframe_idx, cur_frame_visibility_filter, occ_aware_visibility):
        # 配置参数提取放在一起，提高可读性
        config_train = self.config["Training"]
        kf_translation = config_train["kf_translation"]
        kf_min_translation = config_train["kf_min_translation"]
        kf_overlap = config_train["kf_overlap"]
        pol_diff_threshold = config_train.get("polarization_diff_threshold", 0.4)

        # 计算几何位移 (优化矩阵运算)
        curr_frame, last_kf = self.cameras[cur_frame_idx], self.cameras[last_keyframe_idx]
        pose_CW, last_kf_CW = curr_frame.T, last_kf.T
        # 使用更高效的方式计算相对位移
        rel_pose = pose_CW @ torch.linalg.inv(last_kf_CW)
        dist = torch.norm(rel_pose[0:3, 3])

        # 预先计算几何条件
        dist_check = dist > kf_translation * self.median_depth
        dist_check2 = dist > kf_min_translation * self.median_depth

        # 使用torch操作更高效地计算集合操作
        union = torch.logical_or(cur_frame_visibility_filter, occ_aware_visibility[last_keyframe_idx])
        intersection = torch.logical_and(cur_frame_visibility_filter, occ_aware_visibility[last_keyframe_idx])
        union_count = union.sum()

        # 防止除零错误
        point_ratio_2 = intersection.sum() / union_count if union_count > 0 else 0.0

        # 获取设备信息，避免不必要的设备转换
        device = cur_frame_visibility_filter.device

        # 如果基于几何条件就能决定是关键帧，提前返回
        if (point_ratio_2 < kf_overlap and dist_check2) or dist_check:
            print(f"{cur_frame_idx} 被标记为关键帧 (几何条件)")
            return True

        # 计算图像掩码并检查共同可见区域大小
        cur_visible_mask = self.project_visibility_to_image(cur_frame_visibility_filter, cur_frame_idx)
        last_visible_mask = self.project_visibility_to_image(occ_aware_visibility[last_keyframe_idx], last_keyframe_idx)
        common_visible_mask = cur_visible_mask & last_visible_mask
        common_pixels = common_visible_mask.sum()

        # 如果共同可见区域太小，无需进一步计算偏振差异
        min_required_pixels = 100
        if common_pixels < min_required_pixels:
            # print(f"当前帧 {cur_frame_idx} 的共同可见区域太小 ({common_pixels} 像素)，不被标记为关键帧。")
            return False

        # 获取偏振图像，使用to(device)而不是hardcode .cuda()
        cur_aolp = self.cameras[cur_frame_idx].gt_aolp.to(device)
        last_aolp = self.cameras[last_keyframe_idx].gt_aolp.to(device)
        cur_dolp = self.cameras[cur_frame_idx].gt_dolp.to(device)
        last_dolp = self.cameras[last_keyframe_idx].gt_dolp.to(device)

        # 处理维度不匹配问题
        if cur_dolp.dim() > 2:
            cur_dolp = cur_dolp.squeeze(0)  # 从 [1, H, W] 变为 [H, W]
        if last_dolp.dim() > 2:
            last_dolp = last_dolp.squeeze(0)
        if cur_aolp.dim() > 2:
            cur_aolp = cur_aolp.squeeze(0)
        if last_aolp.dim() > 2:
            last_aolp = last_aolp.squeeze(0)

        # 计算偏振差异
        aolp_diff = torch.min(torch.abs(cur_aolp - last_aolp),
                              torch.abs(torch.abs(cur_aolp - last_aolp) - torch.pi))
        dolp_diff = torch.abs(cur_dolp - last_dolp)

        # 应用掩码
        masked_aolp_diff = aolp_diff * common_visible_mask
        masked_dolp_diff = dolp_diff * common_visible_mask

        # 计算加权差异，使用动态权重
        # 根据场景特性调整权重，例如对于主要是镜面反射的场景，DOLP差异可能更重要
        scene_complexity = torch.mean(cur_dolp[common_visible_mask])
        aolp_weight = 0.6 if scene_complexity > 0.5 else 0.4
        dolp_weight = 1.0 - aolp_weight

        aolp_diff_mean = masked_aolp_diff.sum() / common_pixels
        dolp_diff_mean = masked_dolp_diff.sum() / common_pixels
        pol_diff = aolp_weight * aolp_diff_mean + dolp_weight * dolp_diff_mean

        pol_check = pol_diff > pol_diff_threshold

        if pol_check:
            print(f"当前帧 {cur_frame_idx} 由于偏振差异显著 (值: {pol_diff:.3f}) 被标记为关键帧。")

        return pol_check

    def project_visibility_to_image(self, visibility_filter, frame_idx):
        """将Gaussian可见性投影到图像平面上，生成二维掩码"""
        camera = self.cameras[frame_idx]
        height, width = camera.image_height, camera.image_width
        mask = torch.zeros((height, width), dtype=torch.bool, device=visibility_filter.device)

        # 获取可见Gaussian的索引
        visible_indices = torch.where(visibility_filter)[0]

        if len(visible_indices) == 0:
            return mask

        # 获取这些Gaussian投影到图像平面上的坐标
        visible_positions = self.gaussians._xyz[visible_indices]

        # 直接实现投影功能，不依赖Camera.project方法
        # 构建相机内参矩阵
        fx = camera.fx if hasattr(camera, 'fx') else width / (2 * math.tan(camera.FoVx / 2.))
        fy = camera.fy if hasattr(camera, 'fy') else height / (2 * math.tan(camera.FoVy / 2.))
        cx = camera.cx if hasattr(camera, 'cx') else width / 2.
        cy = camera.cy if hasattr(camera, 'cy') else height / 2.

        # 计算点在相机坐标系中的坐标
        camera_matrix = camera.T  # 相机到世界的变换矩阵
        world_to_camera = torch.linalg.inv(camera_matrix)

        # 将点从世界坐标系转换到相机坐标系
        homogeneous_positions = torch.cat([
            visible_positions,
            torch.ones((visible_positions.shape[0], 1), device=visible_positions.device)
        ], dim=1)
        camera_space_positions = (world_to_camera @ homogeneous_positions.T).T

        # 透视投影
        x_cam = camera_space_positions[:, 0]
        y_cam = camera_space_positions[:, 1]
        z_cam = camera_space_positions[:, 2]

        # 处理z为0或负数的情况(在相机后面的点)
        valid_z = z_cam > 0

        # 投影到图像平面
        pixels_x = torch.zeros_like(x_cam)
        pixels_y = torch.zeros_like(y_cam)

        # 只投影有效的点
        pixels_x[valid_z] = (fx * x_cam[valid_z] / z_cam[valid_z]) + cx
        pixels_y[valid_z] = (fy * y_cam[valid_z] / z_cam[valid_z]) + cy

        # 转换为整数坐标
        pixels_x = pixels_x.round().long()
        pixels_y = pixels_y.round().long()

        # 过滤有效范围内的像素
        valid = valid_z & (pixels_x >= 0) & (pixels_x < width) & (pixels_y >= 0) & (pixels_y < height)
        pixels_x = pixels_x[valid]
        pixels_y = pixels_y[valid]

        # 设置掩码
        if len(pixels_x) > 0:
            mask[pixels_y, pixels_x] = True

        return mask

    def add_to_window(
            self, cur_frame_idx, cur_frame_visibility_filter, occ_aware_visibility, window
    ):
        N_dont_touch = 2
        window = [cur_frame_idx] + window
        # remove frames which has little overlap with the current frame
        curr_frame = self.cameras[cur_frame_idx]
        to_remove = []
        removed_frame = None
        for i in range(N_dont_touch, len(window)):
            kf_idx = window[i]
            # szymkiewicz–simpson coefficient
            intersection = torch.logical_and(
                cur_frame_visibility_filter, occ_aware_visibility[kf_idx]
            ).count_nonzero()
            denom = min(
                cur_frame_visibility_filter.count_nonzero(),
                occ_aware_visibility[kf_idx].count_nonzero(),
            )
            point_ratio_2 = intersection / denom
            cut_off = (
                self.config["Training"]["kf_cutoff"]
                if "kf_cutoff" in self.config["Training"]
                else 0.4
            )
            if not self.initialized:
                cut_off = 0.4
            if point_ratio_2 <= cut_off:
                to_remove.append(kf_idx)

        if to_remove:
            window.remove(to_remove[-1])
            removed_frame = to_remove[-1]
        kf_0_WC = torch.linalg.inv(curr_frame.T)

        if len(window) > self.config["Training"]["window_size"]:
            # we need to find the keyframe to remove...
            inv_dist = []
            for i in range(N_dont_touch, len(window)):
                inv_dists = []
                kf_i_idx = window[i]
                kf_i = self.cameras[kf_i_idx]
                kf_i_CW = kf_i.T
                for j in range(N_dont_touch, len(window)):
                    if i == j:
                        continue
                    kf_j_idx = window[j]
                    kf_j = self.cameras[kf_j_idx]
                    kf_j_WC = torch.linalg.inv(kf_j.T)

                    T_CiCj = kf_i_CW @ kf_j_WC
                    inv_dists.append(1.0 / (torch.norm(T_CiCj[0:3, 3]) + 1e-6).item())

                T_CiC0 = kf_i_CW @ kf_0_WC
                k = torch.sqrt(torch.norm(T_CiC0[0:3, 3])).item()
                inv_dist.append(k * sum(inv_dists))

            idx = np.argmax(inv_dist)
            removed_frame = window[N_dont_touch + idx]
            window.remove(removed_frame)

        return window, removed_frame

    def request_keyframe(self, cur_frame_idx, viewpoint, current_window, depthmap):
        msg = ["keyframe", cur_frame_idx, viewpoint, current_window, depthmap]
        self.backend_queue.put(msg)
        self.requested_keyframe += 1

    def reqeust_mapping(self, cur_frame_idx, viewpoint):
        msg = ["map", cur_frame_idx, viewpoint]
        self.backend_queue.put(msg)

    def request_init(self, cur_frame_idx, viewpoint, depth_map):
        msg = ["init", cur_frame_idx, viewpoint, depth_map]
        self.backend_queue.put(msg)
        self.requested_init = True

    def sync_backend(self, data):
        self.gaussians = data[1]
        occ_aware_visibility = data[2]
        keyframes = data[3]
        self.occ_aware_visibility = occ_aware_visibility

        for kf_id, kf_T in keyframes:
            self.cameras[kf_id].T = kf_T

    def cleanup(self, cur_frame_idx):
        self.cameras[cur_frame_idx].clean()
        if cur_frame_idx % 10 == 0:
            torch.cuda.empty_cache()

    def run(self):
        cur_frame_idx = 0
        # 获取投影矩阵
        projection_matrix = getProjectionMatrix2(
            znear=0.01,
            zfar=100.0,
            fx=self.dataset.fx,
            fy=self.dataset.fy,
            cx=self.dataset.cx,
            cy=self.dataset.cy,
            W=self.dataset.width,
            H=self.dataset.height,
        ).transpose(0, 1)
        projection_matrix = projection_matrix.to(device=self.device)
        self.total_frames = len(self.dataset)
        # 主循环，负责调用追踪和管理关键帧
        while True:
            if self.q_vis2main.empty():
                if self.pause:
                    continue
            else:
                # 检查 q_vis2main 队列是否有暂停或继续的信号
                data_vis2main = self.q_vis2main.get()
                self.pause = data_vis2main.flag_pause
                if self.pause:
                    self.backend_queue.put(["pause"])
                    continue
                else:
                    self.backend_queue.put(["unpause"])

            if self.frontend_queue.empty():  # 没有给后端的消息
                if cur_frame_idx >= len(self.dataset):
                    if self.save_results:
                        # eval_ate(
                        #     self.cameras,
                        #     self.kf_indices,
                        #     self.save_dir,
                        #     0,
                        #     final=True,
                        #     monocular=self.monocular,
                        # )
                        save_gaussians(
                            self.gaussians, self.save_dir, "final", final=True
                        )
                    break

                if self.requested_init:
                    time.sleep(0.01)
                    continue

                if self.single_thread and self.requested_keyframe > 0:
                    time.sleep(0.01)
                    continue

                if not self.initialized and self.requested_keyframe > 0:
                    time.sleep(0.001)
                    continue

                viewpoint = Camera.init_from_dataset(
                    self.dataset, cur_frame_idx, projection_matrix
                )
                # viewpoint存储了该帧的所有信息
                viewpoint.compute_grad_mask(self.config)
                # 存储相机位姿在字典中
                self.cameras[cur_frame_idx] = viewpoint

                if self.reset:
                    self.initialize(cur_frame_idx, viewpoint)
                    self.current_window.append(cur_frame_idx)
                    print(f"Processing frame {cur_frame_idx + 1}/{self.total_frames}")
                    cur_frame_idx += 1
                    continue

                self.initialized = self.initialized or (
                        len(self.current_window) == self.window_size
                )

                # Tracking
                # 调用 tracking 方法对当前帧进行跟踪，返回渲染包
                render_pkg = self.tracking(cur_frame_idx, viewpoint)

                # 初始化当前窗口字典，存储当前窗口中的关键帧
                current_window_dict = {}
                current_window_dict[self.current_window[0]] = self.current_window[1:]

                # 每处理 5 帧，将当前窗口中的关键帧和高斯模型发送到可视化队列
                if cur_frame_idx % 5 == 0:
                    keyframes = [CameraMsg(self.cameras[kf_idx]) for kf_idx in self.current_window]
                    self.q_main2vis.put(
                        gui_utils.GaussianPacket(
                            gaussians=clone_obj(self.gaussians),
                            keyframes=keyframes,
                            kf_window=current_window_dict,
                        )
                    )
                # 否则，只发送当前窗口中的关键帧
                else:
                    keyframes = [CameraMsg(self.cameras[kf_idx]) for kf_idx in self.current_window]
                    self.q_main2vis.put(
                        gui_utils.GaussianPacket(
                            keyframes=keyframes,
                            kf_window=current_window_dict,
                        )
                    )

                # 如果有请求关键帧，则清理当前帧的缓存并继续处理下一帧
                if self.requested_keyframe > 0:
                    self.cleanup(cur_frame_idx)
                    print(f"Processing frame {cur_frame_idx + 1}/{self.total_frames}")
                    cur_frame_idx += 1
                    continue

                # 获取上一个关键帧的索引
                last_keyframe_idx = self.current_window[0]
                # 检查是否达到创建新关键帧的时间间隔
                check_time = (cur_frame_idx - last_keyframe_idx) >= self.kf_interval
                # 获取当前帧的可见性过滤器
                curr_visibility = (render_pkg["n_touched"] > 0).long()
                # 判断是否需要创建新的关键帧
                create_kf = self.is_keyframe(
                    cur_frame_idx,
                    last_keyframe_idx,
                    curr_visibility,
                    self.occ_aware_visibility,
                )

                # 如果当前窗口中的关键帧数量小于窗口大小，进一步检查是否需要创建新关键帧
                if len(self.current_window) < self.window_size:
                    union = torch.logical_or(
                        curr_visibility, self.occ_aware_visibility[last_keyframe_idx]
                    ).count_nonzero()
                    intersection = torch.logical_and(
                        curr_visibility, self.occ_aware_visibility[last_keyframe_idx]
                    ).count_nonzero()
                    point_ratio = intersection / union
                    create_kf = (
                            check_time
                            and point_ratio < self.config["Training"]["kf_overlap"]
                    )

                # 如果是单线程模式，确保在时间间隔检查通过的情况下创建新关键帧
                if self.single_thread:
                    create_kf = check_time and create_kf

                # 如果需要创建新关键帧
                if create_kf:
                    # 将当前帧添加到窗口中，并移除重叠较少的关键帧
                    self.current_window, removed = self.add_to_window(
                        cur_frame_idx,
                        curr_visibility,
                        self.occ_aware_visibility,
                        self.current_window,
                    )
                    # 如果是单目模式且未初始化，并且移除了关键帧，则重置系统
                    if self.monocular and not self.initialized and removed is not None:
                        self.reset = True
                        Log("Keyframes lacks sufficient overlap to initialize the map, resetting.")
                        continue
                    # 添加新的关键帧并请求后端处理
                    depth_map = self.add_new_keyframe(
                        cur_frame_idx,
                        depth=render_pkg["depth"],
                        opacity=render_pkg["opacity"],
                        init=False,
                    )
                    self.request_keyframe(
                        cur_frame_idx, viewpoint, self.current_window, depth_map
                    )
                # 如果不需要创建新关键帧，则清理当前帧的缓存
                else:
                    self.cleanup(cur_frame_idx)

                # 打印当前处理的帧数并继续处理下一帧
                print(f"Processing frame {cur_frame_idx + 1}/{self.total_frames}")
                cur_frame_idx += 1

                # if (
                #     self.save_results
                #     and self.save_trj
                #     and create_kf
                #     and len(self.kf_indices) % self.save_trj_kf_intv == 0
                # ):
                #     Log("Evaluating ATE at frame: ", cur_frame_idx)
                #     eval_ate(
                #         self.cameras,
                #         self.kf_indices,
                #         self.save_dir,
                #         cur_frame_idx,
                #         monocular=self.monocular,
                #     )

            else:  # 如果队列中有东西需要传给后端
                # 根据队列中的数据类型进行处理
                data = self.frontend_queue.get()
                if data[0] == "sync_backend":
                    self.sync_backend(data)

                elif data[0] == "keyframe":
                    self.sync_backend(data)
                    self.requested_keyframe -= 1

                elif data[0] == "init":
                    self.sync_backend(data)
                    self.requested_init = False

                elif data[0] == "stop":
                    Log("Frontend Stopped.")
                    break
