import torchvision.transforms as transforms
import os
import numpy as np
import torch
from PIL import Image
from datetime import datetime
import matplotlib

cmap = matplotlib.colormaps.get_cmap('Spectral')


def initialize_visualization_folder(weight_iso_loss, weight_normal_loss, mapping_itr_num_multi_thread):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"render_result/{timestamp}_wiso{weight_iso_loss}_wn{weight_normal_loss}_itr{mapping_itr_num_multi_thread}"
    os.makedirs(f"{folder_name}/step0/depth", exist_ok=True)
    os.makedirs(f"{folder_name}/step0/combine", exist_ok=True)
    os.makedirs(f"{folder_name}/step0/render_rgb", exist_ok=True)
    os.makedirs(f"{folder_name}/step0/depth_normal", exist_ok=True)
    os.makedirs(f"{folder_name}/step0/normal", exist_ok=True)
    os.makedirs(f"{folder_name}/step0/gt_normal_8bit", exist_ok=True)
    os.makedirs(f"{folder_name}/step1/depth", exist_ok=True)
    os.makedirs(f"{folder_name}/step1/combine", exist_ok=True)
    os.makedirs(f"{folder_name}/step1/render_rgb", exist_ok=True)
    os.makedirs(f"{folder_name}/step1/depth_normal", exist_ok=True)
    os.makedirs(f"{folder_name}/step1/normal", exist_ok=True)
    os.makedirs(f"{folder_name}/step1/gt_normal_8bit", exist_ok=True)
    os.makedirs(f"{folder_name}/step2/depth", exist_ok=True)
    os.makedirs(f"{folder_name}/step2/combine", exist_ok=True)
    os.makedirs(f"{folder_name}/step2/render_rgb", exist_ok=True)
    os.makedirs(f"{folder_name}/step2/depth_normal", exist_ok=True)
    os.makedirs(f"{folder_name}/step2/normal", exist_ok=True)
    os.makedirs(f"{folder_name}/step2/gt_normal_8bit", exist_ok=True)
    return f"{folder_name}/step0", f"{folder_name}/step1", f"{folder_name}/step2"

def normalize_with_percentile(img, lower_percent=1, upper_percent=99):
    """
    使用百分位数进行归一化，避免异常值的影响，并返回PIL图像
    """
    # 如果输入是PIL图像，先转换为numpy数组
    if isinstance(img, Image.Image):
        img_array = np.array(img)
    else:
        img_array = img

    lower_val = np.percentile(img_array, lower_percent)
    upper_val = np.percentile(img_array, upper_percent)

    # 裁剪图像到百分位范围
    img_clipped = np.clip(img_array, lower_val, upper_val)

    # 归一化到 [0, 255]
    img_normalized = ((img_clipped - lower_val) / (upper_val - lower_val) * 255.0)
    img_normalized = np.clip(img_normalized, 0, 255).astype(np.uint8)

    # 转换回PIL图像
    output_img = Image.fromarray(img_normalized)

    return output_img

def save_visualizations(folder_name, image, depth, depth_normal, normal, gt_normal, cur_frame_idx, save_raw_normal=False, depth_scale=1000):
    def visualize_depth(depth):
        depth_normalized = depth / depth.max()
        depth_np = depth_normalized.detach().cpu().numpy()
        if len(depth_np.shape) == 3 and depth_np.shape[0] == 1:
            depth_np = depth_np[0]
        colored = cmap(depth_np)
        colored_rgb = colored[..., :3]
        colored_rgb = np.transpose(colored_rgb, (2, 0, 1))
        colored_rgb = (colored_rgb * 255).astype(np.uint8)
        return colored_rgb

    def convert_depth_to_16bit_image(depth, depth_scale=2000):
        depth_16bit = (depth * depth_scale).to(torch.uint16)
        depth_np = depth_16bit.cpu().numpy()
        if len(depth_np.shape) == 3:
            if depth_np.shape[0] == 1:
                depth_np = depth_np[0]
            else:
                raise ValueError(f"Unexpected depth shape: {depth_np.shape}")
        depth_np = depth_np.astype(np.uint16)
        depth_16bit_pil = Image.fromarray(depth_np)
        return depth_16bit_pil

    depth_normalized = depth / depth.max()
    depth_colored = visualize_depth(depth_normalized)

    depth_colored_pil = np.transpose(depth_colored, (1, 2, 0))
    depth_img = Image.fromarray(depth_colored_pil)

    processed_normal = (-normal.detach().cpu() + 1) / 2
    normal_img = transforms.ToPILImage()(processed_normal)

    processed_depth_normal = (-depth_normal.detach().cpu() + 1) / 2
    depth_normal_img = transforms.ToPILImage()(processed_depth_normal)

    # 修改后的代码
    numpy_img = image.detach().cpu().numpy()
    # 如果是形状为[C,H,W]的张量，需要调整通道顺序为[H,W,C]
    if len(numpy_img.shape) == 3 and numpy_img.shape[0] in [1, 3, 4]:  # 如果第一维是通道数
        numpy_img = numpy_img.transpose(1, 2, 0)
        # 如果是单通道，去掉多余的维度
        if numpy_img.shape[2] == 1:
            numpy_img = numpy_img.squeeze(2)
    render_img = normalize_with_percentile(numpy_img, 1, 99)

    # Handle case when gt_normal is None or empty
    # if gt_normal is not None and gt_normal.numel() > 0:
    #     processed_gt_normal = (-gt_normal.detach().cpu() + 1) / 2
    #     gt_normal_img = transforms.ToPILImage()(processed_gt_normal)
    #     # Create combined image with all four images
    #     combined_img = Image.new('RGB', (render_img.width + depth_img.width + depth_normal_img.width + normal_img.width + gt_normal_img.width,
    #                                    render_img.height))
    #     combined_img.paste(render_img, (0, 0))
    #     combined_img.paste(depth_img, (render_img.width, 0))
    #     combined_img.paste(depth_normal_img, (render_img.width + depth_img.width, 0))
    #     combined_img.paste(normal_img, (render_img.width + depth_img.width + depth_normal_img.width, 0))
    #     combined_img.paste(gt_normal_img, (render_img.width + depth_img.width + depth_normal_img.width + normal_img.width, 0))
    #     gt_normal_img.save(f"{folder_name}/gt_normal_8bit/{cur_frame_idx}.png")
    # else:
    #     # Create combined image with only three images
    #     combined_img = Image.new('RGB', (render_img.width + depth_img.width + normal_img.width + depth_normal_img.width,
    #                                    render_img.height))
    #     combined_img.paste(render_img, (0, 0))
    #     combined_img.paste(depth_img, (render_img.width, 0))
    #     combined_img.paste(depth_normal_img, (render_img.width + depth_img.width, 0))
    #     combined_img.paste(normal_img, (render_img.width + depth_img.width + depth_normal_img.width, 0))
    #
    # combined_img.save(f"{folder_name}/combine/{cur_frame_idx}.png")
    render_img.save(f"{folder_name}/render_rgb/{cur_frame_idx}.png")
    depth_normal_img.save(f"{folder_name}/depth_normal/{cur_frame_idx}.png")
    normal_img.save(f"{folder_name}/normal/{cur_frame_idx}.png")
    depth_16bit_pil = convert_depth_to_16bit_image(depth, depth_scale)
    depth_16bit_pil.save(f"{folder_name}/depth/{cur_frame_idx}.png")

    if save_raw_normal:
        raw_normal_path = f"{folder_name}/raw_normal/{cur_frame_idx}.npy"
        np.save(raw_normal_path, processed_normal.numpy())