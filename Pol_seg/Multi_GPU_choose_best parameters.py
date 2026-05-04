# 批量处理不同参数组合以优化图像分割效果

import os
import numpy as np
import cv2
import torch
import cupy as cp
from cuml.preprocessing import StandardScaler as CumlStandardScaler
from cuml.cluster import HDBSCAN
import torch.multiprocessing as mp
from tqdm import tqdm
# 添加缓存装饰器避免重复计算
from functools import lru_cache
from skimage.measure import label, regionprops


def worker_init(gpu_id):
    try:
        torch.cuda.set_device(gpu_id)
        cp.cuda.Device(gpu_id).use()
        print(f"Worker initialized on GPU {gpu_id}")
    except Exception as e:
        print(f"Error initializing worker on GPU {gpu_id}: {str(e)}")
        raise e


class ImageSegmenter:
    def __init__(self, min_cluster_size=20, min_samples=5, min_region_size=100,
                 rgb_weight=None, aolp_weight=None, gpu_id=0, use_min_region_size=True):
        self.min_cluster_size = min_cluster_size  # HDBSCAN parameter
        self.min_samples = min_samples  # HDBSCAN parameter
        self.min_region_size = min_region_size
        self.manual_rgb_weight = rgb_weight  # 添加RGB权重参数
        self.manual_aolp_weight = aolp_weight  # 添加AOLP权重参数
        self.use_min_region_size = use_min_region_size  # 控制是否使用min_region_size
        self.gpu_id = gpu_id
        self.device = f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu'

        # 验证权重参数
        if (rgb_weight is not None and aolp_weight is None) or \
                (rgb_weight is None and aolp_weight is not None):
            raise ValueError("Both rgb_weight and aolp_weight must be provided together or both be None")

        if rgb_weight is not None and aolp_weight is not None:
            if not (0 <= rgb_weight <= 1 and 0 <= aolp_weight <= 1):
                raise ValueError("Weights must be between 0 and 1")

        # Set random seeds，不是序列之间颜色一样，而是每次运行的同一张图，分割颜色一样
        np.random.seed(42)
        torch.manual_seed(42)

    def process_image(self, rgb_path, aolp_path, output_polar_seg_dir, output_polar_seg_vis_dir, param_str):
        """Process a single image with the current parameters"""
        try:
            # 确保输出目录存在
            os.makedirs(output_polar_seg_dir, exist_ok=True)
            os.makedirs(output_polar_seg_vis_dir, exist_ok=True)

            # 获取文件基本名称
            filename_base = os.path.splitext(os.path.basename(aolp_path))[0]

            # 加载并预处理图像
            self.load_and_preprocess(rgb_path, aolp_path)
            features = self.prepare_features()
            clustering_results = self.hdbscan_clustering(features)

            # 处理聚类结果
            for labels, confidences, n_clusters, criterion in clustering_results:
                segmented = self.visualize_segments(
                    labels, confidences,
                    self.rgb.shape[0],
                    self.rgb.shape[1],
                    n_clusters
                )

                # 保存分割结果，添加参数信息到文件名
                raw_seg_filename = os.path.join(
                    output_polar_seg_dir,
                    f"{filename_base}_{param_str}.png"
                )
                cv2.imwrite(
                    raw_seg_filename,
                    labels.reshape(self.rgb.shape[0], self.rgb.shape[1]).astype(np.uint8)
                )

                # 保存可视化结果，添加参数信息到文件名
                vis_seg_filename = os.path.join(
                    output_polar_seg_vis_dir,
                    f"{filename_base}_{param_str}_vis.png"
                )
                cv2.imwrite(
                    vis_seg_filename,
                    (segmented * 255).astype(np.uint8)
                )

            return True

        except Exception as e:
            print(f"Error processing image with parameters {param_str}: {str(e)}")
            return False

    def compute_feature_weights_gpu(self, rgb_features, aolp_features):
        """使用GPU计算RGB和AOLP特征的权重，如果设置了手动权重则使用手动权重"""
        if self.manual_rgb_weight is not None and self.manual_aolp_weight is not None:
            return self.manual_rgb_weight, self.manual_aolp_weight

        # 在GPU上计算方差
        rgb_var = cp.var(rgb_features, axis=0).mean()
        aolp_var = cp.var(aolp_features, axis=0).mean()

        # 使用方差比作为权重
        total_var = rgb_var + aolp_var
        rgb_weight = float(rgb_var / total_var)
        aolp_weight = float(aolp_var / total_var)

        return rgb_weight, aolp_weight

    def prepare_features(self):
        try:
            if not hasattr(self, 'device'):
                self.device = f'cuda:{self.gpu_id}' if torch.cuda.is_available() else 'cpu'

            current_device = self.gpu_id

            # 使用上下文管理器确保GPU资源正确释放
            with cp.cuda.Device(current_device):
                # RGB和AOLP数据转换为GPU张量
                rgb_tensor = torch.from_numpy(self.rgb).float().to(self.device) / 255.0
                aolp_tensor = torch.from_numpy(self.aolp_rgb).float().to(self.device) / 255.0

                # 特征准备
                rgb_features = rgb_tensor.reshape(-1, 3)
                aolp_features = aolp_tensor.reshape(-1, 3)

                # 转换为cupy数组并计算权重
                rgb_features_cp = cp.asarray(rgb_features.cpu().numpy())
                aolp_features_cp = cp.asarray(aolp_features.cpu().numpy())

                rgb_weight, aolp_weight = self.compute_feature_weights_gpu(
                    rgb_features_cp, aolp_features_cp)

                # 组合特征并标准化
                features = cp.concatenate([
                    rgb_features_cp * rgb_weight,
                    aolp_features_cp * aolp_weight
                ], axis=1)

                scaler = CumlStandardScaler()
                features_scaled = scaler.fit_transform(features)

                return features_scaled

        except Exception as e:
            print(f"Error in feature preparation: {str(e)}")
            raise e
        finally:
            # 清理GPU内存
            cp.get_default_memory_pool().free_all_blocks()

    def hdbscan_clustering(self, features):
        try:
            # HDBSCAN聚类
            clusterer = HDBSCAN(
                min_cluster_size=self.min_cluster_size,
                min_samples=self.min_samples,
                metric='euclidean',
                cluster_selection_method='eom',
                prediction_data=True
            )

            # 执行聚类
            clusterer.fit(features)
            labels = clusterer.labels_
            probabilities = clusterer.probabilities_
            confidences = cp.asnumpy(probabilities)

            # 将标签转换为numpy数组
            labels = cp.asnumpy(labels)

            # 添加这段代码：检测并处理纯黑或纯白区域
            height, width = self.rgb.shape[:2]
            rgb_reshaped = self.rgb.reshape(-1, 3)

            # 计算每个像素的亮度和色彩变化
            luminance = 0.299 * rgb_reshaped[:, 2] + 0.587 * rgb_reshaped[:, 1] + 0.114 * rgb_reshaped[:, 0]

            # # 识别纯黑区域 (低亮度)
            # black_mask = luminance < 5  # 可调整阈值

            # 识别纯白区域 (高亮度且低饱和度)
            white_mask = luminance > 250  # 可调整阈值

            # 如果需要考虑AOLP特征
            aolp_reshaped = self.aolp_rgb.reshape(-1, 3)
            aolp_variation = np.std(aolp_reshaped, axis=1)
            aolp_low_feature = aolp_variation < 1  # 可调整阈值

            # 结合上述条件，判断低特征区域
            low_feature_mask = white_mask | aolp_low_feature

            # 将低特征区域标记为噪声，以免出现彩色碎块
            labels[low_feature_mask] = -1

            # 处理噪声点
            noise_mask = labels == -1
            labels[noise_mask] = 0
            confidences[noise_mask] = 0.1

            # 获取非零标签并重新映射为连续值
            non_zero_labels = np.unique(labels[labels > 0])
            new_label_map = {old_label: new_label + 1 for new_label, old_label in enumerate(non_zero_labels)}
            new_label_map[0] = 0  # 保持0标签不变

            # 应用新的标签映射
            new_labels = np.zeros_like(labels)
            for old_label, new_label in new_label_map.items():
                new_labels[labels == old_label] = new_label

            # 更新标签
            labels = new_labels

            # 仅当use_min_region_size为True时才执行区域合并
            if self.use_min_region_size and self.min_region_size > 0:
                labels, confidences = self.merge_small_regions(labels, confidences)

                # 合并后再次进行连续化处理
                non_zero_labels = np.unique(labels[labels > 0])
                new_label_map = {old_label: new_label + 1 for new_label, old_label in enumerate(non_zero_labels)}
                new_label_map[0] = 0

                new_labels = np.zeros_like(labels)
                for old_label, new_label in new_label_map.items():
                    new_labels[labels == old_label] = new_label

                labels = new_labels
                print(f"Unique labels after merging and remapping: {np.unique(labels)}")

            n_clusters = len(np.unique(labels))
            return [(labels, confidences, n_clusters, 'HDBSCAN')]

        except Exception as e:
            print(f"Error in HDBSCAN clustering: {str(e)}")
            raise e

    def merge_small_regions_cpu(self, labels, confidences, window_size=5):
        from skimage.measure import label, regionprops

        height, width = self.rgb.shape[:2]
        labels_2d = labels.reshape(height, width)
        confidences_2d = confidences.reshape(height, width)
        # print(f"Unique labels before merge: {np.unique(labels_2d)}")
        labeled_img, num_labels = label(labels_2d, return_num=True)

        for i in range(1, num_labels + 1):
            region_mask = labeled_img == i
            region_size = np.sum(region_mask)

            if region_size < self.min_region_size:
                props = regionprops(region_mask.astype(int))[0]
                centroid_row, centroid_col = props.centroid  # 保留浮点数质心

                half_window = window_size // 2
                best_dist = float('inf')
                target_label = -1
                target_confidence = 0

                row_start = max(0, int(centroid_row) - half_window)
                row_end = min(height, int(centroid_row) + half_window + 1)
                col_start = max(0, int(centroid_col) - half_window)
                col_end = min(width, int(centroid_col) + half_window + 1)

                window_region = labels_2d[row_start:row_end, col_start:col_end]
                window_confidences = confidences_2d[row_start:row_end, col_start:col_end]
                current_label = labels_2d[int(centroid_row), int(centroid_col)]

                unique_labels = np.unique(window_region)
                for label in unique_labels:
                    if label != current_label:
                        label_mask = window_region == label
                        if np.sum(label_mask) > 0:
                            neighbor_coords = np.argwhere(label_mask)
                            for nr, nc in neighbor_coords:
                                dist = np.sqrt((nr - half_window) ** 2 + (nc - half_window) ** 2)
                                if dist < best_dist:
                                    best_dist = dist
                                    target_label = label
                                    target_confidence = np.mean(window_confidences[label_mask])

                if target_label != -1:
                    labels_2d[region_mask] = target_label
                    current_confidence = np.mean(confidences_2d[region_mask])
                    labels_2d[region_mask] = target_label
                    confidences_2d[region_mask] = (current_confidence * region_size + target_confidence * np.sum(
                        window_region == target_label)) / (region_size + np.sum(window_region == target_label))

        # 只进行一次标签重映射
        non_zero_labels = np.unique(labels_2d[labels_2d > 0])
        new_label_map = {old_label: new_label + 1 for new_label, old_label in enumerate(non_zero_labels)}
        new_label_map[0] = 0
        new_labels = np.zeros_like(labels_2d)
        for old_label, new_label in new_label_map.items():
            new_labels[labels_2d == old_label] = new_label
        labels_2d = new_labels
        # print(f"Unique labels after merge: {np.unique(new_labels)}")
        return labels_2d.reshape(-1), confidences_2d.reshape(-1)

    def merge_small_regions(self, labels, confidences, window_size=5):
        """
        GPU-accelerated version of merge_small_regions function
        Uses CuPy for GPU computations to improve performance
        """
        try:
            height, width = self.rgb.shape[:2]

            # Convert data to CuPy arrays for GPU acceleration
            labels_2d = cp.asarray(labels.reshape(height, width))
            confidences_2d = cp.asarray(confidences.reshape(height, width))

            # Use CuPy's connected component labeling (similar to skimage.measure.label)
            # First, create binary masks for each label
            unique_labels = cp.unique(labels_2d)

            # Create a mask for each label and check its size
            for label_val in unique_labels:
                if label_val == 0:  # Skip background label
                    continue

                # Create binary mask for current label
                region_mask = labels_2d == label_val
                region_size = cp.sum(region_mask).item()

                # Only process regions smaller than the minimum size
                if region_size < self.min_region_size:
                    # Find region properties (centroid)
                    # We'll compute the centroid directly with CuPy
                    y_indices, x_indices = cp.where(region_mask)
                    if len(y_indices) == 0:
                        continue

                    centroid_row = cp.mean(y_indices).item()
                    centroid_col = cp.mean(x_indices).item()

                    # Define window around centroid
                    half_window = window_size // 2
                    row_start = max(0, int(centroid_row) - half_window)
                    row_end = min(height, int(centroid_row) + half_window + 1)
                    col_start = max(0, int(centroid_col) - half_window)
                    col_end = min(width, int(centroid_col) + half_window + 1)

                    # Extract window region
                    window_region = labels_2d[row_start:row_end, col_start:col_end]
                    window_confidences = confidences_2d[row_start:row_end, col_start:col_end]

                    # Find unique labels in the window excluding the current label
                    window_unique_labels = cp.unique(window_region)
                    window_unique_labels = window_unique_labels[window_unique_labels != label_val]

                    if len(window_unique_labels) == 0:
                        continue

                    # Find best neighbor label
                    best_dist = float('inf')
                    target_label = -1
                    target_confidence = 0

                    # Create coordinate matrices for the window
                    y_grid, x_grid = cp.mgrid[row_start:row_end, col_start:col_end]

                    # Calculate center of the window
                    center_y = centroid_row
                    center_x = centroid_col

                    for neighbor_label in window_unique_labels:
                        if neighbor_label != 0:  # Skip background
                            # Create mask for this neighbor label
                            neighbor_mask = window_region == neighbor_label

                            if cp.sum(neighbor_mask) > 0:
                                # Get coordinates of pixels with this label
                                neighbor_y = y_grid[neighbor_mask]
                                neighbor_x = x_grid[neighbor_mask]

                                # Calculate distances to center
                                distances = cp.sqrt((neighbor_y - center_y) ** 2 + (neighbor_x - center_x) ** 2)
                                min_dist = cp.min(distances).item()

                                if min_dist < best_dist:
                                    best_dist = min_dist
                                    target_label = neighbor_label
                                    target_confidence = cp.mean(window_confidences[neighbor_mask]).item()

                    if target_label != -1:
                        # Merge the regions
                        current_confidence = cp.mean(confidences_2d[region_mask]).item()
                        target_mask = labels_2d == target_label
                        target_size = cp.sum(target_mask).item()

                        # Update labels
                        labels_2d[region_mask] = target_label

                        # Update confidences - weighted average
                        new_confidence = (current_confidence * region_size +
                                          target_confidence * target_size) / (region_size + target_size)
                        confidences_2d[region_mask] = new_confidence

            # Convert back to numpy arrays
            result_labels = cp.asnumpy(labels_2d)
            result_confidences = cp.asnumpy(confidences_2d)

            return result_labels.reshape(-1), result_confidences.reshape(-1)

        except Exception as e:
            print(f"Error in GPU merge_small_regions: {str(e)}")
        finally:
            # Free GPU memory
            cp.get_default_memory_pool().free_all_blocks()

    @lru_cache(maxsize=32)
    def generate_distinct_colors(self, n):
        colors = [[0, 0, 0]]  # First color is black for noise
        golden_ratio = 0.618033988749895
        hues = [(i * golden_ratio) % 1.0 for i in range(n - 1)]

        for hue in hues:
            s = 0.95  # High saturation
            v = 0.95  # High value
            h = hue * 360  # Convert to degrees

            c = v * s
            x = c * (1 - abs((h / 60) % 2 - 1))
            m = v - c

            if h < 60:
                r, g, b = c, x, 0
            elif h < 120:
                r, g, b = x, c, 0
            elif h < 180:
                r, g, b = 0, c, x
            elif h < 240:
                r, g, b = 0, x, c
            elif h < 300:
                r, g, b = x, 0, c
            else:
                r, g, b = c, 0, x

            colors.append([r + m, g + m, b + m])

        return np.array(colors)

    def visualize_segments(self, labels, confidences, height, width, n_clusters):
        """Visualize segments with distinct colors"""
        # Generate colors
        colors = self.generate_distinct_colors(n_clusters)
        colors[1:] = np.clip(colors[1:] * 1.2, 0, 1)  # Enhance contrast

        # Create segmentation image
        segmented = np.zeros((height, width, 3))
        labels_2d = labels.reshape(height, width)
        confidences_2d = confidences.reshape(height, width)

        # Apply colors
        for i in range(n_clusters):
            mask = labels_2d == i
            if i == 0:  # Noise cluster
                segmented[mask] = colors[0]
            else:
                for c in range(3):
                    segmented[:, :, c][mask] = colors[i, c] * (0.7 + 0.3 * confidences_2d[mask])

        # Remove the early return that was here
        return segmented

    def load_and_preprocess(self, rgb_path, aolp_path):
        """Load and preprocess images"""
        # Read images
        self.rgb = cv2.imread(rgb_path)
        self.aolp = cv2.imread(aolp_path)

        # Ensure images are the same size
        if self.rgb.shape != self.aolp.shape:
            self.aolp = cv2.resize(self.aolp, (self.rgb.shape[1], self.rgb.shape[0]))

        # Store AOLP RGB values
        self.aolp_rgb = self.aolp.copy()

        # Apply bilateral filter to AOLP
        self.aolp_filtered = cv2.bilateralFilter(self.aolp, d=9, sigmaColor=150, sigmaSpace=150)

        # Create brightness mask
        # gray = cv2.cvtColor(self.aolp_filtered, cv2.COLOR_BGR2GRAY)
        # brightness_mask = gray < 5
        #
        # # Set low brightness areas to zero
        # self.aolp_filtered[brightness_mask] = 0
        self.aolp_rgb = self.aolp_filtered.copy()
        return self.rgb, self.aolp_rgb


def process_with_params(args):
    """Worker function for processing image with specific parameters"""
    gpu_id, rgb_path, aolp_path, output_polar_seg_dir, output_polar_seg_vis_dir, params = args

    # 设置当前GPU
    torch.cuda.set_device(gpu_id)
    cp.cuda.Device(gpu_id).use()

    # 创建参数字符串用于文件名
    use_mrs_str = "use_mrs" if params.get('use_min_region_size', True) else "no_mrs"
    param_str = f"mcs{params['min_cluster_size']}_ms{params['min_samples']}_mrs{params['min_region_size']}_rgbw{params['rgb_weight']}_aolpw{params['aolp_weight']}_{use_mrs_str}"

    # 创建分割器实例
    segmenter = ImageSegmenter(
        min_cluster_size=params['min_cluster_size'],
        min_samples=params['min_samples'],
        min_region_size=params['min_region_size'],
        rgb_weight=params['rgb_weight'],
        aolp_weight=params['aolp_weight'],
        gpu_id=gpu_id,
        use_min_region_size=params.get('use_min_region_size', True)
    )

    # 处理图像
    success = segmenter.process_image(
        rgb_path,
        aolp_path,
        output_polar_seg_dir,
        output_polar_seg_vis_dir,
        param_str
    )

    # 清理GPU内存
    cp.get_default_memory_pool().free_all_blocks()
    torch.cuda.empty_cache()

    return param_str, success


def main():
    # 设置路径
    aolp_dir = r'/root/datasets/my/polar_depth_V2/chair_600_around/aolp_1chanel_cycle_pi_weight/'
    rgb_dir = r'/root/datasets/my/polar_depth_V2/chair_600_around/Id_enhance/'
    output_polar_seg_dir = r'/root/datasets/my/polar_depth_V2/chair_600_around/polar_seg'
    output_polar_seg_vis_dir = r'/root/datasets/my/polar_depth_V2/chair_600_around/polar_seg_vis'

    # 确保输出目录存在
    os.makedirs(output_polar_seg_dir, exist_ok=True)
    os.makedirs(output_polar_seg_vis_dir, exist_ok=True)

    # 获取图像列表
    aolp_images = sorted([f for f in os.listdir(aolp_dir) if f.lower().endswith('.png')])
    rgb_images = sorted([f for f in os.listdir(rgb_dir) if f.lower().endswith('.png')])

    # 均匀采样5对图像
    if len(aolp_images) > 0 and len(rgb_images) > 0:
        total_images = len(aolp_images)
        samples_count = 5  # 采样5对图像

        if total_images <= samples_count:
            # 如果图像总数小于等于需要的采样数，使用所有图像
            indices = list(range(total_images))
        else:
            # 均匀采样
            step = total_images / samples_count
            indices = [int(i * step) for i in range(samples_count)]

        print(f"选择的图像索引: {indices}")
        image_pairs = [(rgb_images[i], aolp_images[i]) for i in indices]

        for idx, (rgb_image_name, aolp_image_name) in enumerate(image_pairs):
            aolp_image_path = os.path.join(aolp_dir, aolp_image_name)
            rgb_image_path = os.path.join(rgb_dir, rgb_image_name)

            filename_base = os.path.splitext(aolp_image_name)[0]
            print(f"\n处理第 {idx + 1} 对图像: {rgb_image_name} 和 {aolp_image_name}")

            # 定义要测试的参数组合
            min_cluster_size_options = [15, 20, 25]
            min_samples_options = [15, 17, 20]
            min_region_size_options = [2000, 4000]
            weight_options = [(0.2, 0.8), (0.4, 0.6), (0.5, 0.5)]
            use_min_region_size_options = [True]

            # 生成所有参数组合
            param_combinations = []
            for mcs in min_cluster_size_options:
                for ms in min_samples_options:
                    for mrs in min_region_size_options:
                        for rgb_w, aolp_w in weight_options:
                            for use_mrs in use_min_region_size_options:
                                param_combinations.append({
                                    'min_cluster_size': mcs,
                                    'min_samples': ms,
                                    'min_region_size': mrs,
                                    'rgb_weight': rgb_w,
                                    'aolp_weight': aolp_w,
                                    'use_min_region_size': use_mrs
                                })

            print(f"总共要测试 {len(param_combinations)} 种参数组合")

            # 设置GPU IDs
            gpu_ids = [0, 1, 2]  # 使用GPU
            num_gpus = len(gpu_ids)

            # 将参数分配到不同的GPU上
            gpu_tasks = []
            for i, params in enumerate(param_combinations):
                gpu_id = gpu_ids[i % num_gpus]
                gpu_tasks.append(
                    (gpu_id, rgb_image_path, aolp_image_path, output_polar_seg_dir, output_polar_seg_vis_dir, params))

            # 设置多进程
            if idx == 0:  # 仅在第一次迭代时设置
                mp.set_start_method('spawn', force=True)

            # 创建进度条
            pbar = tqdm(total=len(param_combinations), desc=f"图像 {idx + 1}/{samples_count} 参数优化进度")

            # 多进程运行不同参数组合
            results = []
            with mp.Pool(processes=num_gpus, initializer=worker_init, initargs=(gpu_ids[0],)) as pool:
                for param_str, success in pool.imap_unordered(process_with_params, gpu_tasks):
                    results.append((param_str, success))
                    if success:
                        print(f"成功处理参数组合: {param_str}")
                    else:
                        print(f"处理参数组合失败: {param_str}")
                    pbar.update(1)

            # 关闭进度条
            pbar.close()

            # 统计结果
            successful = [r[0] for r in results if r[1]]
            failed = [r[0] for r in results if not r[1]]

            print(f"\n图像 {idx + 1}/{samples_count} 参数优化完成。")
            print(f"成功: {len(successful)}/{len(param_combinations)}")
            print(f"失败: {len(failed)}/{len(param_combinations)}")
    else:
        print("未找到任何图像文件！")
        return

    print(f"所有图像处理完成。结果保存在:\n{output_polar_seg_dir}\n{output_polar_seg_vis_dir}")

if __name__ == "__main__":
    # 运行主函数
    main()