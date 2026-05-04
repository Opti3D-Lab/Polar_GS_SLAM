
import os
from time import sleep
import time
import numpy as np
import cv2
import torch
import cupy as cp
from cuml.preprocessing import StandardScaler as CumlStandardScaler
from cuml.cluster import HDBSCAN
import torch.multiprocessing as mp
from tqdm import tqdm
import random
import math
import numpy.random as np_random
# 添加缓存装饰器避免重复计算
from functools import lru_cache
from skimage.measure import label, regionprops
import skimage.measure as measure
from concurrent.futures import ThreadPoolExecutor, TimeoutError

def worker_init(gpu_id):
    try:
        torch.cuda.set_device(gpu_id)
        print(f"Worker initialized on GPU {gpu_id}")
    except Exception as e:
        print(f"Error initializing worker on GPU {gpu_id}: {str(e)}")
        raise e


class ImageSegmenter:
    def __init__(self, min_cluster_size=20, min_samples=5, min_region_size=100,
                 rgb_weight=None, aolp_weight=None, dolp_weight=None, gpu_ids=None,
                 enable_merge_regions=True):
        self.min_cluster_size = min_cluster_size  # HDBSCAN parameter
        self.min_samples = min_samples  # HDBSCAN parameter
        self.min_region_size = min_region_size
        self.manual_rgb_weight = rgb_weight    # 添加RGB权重参数
        self.manual_aolp_weight = aolp_weight  # 添加AOLP权重参数
        self.manual_dolp_weight = dolp_weight  # 添加DOLP权重参数
        self.enable_merge_regions = enable_merge_regions  # 新增控制开关
        self.gpu_ids = gpu_ids if gpu_ids else [0]
        self.num_gpus = len(self.gpu_ids) if isinstance(gpu_ids, (list, tuple)) else 1
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # 权重验证逻辑也需要更新
        if ((rgb_weight is not None or aolp_weight is not None or dolp_weight is not None) and
                not (rgb_weight is not None and aolp_weight is not None and dolp_weight is not None)):
            raise ValueError(
                "All three weights (rgb_weight, aolp_weight, and dolp_weight) must be provided together or all be None")

        if rgb_weight is not None and aolp_weight is not None and dolp_weight is not None:
            if not (0 <= rgb_weight <= 1 and 0 <= aolp_weight <= 1 and 0 <= dolp_weight <= 1):
                raise ValueError("Weights must be between 0 and 1")
            # 验证三个权重之和为1
            if not math.isclose(rgb_weight + aolp_weight + dolp_weight, 1.0, rel_tol=1e-5):
                raise ValueError("Sum of weights must be 1.0")

        # Set random seeds，不是序列之间颜色一样，而是每次运行的同一张图，分割颜色一样
        random.seed(42)
        np_random.seed(42)
        torch.manual_seed(42)

    def process_images_parallel(self, image_list, output_polar_seg_dir, output_polar_seg_vis_dir):
        """Process multiple images in parallel"""
        total_images = len(image_list)
        print(f"\nTotal images to process: {total_images}")

        # Create output directories
        os.makedirs(output_polar_seg_dir, exist_ok=True)
        os.makedirs(output_polar_seg_vis_dir, exist_ok=True)

        # Create global progress bar
        global_progress = tqdm(total=total_images, desc="Total Progress", position=0)
        processed_count = 0

        # Split image list into batches
        batch_size = max(1, len(image_list) // self.num_gpus)
        batches = [image_list[i:i + batch_size] for i in range(0, len(image_list), batch_size)]

        # Ensure number of batches doesn't exceed GPU count
        while len(batches) > self.num_gpus:
            last_batch = batches.pop()
            batches[-1].extend(last_batch)

        # Prepare work arguments
        work_args = [(gpu_id, batch, output_polar_seg_dir, output_polar_seg_vis_dir)
                     for gpu_id, batch in zip(self.gpu_ids, batches)]

        with mp.Pool(
                processes=self.num_gpus,
                initializer=worker_init,
                initargs=(self.gpu_ids[0],)
        ) as pool:
            try:
                for result in pool.imap_unordered(self.process_batch, work_args):
                    processed_count += len(result) if result else 0
                    global_progress.update(len(result) if result else 0)

            except Exception as e:
                print(f"Error in parallel processing: {str(e)}")
                raise e
            finally:
                global_progress.close()
                pool.close()
                pool.join()

        print(f"\nProcessing completed. Total images processed: {processed_count}/{total_images}")

    def compute_feature_weights_gpu(self, rgb_features, aolp_features, dolp_features):
        """使用GPU计算RGB、AOLP和DOLP特征的权重"""
        if (self.manual_rgb_weight is not None and
                self.manual_aolp_weight is not None and
                self.manual_dolp_weight is not None):
            return self.manual_rgb_weight, self.manual_aolp_weight, self.manual_dolp_weight

        # 在GPU上计算方差
        rgb_var = cp.var(rgb_features, axis=0).mean()
        aolp_var = cp.var(aolp_features, axis=0).mean()
        dolp_var = cp.var(dolp_features, axis=0).mean()

        # 使用方差比作为权重
        total_var = rgb_var + aolp_var + dolp_var
        rgb_weight = float(rgb_var / total_var)
        aolp_weight = float(aolp_var / total_var)
        dolp_weight = float(dolp_var / total_var)

        return rgb_weight, aolp_weight, dolp_weight

    def process_batch(self, args):
        """Process a single batch of images with a timeout mechanism using threading"""
        gpu_id, batch_data, output_polar_seg_dir, output_polar_seg_vis_dir = args
        try:
            print(f"Starting batch processing on GPU {gpu_id}")
            torch.cuda.set_device(gpu_id)
            self.device = f'cuda:{gpu_id}'
            print(f"Using device: {self.device}")

            results = []
            skipped_due_to_timeout = []  # Track images skipped due to timeout

            for rgb_path, aolp_path, dolp_path, filename_base in tqdm(
                    batch_data,
                    desc=f"GPU {gpu_id}",
                    leave=False
            ):
                # 使用线程超时机制
                result = None
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(
                        self._process_single_image_wrapper,
                        rgb_path, aolp_path, dolp_path, filename_base, output_polar_seg_dir, output_polar_seg_vis_dir, gpu_id
                    )

                    try:
                        # 等待最多120秒
                        result = future.result(timeout=120)
                        if result is not None:
                            results.append(result)
                    except TimeoutError:
                        # 超时处理
                        print(f"\nTimeout: Image {filename_base} processing exceeded 2 minutes. Skipping.")
                        skipped_due_to_timeout.append(filename_base)
                        # 取消任务
                        future.cancel()
                        # 清理GPU内存
                        torch.cuda.empty_cache()
                        if 'cp' in globals() and hasattr(cp, 'get_default_memory_pool'):
                            cp.get_default_memory_pool().free_all_blocks()
                    except Exception as e:
                        print(f"Error processing image {filename_base}: {str(e)}")

            # 报告跳过的图像
            if skipped_due_to_timeout:
                print(f"\nSkipped {len(skipped_due_to_timeout)} image(s) due to timeout on GPU {gpu_id}:")
                for img in skipped_due_to_timeout:
                    print(f"  - {img}")

            return results

        except Exception as e:
            print(f"Batch processing error on GPU {gpu_id}: {str(e)}")
            return []

    def _process_single_image_wrapper(self, rgb_path, aolp_path, dolp_path, filename_base, output_polar_seg_dir,
                                      output_polar_seg_vis_dir, gpu_id):
        """修改后的处理函数，直接返回结果而不使用队列"""
        try:
            # 设置GPU设备
            torch.cuda.set_device(gpu_id)
            self.device = f'cuda:{gpu_id}'

            # 加载和预处理图像
            self.load_and_preprocess(rgb_path, aolp_path, dolp_path)
            features = self.prepare_features()
            clustering_results = self.hdbscan_clustering(features)

            result_list = []
            # 处理聚类结果
            for labels, confidences, n_clusters, criterion in clustering_results:
                segmented = self.visualize_segments(
                    labels, confidences,
                    self.rgb.shape[0],
                    self.rgb.shape[1],
                    n_clusters
                )

                # 保存分割结果
                raw_seg_filename = os.path.join(
                    output_polar_seg_dir,
                    f"{filename_base}.png"
                )
                cv2.imwrite(
                    raw_seg_filename,
                    labels.reshape(self.rgb.shape[0], self.rgb.shape[1]).astype(np.uint8)
                )

                # 保存可视化结果
                vis_seg_filename = os.path.join(
                    output_polar_seg_vis_dir,
                    f"{filename_base}.png"
                )
                cv2.imwrite(
                    vis_seg_filename,
                    (segmented * 255).astype(np.uint8)
                )

                result_list.append((labels, segmented, filename_base, criterion))

            return result_list[0] if result_list else None

        except Exception as e:
            print(f"Error processing image {filename_base}: {str(e)}")
            return None
        finally:
            # 清理GPU内存
            torch.cuda.empty_cache()
            if 'cp' in globals() and hasattr(cp, 'get_default_memory_pool'):
                cp.get_default_memory_pool().free_all_blocks()

    def prepare_features(self):
        try:
            if not hasattr(self, 'device'):
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

            current_device = int(self.device.split(':')[1]) if ':' in self.device else 0

            # 使用上下文管理器确保GPU资源正确释放
            with cp.cuda.Device(current_device):
                # RGB和AOLP数据转换为GPU张量
                rgb_tensor = torch.from_numpy(self.rgb).float().to(self.device) / 255.0
                aolp_tensor = torch.from_numpy(self.aolp_rgb).float().to(self.device) / 255.0
                # 处理DOLP (注意DOLP是单通道的)
                dolp_tensor = torch.from_numpy(self.dolp_filtered).float().to(self.device) / 255.0

                # 特征准备
                rgb_features = rgb_tensor.reshape(-1, 3)
                aolp_features = aolp_tensor.reshape(-1, 3)
                # DOLP特征需要从2D reshape到1D
                dolp_features = dolp_tensor.reshape(-1, 1)

                # 转换为cupy数组并计算权重
                rgb_features_cp = cp.asarray(rgb_features.cpu().numpy())
                aolp_features_cp = cp.asarray(aolp_features.cpu().numpy())
                dolp_features_cp = cp.asarray(dolp_features.cpu().numpy())

                # 计算权重
                rgb_weight, aolp_weight, dolp_weight = self.compute_feature_weights_gpu(
                    rgb_features_cp, aolp_features_cp, dolp_features_cp)

                # 组合特征
                features = cp.concatenate([
                    rgb_features_cp * rgb_weight,
                    aolp_features_cp * aolp_weight,
                    dolp_features_cp * dolp_weight  # 添加DOLP特征
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

            # 打印调试信息
            # print(f"Unique labels after remapping: {np.unique(labels)}")
            # print(f"Number of clusters: {len(np.unique(labels)) - 1}")  # 减去背景类0

            if self.enable_merge_regions and self.min_region_size > 0:
                labels, confidences = self.merge_small_regions(labels, confidences)

                # 合并后再次进行连续化处理
                non_zero_labels = np.unique(labels[labels > 0])
                new_label_map = {old_label: new_label + 1 for new_label, old_label in enumerate(non_zero_labels)}
                new_label_map[0] = 0

                new_labels = np.zeros_like(labels)
                for old_label, new_label in new_label_map.items():
                    new_labels[labels == old_label] = new_label

                labels = new_labels
                # print(f"Unique labels after merging and remapping: {np.unique(labels)}")

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
            # Fallback to CPU version if GPU version fails
            return self._merge_small_regions_cpu(labels, confidences, window_size)
        finally:
            # Free GPU memory
            cp.get_default_memory_pool().free_all_blocks()

    def visualize_segments(self, labels, confidences, height, width, n_clusters):
        """Visualize segments with distinct colors"""

        @lru_cache(maxsize=32)
        def generate_distinct_colors(n):
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

        # Generate colors
        colors = generate_distinct_colors(n_clusters)
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

        return segmented

    def load_and_preprocess(self, rgb_path, aolp_path, dolp_path):
        """Load and preprocess images"""
        # Read images
        self.rgb = cv2.imread(rgb_path)
        self.aolp = cv2.imread(aolp_path)
        # 读取DOLP图像(单通道)
        self.dolp = cv2.imread(dolp_path, cv2.IMREAD_GRAYSCALE)
        if self.dolp is None:
            raise ValueError(f"Failed to load DOLP image: {dolp_path}")

        # Ensure images are the same size
        if self.rgb.shape != self.aolp.shape:
            self.aolp = cv2.resize(self.aolp, (self.rgb.shape[1], self.rgb.shape[0]))
        # 确保所有图像尺寸一致
        if self.rgb.shape[:2] != self.dolp.shape[:2]:
            self.dolp = cv2.resize(self.dolp, (self.rgb.shape[1], self.rgb.shape[0]))

        # Store AOLP RGB values
        self.aolp_rgb = self.aolp.copy()
        # 保存原始DOLP数据
        self.dolp_original = self.dolp.copy()

        # 1. 先应用高斯模糊降噪
        self.aolp = cv2.GaussianBlur(self.aolp, (5, 5), 0)
        # 2. 然后再应用双边滤波
        self.aolp_filtered = cv2.bilateralFilter(self.aolp, d=9, sigmaColor=100, sigmaSpace=100)

        # 对DOLP进行降噪处理，类似于AOLP
        # 1. 应用高斯模糊
        self.dolp = cv2.GaussianBlur(self.dolp, (5, 5), 0)
        # 2. 应用双边滤波
        self.dolp_filtered = cv2.bilateralFilter(self.dolp, d=9, sigmaColor=100, sigmaSpace=100)

        # Create brightness mask
        gray = cv2.cvtColor(self.aolp_filtered, cv2.COLOR_BGR2GRAY)
        brightness_mask = gray < 1

        # Set low brightness areas to zero
        self.aolp_filtered[brightness_mask] = 0
        self.aolp_rgb = self.aolp_filtered.copy()
        # cv2.imwrite('/root/datasets/my/polar/object/trash_can4_low_620/aolp_filtered.png', self.aolp_filtered)
        # sleep(3)
        return self.rgb, self.aolp_rgb, self.dolp_filtered


def main():
    # Set paths
    aolp_dir = r'/data/datasets/my/polar_depth_V2/water_dispenser_620_around_xyz/aolp_1chanel_cycle_pi_weight/'
    dolp_dir = r'/data/datasets/my/polar_depth_V2/water_dispenser_620_around_xyz/dolp_1chanel/'  # 添加DOLP目录
    rgb_dir = r'/data/datasets/my/polar_depth_V2/water_dispenser_620_around_xyz/Id_enhance/'
    output_polar_seg_dir = r'/data/datasets/my/polar_depth_V2/water_dispenser_620_around_xyz/polar_seg_dolp_test'
    output_polar_seg_vis_dir = r'/data/datasets/my/polar_depth_V2/water_dispenser_620_around_xyz/polar_seg_vis_dolp_test'

    # Create output directories if they don't exist
    os.makedirs(output_polar_seg_dir, exist_ok=True)
    os.makedirs(output_polar_seg_vis_dir, exist_ok=True)

    # Create segmenter instance
    segmenter = ImageSegmenter(
        min_cluster_size=20,  # HDBSCAN参数，建议10-50
        min_samples=15,  # HDBSCAN参数，建议3-15
        min_region_size=4000,  # 小区域合并阈值，建议100-500
        rgb_weight=0.2,
        aolp_weight=0.6,
        dolp_weight=0.2,  # 添加DOLP权重，三个权重和为1
        gpu_ids=gpu_ids,
        enable_merge_regions=True  # 控制是否进行小区域合并
    )
    print("全黑的话修改aolp里面的brightness_mask的阈值")
    # Get image lists
    aolp_images = sorted([f for f in os.listdir(aolp_dir) if f.lower().endswith('.png')])
    rgb_images = sorted([f for f in os.listdir(rgb_dir) if f.lower().endswith('.png')])
    dolp_images = sorted([f for f in os.listdir(dolp_dir) if f.lower().endswith('.png')])

    total_files = len(aolp_images)
    print(f"\nTotal number of files to process: {total_files}")

    # Create processing list
    image_list = []
    skipped_files = 0
    with tqdm(total=total_files, desc="Creating processing list") as pbar:
        for aolp_image_name, rgb_image_name, dolp_image_name in zip(aolp_images, rgb_images, dolp_images):
            dolp_image_path = os.path.join(dolp_dir, dolp_image_name)
            aolp_image_path = os.path.join(aolp_dir, aolp_image_name)
            rgb_image_path = os.path.join(rgb_dir, rgb_image_name)

            if not os.path.exists(rgb_image_path):
                print(f"Warning: RGB image not found for {aolp_image_name}. Skipping.")
                pbar.update(1)
                continue

            filename_base = os.path.splitext(aolp_image_name)[0]

            # Check if this image has already been processed
            seg_output_path = os.path.join(output_polar_seg_dir, f"{filename_base}.png")
            vis_output_path = os.path.join(output_polar_seg_vis_dir, f"{filename_base}.png")

            if os.path.exists(seg_output_path) and os.path.exists(vis_output_path):
                # Both output files exist, skip this image
                skipped_files += 1
                pbar.update(1)
                continue

            # Add to processing list if not already processed
            image_list.append((rgb_image_path, aolp_image_path, dolp_image_path, filename_base))
            pbar.update(1)

    print(f"\nSkipped {skipped_files} already processed files.")
    print(f"Processing {len(image_list)} new files.")

    # Process images in parallel only if there are new images to process
    if image_list:
        print("\nStarting image processing...")
        segmenter.process_images_parallel(
            image_list,
            output_polar_seg_dir,
            output_polar_seg_vis_dir
        )
    else:
        print("\nNo new images to process. All images have been processed already.")


if __name__ == "__main__":
    # Set GPU IDs
    gpu_ids = [0, 1, 2, 3]  # Use GPUs 0,1,2,3

    # Set environment variables
    # os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))

    # Set multiprocessing start method
    import os
    import multiprocessing as mp
    from tqdm import tqdm

    mp.set_start_method('spawn', force=True)

    # Run main function
    main()