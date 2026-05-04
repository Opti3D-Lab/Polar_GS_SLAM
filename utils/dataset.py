import csv
import glob
import os

import cv2
import numpy as np
import torch
import trimesh
import lycon  # __getitem__和非加速的不一样
# 同时支持纯rgb，rgb+depth和rgb+depth+pose

from gaussian_splatting.utils.graphics_utils import focal2fov

try:
    import pyrealsense2 as rs
except Exception:
    pass

# 针对单目模式所做的修改，TODO:需要配置RGB单目模式的时候使用这个
class Replica_RGB_Parser:
    def __init__(self, input_folder):
        self.input_folder = input_folder
        self.color_paths = sorted(glob.glob(f"{self.input_folder}/results/frame*.jpg"))
        # self.depth_paths = sorted(glob.glob(f"{self.input_folder}/results/depth*.png")) # RGBカメラ使用のためコメントアウト
        self.depth_paths = None  # RGBカメラ使用のため追加
        self.n_img = len(self.color_paths)
        self.load_poses(f"{self.input_folder}/traj.txt")

    def load_poses(self, path):
        # self.poses = [] # RGBカメラ使用のためコメントアウト
        self.poses = [np.eye(4) for _ in range(self.n_img)]  # RGBカメラ使用のため追加

        with open(path, "r") as f:
            lines = f.readlines()

        frames = []
        for i in range(self.n_img):
            line = lines[i]
            pose = np.array(list(map(float, line.split()))).reshape(4, 4)
            pose = np.linalg.inv(pose)
            self.poses.append(pose)
            frame = {
                "file_path": self.color_paths[i],
                "depth_path": self.depth_paths[i],
                "transform_matrix": pose.tolist(),
            }

            frames.append(frame)
        self.frames = frames

class ReplicaParser:
    def __init__(self, input_folder):
        self.input_folder = input_folder
        self.color_paths = sorted(glob.glob(f"{self.input_folder}/results/frame*.jpg"))
        self.depth_paths = sorted(glob.glob(f"{self.input_folder}/results/depth*.png"))
        self.n_img = len(self.color_paths)
        self.load_poses(f"{self.input_folder}/traj.txt")

    def load_poses(self, path):
        self.poses = []
        with open(path, "r") as f:
            lines = f.readlines()

        frames = []
        for i in range(self.n_img):
            line = lines[i]
            pose = np.array(list(map(float, line.split()))).reshape(4, 4)
            pose = np.linalg.inv(pose)
            self.poses.append(pose)
            frame = {
                "file_path": self.color_paths[i],
                "depth_path": self.depth_paths[i],
                "transform_matrix": pose.tolist(),
            }

            frames.append(frame)
        self.frames = frames

# 只加载rgb，没有depth和pose
class TUM_RGB_Parser:
    def __init__(self, input_folder):
        self.input_folder = input_folder
        self.load_poses(self.input_folder, frame_rate=32)
        self.n_img = len(self.color_paths)

    def parse_list(self, filepath, skiprows=0):
        data = np.loadtxt(filepath, delimiter=" ", dtype=np.unicode_, skiprows=skiprows)
        return data

    # def associate_frames(self, tstamp_image, tstamp_depth, tstamp_pose, max_dt=0.08):
    #     associations = []
    #     for i, t in enumerate(tstamp_image):
    #         if tstamp_pose is None:
    #             j = np.argmin(np.abs(tstamp_depth - t))
    #             if np.abs(tstamp_depth[j] - t) < max_dt:
    #                 associations.append((i, j))
    #         else:
    #             j = np.argmin(np.abs(tstamp_depth - t))
    #             k = np.argmin(np.abs(tstamp_pose - t))
    #             if (np.abs(tstamp_depth[j] - t) < max_dt) and (
    #                 np.abs(tstamp_pose[k] - t) < max_dt
    #             ):
    #                 associations.append((i, j, k))
    #     return associations

    # def associate_frames(self, tstamp_image, max_dt=0.08):    # 原始的只有rgb输入的代码
    #     associations = [(i, i, i) for i in range(len(tstamp_image))]  # 画像のインデックスのみを持つタプルのリストを作成
    #     # print("associations:", associations)
    #     return associations

    def associate_frames(self, tstamp_image, tstamp_normal=None, max_dt=0.08):
        associations = []
        for i, t in enumerate(tstamp_image):
            if tstamp_normal is None:
                associations.append((i, i, i))
            else:
                j = np.argmin(np.abs(tstamp_normal - t))
                if np.abs(tstamp_normal[j] - t) < max_dt:
                    associations.append((i, j, i))
        return associations

    def load_poses(self, datapath, frame_rate=-1):
        if os.path.isfile(os.path.join(datapath, "groundtruth.txt")):
            pose_list = os.path.join(datapath, "groundtruth.txt")
        if os.path.isfile(os.path.join(datapath, "normal.txt")):
            normal_list = os.path.join(datapath, "normal.txt")
            normal_data = self.parse_list(normal_list)
            tstamp_normal = normal_data[:, 0].astype(np.float64)
        # elif os.path.isfile(os.path.join(datapath, "pose.txt")): # コメントアウト
        #     pose_list = os.path.join(datapath, "pose.txt") # コメントアウト

        image_list = os.path.join(datapath, "rgb.txt")
        # depth_list = os.path.join(datapath, "depth.txt") # コメントアウト

        image_data = self.parse_list(image_list)
        # depth_data = self.parse_list(depth_list) # コメントアウト
        # pose_data = self.parse_list(pose_list, skiprows=1) # コメントアウト
        # pose_vecs = pose_data[:, 0:].astype(np.float64) # コメントアウト

        tstamp_image = image_data[:, 0].astype(np.float64)
        # tstamp_depth = depth_data[:, 0].astype(np.float64) # コメントアウト
        # tstamp_pose = pose_data[:, 0].astype(np.float64) # コメントアウト
        associations = self.associate_frames(tstamp_image, tstamp_normal)  # コメントアウト

        indicies = [0]
        for i in range(1, len(associations)):
            t0 = tstamp_image[associations[indicies[-1]][0]]
            t1 = tstamp_image[associations[i][0]]
            if t1 - t0 > 1.0 / frame_rate:
                indicies += [i]

        # self.color_paths, self.poses, self.depth_paths, self.frames = [], [], [], []
        self.color_paths, self.poses, self.depth_paths, self.frames, self.normal_paths = [], [], [], [], []

        for ix in indicies:
            (i, j, k) = associations[ix]
            self.color_paths += [os.path.join(datapath, image_data[i, 1])]
            # self.depth_paths += [os.path.join(datapath, depth_data[j, 1])] # コメントアウト

            # quat = pose_vecs[k][4:] # コメントアウト
            # trans = pose_vecs[k][1:4] # コメントアウト
            # T = trimesh.transformations.quaternion_matrix(np.roll(quat, 1)) # コメントアウト
            # T[:3, 3] = trans # コメントアウト
            # self.poses += [np.linalg.inv(T)] # コメントアウト

            self.poses += [np.eye(4)]  # 代わりに単位行列を追加

            frame = {
                "file_path": str(os.path.join(datapath, image_data[i, 1])),
                # "depth_path": str(os.path.join(datapath, depth_data[j, 1])), # コメントアウト
                # "transform_matrix": (np.linalg.inv(T)).tolist(), # コメントアウト
            }

            if os.path.isfile(os.path.join(datapath, "normal.txt")):
                normal_path = os.path.join(datapath, normal_data[i, 1])
                self.normal_paths.append(normal_path)
                frame["normal_path"] = normal_path
            self.frames.append(frame)


# 最全解析器，加载rgb、depth和pose（如果有的话）
class TUMParser:
    def __init__(self, input_folder):
        self.input_folder = input_folder
        self.load_poses(self.input_folder, frame_rate=32)
        self.n_img = len(self.color_paths)

    def parse_list(self, filepath, skiprows=0):
        data = np.loadtxt(filepath, delimiter=" ", dtype=np.unicode_, skiprows=skiprows)
        return data

    def associate_frames(self, tstamp_image, tstamp_depth, tstamp_pose, tstamp_normal=None, max_dt=0.08):
        associations = []
        for i, t in enumerate(tstamp_image):
            if tstamp_pose is None:
                j = np.argmin(np.abs(tstamp_depth - t))
                if np.abs(tstamp_depth[j] - t) < max_dt:
                    associations.append((i, j))

            else:
                j = np.argmin(np.abs(tstamp_depth - t))
                k = np.argmin(np.abs(tstamp_pose - t))

                if tstamp_normal is not None:
                    n = np.argmin(np.abs(tstamp_normal - t))
                    if (np.abs(tstamp_depth[j] - t) < max_dt) and (np.abs(tstamp_pose[k] - t) < max_dt) and (np.abs(tstamp_normal[n] - t) < max_dt):
                        associations.append((i, j, k, n))
                else:
                    if (np.abs(tstamp_depth[j] - t) < max_dt) and (np.abs(tstamp_pose[k] - t) < max_dt):
                        associations.append((i, j, k))

        return associations

    def load_poses(self, datapath, frame_rate=-1):
        if os.path.isfile(os.path.join(datapath, "groundtruth.txt")):
            pose_list = os.path.join(datapath, "groundtruth.txt")
        elif os.path.isfile(os.path.join(datapath, "pose.txt")):
            pose_list = os.path.join(datapath, "pose.txt")
        else:
            print("未加载pose文件")
            pose_list = None

        image_list = os.path.join(datapath, "rgb.txt")
        depth_list = os.path.join(datapath, "depth.txt")
        normal_list = os.path.join(datapath, "normal.txt") if os.path.isfile(
            os.path.join(datapath, "normal.txt")) else None

        image_data = self.parse_list(image_list)
        depth_data = self.parse_list(depth_list)
        if pose_list:
            print("已使用pose.txt")
            pose_data = self.parse_list(pose_list, skiprows=1)
            pose_vecs = pose_data[:, 0:].astype(np.float64)
        else:
            pose_data = None
            pose_vecs = None

        if normal_list:
            print("已使用normal.txt")
            normal_data = self.parse_list(normal_list)
            tstamp_normal = normal_data[:, 0].astype(np.float64)
        else:
            normal_data = None
            tstamp_normal = None

        tstamp_image = image_data[:, 0].astype(np.float64)
        tstamp_depth = depth_data[:, 0].astype(np.float64)
        tstamp_pose = pose_data[:, 0].astype(np.float64) if pose_data is not None else None
        associations = self.associate_frames(tstamp_image, tstamp_depth, tstamp_pose, tstamp_normal)

        indicies = [0]
        for i in range(1, len(associations)):
            t0 = tstamp_image[associations[indicies[-1]][0]]
            t1 = tstamp_image[associations[i][0]]
            if t1 - t0 > 1.0 / frame_rate:
                indicies += [i]

        self.color_paths, self.poses, self.depth_paths, self.frames, self.normal_paths = [], [], [], [], []

        for ix in indicies:
            if pose_vecs is not None:
                if normal_data is not None:
                    (i, j, k, n) = associations[ix]
                else:
                    (i, j, k) = associations[ix]
                    n = None
            else:
                (i, j) = associations[ix]
                k = None
                n = None

            self.color_paths += [os.path.join(datapath, image_data[i, 1])]
            self.depth_paths += [os.path.join(datapath, depth_data[j, 1])]

            if pose_vecs is not None:
                quat = pose_vecs[k][4:]
                trans = pose_vecs[k][1:4]
                T = trimesh.transformations.quaternion_matrix(np.roll(quat, 1))
                T[:3, 3] = trans
                self.poses += [np.linalg.inv(T)]
            else:
                self.poses += [np.eye(4)]

            frame = {
                "file_path": str(os.path.join(datapath, image_data[i, 1])),
                "depth_path": str(os.path.join(datapath, depth_data[j, 1])),
                "transform_matrix": (np.linalg.inv(T)).tolist() if pose_vecs is not None else None,
            }

            if normal_data is not None:
                if isinstance(normal_data[n, 1], np.ndarray):
                    normal_path = os.path.join(datapath, normal_data[n, 1].squeeze()[1])
                else:
                    normal_path = os.path.join(datapath, normal_data[n, 1])
                # print("normal_path:", normal_path)
                self.normal_paths.append(normal_path)
                frame["normal_path"] = normal_path

            self.frames.append(frame)

# 去掉pose，增加aolp和dolp的输入，depth和normal是可选
# 这里不用单独弄个单目解析器，因为只有depth的区别，肯定没有pose，根据depth.txt是否存在来判断即可
class TUM_Polar_Parser:
    def __init__(self, input_folder):
        self.input_folder = input_folder
        self.load_poses(self.input_folder, frame_rate=32)
        self.n_img = len(self.color_paths)

    def parse_list(self, filepath, skiprows=0):
        data = np.loadtxt(filepath, delimiter=" ", dtype=np.unicode_, skiprows=skiprows)
        return data

    # tstamp_aolp, tstamp_dolp和tstamp_image是一致的，所以只需要一个就行
    # depth和normal的是一致的
    # i和j分别代表二者的索引，即txt文件中的第几行

    def associate_frames(self, tstamp_image, tstamp_depth=None, tstamp_seg=None, max_dt=0.08):
        associations = []
        for i, t in enumerate(tstamp_image):
            assoc_indices = [i]  # 总是包含图像索引

            # 尝试匹配深度数据
            if tstamp_depth is not None:
                j = np.argmin(np.abs(tstamp_depth - t))  # 找到最接近的时间戳索引
                if np.abs(tstamp_depth[j] - t) < max_dt:
                    assoc_indices.append(j)
                else:
                    assoc_indices.append(None)
            else:
                assoc_indices.append(None)

            # 尝试匹配分割数据
            if tstamp_seg is not None:
                k = np.argmin(np.abs(tstamp_seg - t))  # 找到最接近的时间戳索引
                if np.abs(tstamp_seg[k] - t) < max_dt:
                    assoc_indices.append(k)
                else:
                    assoc_indices.append(None)
            else:
                assoc_indices.append(None)

            associations.append(tuple(assoc_indices))

        return associations


    # 下面的代码是处理seg.txt文件中，部分图像没有seg的情况
    def load_poses(self, datapath, frame_rate=-1):
        print("未加载pose文件，默认aolp, dolp, rgb三者的时间戳一致")

        image_list = os.path.join(datapath, "rgb.txt")
        # depth_list = os.path.join(datapath, "depth.txt")
        aolp_list = os.path.join(datapath, "aolp.txt")
        dolp_list = os.path.join(datapath, "dolp.txt")
        normal_list = os.path.join(datapath, "normal.txt") if os.path.isfile(
            os.path.join(datapath, "normal.txt")) else None
        depth_list = os.path.join(datapath, "depth.txt") if os.path.isfile(
            os.path.join(datapath, "depth.txt")) else None
        seg_list = os.path.join(datapath, "seg.txt") if os.path.isfile(
            os.path.join(datapath, "seg.txt")) else None

        image_data = self.parse_list(image_list)
        # depth_data = self.parse_list(depth_list)
        aolp_data = self.parse_list(aolp_list)
        dolp_data = self.parse_list(dolp_list)

        # 使用步长2获取数据(每隔开1行),测试帧率是否影响
        # image_data = image_data[::3]
        # aolp_data = aolp_data[::3]
        # dolp_data = dolp_data[::3]

        tstamp_image = image_data[:, 0].astype(np.float64)

        if normal_list:
            normal_data = self.parse_list(normal_list)
        else:
            normal_data = None

        if depth_list:
            depth_data = self.parse_list(depth_list)
            tstamp_depth = depth_data[:, 0].astype(np.float64)
        else:
            depth_data = None
            tstamp_depth = None

        if seg_list:
            seg_data = self.parse_list(seg_list)
            tstamp_seg = seg_data[:, 0].astype(np.float64)
        else:
            seg_data = None
            tstamp_seg = None

        # 根据需要匹配的数据类型进行关联
        associations = self.associate_frames(tstamp_image, tstamp_depth, tstamp_seg)

        # 确保所选帧之间的时间间隔足够大，以满足指定的帧率要求
        indicies = [0]
        for i in range(1, len(associations)):
            t0 = tstamp_image[associations[indicies[-1]][0]]
            t1 = tstamp_image[associations[i][0]]
            if t1 - t0 > 1.0 / frame_rate:
                indicies += [i]

        (self.color_paths, self.poses, self.depth_paths, self.frames, self.seg_paths,
         self.normal_paths, self.aolp_paths, self.dolp_paths) = [], [], [], [], [], [], [], []

        for ix in indicies:
            association = associations[ix]
            i = association[0]  # 图像索引
            j = association[1]  # 深度索引，可能为None
            k = association[2]  # 分割索引，可能为None

            # 从txt文件中读取指定行的路径
            self.color_paths += [os.path.join(datapath, image_data[i, 1])]
            self.aolp_paths += [os.path.join(datapath, aolp_data[i, 1])]
            self.dolp_paths += [os.path.join(datapath, dolp_data[i, 1])]

            self.poses += [np.eye(4)]  # 单位矩阵，用于占位

            frame = {
                "file_path": str(os.path.join(datapath, image_data[i, 1])),
                "aolp_path": str(os.path.join(datapath, aolp_data[i, 1])),
                "dolp_path": str(os.path.join(datapath, dolp_data[i, 1])),
            }

            # 处理深度和法线数据
            if depth_data is not None and j is not None:
                if isinstance(depth_data[j, 1], np.ndarray):
                    depth_path = str(os.path.join(datapath, depth_data[j, 1].squeeze()[1]))
                    normal_path = str(os.path.join(datapath, normal_data[j, 1].squeeze()[1]))
                else:
                    depth_path = str(os.path.join(datapath, depth_data[j, 1]))
                    normal_path = str(os.path.join(datapath, normal_data[j, 1]))
                self.depth_paths.append(depth_path)
                self.normal_paths.append(normal_path)
                frame["depth_path"] = depth_path
                frame["normal_path"] = normal_path
            else:
                self.depth_paths.append(None)
                self.normal_paths.append(None)
                frame["depth_path"] = None
                frame["normal_path"] = None

            # 处理分割数据
            if seg_data is not None and k is not None:
                if isinstance(seg_data[k, 1], np.ndarray):
                    seg_path = str(os.path.join(datapath, seg_data[k, 1].squeeze()[1]))
                else:
                    seg_path = str(os.path.join(datapath, seg_data[k, 1]))
                self.seg_paths.append(seg_path)
                frame["seg_path"] = seg_path
            else:
                self.seg_paths.append(None)
                frame["seg_path"] = None

            self.frames.append(frame)


class EuRoCParser:
    def __init__(self, input_folder, start_idx=0):
        self.input_folder = input_folder
        self.start_idx = start_idx
        self.color_paths = sorted(
            glob.glob(f"{self.input_folder}/mav0/cam0/data/*.png")
        )
        self.color_paths_r = sorted(
            glob.glob(f"{self.input_folder}/mav0/cam1/data/*.png")
        )
        assert len(self.color_paths) == len(self.color_paths_r)
        self.color_paths = self.color_paths[start_idx:]
        self.color_paths_r = self.color_paths_r[start_idx:]
        self.n_img = len(self.color_paths)
        self.load_poses(
            f"{self.input_folder}/mav0/state_groundtruth_estimate0/data.csv"
        )

    def associate(self, ts_pose):
        pose_indices = []
        for i in range(self.n_img):
            color_ts = float((self.color_paths[i].split("/")[-1]).split(".")[0])
            k = np.argmin(np.abs(ts_pose - color_ts))
            pose_indices.append(k)

        return pose_indices

    def load_poses(self, path):
        self.poses = []
        with open(path) as f:
            reader = csv.reader(f)
            header = next(reader)
            data = [list(map(float, row)) for row in reader]
        data = np.array(data)
        T_i_c0 = np.array(
            [
                [0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975],
                [0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768],
                [-0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        pose_ts = data[:, 0]
        pose_indices = self.associate(pose_ts)

        frames = []
        for i in range(self.n_img):
            trans = data[pose_indices[i], 1:4]
            quat = data[pose_indices[i], 4:8]

            T_w_i = trimesh.transformations.quaternion_matrix(np.roll(quat, 1))
            T_w_i[:3, 3] = trans
            T_w_c = np.dot(T_w_i, T_i_c0)

            self.poses += [np.linalg.inv(T_w_c)]

            frame = {
                "file_path": self.color_paths[i],
                "transform_matrix": (np.linalg.inv(T_w_c)).tolist(),
            }

            frames.append(frame)
        self.frames = frames


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, args, path, config):
        self.args = args
        self.path = path
        self.config = config
        self.device = "cuda:0"  # TODO: 根据需要修改
        self.dtype = torch.float32
        self.num_imgs = 999999

    def __len__(self):
        return self.num_imgs

    def __getitem__(self, idx):
        pass


class MonocularDataset(BaseDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        calibration = config["Dataset"]["Calibration"]
        # Camera prameters
        self.fx = calibration["fx"]
        self.fy = calibration["fy"]
        self.cx = calibration["cx"]
        self.cy = calibration["cy"]
        self.width = calibration["width"]
        self.height = calibration["height"]
        self.fovx = focal2fov(self.fx, self.width)
        self.fovy = focal2fov(self.fy, self.height)
        self.K = np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]]
        )
        # distortion parameters
        self.disorted = calibration["distorted"]
        self.dist_coeffs = np.array(
            [
                calibration["k1"],
                calibration["k2"],
                calibration["p1"],
                calibration["p2"],
                calibration["k3"],
            ]
        )
        self.map1x, self.map1y = cv2.initUndistortRectifyMap(
            self.K,
            self.dist_coeffs,
            np.eye(3),
            self.K,
            (self.width, self.height),
            cv2.CV_32FC1,
        )
        # depth parameters
        self.has_depth = True if "depth_scale" in calibration.keys() else False # RGBカメラ使用のためコメントアウト
        if self.has_depth:
            print("has_depth_scale=true")
        else:
            print("has_depth_scale=false")
        # self.has_depth = False  # RGBカメラ使用のため追加
        self.depth_scale = calibration["depth_scale"] if self.has_depth else None
        self.has_normal = True if os.path.isfile(
            os.path.join(config["Dataset"]["dataset_path"], "normal.txt")) else False
        self.has_polar = True if os.path.isfile(
            os.path.join(config["Dataset"]["dataset_path"], "aolp.txt")) else False
        self.has_seg = True if os.path.isfile(
            os.path.join(config["Dataset"]["dataset_path"], "seg.txt")) else False
        if self.has_normal:
            print("已加载normal.txt")
        else:
            print("未加载normal.txt")
        if self.has_polar:
            print("已加载aolp.txt和dolp.txt")
        else:
            print("未加载aolp.txt和dolp.txt")
        if self.has_seg:
            print("已加载seg.txt")
        else:
            print("未加载seg.txt")
        # Default scene scale
        nerf_normalization_radius = 5
        self.scene_info = {
            "nerf_normalization": {
                "radius": nerf_normalization_radius,
                "translation": np.zeros(3),
            },
        }

    def __getitem__(self, idx):
        color_path = self.color_paths[idx]
        pose = self.poses[idx]
        image = lycon.load(color_path)
        depth = None
        normal = None
        aolp = None
        dolp = None
        seg = None

        if self.disorted:
            image = cv2.remap(image, self.map1x, self.map1y, cv2.INTER_LINEAR)

        if self.has_depth:
            depth_path = self.depth_paths[idx]
            depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
            if depth is None:
                raise ValueError("读取depth文件失败，请检查路径，或者去掉depth_scale进行单目模式运行")
            depth = depth / self.depth_scale

        if self.has_normal: # 只要有就加载
            normal_path = self.normal_paths[idx]
            normal = cv2.imread(normal_path, cv2.IMREAD_UNCHANGED)
            normal = normal[:, :, ::-1]   # BGR to RGB
            if normal is None:
                raise ValueError("读取normal文件失败，请检查路径")
            normal = 1 - normal / 65535.0 * 2  # 归一化到 -1 到 1 的范围，方便直接跟输出做loss
            # normal = torch.from_numpy(normal).permute(2, 0, 1).to(device=self.device, dtype=self.dtype)   # 在后面处理了
        image = (
            torch.from_numpy(image / 255.0)
            .clamp(0.0, 1.0)
            .permute(2, 0, 1)
            .to(device=self.device, dtype=self.dtype)
        )
        pose = torch.from_numpy(pose).to(device=self.device)

        if self.has_polar:
            aolp_path = self.aolp_paths[idx]
            dolp_path = self.dolp_paths[idx]
            aolp = cv2.imread(aolp_path, cv2.IMREAD_UNCHANGED)
            dolp = cv2.imread(dolp_path, cv2.IMREAD_UNCHANGED)
            # AOLP表示偏振方向，由于偏振的物理特性，它只能表示[0, π]
            # 范围的角度因为偏振态在旋转180度后会回到相同状态
            # 而方位角（azimuth）是表示在平面内的完整角度，范围是[0, 2π]
            # aolp = torch.from_numpy(aolp / 255.0 * np.pi).to(device=self.device, dtype=self.dtype)
            # dolp = torch.from_numpy(dolp / 255.0).to(device=self.device, dtype=self.dtype)
            # 保留aolp和dolp为NumPy数组，后面和depth一起处理转成tensor
            if aolp is None:
                raise ValueError("读取aolp文件失败，请检查路径")
            if dolp is None:
                raise ValueError("读取dolp文件失败，请检查路径")
            aolp = aolp / 255.0 * np.pi
            dolp = dolp / 255.0

        if self.has_seg:    # 默认seg.txt里就是实际的seg路径
            seg_path = self.seg_paths[idx]
            if seg_path is not None and os.path.isfile(seg_path):
                seg = cv2.imread(seg_path, cv2.IMREAD_UNCHANGED)
                if seg is None:
                    raise ValueError(f"路径 {seg_path} 的seg文件存在但可能损坏")
            elif seg_path is None:  # 表示有rgb_path而没有seg_path的文件
                print("提示：该帧无seg_path")
            else:
                raise ValueError(f"路径 {seg_path} 的没有对应的seg数据，请检查seg.txt")

        return image, depth, normal, pose, aolp, dolp, seg


class StereoDataset(BaseDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        calibration = config["Dataset"]["Calibration"]
        self.width = calibration["width"]
        self.height = calibration["height"]

        cam0raw = calibration["cam0"]["raw"]
        cam0opt = calibration["cam0"]["opt"]
        cam1raw = calibration["cam1"]["raw"]
        cam1opt = calibration["cam1"]["opt"]
        # Camera prameters
        self.fx_raw = cam0raw["fx"]
        self.fy_raw = cam0raw["fy"]
        self.cx_raw = cam0raw["cx"]
        self.cy_raw = cam0raw["cy"]
        self.fx = cam0opt["fx"]
        self.fy = cam0opt["fy"]
        self.cx = cam0opt["cx"]
        self.cy = cam0opt["cy"]

        self.fx_raw_r = cam1raw["fx"]
        self.fy_raw_r = cam1raw["fy"]
        self.cx_raw_r = cam1raw["cx"]
        self.cy_raw_r = cam1raw["cy"]
        self.fx_r = cam1opt["fx"]
        self.fy_r = cam1opt["fy"]
        self.cx_r = cam1opt["cx"]
        self.cy_r = cam1opt["cy"]

        self.fovx = focal2fov(self.fx, self.width)
        self.fovy = focal2fov(self.fy, self.height)
        self.K_raw = np.array(
            [
                [self.fx_raw, 0.0, self.cx_raw],
                [0.0, self.fy_raw, self.cy_raw],
                [0.0, 0.0, 1.0],
            ]
        )

        self.K = np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]]
        )

        self.Rmat = np.array(calibration["cam0"]["R"]["data"]).reshape(3, 3)
        self.K_raw_r = np.array(
            [
                [self.fx_raw_r, 0.0, self.cx_raw_r],
                [0.0, self.fy_raw_r, self.cy_raw_r],
                [0.0, 0.0, 1.0],
            ]
        )

        self.K_r = np.array(
            [[self.fx_r, 0.0, self.cx_r], [0.0, self.fy_r, self.cy_r], [0.0, 0.0, 1.0]]
        )
        self.Rmat_r = np.array(calibration["cam1"]["R"]["data"]).reshape(3, 3)

        # distortion parameters
        self.disorted = calibration["distorted"]
        self.dist_coeffs = np.array(
            [cam0raw["k1"], cam0raw["k2"], cam0raw["p1"], cam0raw["p2"], cam0raw["k3"]]
        )
        self.map1x, self.map1y = cv2.initUndistortRectifyMap(
            self.K_raw,
            self.dist_coeffs,
            self.Rmat,
            self.K,
            (self.width, self.height),
            cv2.CV_32FC1,
        )

        self.dist_coeffs_r = np.array(
            [cam1raw["k1"], cam1raw["k2"], cam1raw["p1"], cam1raw["p2"], cam1raw["k3"]]
        )
        self.map1x_r, self.map1y_r = cv2.initUndistortRectifyMap(
            self.K_raw_r,
            self.dist_coeffs_r,
            self.Rmat_r,
            self.K_r,
            (self.width, self.height),
            cv2.CV_32FC1,
        )

    def __getitem__(self, idx):
        color_path = self.color_paths[idx]
        color_path_r = self.color_paths_r[idx]

        pose = self.poses[idx]
        image = cv2.imread(color_path, 0)
        image_r = cv2.imread(color_path_r, 0)
        depth = None
        if self.disorted:
            image = cv2.remap(image, self.map1x, self.map1y, cv2.INTER_LINEAR)
            image_r = cv2.remap(image_r, self.map1x_r, self.map1y_r, cv2.INTER_LINEAR)
        stereo = cv2.StereoSGBM_create(minDisparity=0, numDisparities=64, blockSize=20)
        stereo.setUniquenessRatio(40)
        disparity = stereo.compute(image, image_r) / 16.0
        disparity[disparity == 0] = 1e10
        depth = 47.90639384423901 / (
            disparity
        )  ## Following ORB-SLAM2 config, baseline*fx
        depth[depth < 0] = 0
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        image = (
            torch.from_numpy(image / 255.0)
            .clamp(0.0, 1.0)
            .permute(2, 0, 1)
            .to(device=self.device, dtype=self.dtype)
        )
        pose = torch.from_numpy(pose).to(device=self.device)

        return image, depth, pose


class TUMDataset(MonocularDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        dataset_path = config["Dataset"]["dataset_path"]
        print("正在上使用TUM RGBD模式")
        parser = TUMParser(dataset_path)
        self.num_imgs = parser.n_img
        self.color_paths = parser.color_paths
        self.depth_paths = parser.depth_paths
        self.normal_paths = parser.normal_paths
        self.poses = parser.poses

class TUM_RGB_Dataset(MonocularDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        dataset_path = config["Dataset"]["dataset_path"]
        print("正在上使用TUM单目RGB模式")
        parser = TUM_RGB_Parser(dataset_path)
        self.num_imgs = parser.n_img
        self.color_paths = parser.color_paths
        self.depth_paths = parser.depth_paths
        self.normal_paths = parser.normal_paths
        self.poses = parser.poses

class TUM_Polar_Dataset(MonocularDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        dataset_path = config["Dataset"]["dataset_path"]
        if config["Dataset"]["sensor_type"] == "monocular":
            print("正在上使用偏振单目模式")
        elif config["Dataset"]["sensor_type"] == "depth":
            print("正在上使用偏振RGBD模式")
        else:
            print("Error: 请检查sensor_type是否正确")
        parser = TUM_Polar_Parser(dataset_path)
        self.num_imgs = parser.n_img
        self.color_paths = parser.color_paths
        self.depth_paths = parser.depth_paths
        self.normal_paths = parser.normal_paths
        self.poses = parser.poses   # 虽然就是单位矩阵，但是为了保持一致性，还是加上
        # 下面这两个必须放最后，因为其他的解析器没有
        self.aolp_paths = parser.aolp_paths
        self.dolp_paths = parser.dolp_paths
        self.seg_paths = parser.seg_paths

class ReplicaDataset(MonocularDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        dataset_path = config["Dataset"]["dataset_path"]
        parser = ReplicaParser(dataset_path)
        self.num_imgs = parser.n_img
        self.color_paths = parser.color_paths
        self.depth_paths = parser.depth_paths
        self.poses = parser.poses


class EurocDataset(StereoDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        dataset_path = config["Dataset"]["dataset_path"]
        parser = EuRoCParser(dataset_path, start_idx=config["Dataset"]["start_idx"])
        self.num_imgs = parser.n_img
        self.color_paths = parser.color_paths
        self.color_paths_r = parser.color_paths_r
        self.poses = parser.poses


class RealsenseDataset(BaseDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        self.device = "cuda:1"
        self.pipeline = rs.pipeline()
        self.h, self.w = 720, 1280
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, self.w, self.h, rs.format.bgr8, 30)
        self.profile = self.pipeline.start(self.config)

        self.rgb_sensor = self.profile.get_device().query_sensors()[1]
        self.rgb_sensor.set_option(rs.option.enable_auto_exposure, False)
        # rgb_sensor.set_option(rs.option.enable_auto_white_balance, True)
        self.rgb_sensor.set_option(rs.option.enable_auto_white_balance, False)
        self.rgb_sensor.set_option(rs.option.exposure, 200)
        self.rgb_profile = rs.video_stream_profile(
            self.profile.get_stream(rs.stream.color)
        )

        self.rgb_intrinsics = self.rgb_profile.get_intrinsics()

        self.fx = self.rgb_intrinsics.fx
        self.fy = self.rgb_intrinsics.fy
        self.cx = self.rgb_intrinsics.ppx
        self.cy = self.rgb_intrinsics.ppy
        self.width = self.rgb_intrinsics.width
        self.height = self.rgb_intrinsics.height
        self.fovx = focal2fov(self.fx, self.width)
        self.fovy = focal2fov(self.fy, self.height)
        self.K = np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]]
        )

        self.disorted = True
        self.dist_coeffs = np.asarray(self.rgb_intrinsics.coeffs)
        self.map1x, self.map1y = cv2.initUndistortRectifyMap(
            self.K, self.dist_coeffs, np.eye(3), self.K, (self.w, self.h), cv2.CV_32FC1
        )

        # depth parameters
        self.has_depth = False
        self.depth_scale = None
        self.has_normal = False

    def __getitem__(self, idx):
        pose = torch.eye(4, device=self.device, dtype=self.dtype)

        frameset = self.pipeline.wait_for_frames()
        rgb_frame = frameset.get_color_frame()
        image = np.asanyarray(rgb_frame.get_data())
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.disorted:
            image = cv2.remap(image, self.map1x, self.map1y, cv2.INTER_LINEAR)

        image = (
            torch.from_numpy(image / 255.0)
            .clamp(0.0, 1.0)
            .permute(2, 0, 1)
            .to(device=self.device, dtype=self.dtype)
        )
        return image, None, pose

# 在slam.py里面调用
def load_dataset(args, path, config):
    if config["Dataset"]["type"] == "tum":  # 支持不输入pose，读取depth，但是单目里不使用
        return TUMDataset(args, path, config)
    elif config["Dataset"]["type"] == "tum_rgb":
        return TUM_RGB_Dataset(args, path, config)  # 只使用rgb
    elif config["Dataset"]["type"] == "polar":
        return TUM_Polar_Dataset(args, path, config)
    elif config["Dataset"]["type"] == "replica":
        return ReplicaDataset(args, path, config)
    elif config["Dataset"]["type"] == "euroc":
        return EurocDataset(args, path, config)
    elif config["Dataset"]["type"] == "realsense":
        return RealsenseDataset(args, path, config)
    else:
        raise ValueError("Unknown dataset type")
