import os
import sys
import time
from argparse import ArgumentParser
from datetime import datetime

import torch
import torch.multiprocessing as mp
import yaml
from munch import munchify

import wandb
from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.utils.system_utils import mkdir_p
from gui import gui_utils, slam_gui
from utils.config_utils import load_config
from utils.dataset import load_dataset
from utils.eval_utils import eval_ate, eval_rendering, save_gaussians
from utils.logging_utils import Log
from utils.multiprocessing_utils import FakeQueue
from utils.slam_backend import BackEnd
from utils.slam_frontend import FrontEnd

from gaussian_splatting.gaussian_renderer import render
from utils.depth_utils import depth_to_normal
from utils.save_render import initialize_visualization_folder,save_visualizations   # 自己新建的文件
import numpy as np

from tqdm import tqdm

class SLAM:
    def __init__(self, config, save_dir=None):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        # 初始化配置和保存目录
        self.config = config
        self.save_dir = save_dir
        model_params = munchify(config["model_params"])
        opt_params = munchify(config["opt_params"])
        pipeline_params = munchify(config["pipeline_params"])
        self.model_params, self.opt_params, self.pipeline_params = (
            model_params,
            opt_params,
            pipeline_params,
        )
        # 判断是否为实时模式和单目模式
        self.live_mode = self.config["Dataset"]["type"] == "realsense"
        self.monocular = self.config["Dataset"]["sensor_type"] == "monocular"
        self.use_spherical_harmonics = self.config["Training"]["spherical_harmonics"]
        self.use_gui = self.config["Results"]["use_gui"]
        if self.live_mode:
            self.use_gui = True
        self.eval_rendering = self.config["Results"]["eval_rendering"]
        # 设置球谐函数的阶数
        # model_params.sh_degree = 3 if self.use_spherical_harmonics else 0

        model_params.sh_degree = self.config["model_params"]["sh_degree"] if self.use_spherical_harmonics else 0
        # 初始化高斯模型
        self.gaussians = GaussianModel(model_params.sh_degree, config=self.config)
        self.gaussians.init_lr(6.0)
        self.dataset = load_dataset(
            model_params, model_params.source_path, config=config
        )
        self.total_frames = len(self.dataset)  # 获取总帧数
        self.gaussians.training_setup(opt_params)
        bg_color = [0, 0, 0]    # 背景颜色
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        # 初始化前端和后端队列，使用 multiprocessing 模块提供的 Queue 类
        frontend_queue = mp.Queue()
        backend_queue = mp.Queue()

        # 这两个变量分别用于不同的方向通信：q_main2vis 用于从主进程到可视化进程的通信，而 q_vis2main 用于从可视化进程到主进程的通信。
        q_main2vis = mp.Queue() if self.use_gui else FakeQueue()
        q_vis2main = mp.Queue() if self.use_gui else FakeQueue()

        self.config["Results"]["save_dir"] = save_dir
        self.config["Training"]["monocular"] = self.monocular
        # 初始化前端和后端
        self.frontend = FrontEnd(self.config)
        self.backend = BackEnd(self.config)

        self.frontend.dataset = self.dataset
        self.frontend.background = self.background
        self.frontend.pipeline_params = self.pipeline_params
        self.frontend.frontend_queue = frontend_queue   # 向后端发送关键帧请求、初始化请求等。
        self.frontend.backend_queue = backend_queue
        self.frontend.q_main2vis = q_main2vis
        self.frontend.q_vis2main = q_vis2main
        self.frontend.set_hyperparams()

        self.backend.gaussians = self.gaussians
        self.backend.background = self.background
        self.backend.cameras_extent = 6.0
        self.backend.pipeline_params = self.pipeline_params
        self.backend.opt_params = self.opt_params
        self.backend.frontend_queue = frontend_queue    # 用于从后端进程向前端进程发送数据，例如关键帧数据、同步请求等。
        self.backend.backend_queue = backend_queue  # 用于从主进程向后端进程发送指令，例如暂停、停止或执行特定任务。
        # 后端会不断地从 backend_queue 中获取指令并执行相应的任务。
        self.backend.live_mode = self.live_mode
        # 几个线程的区别，其中前端线程包含在主线程中，但是控制暂停、停止等操作的是主线程，而不是前端线程。
        # self.frontend.frontend_queue：用于前端进程向后端进程发送数据，例如关键帧请求、初始化请求等。
        # self.frontend.backend_queue：用于主进程向后端进程发送指令，例如暂停、停止或执行特定任务。
        # self.backend.frontend_queue：用于后端进程向前端进程发送数据，例如关键帧数据、同步请求等。
        # self.backend.backend_queue：用于主进程向后端进程发送指令，例如暂停、停止或执行特定任务。后端进程会不断地从backend_queue中获取指令并执行相应的任务。

        self.backend.set_hyperparams()
        # 初始化GUI参数
        self.params_gui = gui_utils.ParamsGUI(
            pipe=self.pipeline_params,
            background=self.background,
            gaussians=self.gaussians,
            q_main2vis=q_main2vis,
            q_vis2main=q_vis2main,
        )
        # 创建并启动后端进程
        backend_process = mp.Process(target=self.backend.run)   # 创建目标为 self.backend.run 的进程
        if self.use_gui:
            gui_process = mp.Process(target=slam_gui.run, args=(self.params_gui,))
            gui_process.start()
            time.sleep(5)

        start.record()  # 记录开始时间，后面会有end.record()记录结束时间
        backend_process.start()
        self.frontend.run() # 主线程运行前端
        backend_queue.put(["pause"])

        end.record()
        torch.cuda.synchronize()    # 等待所有 CUDA 设备上的所有流中的所有核心完成。
        # 清空前端队列
        N_frames = len(self.frontend.cameras)
        FPS = N_frames / (start.elapsed_time(end) * 0.001)
        Log("Total time", start.elapsed_time(end) * 0.001, tag="Eval")
        Log("Total FPS", N_frames / (start.elapsed_time(end) * 0.001), tag="Eval")

        if self.eval_rendering:
            self.gaussians = self.frontend.gaussians
            kf_indices = self.frontend.kf_indices
            ATE = eval_ate(
                self.frontend.cameras,
                self.frontend.kf_indices,
                self.save_dir,
                0,
                final=True,
                monocular=self.monocular,
            )

            rendering_result = eval_rendering(
                self.frontend.cameras,
                self.gaussians,
                self.dataset,
                self.save_dir,
                self.pipeline_params,
                self.background,
                kf_indices=kf_indices,
                iteration="before_opt",
            )
            columns = ["tag", "psnr", "ssim", "lpips", "RMSE ATE", "FPS"]
            metrics_table = wandb.Table(columns=columns)
            metrics_table.add_data(
                "Before",
                rendering_result["mean_psnr"],
                rendering_result["mean_ssim"],
                rendering_result["mean_lpips"],
                ATE,
                FPS,
            )

            # re-used the frontend queue to retrive the gaussians from the backend.
            while not frontend_queue.empty():
                frontend_queue.get()
            backend_queue.put(["color_refinement"])
            # 从前端队列中获取优化后的高斯模型
            while True:
                if frontend_queue.empty():
                    time.sleep(0.01)
                    continue
                data = frontend_queue.get()
                if data[0] == "sync_backend" and frontend_queue.empty():
                    gaussians = data[1]
                    self.gaussians = gaussians
                    break
            # 评估优化后的渲染质量
            rendering_result = eval_rendering(
                self.frontend.cameras,
                self.gaussians,
                self.dataset,
                self.save_dir,
                self.pipeline_params,
                self.background,
                kf_indices=kf_indices,
                iteration="after_opt",
            )
            metrics_table.add_data(
                "After",
                rendering_result["mean_psnr"],
                rendering_result["mean_ssim"],
                rendering_result["mean_lpips"],
                ATE,
                FPS,
            )
            wandb.log({"Metrics": metrics_table})
            print("已完成优化，正在保存...")

        else:   # 即便不评估渲染质量，也要执行优化步骤
            # re-used the frontend queue to retrive the gaussians from the backend.
            while not frontend_queue.empty():
                frontend_queue.get()
            backend_queue.put(["color_refinement"])

        # # 从前端队列中获取优化后的高斯模型
        # while True:
        #     if frontend_queue.empty():
        #         time.sleep(0.01)
        #         continue
        #     data = frontend_queue.get()
        #     if data[0] == "sync_backend" and frontend_queue.empty():
        #         gaussians = data[1]
        #         self.gaussians = gaussians
        #         break
        # print("已完成优化，正在保存...")
        # save_gaussians(self.gaussians, self.save_dir, "final_after_opt", final=True)
        # print("已保存优化后的高斯模型")
        #
        # print("正在按顺序保存所有帧的渲染结果...")
        # folder_name0, folder_name1, folder_name2 = initialize_visualization_folder(1, 1, 1)
        #
        # # 添加进度条，获取总视点数用于显示
        # total_viewpoints = len(sorted(self.frontend.cameras))
        # # 使用tqdm创建进度条
        # for viewpoint_cam_idx in tqdm(sorted(self.frontend.cameras), desc="渲染进度", total=total_viewpoints):
        #     viewpoint_cam = self.frontend.cameras[viewpoint_cam_idx]
        #     # 渲染当前视点
        #     render_pkg = render(
        #         viewpoint_cam, self.gaussians, self.pipeline_params, self.background
        #     )
        #     image, depth, normal = (
        #         render_pkg["render"],
        #         render_pkg["depth"],
        #         render_pkg["normal"],
        #     )
        #
        #     # 计算深度法线
        #     depth_normal, _ = depth_to_normal(viewpoint_cam, depth)
        #     depth_normal = depth_normal.permute(2, 0, 1)
        #
        #     # 准备GT法线（如果有）
        #     gt_normal = None
        #     if viewpoint_cam.normal is not None:
        #         if isinstance(viewpoint_cam.normal, np.ndarray):
        #             gt_normal = torch.from_numpy(viewpoint_cam.normal).permute(2, 0, 1).to(
        #                 dtype=torch.float32, device="cuda"
        #             )
        #         else:
        #             gt_normal = viewpoint_cam.normal.cuda()
        #
        #     # 保存渲染结果
        #     save_visualizations(folder_name2, image, depth, depth_normal, normal, gt_normal,
        #                         viewpoint_cam_idx, save_raw_normal=False, depth_scale=1000)
        #
        # print(f"已完成所有 {len(self.frontend.cameras)} 帧的渲染结果保存，保存路径：{folder_name2}，不再拼接图像")


        # 尝试并行保存
        # 从前端队列中获取优化后的高斯模型
        while True:
            if frontend_queue.empty():
                time.sleep(0.01)
                continue
            data = frontend_queue.get()
            if data[0] == "sync_backend" and frontend_queue.empty():
                gaussians = data[1]
                self.gaussians = gaussians
                break
        print("已完成优化，正在保存...")
        save_gaussians(self.gaussians, self.save_dir, "final_after_opt", final=True)
        print("已保存优化后的高斯模型")

        print("正在并行保存所有帧的渲染结果...")
        folder_name0, folder_name1, folder_name2 = initialize_visualization_folder(1, 1, 1)

        import concurrent.futures
        from functools import partial

        def render_and_save(viewpoint_cam_idx, cameras, gaussians, pipeline_params, background, folder_name):
            """渲染单个视角并保存结果"""
            viewpoint_cam = cameras[viewpoint_cam_idx]
            # 渲染当前视点
            render_pkg = render(
                viewpoint_cam, gaussians, pipeline_params, background
            )
            image, depth, normal = (
                render_pkg["render"],
                render_pkg["depth"],
                render_pkg["normal"],
            )

            # 计算深度法线
            depth_normal, _ = depth_to_normal(viewpoint_cam, depth)
            depth_normal = depth_normal.permute(2, 0, 1)

            # 准备GT法线（如果有）
            gt_normal = None
            if viewpoint_cam.normal is not None:
                if isinstance(viewpoint_cam.normal, np.ndarray):
                    gt_normal = torch.from_numpy(viewpoint_cam.normal).permute(2, 0, 1).to(
                        dtype=torch.float32, device="cuda"
                    )
                else:
                    gt_normal = viewpoint_cam.normal.cuda()

            # 保存渲染结果
            save_visualizations(folder_name, image, depth, depth_normal, normal, gt_normal,
                                viewpoint_cam_idx, save_raw_normal=False, depth_scale=1000)

            return viewpoint_cam_idx

        # 获取所有视角索引并排序
        viewpoint_indices = sorted(self.frontend.cameras)
        total_viewpoints = len(viewpoint_indices)

        # 创建部分函数，固定除了视角索引外的所有参数
        render_save_partial = partial(
            render_and_save,
            cameras=self.frontend.cameras,
            gaussians=self.gaussians,
            pipeline_params=self.pipeline_params,
            background=self.background,
            folder_name=folder_name2
        )

        # 使用线程池执行并行渲染和保存
        # 设置合适的max_workers数量，通常为CPU核心数的1-2倍
        max_workers = min(32, os.cpu_count() * 2)  # 限制最大线程数
        print(f"使用 {max_workers} 个线程并行处理...")

        # 使用tqdm显示总体进度
        from tqdm import tqdm

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_idx = {executor.submit(render_save_partial, idx): idx for idx in viewpoint_indices}

            # 创建进度条
            with tqdm(total=total_viewpoints, desc="渲染进度") as pbar:
                # 按任务完成顺序获取结果
                for future in concurrent.futures.as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        # 获取结果（这里只是为了检查是否有异常）
                        completed_idx = future.result()
                        pbar.update(1)
                    except Exception as e:
                        print(f"视角 {idx} 处理失败: {e}")

        print(f"已完成所有 {total_viewpoints} 帧的渲染结果保存，保存路径：{folder_name2}，不再拼接图像")

        # 向后端队列发送 stop 指令并等待后端进程结束
        backend_queue.put(["stop"])
        backend_process.join()
        Log("Backend stopped and joined the main thread")


        if self.use_gui:
            q_main2vis.put(gui_utils.GaussianPacket(finish=True))
            gui_process.join()
            Log("GUI Stopped and joined the main thread")

    def run(self):
        pass


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--config", type=str)
    parser.add_argument("--eval", action="store_true")

    args = parser.parse_args(sys.argv[1:])

    mp.set_start_method("spawn")

    with open(args.config, "r") as yml:
        config = yaml.safe_load(yml)

    config = load_config(args.config)
    save_dir = None

    if args.eval:
        Log("Running MonoGS in Evaluation Mode")
        Log("Following config will be overriden")
        Log("\tsave_results=True")
        config["Results"]["save_results"] = True
        Log("\tuse_gui=False")
        config["Results"]["use_gui"] = False
        Log("\teval_rendering=True")
        config["Results"]["eval_rendering"] = True
        Log("\tuse_wandb=True")
        config["Results"]["use_wandb"] = True

    if config["Results"]["save_results"]:
        mkdir_p(config["Results"]["save_dir"])
        current_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        path = config["Dataset"]["dataset_path"].split("/")
        save_dir = os.path.join(
            config["Results"]["save_dir"], path[-3] + "_" + path[-2], current_datetime
        )
        tmp = args.config
        tmp = tmp.split(".")[0]
        config["Results"]["save_dir"] = save_dir
        mkdir_p(save_dir)
        with open(os.path.join(save_dir, "config.yml"), "w") as file:
            documents = yaml.dump(config, file)
        Log("saving results in " + save_dir)
        run = wandb.init(
            project="MonoGS",
            name=f"{tmp}_{current_datetime}",
            config=config,
            mode=None if config["Results"]["use_wandb"] else "disabled",
        )
        wandb.define_metric("frame_idx")
        wandb.define_metric("ate*", step_metric="frame_idx")

    slam = SLAM(config, save_dir=save_dir)

    slam.run()
    wandb.finish()

    # All done
    Log("Done.")
