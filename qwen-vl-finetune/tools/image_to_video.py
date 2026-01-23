import os
import shutil
import argparse
from pathlib import Path
from copy import deepcopy
import torch
import tqdm

# 设置环境变量，与您提供的脚本保持一致
os.environ["HF_LEROBOT_HOME"] = "data"  # 或者您希望保存数据的根目录

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

def convert_dataset_mode(
    source_root: Path,
    repo_id: str,
    output_root: Path = None,
    force_override: bool = False
):
    """
    读取现有的 LeRobot 数据集 (Image模式)，并将其重新保存为 Video 模式。
    """
    print(f"正在加载源数据集: {source_root}")
    # 1. 加载源数据集 (Image 模式)
    # 这里的 root 应该是包含 meta/ 文件夹的父级目录
    src_ds = LeRobotDataset(str(source_root))
    
    # 检查源数据是否已经加载成功
    print(f"源数据集信息: {src_ds.num_episodes} episodes, {len(src_ds)} frames, FPS={src_ds.fps}")

    # 2. 准备新数据集的配置
    # 我们需要修改 features 中的 dtype，从 image -> video
    new_features = deepcopy(src_ds.features)
    
    # 遍历特征，找到 image 类型的，准备标记为 video
    # 注意：LeRobotDataset.create(use_videos=True) 会自动处理大部分逻辑，
    # 但显式更新 feature 定义是个好习惯
    for key, feat in new_features.items():
        if feat["dtype"] == "image":
            feat["dtype"] = "video"
            print(f"  - 将特征 [{key}] 转换为 Video 模式")

    # 确定输出路径
    if output_root is None:
        # 如果未指定，默认保存在 source_root 同级的 _video 后缀目录
        output_root = source_root.parent / f"{source_root.name}_video"
    
    target_path = output_root
    
    if target_path.exists():
        if force_override:
            print(f"警告: 目标路径 {target_path} 已存在，正在删除...")
            shutil.rmtree(target_path)
        else:
            raise FileExistsError(f"目标路径 {target_path} 已存在。请使用 --force 覆盖。")

    print(f"正在创建目标数据集: {repo_id} @ {target_path}")

    # 3. 初始化目标数据集 (开启 use_videos=True)
    tgt_ds = LeRobotDataset.create(
        repo_id=repo_id,
        root=target_path,
        fps=src_ds.fps,
        robot_type=src_ds.meta.robot_type,
        features=new_features,
        use_videos=True,  # <--- 关键点：强制开启视频模式
        image_writer_processes=10, # 并行写入加速
        image_writer_threads=5,
    )

    # 4. 逐个 Episode 迁移数据
    # 我们不直接遍历 src_ds (那样是逐帧)，而是按 Episode 遍历，以确保 save_episode 正确调用
    print("开始迁移数据...")
    
    for ep_idx in tqdm.tqdm(range(src_ds.num_episodes)):
        # 获取当前 Episode 在总帧数中的范围
        # 在旧版本中，episode_data_index 可能是 tensor 或 dict
        if isinstance(src_ds.episode_data_index, dict):
             # 兼容某些版本的结构
             from_idx = src_ds.episode_data_index["from"][ep_idx].item()
             to_idx = src_ds.episode_data_index["to"][ep_idx].item()
        else:
             # 假设是 tensor (N_episodes, 2)
             from_idx = src_ds.episode_data_index[ep_idx][0].item()
             to_idx = src_ds.episode_data_index[ep_idx][1].item()

        # 获取该 Episode 对应的任务描述 (Task / Instruction)
        # 尝试从 metadata 中获取
        task = "default_task"
        try:
            # 不同的旧版本存储 task 的方式可能不同
            # 方式A: meta.tasks 列表配合 episode 索引
            # 方式B: data 中包含 language_instruction 列
            if "language_instruction" in src_ds.features:
                # 如果每一帧都有指令，取第一帧的即可
                task = src_ds[from_idx]["language_instruction"]
            elif hasattr(src_ds.meta, "tasks") and src_ds.meta.tasks:
                # 如果有任务列表，通常 episodes 属性里存了 task_index
                # 这里做简化的假设，或者直接取默认值，视具体数据而定
                pass
        except Exception:
            pass

        # 逐帧读取源数据并写入目标数据
        for frame_idx in range(from_idx, to_idx):
            # src_ds[i] 返回的是一个字典，包含该帧所有数据 (Tensor格式)
            frame_data = src_ds[frame_idx]
            
            # 注意：LeRobotDataset 读取出的图像通常是 Float Tensor (0-1) 或 Int Tensor
            # add_frame 能够处理 Tensor 输入。
            # 关键：直接透传 frame_data 字典
            tgt_ds.add_frame(frame_data, task=task)

        # 当前 Episode 结束，保存并触发视频编码
        tgt_ds.save_episode()

    print("转换完成！")
    print(f"新数据集保存在: {target_path}")
    print("请检查 videos/ 目录是否生成了 MP4 文件。")

    # 可选：如果需要在 Hub 上使用，执行 tgt_ds.push_to_hub()
    # tgt_ds.push_to_hub()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True, help="源数据集路径 (包含 meta/ 的目录)")
    parser.add_argument("--repo_id", type=str, default="converted_dataset", help="新数据集的 Repo ID")
    parser.add_argument("--output", type=str, default=None, help="输出路径 (可选)")
    parser.add_argument("--force", action="store_true", help="强制覆盖输出目录")
    
    args = parser.parse_args()
    
    # 转换 Path 对象
    src_path = Path(args.source)
    out_path = Path(args.output) if args.output else None
    
    convert_dataset_mode(src_path, args.repo_id, out_path, args.force)