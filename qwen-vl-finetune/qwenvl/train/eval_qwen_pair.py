import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import transformers
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from qwenvl.data.data_loader_pair_clean import PAIR_PROMPT_TEMPLATE, LeRobotPairDataset
from qwenvl.train.argument import ModelArguments, DataArguments, EvalArguments


def rank0_print(*args):
    print(*args)


def _to_pil_image(img_data):
    img_rgb = img_data.permute(1, 2, 0)
    img_rgb_np = img_rgb.numpy()
    if img_rgb_np.dtype == np.float32:
        img_rgb_np = np.clip(img_rgb_np * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(img_rgb_np, mode="RGB")


def _load_pair_state_if_exists(model, checkpoint_dir: str):
    pt_path = Path(checkpoint_dir) / "pytorch_model.bin"
    st_path = Path(checkpoint_dir) / "model.safetensors"
    if pt_path.exists():
        state = torch.load(pt_path, map_location="cpu")
        model.load_state_dict(state, strict=False)
        rank0_print(f"Loaded pair state dict from {pt_path}")
    elif st_path.exists():
        from safetensors.torch import load_file
        state = load_file(str(st_path))
        model.load_state_dict(state, strict=False)
        rank0_print(f"Loaded pair state dict from {st_path}")
    else:
        rank0_print("No consolidated pair checkpoint file found; using backbone-pretrained + in-model weights.")


def load_model_and_processor(model_name_or_path, attn_implementation=None):
    rank0_print(f"Loading pair model from: {model_name_or_path}")
    model = QwenPairDeltaModel(
        model_name_or_path=model_name_or_path,
        cache_dir=None,
        attn_implementation=attn_implementation,
        bf16=True,
    )
    _load_pair_state_if_exists(model, model_name_or_path)

    processor = AutoProcessor.from_pretrained(model_name_or_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    model.backbone.config.use_cache = False
    return model, processor, device


def build_pair_messages(instruction: str):
    return [
        {"role": "user", "content": [{"type": "text", "text": PAIR_PROMPT_TEMPLATE.format(instruction=instruction)}]},
        {"role": "assistant", "content": [{"type": "text", "text": ""}]},
    ]


def _expand_message_with_images(prompt_text: str, images: List[Image.Image]):
    content = []
    text_parts = prompt_text.split("<image>")
    image_idx = 0
    for i, part in enumerate(text_parts):
        if part.strip():
            content.append({"type": "text", "text": part.strip()})
        if i < len(text_parts) - 1:
            content.append({"type": "image", "image": images[image_idx]})
            image_idx += 1
    return content


@torch.no_grad()
def predict_delta(model, processor, device, instruction: str, images: List[Image.Image]) -> float:
    prompt = PAIR_PROMPT_TEMPLATE.format(instruction=instruction)
    user_content = _expand_message_with_images(prompt, images)
    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": [{"type": "text", "text": ""}]},
    ]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, return_dict=True, return_tensors="pt"
    )
    for k, v in list(inputs.items()):
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(device)

    outputs = model(**inputs)
    return float(outputs["delta_pred"][0].item())


def _get_episode_instruction(meta_episode) -> str:
    instruction = meta_episode.get("tasks")
    if isinstance(instruction, list) and len(instruction) > 0:
        return str(instruction[0])
    if instruction is None:
        return "perform the task"
    return str(instruction)


def _frame_triplet(frame_cache, step: int, camera_names: List[str]):
    return [frame_cache[step][cam] for cam in camera_names]


def evaluate_episode(model, processor, device, dataset, ep_info, data_args) -> Dict:
    camera_names = getattr(data_args, "camera_names", ["cam_high", "cam_left_wrist", "cam_right_wrist"])
    T = ep_info["length"]
    start = ep_info["global_start_index"]
    instruction = ep_info["instruction"]

    frame_cache = {}
    for step in range(T):
        row = dataset[start + step]
        frame_cache[step] = {}
        for cam in camera_names:
            frame_cache[step][cam] = _to_pil_image(row[f"observation.images.{cam}"])

    anchor_errors = []
    zero_abs_vals = []
    fb_errors = []
    phi = []
    delta16_gt_delta8 = []

    short_step = int(getattr(data_args, "pair_short_step", 8))
    mid_step = int(getattr(data_args, "pair_mid_step", 16))
    random_min = int(getattr(data_args, "pair_random_min", 1))
    rng = np.random.default_rng(42 + ep_info["episode_idx"])

    for t in range(T):
        imgs_anchor = _frame_triplet(frame_cache, 0, camera_names) + _frame_triplet(frame_cache, t, camera_names)
        pred_anchor = predict_delta(model, processor, device, instruction, imgs_anchor)
        target_anchor = t / T
        phi.append(pred_anchor)
        anchor_errors.append(abs(pred_anchor - target_anchor))

        imgs_zero = _frame_triplet(frame_cache, t, camera_names) + _frame_triplet(frame_cache, t, camera_names)
        pred_zero = predict_delta(model, processor, device, instruction, imgs_zero)
        zero_abs_vals.append(abs(pred_zero))

        forward_pairs = [(0, t, t / T)]
        if t >= short_step:
            forward_pairs.append((t - short_step, t, short_step / T))
        if t >= mid_step:
            forward_pairs.append((t - mid_step, t, mid_step / T))
        forward_pairs.append((t, t, 0.0))
        if t >= random_min:
            r = int(rng.integers(random_min, t + 1))
            forward_pairs.append((t - r, t, r / T))

        for a, b, _ in forward_pairs:
            imgs_f = _frame_triplet(frame_cache, a, camera_names) + _frame_triplet(frame_cache, b, camera_names)
            imgs_b = _frame_triplet(frame_cache, b, camera_names) + _frame_triplet(frame_cache, a, camera_names)
            pred_f = predict_delta(model, processor, device, instruction, imgs_f)
            pred_b = predict_delta(model, processor, device, instruction, imgs_b)
            fb_errors.append(abs(pred_f + pred_b))

        if t >= mid_step and t >= short_step:
            imgs16 = _frame_triplet(frame_cache, t - mid_step, camera_names) + _frame_triplet(frame_cache, t, camera_names)
            imgs8 = _frame_triplet(frame_cache, t - short_step, camera_names) + _frame_triplet(frame_cache, t, camera_names)
            pred16 = abs(predict_delta(model, processor, device, instruction, imgs16))
            pred8 = abs(predict_delta(model, processor, device, instruction, imgs8))
            delta16_gt_delta8.append(float(pred16 > pred8))

    monotonic = []
    for t in range(len(phi) - 1):
        monotonic.append(float(phi[t + 1] >= phi[t]))

    return {
        "episode_idx": int(ep_info["episode_idx"]),
        "episode_len": int(T),
        "E_anchor": float(np.mean(anchor_errors)) if anchor_errors else None,
        "E_zero": float(np.mean(zero_abs_vals)) if zero_abs_vals else None,
        "E_fb": float(np.mean(fb_errors)) if fb_errors else None,
        "scale_consistency": {
            "monotonicity_ratio": float(np.mean(monotonic)) if monotonic else None,
            "delta16_gt_delta8_ratio": float(np.mean(delta16_gt_delta8)) if delta16_gt_delta8 else None,
        },
    }


def evaluate(model_args=None, data_args=None, eval_args=None):
    if model_args is None or data_args is None or eval_args is None:
        parser = transformers.HfArgumentParser((ModelArguments, DataArguments, EvalArguments, TrainingArguments))
        model_args, data_args, eval_args, _ = parser.parse_args_into_dataclasses()

    eval_output_dir = Path(eval_args.eval_output_dir)
    eval_output_dir.mkdir(parents=True, exist_ok=True)

    model_name_or_path = os.path.expanduser(model_args.model_name_or_path)
    model, processor, device = load_model_and_processor(model_name_or_path)

    dataset_dir = data_args.eval_dataset_use or data_args.dataset_use
    if not os.path.isabs(dataset_dir):
        abs_path = os.path.abspath(dataset_dir)
        if os.path.exists(abs_path):
            dataset_dir = abs_path
        else:
            dataset_dir = str(Path(__file__).parent.parent.parent.parent / dataset_dir)
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    rank0_print(f"Loading evaluation dataset from: {dataset_dir}")

    base_seed = getattr(data_args, "seed", 42)
    val_ratio = getattr(data_args, "val_ratio", 0.1)
    camera_names = getattr(data_args, "camera_names", ["cam_high", "cam_left_wrist", "cam_right_wrist"])

    pair_dataset = LeRobotPairDataset(
        dataset_dir=dataset_dir,
        transform=None,
        tokenizer=None,
        language_instruction="perform the task",
        seed=base_seed,
        val_ratio=val_ratio,
        buffer_size=500,
        camera_names=camera_names,
        split="val",
        max_episodes=None,
        pair_short_step=getattr(data_args, "pair_short_step", 8),
        pair_mid_step=getattr(data_args, "pair_mid_step", 16),
        pair_random_min=getattr(data_args, "pair_random_min", 1),
        pair_add_backward=getattr(data_args, "pair_add_backward", True),
        pair_prompt_style=getattr(data_args, "pair_prompt_style", "explicit_t0_t1"),
    )

    lerobot_dataset = pair_dataset.lerobot_dataset
    episodes = list(pair_dataset.episodes_meta)

    max_episodes = eval_args.max_episodes if eval_args.max_episodes is not None else len(episodes)
    episodes = episodes[: max_episodes]

    rank0_print(
        f"Using val split from LeRobotPairDataset for offline eval: "
        f"{len(episodes)} episodes (val_ratio={val_ratio}, seed={base_seed})"
    )

    episode_metrics = []
    for ep_info in tqdm(episodes, desc="Evaluating pair episodes"):
        m = evaluate_episode(model, processor, device, lerobot_dataset, ep_info, data_args)
        episode_metrics.append(m)

    def _collect(key):
        vals = [m[key] for m in episode_metrics if m[key] is not None]
        return float(np.mean(vals)) if vals else None

    mono_vals = [
        m["scale_consistency"]["monotonicity_ratio"]
        for m in episode_metrics
        if m["scale_consistency"]["monotonicity_ratio"] is not None
    ]
    d16_gt_d8_vals = [
        m["scale_consistency"]["delta16_gt_delta8_ratio"]
        for m in episode_metrics
        if m["scale_consistency"]["delta16_gt_delta8_ratio"] is not None
    ]

    summary = {
        "num_episodes": len(episode_metrics),
        "global_metrics": {
            "E_anchor": _collect("E_anchor"),
            "E_zero": _collect("E_zero"),
            "E_fb": _collect("E_fb"),
            "scale_consistency": {
                "monotonicity_ratio": float(np.mean(mono_vals)) if mono_vals else None,
                "delta16_gt_delta8_ratio": float(np.mean(d16_gt_d8_vals)) if d16_gt_d8_vals else None,
            },
        },
    }

    summary_path = eval_output_dir / "evaluation_summary_pair.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    episode_path = eval_output_dir / "episode_metrics_pair.jsonl"
    with open(episode_path, "w", encoding="utf-8") as f:
        for m in episode_metrics:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    rank0_print(f"Saved: {summary_path}")
    rank0_print(f"Saved: {episode_path}")
    rank0_print(f"Done. Episodes: {len(episode_metrics)}")


if __name__ == "__main__":
    evaluate()
