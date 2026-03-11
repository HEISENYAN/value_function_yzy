import json
import logging
import os
import pathlib
import sys
from dataclasses import asdict
from pathlib import Path

import torch
import torch.distributed as dist
import transformers
from transformers import AutoProcessor, Trainer, TrainerCallback
from transformers import Qwen2_5_VLForConditionalGeneration

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from qwenvl.data.data_processor_pi06_map import make_supervised_pi06_map_data_module
from qwenvl.train.argument import DataArguments, ModelArguments, TrainingArguments
from qwenvl.utils.value_tokenizer import ValueTokenizer

import datasets.features.features as hf_features

if "List" not in hf_features._FEATURE_TYPES:
    hf_features._FEATURE_TYPES["List"] = hf_features.Sequence

local_rank = 0


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)


def set_model(model_args, model):
    if model_args.tune_mm_vision:
        for _, p in model.visual.named_parameters():
            p.requires_grad = True
    else:
        for _, p in model.visual.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_mlp:
        for _, p in model.visual.merger.named_parameters():
            p.requires_grad = True
    else:
        for _, p in model.visual.merger.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_llm:
        for _, p in model.language_model.named_parameters():
            p.requires_grad = True
        model.lm_head.requires_grad = True
    else:
        for _, p in model.language_model.named_parameters():
            p.requires_grad = False
        model.lm_head.requires_grad = False


class StructuredMetricsCallback(TrainerCallback):
    def __init__(self, output_dir: str, config_payload: dict):
        self.output_dir = Path(output_dir)
        self.metrics_path = self.output_dir / "train_metrics.jsonl"
        self.config_path = self.output_dir / "run_config.json"
        self.config_payload = config_payload

    def on_train_begin(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, "w", encoding="utf-8") as fout:
            json.dump(self.config_payload, fout, ensure_ascii=True, indent=2, sort_keys=True)
        if self.metrics_path.exists():
            self.metrics_path.unlink()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero or not logs:
            return
        record = {
            "global_step": int(state.global_step),
            "epoch": float(state.epoch) if state.epoch is not None else None,
        }
        record.update(logs)
        with open(self.metrics_path, "a", encoding="utf-8") as fout:
            fout.write(json.dumps(record, ensure_ascii=True) + "\n")


def write_training_summary(trainer: Trainer, output_dir: str, extra: dict):
    if not trainer.is_world_process_zero():
        return
    summary_path = Path(output_dir) / "train_summary.json"
    last_logged = {}
    for entry in reversed(trainer.state.log_history):
        if "loss" in entry or "eval_loss" in entry:
            last_logged = dict(entry)
            break
    payload = {
        "global_step": int(trainer.state.global_step),
        "best_metric": trainer.state.best_metric,
        "last_log": last_logged,
    }
    payload.update(extra)
    with open(summary_path, "w", encoding="utf-8") as fout:
        json.dump(payload, fout, ensure_ascii=True, indent=2, sort_keys=True)


def train(attn_implementation="flash_attention_2"):
    global local_rank

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if not getattr(data_args, "use_value_tokenizer", False):
        raise ValueError("train_qwen_pi06.py requires --use_value_tokenizer True.")

    local_rank = training_args.local_rank if training_args.local_rank != -1 else 0
    os.makedirs(training_args.output_dir, exist_ok=True)
    if training_args.save_interval is not None:
        training_args.save_strategy = "steps"
        training_args.save_steps = int(training_args.save_interval)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        attn_implementation=attn_implementation,
        torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
    )
    data_args.model_type = "qwen2.5vl"
    model.config.use_cache = False

    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path)
    if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
        processor.tokenizer.padding_side = "right"
        processor.tokenizer.model_max_length = training_args.model_max_length
    tokenizer = processor.tokenizer
    if tokenizer is None:
        raise ValueError("AutoProcessor did not provide tokenizer for pi06 training.")

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, _input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft import LoraConfig, TaskType, get_peft_model

        for param in model.parameters():
            param.requires_grad = False
        lora_config = LoraConfig(
            r=training_args.lora_r or 64,
            lora_alpha=training_args.lora_alpha or 128,
            lora_dropout=training_args.lora_dropout or 0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
    else:
        set_model(model_args, model)

    if training_args.dataloader_num_workers > 0:
        training_args.dataloader_prefetch_factor = 2
    training_args.dataloader_pin_memory = True
    training_args.dataloader_persistent_workers = training_args.dataloader_num_workers > 0

    value_tokenizer = ValueTokenizer(
        llm_path=model_args.model_name_or_path,
        bins=data_args.value_tokenizer_bins,
        min_value=data_args.value_tokenizer_min,
        max_value=data_args.value_tokenizer_max,
    )
    rank0_print(
        f"[PI06-MAP] ValueTokenizer bins={data_args.value_tokenizer_bins}, "
        f"range=[{data_args.value_tokenizer_min}, {data_args.value_tokenizer_max}]"
    )

    data_module, stats = make_supervised_pi06_map_data_module(
        processor,
        data_args=data_args,
        model_args=model_args,
        value_tokenizer=value_tokenizer,
        return_stats=True,
    )

    train_len = len(data_module["train_dataset"])
    eval_len = len(data_module["eval_dataset"])
    per_rank_batch = int(training_args.per_device_train_batch_size)
    grad_accum = int(training_args.gradient_accumulation_steps)
    est_steps = max(1, (train_len + (per_rank_batch * grad_accum) - 1) // (per_rank_batch * grad_accum))
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank0_print(
        f"[PI06-MAP-START] rank={local_rank}, world_size={world_size}, repos={len(stats['repo_paths'])}, "
        f"train_samples_per_rank={train_len}, eval_samples_per_rank={eval_len}, est_steps_per_epoch={est_steps}"
    )
    rank0_print(
        f"[PI06-MAP-START] tokenizer_padding_side={tokenizer.padding_side}, "
        f"workers={training_args.dataloader_num_workers}, micro_batch={per_rank_batch}, grad_accum={grad_accum}"
    )

    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        callbacks=[
            StructuredMetricsCallback(
                output_dir=training_args.output_dir,
                config_payload={
                    "model_args": asdict(model_args),
                    "data_args": asdict(data_args),
                    "training_args": training_args.to_dict(),
                },
            )
        ],
        **data_module,
    )

    ckpt_list = list(pathlib.Path(training_args.output_dir).glob("checkpoint-*"))
    rank0_print(f"[PI06-MAP] output_dir={training_args.output_dir}, ckpts={ckpt_list}")

    try:
        if ckpt_list:
            logging.info("checkpoint found, resume training")
            trainer.train(resume_from_checkpoint=True)
        else:
            trainer.train()
    except Exception:
        logging.exception("Pi06 map training failed on this rank.")
        if dist.is_initialized():
            try:
                dist.abort()
            except Exception:
                pass
        raise

    trainer.save_state()
    model.config.use_cache = True
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)
    final_model_dir = os.path.join(training_args.output_dir, "final_model")
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=final_model_dir)
    processor.save_pretrained(final_model_dir)
    write_training_summary(
        trainer,
        training_args.output_dir,
        extra={
            "final_model_dir": final_model_dir,
            "repo_paths": stats["repo_paths"],
            "train_total_samples": stats["train_total_samples"],
            "eval_total_samples": stats["eval_total_samples"],
        },
    )


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
