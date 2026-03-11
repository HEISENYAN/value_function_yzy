import logging
import os
import pathlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch
import torch.distributed as dist
import transformers
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoProcessor, Trainer, TrainerCallback

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from qwenvl.data.data_processor_pair_map import make_supervised_pair_map_data_module
from qwenvl.train.argument import DataArguments, ModelArguments, TrainingArguments
from qwenvl.train.pair_model import (
    QwenPairDeltaModel,
    export_pair_model,
    get_visual_bridge_module,
    set_backbone_trainable,
)

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
    trainer.save_model(output_dir)


class EpochAwareDataLoader(DataLoader):
    def set_epoch(self, epoch: int):
        if hasattr(self.sampler, "set_epoch"):
            self.sampler.set_epoch(epoch)


class PairRegressionTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._seen_samples_local = 0
        self._latest_value_loss = None
        self._latest_value_mse = None

    def _build_loader(self, dataset, batch_size: int, shuffle: bool, drop_last: bool):
        sampler = None
        if dist.is_initialized():
            sampler = DistributedSampler(
                dataset,
                num_replicas=dist.get_world_size(),
                rank=dist.get_rank(),
                shuffle=shuffle,
                drop_last=drop_last,
            )

        kwargs = {
            "dataset": dataset,
            "batch_size": batch_size,
            "shuffle": sampler is None and shuffle,
            "sampler": sampler,
            "collate_fn": self.data_collator,
            "drop_last": drop_last,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": True,
            "persistent_workers": self.args.dataloader_num_workers > 0,
        }
        if self.args.dataloader_num_workers > 0:
            kwargs["prefetch_factor"] = 2
        return EpochAwareDataLoader(**kwargs)

    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        return self._build_loader(
            dataset=self.train_dataset,
            batch_size=self._train_batch_size,
            shuffle=True,
            drop_last=bool(self.args.dataloader_drop_last),
        )

    def get_eval_dataloader(self, eval_dataset=None):
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        return self._build_loader(
            dataset=eval_dataset,
            batch_size=self.args.eval_batch_size,
            shuffle=False,
            drop_last=bool(self.args.dataloader_drop_last),
        )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        delta_labels = inputs.pop("delta_labels")
        self._seen_samples_local += int(delta_labels.shape[0])
        t_group_weights = inputs.pop("t_group_weights", None)
        if not getattr(self.args, "pair_use_t_group_weight", True):
            t_group_weights = None
        outputs = model(**inputs, delta_labels=delta_labels, t_group_weights=t_group_weights)
        loss = outputs["loss"]
        if outputs.get("value_loss") is not None:
            self._latest_value_loss = float(outputs["value_loss"].detach().float().item())
        if outputs.get("value_mse") is not None:
            self._latest_value_mse = float(outputs["value_mse"].detach().float().item())
        return (loss, outputs) if return_outputs else loss

    def log(self, logs, *args, **kwargs):
        logs = dict(logs)
        if self._latest_value_loss is not None and "value_loss" not in logs:
            logs["value_loss"] = float(self._latest_value_loss)
        if self._latest_value_mse is not None and "value_mse" not in logs:
            logs["value_mse"] = float(self._latest_value_mse)
        logs["samples_seen_local"] = float(self._seen_samples_local)
        if dist.is_initialized():
            device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
            tensor = torch.tensor([float(self._seen_samples_local)], device=device)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            if self.is_world_process_zero():
                logs["samples_seen_global"] = float(tensor.item())
        super().log(logs, *args, **kwargs)

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        if getattr(self.args, "value_head_lr", None) is None:
            return super().create_optimizer()

        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        head_params_decay = []
        head_params_nodecay = []
        base_params_decay = []
        base_params_nodecay = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            in_head = name.startswith("value_head")
            is_no_decay = any(token in name for token in no_decay)
            if in_head and not is_no_decay:
                head_params_decay.append(param)
            elif in_head and is_no_decay:
                head_params_nodecay.append(param)
            elif not in_head and not is_no_decay:
                base_params_decay.append(param)
            else:
                base_params_nodecay.append(param)

        optimizer_grouped_parameters = [
            {
                "params": base_params_decay,
                "weight_decay": self.args.weight_decay,
                "lr": self.args.learning_rate,
            },
            {
                "params": base_params_nodecay,
                "weight_decay": 0.0,
                "lr": self.args.learning_rate,
            },
            {
                "params": head_params_decay,
                "weight_decay": getattr(self.args, "value_head_weight_decay", 0.0),
                "lr": self.args.value_head_lr,
            },
            {
                "params": head_params_nodecay,
                "weight_decay": 0.0,
                "lr": self.args.value_head_lr,
            },
        ]
        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
        if self.is_world_process_zero():
            for idx, group in enumerate(optimizer_grouped_parameters):
                rank0_print(
                    f"[PAIR-OPT] group={idx}, lr={group['lr']}, weight_decay={group['weight_decay']}, "
                    f"params={sum(p.numel() for p in group['params'])}"
                )
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        return self.optimizer


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


def _final_samples_seen_global(local_count: int) -> float:
    if not dist.is_initialized():
        return float(local_count)
    device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
    tensor = torch.tensor([float(local_count)], device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item())


def write_training_summary(trainer: PairRegressionTrainer, output_dir: str, extra: dict):
    samples_seen_global = _final_samples_seen_global(trainer._seen_samples_local)
    if not trainer.is_world_process_zero():
        return
    summary_path = Path(output_dir) / "train_summary.json"
    last_logged = {}
    for entry in reversed(trainer.state.log_history):
        if "loss" in entry or "eval_loss" in entry or "value_loss" in entry:
            last_logged = dict(entry)
            break
    payload = {
        "global_step": int(trainer.state.global_step),
        "best_metric": trainer.state.best_metric,
        "samples_seen_local": float(trainer._seen_samples_local),
        "samples_seen_global": samples_seen_global,
        "last_log": last_logged,
    }
    payload.update(extra)
    with open(summary_path, "w", encoding="utf-8") as fout:
        json.dump(payload, fout, ensure_ascii=True, indent=2, sort_keys=True)


def train(attn_implementation="flash_attention_2"):
    global local_rank

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank if training_args.local_rank != -1 else 0
    os.makedirs(training_args.output_dir, exist_ok=True)
    if training_args.save_interval is not None:
        training_args.save_strategy = "steps"
        training_args.save_steps = int(training_args.save_interval)

    model = QwenPairDeltaModel(
        model_name_or_path=model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        attn_implementation=attn_implementation,
        bf16=training_args.bf16,
    )
    data_args.model_type = model.model_type
    model.backbone.config.use_cache = False

    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path)
    if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
        processor.tokenizer.padding_side = "left"
        processor.tokenizer.model_max_length = training_args.model_max_length
    tokenizer = processor.tokenizer
    if tokenizer is None:
        raise ValueError("AutoProcessor did not provide tokenizer for pair training.")

    if training_args.gradient_checkpointing:
        if hasattr(model.backbone, "enable_input_require_grads"):
            model.backbone.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, _input, output):
                output.requires_grad_(True)
            model.backbone.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft import LoraConfig, TaskType, get_peft_model

        for param in model.backbone.parameters():
            param.requires_grad = False
        lora_config = LoraConfig(
            r=training_args.lora_r or 64,
            lora_alpha=training_args.lora_alpha or 128,
            lora_dropout=training_args.lora_dropout or 0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model.backbone = get_peft_model(model.backbone, lora_config)
        for param in model.value_head.parameters():
            param.requires_grad = True
    else:
        set_backbone_trainable(model_args, model.backbone)
        for param in model.value_head.parameters():
            param.requires_grad = True

    training_args.dataloader_drop_last = True
    training_args.dataloader_pin_memory = True
    training_args.dataloader_persistent_workers = training_args.dataloader_num_workers > 0
    if training_args.dataloader_num_workers > 0:
        training_args.dataloader_prefetch_factor = 2

    data_module, stats = make_supervised_pair_map_data_module(
        processor,
        data_args=data_args,
        model_args=model_args,
        return_stats=True,
    )

    train_len = len(data_module["train_dataset"])
    eval_len = len(data_module["eval_dataset"])
    per_rank_batch = int(training_args.per_device_train_batch_size)
    grad_accum = int(training_args.gradient_accumulation_steps)
    est_steps = max(1, (train_len + (per_rank_batch * grad_accum) - 1) // (per_rank_batch * grad_accum))
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank0_print(
        f"[PAIR-MAP-START] rank={local_rank}, world_size={world_size}, repos={len(stats['repo_paths'])}, "
        f"train_samples_per_rank={train_len}, eval_samples_per_rank={eval_len}, est_steps_per_epoch={est_steps}"
    )
    rank0_print(
        f"[PAIR-MAP-START] tokenizer_padding_side={tokenizer.padding_side}, "
        f"workers={training_args.dataloader_num_workers}, micro_batch={per_rank_batch}, grad_accum={grad_accum}"
    )

    trainer = PairRegressionTrainer(
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
    rank0_print(f"[PAIR-MAP] output_dir={training_args.output_dir}, ckpts={ckpt_list}")

    try:
        if ckpt_list:
            logging.info("checkpoint found, resume training")
            trainer.train(resume_from_checkpoint=True)
        else:
            trainer.train()
    except Exception:
        logging.exception("Pair map training failed on this rank.")
        if dist.is_initialized():
            try:
                dist.abort()
            except Exception:
                pass
        raise

    trainer.save_state()
    model.backbone.config.use_cache = True
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)
    final_model_dir = os.path.join(training_args.output_dir, "final_model")
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=final_model_dir)
    processor.save_pretrained(final_model_dir)
    export_pair_model(
        model=trainer.model,
        processor=processor,
        output_dir=final_model_dir,
        metadata={
            "save_format": training_args.save_format,
            "run_name": training_args.run_name,
            "output_dir": training_args.output_dir,
            "lora_enable": bool(training_args.lora_enable),
        },
    )
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
