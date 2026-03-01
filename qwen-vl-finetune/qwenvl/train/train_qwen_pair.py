# Pairwise delta training entrypoint for Qwen2.5-VL.

import logging
import os
import pathlib
import sys
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
import transformers
from transformers import AutoProcessor, Trainer
from transformers import Qwen2_5_VLForConditionalGeneration

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from qwenvl.data.data_processor_pair import make_supervised_data_module
from qwenvl.train.argument import ModelArguments, DataArguments, TrainingArguments


local_rank = None


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


def set_backbone_trainable(model_args, backbone):
    if model_args.tune_mm_vision:
        for _, p in backbone.visual.named_parameters():
            p.requires_grad = True
    else:
        for _, p in backbone.visual.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_mlp:
        for _, p in backbone.visual.merger.named_parameters():
            p.requires_grad = True
    else:
        for _, p in backbone.visual.merger.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_llm:
        for _, p in backbone.language_model.named_parameters():
            p.requires_grad = True
        backbone.lm_head.requires_grad = True
    else:
        for _, p in backbone.language_model.named_parameters():
            p.requires_grad = False
        backbone.lm_head.requires_grad = False


class QwenPairDeltaModel(nn.Module):
    def __init__(self, model_name_or_path: str, cache_dir=None, attn_implementation=None, bf16=False):
        super().__init__()
        self.backbone = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if bf16 else None),
        )
        hidden_size = self.backbone.config.hidden_size
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 1),
            nn.Tanh(),
        )
        self.gradient_checkpointing_enabled = False

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        self.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
        logging.info("Enabled gradient checkpointing for QwenPairDeltaModel")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.backbone.gradient_checkpointing_disable()
        logging.info("Disabled gradient checkpointing for QwenPairDeltaModel")

    def is_gradient_checkpointing_enabled(self):
        """Check if gradient checkpointing is enabled."""
        return self.gradient_checkpointing_enabled

    @property
    def config(self):
        return self.backbone.config

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        pixel_values=None,
        image_grid_thw=None,
        pixel_values_videos=None,
        video_grid_thw=None,
        position_ids=None,
        delta_labels=None,
        t_group_weights=None,
        **kwargs,
    ):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )

        last_hidden = outputs.hidden_states[-1]  # [B, L, W]
        if attention_mask is None:
            pooled = last_hidden[:, -1, :]
        else:
            if attention_mask.dtype != torch.long:
                mask_long = attention_mask.long()
            else:
                mask_long = attention_mask
            last_idx = torch.clamp(mask_long.sum(dim=1) - 1, min=0)
            batch_idx = torch.arange(last_hidden.size(0), device=last_hidden.device)
            pooled = last_hidden[batch_idx, last_idx, :]

        delta_pred = self.value_head(pooled).squeeze(-1)

        loss = None
        if delta_labels is not None:
            target = torch.clamp(delta_labels, -1.0, 1.0)
            mse = (delta_pred - target) ** 2
            if t_group_weights is not None:
                w = torch.clamp(t_group_weights, min=0.0)
                w_sum = torch.clamp(w.sum(), min=1e-8)
                loss = (mse * w).sum() / w_sum
            else:
                loss = mse.mean()

        return {"loss": loss, "delta_pred": delta_pred}


class PairRegressionTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        delta_labels = inputs.pop("delta_labels")
        t_group_weights = inputs.pop("t_group_weights", None)
        if not getattr(self.args, "pair_use_t_group_weight", True):
            t_group_weights = None
        outputs = model(**inputs, delta_labels=delta_labels, t_group_weights=t_group_weights)
        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss

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

        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            in_head = n.startswith("value_head")
            is_no_decay = any(nd in n for nd in no_decay)
            if in_head and not is_no_decay:
                head_params_decay.append(p)
            elif in_head and is_no_decay:
                head_params_nodecay.append(p)
            elif not in_head and not is_no_decay:
                base_params_decay.append(p)
            else:
                base_params_nodecay.append(p)

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
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        return self.optimizer


def train(attn_implementation=None):
    global local_rank

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)

    if not getattr(data_args, "pair_mode", False):
        rank0_print("[WARN] --pair_mode is False. train_qwen_pair.py still runs pair training by design.")

    model = QwenPairDeltaModel(
        model_name_or_path=model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        attn_implementation=attn_implementation,
        bf16=training_args.bf16,
    )
    data_args.model_type = "qwen2.5vl"
    model.backbone.config.use_cache = False

    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path)

    if training_args.gradient_checkpointing:
        if hasattr(model.backbone, "enable_input_require_grads"):
            model.backbone.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.backbone.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    if training_args.lora_enable:
        from peft import LoraConfig, TaskType, get_peft_model
        for p in model.backbone.parameters():
            p.requires_grad = False
        lora_config = LoraConfig(
            r=training_args.lora_r or 64,
            lora_alpha=training_args.lora_alpha or 128,
            lora_dropout=training_args.lora_dropout or 0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model.backbone = get_peft_model(model.backbone, lora_config)
        for p in model.value_head.parameters():
            p.requires_grad = True
    else:
        set_backbone_trainable(model_args, model.backbone)
        for p in model.value_head.parameters():
            p.requires_grad = True

    data_module = make_supervised_data_module(processor, data_args=data_args, model_args=model_args, value_tokenizer=None)

    trainer = PairRegressionTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        **data_module,
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logging.info("checkpoint found, resume training")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.backbone.config.use_cache = True
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    train(attn_implementation=None)
