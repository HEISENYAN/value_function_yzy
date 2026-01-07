import os
import torch
import transformers
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from transformers import Qwen2_5_VLForConditionalGeneration

from qwenvl.data.data_loader import LeRobotValueDataset
from qwenvl.train.argument import (
    ModelArguments,
    DataArguments,
    EvalArguments,
)
from transformers import AutoProcessor
from qwenvl.utils.value_tokenizer import ValueTokenizer

def rank0_print(*args):
    print(*args)


def detect_dataset_type(dataset_path):
    """Always return 'lerobot' since we now only support LeRobot datasets."""
    return "lerobot"


def create_evaluation_dataset(dataset_path, processor, **kwargs):
    """
    Create LeRobot dataset for evaluation.

    Args:
        dataset_path: Path to LeRobot dataset
        processor: Qwen processor
        **kwargs: Additional arguments

    Returns:
        LeRobotValueDataset instance
    """
    import os
    import torch

    # Resolve dataset directory
    if not os.path.isabs(dataset_path):
        abs_path = os.path.abspath(dataset_path)
        if os.path.exists(abs_path):
            dataset_path = abs_path

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")

    # Get distributed training info
    local_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

    # Create LeRobot dataset
    dataset = LeRobotValueDataset(
        dataset_name="lerobot_eval",
        transform=None,
        tokenizer=processor.tokenizer,
        dataset_dir=dataset_path,
        shuffle=False,  # Keep episode order for evaluation
        local_rank=local_rank,
        world_size=world_size,
        **kwargs
    )

    return dataset


def load_model_and_processor(model_name_or_path, attn_implementation=None):
    """Load model and processor from checkpoint."""
    rank0_print(f"Loading Qwen2.5-VL model from checkpoint: {model_name_or_path}")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name_or_path,
        attn_implementation=attn_implementation,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    processor = AutoProcessor.from_pretrained(model_name_or_path)
    model.eval()
    model.config.use_cache = True

    rank0_print(f"Model loaded: {model.__class__.__name__}")
    return model, processor


def decode_value_token_with_interpolation(value_tokenizer, generated_outputs, input_length):
    """Decode value tokens with interpolation based on logits for continuous values."""
    # Get the logits for the generated token (last token)
    # logits shape: [batch_size, seq_len, vocab_size]
    logits = generated_outputs.logits[0]  # Get first batch

    if len(logits) == 0:
        return 0.0

    # Get logits for the last generated token
    last_token_logits = logits[-1]  # Shape: [vocab_size]

    # Extract logits only for value tokens
    value_token_ids = value_tokenizer.extra_id_token_ids
    value_logits = last_token_logits[value_token_ids]  # Shape: [n_bins]

    # Apply softmax to get probabilities
    value_probs = torch.softmax(value_logits, dim=0).cpu().numpy()

    # Interpolate between bin centers using probabilities as weights
    bin_centers = value_tokenizer.bin_centers  # Shape: [n_bins]
    interpolated_value = np.sum(value_probs * bin_centers)

    return interpolated_value


def decode_value_token(value_tokenizer, generated_token_id):
    """Decode a single token ID to continuous value using ValueTokenizer."""
    # Convert token ID to numpy array
    token_id_array = np.array([generated_token_id])
    # Decode using value_tokenizer
    value = value_tokenizer.decode_token_ids_to_values(token_id_array)
    return value[0] if len(value) > 0 else 0.0


def evaluate_episode(model, processor, value_tokenizer, episode_data, device):
    """Evaluate a single episode and return predicted values."""
    predicted_values = []
    true_values = []
    steps = []
    
    for step_idx, step_data in enumerate(episode_data):
        try:
            # Extract image and conversation
            image = step_data['image']
            conversation = step_data['conversations']
            true_value = step_data.get('value', None)
            
            # Build messages for processor - only include user message (question)
            # We want to predict the assistant's response (value)
            # Use the exact same prompt format as training
            messages = []

            # Extract user message from conversation (should match training format exactly)
            user_message = None
            for turn in conversation:
                if turn["from"] == "human":
                    user_message = turn["value"]
                    break

            # If no user message found in conversation, this indicates a data loading issue
            if user_message is None:
                raise ValueError(f"No user message found in conversation for step data: {step_data}")

            # The user_message should already be in the correct training format:
            # "You are estimating task progress for robotic manipulation.\n\nGiven a task instruction and a single image, estimate the current progress toward completing the task.\n\nObservation: <image>\n\nInstruction: {lang_instruction}"

            # Build message content with proper image handling
            content = []
            text_parts = user_message.split("<image>")
            for i, part in enumerate(text_parts):
                if part.strip():
                    content.append({"type": "text", "text": part.strip()})
                if i < len(text_parts) - 1:  # Add image after each <image> token except the last
                    content.append({"type": "image", "image": image[0]})

            messages.append({"role": "user", "content": content})
            
            # Prepare inputs with generation prompt
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,  # Add generation prompt to get model to generate
                return_dict=True,
                return_tensors="pt"
            )
            # Move to device
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            with torch.no_grad():
                generation_config = {
                    "max_new_tokens": 1,  # We only need one value token
                    "do_sample": False,   # Greedy decoding for deterministic prediction
                    "pad_token_id": processor.tokenizer.pad_token_id,
                    "eos_token_id": processor.tokenizer.eos_token_id,
                }
                generated_outputs = model.generate(
                    **inputs,
                    **generation_config,
                    return_dict_in_generate=True,
                    output_logits=True
                )

                generated_token_ids = generated_outputs.sequences[0][len(inputs['input_ids'][0]):]
                generated_token_id = generated_token_ids[0].item() if len(generated_token_ids) > 0 else None

                # Use interpolation for continuous value prediction
                if generated_token_id and generated_token_id in value_tokenizer.extra_id_token_ids:
                    predicted_value = decode_value_token_with_interpolation(
                        value_tokenizer, generated_outputs, len(inputs['input_ids'][0])
                    )
                else:
                    raise ValueError(f"Model generated non-value token {generated_token_id} ({processor.tokenizer.decode([generated_token_id] if generated_token_id else [], skip_special_tokens=False)}), falling back to logits")

                # Optional: debug output for first and last steps
                if step_idx == 0 or step_idx == len(episode_data) - 1:
                    rank0_print(f"[DEBUG] Step {step_idx}")
                    if generated_token_id and generated_token_id in value_tokenizer.extra_id_token_ids:
                        rank0_print(f"Generated value token: {processor.tokenizer.decode([generated_token_id], skip_special_tokens=False)}")
                        bin_idx = np.where(value_tokenizer.extra_id_token_ids == generated_token_id)[0][0]
                        rank0_print(f"Bin index: {bin_idx}, Value: {predicted_value:.4f}")
                    else:
                        rank0_print(f"Used fallback prediction: {predicted_value:.4f}")
                    rank0_print("=" * 80)
            
            predicted_values.append(predicted_value)
            
            if true_value is not None:
                true_values.append(true_value)
            steps.append(step_idx)
            
        except Exception as e:
            rank0_print(f"Error processing step {step_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return {
        'steps': steps,
        'predicted_values': predicted_values,
        'true_values': true_values if true_values else None,
    }


def plot_episode_values(episode_idx, episode_results, output_dir):
    """Plot value changes for a single episode."""
    steps = episode_results['steps']
    predicted_values = episode_results['predicted_values']

    plt.figure(figsize=(12, 6))
    plt.plot(steps, predicted_values, 'b-', label='Predicted Value', linewidth=2, marker='o', markersize=4)

    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.title(f'Episode {episode_idx} - Value Prediction', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save figure
    output_path = Path(output_dir) / f"episode_{episode_idx}_values.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    rank0_print(f"Saved plot for episode {episode_idx} to {output_path}")


def create_episode_animation(episode_idx, episode_data, episode_results, output_dir, filename=None):
    """Create animation for a single episode showing observation and predicted value changes."""
    if filename is None:
        filename = f"episode_{episode_idx}_animation.gif"

    output_path = Path(output_dir) / filename
    rank0_print(f"Creating animation for episode {episode_idx}...")

    steps = episode_results['steps']
    predicted_values = episode_results['predicted_values']
    max_steps = len(steps)

    # Create figure with dual subplot layout
    fig = plt.figure(figsize=(16, 8))

    # Left subplot: observation image
    ax1 = fig.add_subplot(121)
    ax1.set_title(f'Episode {episode_idx} - Robot Observation', fontsize=14, fontweight='bold')
    ax1.axis('off')  # Hide axes for image display

    # Right subplot: predicted value curve
    ax2 = fig.add_subplot(122)
    ax2.set_title('Predicted Value Changes', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Step', fontsize=12)
    ax2.set_ylabel('Predicted Value', fontsize=12)
    ax2.grid(True, alpha=0.3)

    # Initialize image display
    img_display = None
    current_image = None

    # Initialize prediction curve
    line, = ax2.plot([], [], 'b-', linewidth=3, marker='o', markersize=6, label='Predicted Value')
    ax2.legend(fontsize=11)

    def update(frame):
        nonlocal img_display, current_image

        step_idx = frame
        if step_idx >= max_steps:
            return []

        # Update left subplot: observation image
        if step_idx < len(episode_data):
            step_data = episode_data[step_idx]
            if 'image' in step_data and len(step_data['image']) > 0:
                # Get the first image (usually the primary camera view)
                image = step_data['image'][0]

                # Convert PIL Image to numpy array if needed
                if hasattr(image, 'convert'):
                    # PIL Image
                    current_image = np.array(image.convert('RGB'))
                else:
                    # Assume it's already a numpy array
                    current_image = np.array(image)

                # Display image
                if img_display is None:
                    img_display = ax1.imshow(current_image)
                else:
                    img_display.set_data(current_image)

                ax1.set_title(f'Episode {episode_idx} - Robot Observation (Step {step_idx})',
                            fontsize=14, fontweight='bold')

        # Update right subplot: prediction curve
        # Show all predictions up to current step
        current_steps = steps[:step_idx+1]
        current_predictions = predicted_values[:step_idx+1]

        line.set_data(current_steps, current_predictions)

        # Update plot limits dynamically
        if current_predictions:
            y_min = min(current_predictions) * 0.95
            y_max = max(current_predictions) * 1.05
            ax2.set_xlim(0, max(current_steps) + 1)
            ax2.set_ylim(y_min, y_max)

            # Add current value annotation
            current_value = current_predictions[-1]
            ax2.text(0.02, 0.98, f'Current: {current_value:.4f}',
                    transform=ax2.transAxes, fontsize=12,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        return [line, img_display] if img_display else [line]

    # Create animation
    try:
        ani = animation.FuncAnimation(fig, update, frames=max_steps, interval=500, blit=False)

        # Save animation
        rank0_print(f"Saving animation to {output_path}...")
        ani.save(str(output_path), writer='pillow', fps=2, dpi=100)
        rank0_print(f"Animation saved as {output_path}")

        # Save the final frame as a static image
        rank0_print("Saving final frame as static image...")
        fig_final = plt.figure(figsize=(16, 8))

        # Left subplot: final observation
        ax1_final = fig_final.add_subplot(121)
        ax1_final.set_title(f'Episode {episode_idx} - Robot Observation (Final)', fontsize=14, fontweight='bold')
        ax1_final.axis('off')

        if max_steps > 0 and max_steps <= len(episode_data):
            final_step_data = episode_data[max_steps - 1]
            if 'image' in final_step_data and len(final_step_data['image']) > 0:
                final_image = final_step_data['image'][0]
                if hasattr(final_image, 'convert'):
                    final_image = np.array(final_image.convert('RGB'))
                else:
                    final_image = np.array(final_image)
                ax1_final.imshow(final_image)

        # Right subplot: complete prediction curve
        ax2_final = fig_final.add_subplot(122)
        ax2_final.set_title('Predicted Value Changes (Complete)', fontsize=14, fontweight='bold')
        ax2_final.set_xlabel('Step', fontsize=12)
        ax2_final.set_ylabel('Predicted Value', fontsize=12)
        ax2_final.grid(True, alpha=0.3)

        # Plot complete curve
        ax2_final.plot(steps, predicted_values, 'b-', linewidth=3, marker='o', markersize=6, label='Predicted Value')
        ax2_final.legend(fontsize=11)

        if predicted_values:
            ax2_final.set_xlim(0, max(steps) + 1)
            y_min = min(predicted_values) * 0.95
            y_max = max(predicted_values) * 1.05
            ax2_final.set_ylim(y_min, y_max)

            # Add final value annotation
            final_value = predicted_values[-1]
            ax2_final.text(0.02, 0.98, f'Final: {final_value:.4f}',
                          transform=ax2_final.transAxes, fontsize=12,
                          verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # Save final frame as PNG
        final_image_path = output_path.with_suffix('.png')
        plt.tight_layout()
        fig_final.savefig(final_image_path, dpi=150, bbox_inches='tight')
        rank0_print(f"Final frame saved as {final_image_path}")

        plt.close(fig)
        plt.close(fig_final)
        return True

    except Exception as e:
        rank0_print(f"Animation creation failed for episode {episode_idx}: {e}")
        plt.close(fig)
        return False


def evaluate(model_args=None, data_args=None, eval_args=None, model=None, processor=None, value_tokenizer=None, output_dir=None):
    """Main evaluation function.

    Args:
        model_args, data_args, eval_args: Configuration arguments (for standalone evaluation)
        model: Pre-initialized model (for calling during training)
        processor: Pre-initialized processor (for calling during training)
        value_tokenizer: Pre-initialized value tokenizer (for calling during training)
        output_dir: Output directory (for calling during training)
    """

    # Check if called during training (with pre-initialized components)
    called_from_training = model is not None and processor is not None and value_tokenizer is not None

    if called_from_training:
        assert model is not None and processor is not None  # For type checker
        # Use provided components from training
        rank0_print("Evaluating with pre-initialized components from training...")
        device = next(model.parameters()).device  # type: ignore
        eval_output_dir = Path(output_dir) if output_dir else Path("./eval_output")
        # data_args should be provided when called from training
        if data_args is None:
            raise ValueError("data_args must be provided when calling evaluate from training")
    else:
        # Standalone evaluation - parse arguments and initialize components
        if model_args is None or data_args is None or eval_args is None:
            parser = transformers.HfArgumentParser(  # type: ignore
                (ModelArguments, DataArguments, EvalArguments)  # type: ignore
            )
            model_args, data_args, eval_args = parser.parse_args_into_dataclasses()

        # Create output directory
        eval_output_dir = Path(eval_args.output_dir)
        eval_output_dir.mkdir(parents=True, exist_ok=True)

        # Load model and processor
        model_name_or_path = os.path.expanduser(model_args.model_name_or_path)
        model, processor = load_model_and_processor(model_name_or_path)
        assert model is not None  # For type checker
        device = next(model.parameters()).device  # type: ignore
        assert processor is not None  # For type checker

        # Create value tokenizer
        value_tokenizer = ValueTokenizer(
            llm_path=model_name_or_path,  # Use same model path to ensure consistency
            bins=data_args.value_tokenizer_bins,
            min_value=data_args.value_tokenizer_min,
            max_value=data_args.value_tokenizer_max,
        )
        rank0_print(f"ValueTokenizer created: bins={data_args.value_tokenizer_bins}, "
                   f"range=[{data_args.value_tokenizer_min}, {data_args.value_tokenizer_max}]")

        # Check if value token embeddings were trained (diagnostic)
        emb = model.get_input_embeddings().weight.detach().cpu().float().numpy()  # type: ignore
        # Use value_tokenizer's extra_id_token_ids (which are the correct token IDs for <extra_id_{i}>)
        value_token_ids_check = value_tokenizer.extra_id_token_ids

        # Check first few value token embeddings
        if len(value_token_ids_check) >= 5:
            value_embs = emb[value_token_ids_check[:5]]  # First 5 value tokens
            emb_norms = np.linalg.norm(value_embs, axis=1)
            rank0_print(f"Value token embedding norms (first 5): min={emb_norms.min():.4f}, "
                       f"mean={emb_norms.mean():.4f}, max={emb_norms.max():.4f}")
            # Compare to some regular token embeddings
            regular_embs = emb[:1000]  # First 1000 tokens
            regular_norms = np.linalg.norm(regular_embs, axis=1)
            rank0_print(f"Regular token embedding norms (first 1000): min={regular_norms.min():.4f}, "
                       f"mean={regular_norms.mean():.4f}, max={regular_norms.max():.4f}")
        else:
            raise ValueError("Not enough value tokens to check embeddings!")
    
    # Load dataset
    dataset_dir = data_args.dataset_use
    # Convert relative path to absolute if needed
    if not os.path.isabs(dataset_dir):
        # Try to resolve relative to current working directory first
        abs_path = os.path.abspath(dataset_dir)
        if os.path.exists(abs_path):
            dataset_dir = abs_path
        else:
            # Fallback: assume relative to project root (qwen-vl-finetune)
            project_root = Path(__file__).parent.parent.parent.parent
            dataset_dir = str(project_root / dataset_dir)

    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    rank0_print(f"Loading dataset from: {dataset_dir}")

    # Load LeRobot dataset for evaluation
    rank0_print(f"Loading LeRobot dataset for evaluation")

    dataset = create_evaluation_dataset(
        dataset_dir,
        processor,
        local_rank=0,
        world_size=1,
        num_workers=0,  # No multiprocessing for evaluation
        value_tokenizer=value_tokenizer,  # Pass the created value_tokenizer
    )
    
    # Group data by episode
    # Different handling for different dataset types
    rank0_print("Loading and grouping data by episode...")

    episode_data = defaultdict(list)

    # LeRobot datasets have indices, use them for structured loading
    if hasattr(dataset, 'indices') and dataset.indices:
        # Sort indices by episode_idx, then by step to maintain order
        sorted_indices = sorted(dataset.indices, key=lambda x: (x[0], x[1]))

        for episode_idx, step in tqdm(sorted_indices, desc="Loading dataset"):
            # Load the data for this episode and step
            try:
                lerobot_sample = dataset._load_frame(episode_idx, step)
                data = dataset.batch_transform(lerobot_sample)

                qwen_data = {
                    "conversations": data['conversations'],
                    "data_path": "",
                    "image": data['image'],
                    "value": data.get('value', None),  # Include true value if available
                }

                episode_data[episode_idx].append(qwen_data)
            except Exception as e:
                rank0_print(f"Error loading episode {episode_idx}, step {step}: {e}")
                import traceback
                traceback.print_exc()
                continue
    else:
        raise ValueError("LeRobot dataset must provide indices for evaluation.")
    
    # Sort episodes by episode index
    episode_data = dict(sorted(episode_data.items()))
    
    # Steps should already be in order since we sorted indices by (episode_idx, step)
    # But let's verify and ensure they're in the correct order
    rank0_print(f"Loaded {len(episode_data)} episodes")
    for episode_idx, items in episode_data.items():
        rank0_print(f"  Episode {episode_idx}: {len(items)} steps")
    
    # Limit number of episodes if specified
    max_episodes = eval_args.max_episodes if eval_args else None
    if max_episodes:
        episode_data = {k: v for k, v in list(episode_data.items())[:max_episodes]}
    
    rank0_print(f"Found {len(episode_data)} episodes to evaluate")
    
    # Evaluate each episode
    all_results = {}
    for episode_idx, episode_items in tqdm(episode_data.items(), desc="Evaluating episodes"):
        try:
            episode_results = evaluate_episode(
                model, processor, value_tokenizer, episode_items, device
            )
            all_results[episode_idx] = episode_results
            
            # Plot episode values
            plot_episode_values(episode_idx, episode_results, eval_output_dir)

            # Create animation for episode
            create_episode_animation(episode_idx, episode_items, episode_results, eval_output_dir)
            
        except Exception as e:
            rank0_print(f"Error evaluating episode {episode_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save summary results
    summary = {
        'num_episodes': len(all_results),
        'episodes': {}
    }
    
    for episode_idx, results in all_results.items():
        summary['episodes'][episode_idx] = {
            'num_steps': len(results['steps']),
            'predicted_values': results['predicted_values'],
            'true_values': results['true_values'] if results['true_values'] else None,
        }
    
    summary_path = eval_output_dir / "evaluation_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    rank0_print(f"\nEvaluation complete!")
    rank0_print(f"Results saved to: {eval_output_dir}")
    rank0_print(f"Summary saved to: {summary_path}")
    rank0_print(f"Total episodes evaluated: {len(all_results)}")


if __name__ == "__main__":
    evaluate()

