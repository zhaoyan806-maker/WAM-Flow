import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from flow_matching.data.navsim import SupervisedDataset, VOCABULARY_SIZE_TXT, VOCABULARY_SIZE_IMG
from flow_matching.path import MixtureDiscreteSoftmaxProbPath
from flow_matching.rl import (
    GRPOLossConfig,
    clipped_grpo_loss,
    expand_data_info,
    grouped_advantages,
    sample_grouped_rollouts,
    sequence_logprobs,
)
from flow_matching.rl.navsim_runner import (
    collect_metric_tensors,
    extract_trajectory_numbers,
    proxy_navsim_metrics_from_numbers,
)
from flow_matching.rl.reward import RewardWeights, compute_navsim_reward
from flow_matching.solver import MixtureDiscreteSoftmaxEulerSolver
from fudoki.eval_loop import CFGScaledModel
from fudoki.janus.models import VLChatProcessor
from fudoki.model import instantiate_model


logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Simulated NAVSIM GRPO training for WAM-Flow.")
    parser.add_argument("--config", default="config/grpo_navsim.yaml", help="Path to GRPO config file.")
    parser.add_argument("--output_dir", default="output/train/grpo_navsim", help="Directory for checkpoints/logs.")
    parser.add_argument("--dry_run", action="store_true", help="Validate config and exit before loading models/data.")
    args, overrides = parser.parse_known_args()
    args.overrides = overrides
    return args


def add_numeric_tokens(vl_chat_processor):
    origin_len = len(vl_chat_processor.tokenizer)
    num_tokens = [f"{x:.2f}" for x in np.linspace(-100, 100, 20001)]
    vl_chat_processor.tokenizer.add_tokens(num_tokens)
    vl_chat_processor.num_start_id = origin_len
    vl_chat_processor.num_end_id = origin_len + len(num_tokens) - 1
    vl_chat_processor.min_num = -100
    vl_chat_processor.max_num = 100
    vl_chat_processor.interval = 0.01
    return len(num_tokens)


def move_batch_to_device(batch, device):
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if isinstance(value, torch.Tensor) else value
    return moved


def build_solver(model, cfg, vocab_size_txt):
    cfg_model = CFGScaledModel(model=model, g_or_u="understanding", mode="train-top", top=cfg.grpo.group_size)
    path_txt = MixtureDiscreteSoftmaxProbPath(mode="text", embedding_path=cfg.model.text_embedding_path)
    path_txt.set_embedding(model.language_model.get_input_embeddings())
    path_img = MixtureDiscreteSoftmaxProbPath(mode="image", embedding_path=cfg.model.image_embedding_path)
    return MixtureDiscreteSoftmaxEulerSolver(
        model=cfg_model,
        path_txt=path_txt,
        path_img=path_img,
        vocabulary_size_txt=vocab_size_txt,
        vocabulary_size_img=VOCABULARY_SIZE_IMG,
    )


def configure_trainable_parameters(model, train_llm_embedding=False):
    model.requires_grad_(False)
    model.language_model.requires_grad_(True)
    if not train_llm_embedding:
        model.language_model.model.embed_tokens.requires_grad_(False)
    return [param for param in model.parameters() if param.requires_grad]


def model_sequence_logprobs(model, tokens, data_info):
    _, txt_logits = model(tokens, data_info)
    return sequence_logprobs(txt_logits, tokens, data_info["text_token_mask"])


def decode_proxy_metric_rows(tokenizer, tokens):
    metric_rows = []
    for token_row in tokens:
        text = tokenizer.decode(token_row.detach().cpu().tolist(), skip_special_tokens=True)
        numbers, _ = extract_trajectory_numbers(text)
        metric_rows.append(proxy_navsim_metrics_from_numbers(numbers))
    return metric_rows


def validate_config(cfg):
    missing = []
    for dotted_key in [
        "model.policy_checkpoint",
        "model.reference_checkpoint",
        "model.processor_path",
        "model.text_embedding_path",
        "model.image_embedding_path",
        "data.data_list",
    ]:
        value = OmegaConf.select(cfg, dotted_key)
        if value in (None, ""):
            missing.append(dotted_key)
    if missing:
        raise ValueError(f"Missing required GRPO config fields: {', '.join(missing)}")

    if cfg.grpo.kl.beta is None:
        raise ValueError("grpo.kl.beta must be set before training; the paper does not provide a fixed value.")
    if cfg.grpo.clip_epsilon is None:
        raise ValueError("grpo.clip_epsilon must be set before training; the paper does not provide a fixed value.")


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    cfg = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_dotlist(args.overrides))

    if args.dry_run:
        logger.info("Loaded GRPO config:\n%s", OmegaConf.to_yaml(cfg))
        logger.info("Dry run complete. Set grpo.kl.beta and grpo.clip_epsilon before launching training.")
        return

    validate_config(cfg)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.get("seed", 42))
    np.random.seed(cfg.get("seed", 42))

    vl_chat_processor = VLChatProcessor.from_pretrained(cfg.model.processor_path)
    num_tokens_length = add_numeric_tokens(vl_chat_processor)
    vocab_size_txt = VOCABULARY_SIZE_TXT + num_tokens_length

    dataset = SupervisedDataset(
        data_list=list(cfg.data.data_list),
        vl_chat_processor=vl_chat_processor,
        txt_max_length=cfg.data.txt_max_length,
    )
    if cfg.grpo.train_size is not None:
        dataset = Subset(dataset, range(min(int(cfg.grpo.train_size), len(dataset))))

    dataloader = DataLoader(
        dataset,
        shuffle=True,
        batch_size=cfg.grpo.batch_size,
        num_workers=cfg.data.dataloader_num_workers,
        pin_memory=True,
    )

    policy = instantiate_model(cfg.model.policy_checkpoint).to(device)
    reference = instantiate_model(cfg.model.reference_checkpoint).to(device)
    reference.eval()
    reference.requires_grad_(False)

    trainable_params = configure_trainable_parameters(policy, cfg.model.get("train_llm_embedding", False))
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=cfg.grpo.lr,
        weight_decay=cfg.grpo.weight_decay,
    )

    def lr_lambda(step):
        warmup_steps = max(1, int(cfg.grpo.warmup_steps))
        return min(1.0, float(step + 1) / warmup_steps)

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    solver = build_solver(policy, cfg, vocab_size_txt)

    loss_config = GRPOLossConfig(beta=float(cfg.grpo.kl.beta), clip_epsilon=float(cfg.grpo.clip_epsilon))
    reward_weights = RewardWeights(
        safety_metrics=tuple(cfg.grpo.reward.safety),
        performance_weights={key: float(value) for key, value in cfg.grpo.reward.performance.items()},
    )

    max_steps = int(np.ceil(len(dataloader) * float(cfg.grpo.epochs)))
    progress_bar = tqdm(total=max_steps, desc="GRPO")
    global_step = 0

    policy.train()
    for epoch in range(max(1, int(np.ceil(float(cfg.grpo.epochs))))):
        for batch in dataloader:
            if global_step >= max_steps:
                break
            batch = move_batch_to_device(batch, device)
            input_ids = batch["input_ids"]
            data_info = {key: value for key, value in batch.items() if key != "input_ids"}

            rollout = sample_grouped_rollouts(
                solver=solver,
                input_ids=input_ids,
                data_info=data_info,
                vocab_size=vocab_size_txt,
                group_size=cfg.grpo.group_size,
                denoise_steps=list(cfg.grpo.sampling.denoise_steps),
            )
            grouped_data_info = expand_data_info(data_info, input_ids.shape[0], cfg.grpo.group_size)

            metric_rows = decode_proxy_metric_rows(vl_chat_processor.tokenizer, rollout.tokens)
            metric_tensors = collect_metric_tensors(metric_rows, device=device)
            rewards = compute_navsim_reward(metric_tensors, reward_weights)
            advantages = grouped_advantages(
                rewards,
                group_size=cfg.grpo.group_size,
                normalize=loss_config.normalize_advantages,
                eps=loss_config.eps,
            )

            current_logprobs = model_sequence_logprobs(policy, rollout.tokens, grouped_data_info)
            with torch.no_grad():
                old_logprobs = current_logprobs.detach()
                reference_logprobs = model_sequence_logprobs(reference, rollout.tokens, grouped_data_info)

            loss, loss_logs = clipped_grpo_loss(
                current_logprobs=current_logprobs,
                old_logprobs=old_logprobs,
                reference_logprobs=reference_logprobs,
                advantages=advantages,
                config=loss_config,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, cfg.grpo.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            if global_step % cfg.logging.log_every == 0:
                logger.info(
                    "step=%s reward=%.4f adv=%.4f loss=%s",
                    global_step,
                    float(rewards.mean().detach().cpu()),
                    float(advantages.mean().detach().cpu()),
                    loss_logs,
                )

            if cfg.checkpointing.save_steps and global_step > 0 and global_step % cfg.checkpointing.save_steps == 0:
                ckpt_dir = Path(args.output_dir) / f"checkpoint-{global_step}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                policy.save_pretrained(ckpt_dir)
                vl_chat_processor.save_pretrained(ckpt_dir)

            global_step += 1
            progress_bar.update(1)

        if global_step >= max_steps:
            break

    final_dir = Path(args.output_dir) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(final_dir)
    vl_chat_processor.save_pretrained(final_dir)
    logger.info("Saved final GRPO policy to %s", final_dir)


if __name__ == "__main__":
    main()
