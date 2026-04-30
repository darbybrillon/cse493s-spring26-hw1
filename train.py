import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from typing import Callable, List
from model import GPT, GPTConfig
from tokenizer import tokenizer

import json
import numpy as np
from pathlib import Path
import random
from datetime import datetime


def _save_checkpoint(path, config, model, tok):
    torch.save({
        "model_config": config.__dict__.copy(),
        "model": model.state_dict(),
        "tokenizer": {
            "vocab": sorted(tok.vocab),
            "vocab_index_mapping": tok.vocab_index_mapping,
            "index_vocab_mapping": tok.index_vocab_mapping,
        },
    }, path)


OPERATORS = ["+", "-", "/"]
OPERATOR_TO_ID = {op: i for i, op in enumerate(OPERATORS)}


def _save_metrics(path, metrics):
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)


def _new_run_dir(output_dir=None):
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_dir = Path("runs") / f"run_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=False)
    else:
        run_dir = Path(output_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def all_tokens_mask(ids, text, tok):
    return [1] * (len(ids) - 1)


def suffix_after_n_input_tokens(n):
    def mask_fn(ids, text, tok):
        mask = [0] * (len(ids) - 1)
        start = min(n, len(mask))
        mask[start:] = [1] * (len(mask) - start)
        return mask
    return mask_fn


def token_after_marker(marker):
    def mask_fn(ids, text, tok):
        marker_id = tok.vocab_index_mapping[marker]
        marker_idx = ids.index(marker_id)
        mask = [0] * (len(ids) - 1)
        if marker_idx < len(mask):
            mask[marker_idx] = 1
        return mask
    mask_fn.marker = marker
    return mask_fn


modulo_answer_mask = token_after_marker("=")
modulo_answer_mask.is_modulo_answer_mask = True


class Custom_Dataset(Dataset):
    def __init__(self, strings, tok, mask_fn, collect_operator_metrics=False):
        self.strings = strings
        self.tok = tok
        self.mask_fn = mask_fn
        self.collect_operator_metrics = collect_operator_metrics

    def __len__(self):
        return len(self.strings)

    def __getitem__(self, idx):
        text = self.strings[idx]
        ids = self.tok.encoding(text)
        mask = self.mask_fn(ids, text, self.tok)
        op_id = -1
        if self.collect_operator_metrics:
            for op in OPERATORS:
                if f" {op} " in text:
                    op_id = OPERATOR_TO_ID[op]
                    break
        return torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.float), torch.tensor(op_id, dtype=torch.long)


def _empty_epoch_stats():
    return {"loss_sum": 0.0, "correct": 0, "total": 0.0}


def _empty_operator_epoch_stats():
    return {op: _empty_epoch_stats() for op in OPERATORS}


def _append_epoch_metrics(metrics, split, stats, operator_stats=None):
    metrics["combined"][f"{split}_losses"].append(stats["loss_sum"] / stats["total"])
    metrics["combined"][f"{split}_accs"].append(stats["correct"] / stats["total"])

    if operator_stats is None:
        return

    for op, op_stats in operator_stats.items():
        if op_stats["total"] == 0:
            metrics["by_operator"][op][f"{split}_losses"].append(None)
            metrics["by_operator"][op][f"{split}_accs"].append(None)
        else:
            metrics["by_operator"][op][f"{split}_losses"].append(op_stats["loss_sum"] / op_stats["total"])
            metrics["by_operator"][op][f"{split}_accs"].append(op_stats["correct"] / op_stats["total"])


def __infer_step(batch, model, stats, device, operator_stats=None):
    batch, loss_mask, op_ids = batch
    batch = batch.to(device)
    loss_mask = loss_mask.to(device)
    op_ids = op_ids.to(device)
    x = batch[:, :-1]
    y = batch[:, 1:]

    logits = model(x)
    token_losses = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        y.reshape(-1),
        reduction="none",
    ).view_as(y)
    num_masked = loss_mask.sum()
    if num_masked.item() == 0:
        raise ValueError("loss mask must select at least one token in each batch")
    loss = (token_losses * loss_mask).sum() / num_masked

    preds = logits.argmax(dim=-1)
    correct_by_token = ((preds == y) * loss_mask.bool()).float()

    loss_by_example = (token_losses * loss_mask).sum(dim=1)
    correct_by_example = correct_by_token.sum(dim=1)
    total_by_example = loss_mask.sum(dim=1)

    stats["loss_sum"] += loss_by_example.sum().item()
    stats["correct"] += correct_by_example.sum().item()
    stats["total"] += total_by_example.sum().item()

    if operator_stats is not None:
        for op, op_id in OPERATOR_TO_ID.items():
            op_mask = op_ids == op_id
            if op_mask.any():
                operator_stats[op]["loss_sum"] += loss_by_example[op_mask].sum().item()
                operator_stats[op]["correct"] += correct_by_example[op_mask].sum().item()
                operator_stats[op]["total"] += total_by_example[op_mask].sum().item()

    return loss


def train(
        tok:tokenizer,
        train_data:List[str], 
        val_data:List[str],  
        batch_size:int,
        block_size:int,
        n_layer:int,
        n_head:int,
        n_embd:int,
        dropout:float,
        bias:bool, 
        learning_rate:float, 
        epochs:int, 
        seed:int,
        mask_fn:Callable = modulo_answer_mask,
        output_dir:str = None,
        intermediate_ckpt:bool = True,
        verbose:bool = False,
        weight_decay:float = 1.0,
        beta1:float = 0.9,
        beta2:float = 0.98,
        warmup_steps:int = 10):
    
    device = 'cuda' if torch.cuda.is_available() else "cpu"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    
    run_dir = _new_run_dir(output_dir)
    collect_operator_metrics = getattr(mask_fn, "is_modulo_answer_mask", False)

    train_ds = Custom_Dataset(train_data, tok, mask_fn, collect_operator_metrics)
    train_loader = DataLoader(train_ds, batch_size=batch_size)
    val_ds = Custom_Dataset(val_data, tok, mask_fn, collect_operator_metrics)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    config = GPTConfig(
        vocab_size=len(tok.vocab_index_mapping),
        block_size=block_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        dropout=dropout,
        bias=bias,
    )

    model = GPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(beta1, beta2), weight_decay=weight_decay)

    ckpt_iter = epochs // 2

    metrics = {
        "combined": {
            "train_losses": [],
            "train_accs": [],
            "val_losses": [],
            "val_accs": [],
        },
    }
    if collect_operator_metrics:
        metrics["by_operator"] = {
            op: {
                "train_losses": [],
                "train_accs": [],
                "val_losses": [],
                "val_accs": [],
            }
            for op in OPERATORS
        }

    epoch_iter = range(epochs)
    progress = None
    if verbose:
        try:
            from tqdm import tqdm
            progress = tqdm(epoch_iter, desc="training", leave=True)
            epoch_iter = progress
        except ImportError:
            progress = None

    global_step = 0
    for i in epoch_iter:

        model.train()
        train_stats = _empty_epoch_stats()
        train_operator_stats = _empty_operator_epoch_stats() if collect_operator_metrics else None

        for batch in train_loader:
            loss = __infer_step(batch, model, train_stats, device, train_operator_stats)
            
            global_step += 1
            if warmup_steps > 0 and global_step <= warmup_steps:
                lr = learning_rate * global_step / warmup_steps
            else:
                lr = learning_rate

            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        _append_epoch_metrics(metrics, "train", train_stats, train_operator_stats)


        model.eval()
        val_stats = _empty_epoch_stats()
        val_operator_stats = _empty_operator_epoch_stats() if collect_operator_metrics else None
        with torch.no_grad():
            for batch in val_loader:
                __infer_step(batch, model, val_stats, device, val_operator_stats)
        _append_epoch_metrics(metrics, "val", val_stats, val_operator_stats)

        if progress is not None:
            progress.set_postfix(
                train_loss=metrics["combined"]["train_losses"][-1],
                train_acc=metrics["combined"]["train_accs"][-1],
                val_loss=metrics["combined"]["val_losses"][-1],
                val_acc=metrics["combined"]["val_accs"][-1],
            )
        elif verbose:
            print(
                f"epoch {i + 1}/{epochs} "
                f"train_loss={metrics['combined']['train_losses'][-1]:.6f} "
                f"train_acc={metrics['combined']['train_accs'][-1]:.4f} "
                f"val_loss={metrics['combined']['val_losses'][-1]:.6f} "
                f"val_acc={metrics['combined']['val_accs'][-1]:.4f}"
            )

        completed_epochs = i + 1
        if intermediate_ckpt and completed_epochs == ckpt_iter:
            _save_checkpoint(run_dir / f'ckpt_{completed_epochs}_epochs.pt', config, model, tok)

    

    _save_checkpoint(run_dir / f'ckpt_{epochs}_epochs.pt', config, model, tok)
    metrics["run_dir"] = str(run_dir)
    _save_metrics(run_dir / "metrics.json", metrics)

    return metrics
