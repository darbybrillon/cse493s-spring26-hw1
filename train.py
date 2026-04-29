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


def _save_metrics(path, val_losses, training_losses, training_accs, val_accs):
    with open(path, "w") as f:
        json.dump({
            "val_losses": val_losses,
            "training_losses": training_losses,
            "training_accs": training_accs,
            "val_accs": val_accs,
        }, f, indent=2)


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
    return mask_fn


modulo_answer_mask = token_after_marker("=")


class Custom_Dataset(Dataset):
    def __init__(self, strings, tok, mask_fn):
        self.strings = strings
        self.tok = tok
        self.mask_fn = mask_fn

    def __len__(self):
        return len(self.strings)

    def __getitem__(self, idx):
        text = self.strings[idx]
        ids = self.tok.encoding(text)
        mask = self.mask_fn(ids, text, self.tok)
        return torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.float)


def __infer_step(batch, model, cum_loss, correct, total, device):
    batch, loss_mask = batch
    batch = batch.to(device)
    loss_mask = loss_mask.to(device)
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
    cum_loss += (token_losses * loss_mask).sum().item()

    preds = logits.argmax(dim=-1)
    correct += ((preds == y) * loss_mask.bool()).sum().item()
    total += num_masked.item()

    return (cum_loss, correct, total, loss)


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

    train_ds = Custom_Dataset(train_data, tok, mask_fn)
    train_loader = DataLoader(train_ds, batch_size=batch_size)
    val_ds = Custom_Dataset(val_data, tok, mask_fn)
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

    training_accs = []
    training_losses = []
    val_accs = []
    val_losses = []

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
        training_loss = 0
        correct = 0
        total = 0

        for batch in train_loader:
            training_loss, correct, total, loss = __infer_step(batch, model, training_loss, correct, total, device)
            
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
        
        training_losses.append(training_loss/total)
        training_accs.append(correct/total)


        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                val_loss, correct, total, _ = __infer_step(batch, model, val_loss, correct, total, device)
                
        val_losses.append(val_loss/total)
        val_accs.append(correct/total)

        if progress is not None:
            progress.set_postfix(
                train_loss=training_losses[-1],
                train_acc=training_accs[-1],
                val_loss=val_losses[-1],
                val_acc=val_accs[-1],
            )
        elif verbose:
            print(
                f"epoch {i + 1}/{epochs} "
                f"train_loss={training_losses[-1]:.6f} "
                f"train_acc={training_accs[-1]:.4f} "
                f"val_loss={val_losses[-1]:.6f} "
                f"val_acc={val_accs[-1]:.4f}"
            )

        completed_epochs = i + 1
        if intermediate_ckpt and completed_epochs == ckpt_iter:
            _save_checkpoint(run_dir / f'ckpt_{completed_epochs}_epochs.pt', config, model, tok)

    

    _save_checkpoint(run_dir / f'ckpt_{epochs}_epochs.pt', config, model, tok)
    _save_metrics(run_dir / "metrics.json", val_losses, training_losses, training_accs, val_accs)

    return (val_losses, training_losses, training_accs, val_accs, str(run_dir))
