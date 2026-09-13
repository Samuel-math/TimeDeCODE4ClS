"""From-scratch VQ-VAE -> masked-token prediction -> UEA classification."""
import argparse
import copy
import json
import random
import subprocess
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from data import DATASETS, prepare, prepare_test, digest
from models import PatchVQVAE, MaskedClassifier


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def loader(x, y, batch, shuffle=False):
    return DataLoader(TensorDataset(torch.as_tensor(x), torch.as_tensor(y)),
                      batch_size=batch, shuffle=shuffle, num_workers=0)


@torch.no_grad()
def encode_data(vq, x, batch, device):
    vq.eval()
    return torch.cat([vq.tokens(torch.as_tensor(x[i:i+batch], device=device)).cpu()
                      for i in range(0, len(x), batch)])


@torch.no_grad()
def evaluate(model, batches, device):
    model.eval()
    correct, count, total = 0, 0, 0.
    for x, y in batches:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total += F.cross_entropy(logits, y, reduction='sum').item()
        correct += (logits.argmax(-1) == y).sum().item()
        count += len(y)
    return dict(accuracy=correct/count, loss=total/count)


def run(args, name):
    seed_all(args.seed)
    out = Path(args.output) / name
    out.mkdir(parents=True, exist_ok=False)
    (out / 'args.json').write_text(json.dumps(vars(args), indent=2))
    source_root = Path(__file__).resolve().parent
    (out / 'source_hashes.json').write_text(json.dumps(
        {name: digest(source_root / name) for name in ('models.py', 'data.py', 'train.py', 'requirements.txt')}, indent=2))
    (out / 'environment.txt').write_text(subprocess.check_output(
        [__import__('sys').executable, '-m', 'pip', 'freeze'], text=True))
    def log(**row):
        print(json.dumps(dict(dataset=name, **row)), flush=True)
        with (out / 'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(row) + '\n')
    (xt, yt), (xv, yv), meta = prepare(args.data_root, name, args.patch_size,
                                        args.max_length, args.seed, args.val_ratio, args.protocol)
    (out / 'data_protocol.json').write_text(json.dumps(meta, indent=2))
    train = loader(xt, yt, args.batch_size, True)
    val = loader(xv, yv, args.batch_size)
    vq = PatchVQVAE(args.patch_size, args.dim, args.codes).to(args.device)
    # Data initialization is TRAIN-only and initializes a standard VQ codebook.
    with torch.no_grad():
        patches = vq.patches(torch.as_tensor(xt[:min(32, len(xt))], device=args.device)).reshape(-1, args.patch_size)
        sample = torch.randint(len(patches), (args.codes,), device=args.device)
        vq.codebook.weight.copy_(vq.encoder(patches[sample]))
    opt = torch.optim.AdamW(vq.parameters(), lr=args.cb_lr)
    best_loss, best_state = float('inf'), None
    for epoch in range(args.cb_epochs):
        vq.train()
        totals = []
        for x, _ in train:
            patches = vq.patches(x.to(args.device)).reshape(-1, args.patch_size)
            if len(patches) > args.cb_patch_batch:
                patches = patches[torch.randperm(len(patches), device=args.device)[:args.cb_patch_batch]]
            rec, penalty, _ = vq(patches)
            loss = F.mse_loss(rec, patches) + penalty
            opt.zero_grad(); loss.backward(); opt.step()
            totals.append(loss.item())
        vq.eval()
        numerator, denominator = 0., 0
        with torch.no_grad():
            for x, _ in val:
                patches = vq.patches(x.to(args.device)).reshape(-1, args.patch_size)
                for part in patches.split(args.cb_patch_batch):
                    rec, penalty, _ = vq(part)
                    numerator += (F.mse_loss(rec, part).item() + penalty.item()) * len(part)
                    denominator += len(part)
        vl = numerator / denominator
        log(stage='vqvae', epoch=epoch+1, train_loss=float(np.mean(totals)), val_loss=vl)
        if vl < best_loss:
            best_loss, best_state = vl, copy.deepcopy(vq.state_dict())
    vq.load_state_dict(best_state)
    torch.save(dict(model=vq.state_dict(), args=vars(args), metadata=meta), out / 'vqvae.pt')
    vq.requires_grad_(False)
    train_tokens = encode_data(vq, xt, args.batch_size, args.device)
    val_tokens = encode_data(vq, xv, args.batch_size, args.device)
    train = loader(train_tokens, yt, args.batch_size, True)
    val = loader(val_tokens, yv, args.batch_size)
    model = MaskedClassifier(args.codes, args.dim, train_tokens.shape[-1], meta['channels'],
                              len(meta['label_mapping']), args.layers, args.dropout).to(args.device)
    with torch.no_grad():
        model.embedding.weight[:args.codes].copy_(vq.codebook.weight)
    opt = torch.optim.AdamW(model.parameters(), lr=args.pre_lr, weight_decay=1e-4)
    best_loss, best_state = float('inf'), None
    for epoch in range(args.pre_epochs):
        model.train(); totals = []
        for ids, _ in train:
            loss = model.masked_loss(ids.to(args.device), args.mask_ratio)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step(); totals.append(loss.item())
        model.eval(); numerator, denominator = 0., 0
        generator = torch.Generator(device=args.device).manual_seed(args.seed + 1000)
        with torch.no_grad():
            for ids, _ in val:
                numerator += model.masked_loss(ids.to(args.device), args.mask_ratio, generator).item() * len(ids)
                denominator += len(ids)
        vl = numerator / denominator
        log(stage='mask', epoch=epoch+1, train_loss=float(np.mean(totals)), val_loss=vl)
        if vl < best_loss:
            best_loss, best_state = vl, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    torch.save(dict(model=model.state_dict(), args=vars(args), metadata=meta), out / 'masked_pretrain.pt')
    opt = torch.optim.AdamW(model.parameters(), lr=args.cls_lr, weight_decay=args.weight_decay)
    best_key, best_epoch, stale = (-1., -float('inf')), 0, 0
    for epoch in range(args.cls_epochs):
        model.train(); total, count = 0., 0
        for ids, labels in train:
            ids, labels = ids.to(args.device), labels.to(args.device)
            loss = F.cross_entropy(model(ids), labels)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step(); total += loss.item() * len(labels); count += len(labels)
        metrics = evaluate(model, val, args.device)
        key = (metrics['accuracy'], -metrics['loss'])
        log(stage='classification', epoch=epoch+1, train_loss=total/count, **{'val_'+k:v for k,v in metrics.items()})
        if key > best_key:
            best_key, best_epoch, stale = key, epoch+1, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if stale >= args.patience:
            break
    model.load_state_dict(best_state)
    x_test, y_test = prepare_test(meta)
    test_tokens = encode_data(vq, x_test, args.batch_size, args.device)
    metrics = evaluate(model, loader(test_tokens, y_test, args.batch_size), args.device)
    result = dict(dataset=name, seed=args.seed, protocol=args.protocol,
                  selection_split=meta['selection_split'], independent_test=meta['independent_test'],
                  best_epoch=best_epoch, val_accuracy=best_key[0],
                  test_accuracy=metrics['accuracy'], test_loss=metrics['loss'],
                  train_size=len(yt), val_size=len(yv), test_size=len(y_test),
                  test_sha256=digest(meta['test_file']))
    torch.save(dict(model=model.state_dict(), args=vars(args), metadata=meta, result=result), out / 'classifier.pt')
    (out / 'result.json').write_text(json.dumps(result, indent=2))
    log(stage='final_test', **{k:v for k,v in result.items() if k != 'dataset'})
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--datasets', nargs='+', choices=DATASETS, default=DATASETS)
    p.add_argument('--data-root', default='datasets/UEA')
    p.add_argument('--output', default='results/r1')
    p.add_argument('--protocol', choices=['train_val', 'test_selection'], default='train_val',
                   help='test_selection uses official TEST for stage selection and classification early stopping; not independent test evaluation')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    for key, value in dict(seed=42, patch_size=8, max_length=512, dim=64, codes=64, layers=2,
                           batch_size=16, cb_patch_batch=4096, cb_epochs=20, pre_epochs=30,
                           cls_epochs=100, patience=15).items():
        p.add_argument('--'+key.replace('_','-'), type=int, default=value)
    for key, value in dict(val_ratio=.2, mask_ratio=.35, dropout=.1, cb_lr=.001,
                           pre_lr=.0005, cls_lr=.0003, weight_decay=.001).items():
        p.add_argument('--'+key.replace('_','-'), type=float, default=value)
    args = p.parse_args()
    if min(args.cb_epochs, args.pre_epochs, args.cls_epochs, args.patch_size) < 1:
        p.error('Each training stage and patch size must be positive')
    if args.dim % 4 or args.max_length < args.patch_size:
        p.error('dim must be divisible by 4 and max_length >= patch_size')
    results = []
    for name in args.datasets:
        results.append(run(args, name))
        summary = dict(completed=len(results), requested=len(args.datasets), results=results,
                       mean_test_accuracy=float(np.mean([r['test_accuracy'] for r in results])))
        (Path(args.output) / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
