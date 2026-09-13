"""UEA .ts reader, stratified TRAIN split, and TRAIN-only normalization."""
from pathlib import Path
import hashlib
import math
import numpy as np
from sklearn.model_selection import train_test_split

DATASETS = ['EthanolConcentration', 'FaceDetection', 'Handwriting', 'Heartbeat',
            'JapaneseVowels', 'PEMS-SF', 'SelfRegulationSCP1', 'SelfRegulationSCP2',
            'SpokenArabicDigits', 'UWaveGestureLibrary']


def read_ts(path):
    samples, labels, active = [], [], False
    with Path(path).open() as stream:
        for raw in stream:
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            if not active:
                if line.lower().startswith('@timestamps true'):
                    raise ValueError('Timestamped .ts files are not supported')
                active = line.lower() == '@data'
                continue
            parts = line.split(':')
            labels.append(parts[-1].strip())
            channels = [np.asarray([float(v) if v != '?' else np.nan for v in part.split(',')],
                                   dtype=np.float32) for part in parts[:-1]]
            samples.append(channels)
    if not samples:
        raise ValueError(f'No observations: {path}')
    return samples, labels


def resize(samples, length):
    out = np.empty((len(samples), length, len(samples[0])), dtype=np.float32)
    for i, channels in enumerate(samples):
        for c, values in enumerate(channels):
            valid = np.flatnonzero(np.isfinite(values))
            values = np.interp(np.arange(len(values)), valid, values[valid]) if len(valid) else np.zeros(len(values))
            out[i, :, c] = np.interp(np.linspace(0, 1, length), np.linspace(0, 1, len(values)), values)
    return out


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def prepare(root, name, patch_size, max_length, seed, val_ratio):
    folder = Path(root) / name
    train_file, test_file = [folder / f'{name}_{s}.ts' for s in ('TRAIN', 'TEST')]
    samples, labels = read_ts(train_file)
    mapping = {label: i for i, label in enumerate(sorted(set(labels)))}
    y = np.asarray([mapping[label] for label in labels], dtype=np.int64)
    train_ids, val_ids = train_test_split(np.arange(len(y)), test_size=val_ratio,
                                        random_state=seed, stratify=y)
    native = int(np.median([len(c) for i in train_ids for c in samples[i]]))
    length = max(patch_size, min(max_length // patch_size, math.ceil(native / patch_size)) * patch_size)
    x = resize(samples, length)
    mean = x[train_ids].mean((0, 1), keepdims=True)
    std = x[train_ids].std((0, 1), keepdims=True) + 1e-5
    x = (x - mean) / std
    # TEST never defines labels, length, normalization, or validation membership.
    metadata = dict(dataset=name, train_indices=train_ids.tolist(), val_indices=val_ids.tolist(),
                    label_mapping=mapping, length=length, channels=x.shape[-1],
                    normalization_mean=mean.tolist(), normalization_std=std.tolist(),
                    train_sha256=digest(train_file), test_file=str(test_file),
                    protocol='official TRAIN stratified 80/20 by default; TEST final evaluation only')
    return (x[train_ids], y[train_ids]), (x[val_ids], y[val_ids]), metadata


def prepare_test(metadata):
    samples, labels = read_ts(metadata['test_file'])
    mapping = metadata['label_mapping']
    y = np.asarray([mapping[label] for label in labels], dtype=np.int64)
    x = resize(samples, metadata['length'])
    x = (x - np.asarray(metadata['normalization_mean'], dtype=np.float32)) / np.asarray(metadata['normalization_std'], dtype=np.float32)
    return x, y
