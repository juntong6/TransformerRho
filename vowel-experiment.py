from __future__ import annotations
import argparse
import copy
import csv
import gc
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
import urllib.request
from dataclasses import asdict, dataclass, replace
from pathlib import Path
for variable, value in {'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '1', 'VECLIB_MAXIMUM_THREADS': '1'}.items():
    os.environ[variable] = value
import numpy as np
import torch
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn import functional as F
MODE = 'experiment'
DATASETS = ('vowel',)
METHODS = ('MLE', 'rho', 'Huber', 'MOM')
N_REPEATS = 50
CONTAMINATION_RATES = (0.0, 0.05, 0.1, 0.2, 0.3)
SEEDS = tuple(range(260926100, 260926150))
RATES = CONTAMINATION_RATES
TUNING = {'MLE': None, 'rho': None, 'Huber': 1.0, 'MOM': 10}
INPUT_SHA256 = {'vowel.scale.txt': '89ecbd4590cabc23b001cbd2d90128efe3c8ad02540ce1a08ef367b98d765585', 'vowel.scale.t': 'b96670a92b4d4d711db91ecc6c89bf7d75347a1ef1f196f5876575f1c5f9a240'}

@dataclass(frozen=True)
class Config:
    mode: str = 'experiment'
    datasets: tuple = DATASETS
    methods: tuple = METHODS
    repeats: int = N_REPEATS
    contamination_rates: tuple = CONTAMINATION_RATES
    huber_grid: tuple = (0.5, 1.0, 4.0)
    mom_grid: tuple = (5, 10, 25)
    seed: int = 1000
    test_fraction: float = 0.3
    validation_fraction: float = 0.1
    sample_size: int = 1000
    d_model: int = 48
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 96
    logit_bound: float = 8.0
    parameter_bound: float = 5.0
    batch_size: int = 128
    lr: float = 0.001
    baseline_epochs: int = 120
    baseline_patience: int = 40
    eval_every: int = 5
    global_grad_clip: float = 5.0
    per_sample_chunk: int = 16
    rho_rounds: int = 10
    rho_early_stop: bool = True
    rho_first_epochs: int = 100
    rho_inner_epochs: int = 40
    rho_restarts: int = 3
    rho_inner_patience: int = 15
    rho_noise: float = 0.02
    rho_threshold: float = 1.0
    save_models: bool = True

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

class AttentionBlock(nn.Module):

    def __init__(self, width, heads, ff_width):
        super().__init__()
        if width % heads:
            raise ValueError('d_model must be divisible by n_heads')
        self.heads = heads
        self.head_width = width // heads
        self.norm1 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.attention_output = nn.Linear(width, width)
        self.norm2 = nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, ff_width), nn.GELU(), nn.Linear(ff_width, width))

    def forward(self, tokens):
        batch, length, width = tokens.shape
        qkv = self.qkv(self.norm1(tokens)).reshape(batch, length, 3, self.heads, self.head_width)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        weights = (q @ k.transpose(-1, -2) / math.sqrt(self.head_width)).softmax(-1)
        attended = (weights @ v).transpose(1, 2).reshape(batch, length, width)
        tokens = tokens + self.attention_output(attended)
        return tokens + self.ffn(self.norm2(tokens))

class TabularTransformer(nn.Module):

    def __init__(self, features, classes, config):
        super().__init__()
        self.features = features
        self.classes = classes
        self.config = config
        self.value_weight = nn.Parameter(torch.empty(features, config.d_model))
        self.feature_bias = nn.Parameter(torch.empty(features, config.d_model))
        self.cls_token = nn.Parameter(torch.empty(1, 1, config.d_model))
        self.blocks = nn.ModuleList([AttentionBlock(config.d_model, config.n_heads, config.d_ff) for _ in range(config.n_layers)])
        self.head = nn.Sequential(nn.LayerNorm(2 * config.d_model), nn.Linear(2 * config.d_model, config.d_model), nn.GELU(), nn.Linear(config.d_model, classes))
        nn.init.normal_(self.value_weight, std=0.1)
        nn.init.normal_(self.feature_bias, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, x):
        tokens = x.unsqueeze(-1) * self.value_weight + self.feature_bias
        tokens = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), tokens), dim=1)
        for block in self.blocks:
            tokens = block(tokens)
        raw = self.head(torch.cat((tokens[:, 0], tokens[:, 1:].mean(1)), dim=1))
        bound = self.config.logit_bound
        return bound * torch.tanh(raw / bound)

@torch.no_grad()
def project_parameters(model):
    bound = model.config.parameter_bound
    for parameter in model.parameters():
        parameter.clamp_(-bound, bound)

def observed_log_prob(model, x, y):
    return F.log_softmax(model(x), dim=1).gather(1, y[:, None]).squeeze(1)

def rho_terms(current_log_prob, challenger_log_prob):
    return torch.tanh((challenger_log_prob - current_log_prob) / 4.0)

@torch.no_grad()
def all_log_prob(model, x, y, batch_size=256):
    model.eval()
    return torch.cat([observed_log_prob(model, x[i:i + batch_size], y[i:i + batch_size]) for i in range(0, len(y), batch_size)]).detach()

@torch.no_grad()
def comparison_value(model, x, y, reference):
    values = rho_terms(reference, all_log_prob(model, x, y))
    return float(values.double().sum().item())

@torch.no_grad()
def evaluate(model, x, y):
    model.eval()
    errors, loss = (0, 0.0)
    for start in range(0, len(y), 256):
        logits = model(x[start:start + 256])
        labels = y[start:start + 256]
        errors += int((logits.argmax(1) != labels).sum().item())
        loss += float(F.cross_entropy(logits, labels, reduction='sum').item())
    return {'error': errors / len(y), 'nll': loss / len(y)}

def snapshot(model):
    return {key: value.detach().clone() for key, value in model.state_dict().items()}

def checked_step(model, optimizer, config):
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.global_grad_clip, error_if_nonfinite=True)
    optimizer.step()
    project_parameters(model)
    return float(norm.item())

def clipped_score_weights(model, x, y, threshold, chunk_size):
    from torch.func import functional_call, grad, vmap
    params = {name: p.detach() for name, p in model.named_parameters()}
    buffers = dict(model.named_buffers())

    def single_loss(parameters, buffer_values, observation, label):
        logits = functional_call(model, (parameters, buffer_values), (observation.unsqueeze(0),))
        return F.cross_entropy(logits, label.unsqueeze(0))
    per_sample = vmap(grad(single_loss), in_dims=(None, None, 0, 0))
    weights = []
    for start in range(0, len(y), chunk_size):
        xb, yb = (x[start:start + chunk_size], y[start:start + chunk_size])
        gradients = per_sample(params, buffers, xb, yb)
        squared = sum((g.detach().reshape(len(yb), -1).square().sum(1) for g in gradients.values()))
        weights.append((threshold / squared.sqrt().clamp_min(1e-12)).clamp(max=1.0))
    return torch.cat(weights).detach()

def geometric_median(vectors, max_iter=30, tolerance=1e-05):
    center = vectors.mean(0)
    for _ in range(max_iter):
        distances = torch.linalg.vector_norm(vectors - center, dim=1).clamp_min(1e-07)
        weights = distances.reciprocal()
        new_center = (weights[:, None] * vectors).sum(0) / weights.sum()
        if float(torch.linalg.vector_norm(new_center - center).item()) <= tolerance:
            return new_center
        center = new_center
    return center

def mom_step(model, x, y, blocks, optimizer, config):
    parameters = list(model.parameters())
    block_gradients = []
    for indices in blocks:
        loss = F.cross_entropy(model(x[indices]), y[indices])
        gradients = torch.autograd.grad(loss, parameters)
        block_gradients.append(torch.cat([g.detach().reshape(-1) for g in gradients]))
    robust_gradient = geometric_median(torch.stack(block_gradients))
    optimizer.zero_grad(set_to_none=True)
    offset = 0
    for p in parameters:
        p.grad = robust_gradient[offset:offset + p.numel()].reshape_as(p).clone()
        offset += p.numel()
    checked_step(model, optimizer, config)

def load_dataset(root, name, config):
    if name != 'vowel':
        raise ValueError('This standalone experiment is for Vowel only.')
    train = root / name / f'{name}.scale.txt'
    test = root / name / f'{name}.scale.t'
    for path in (train, test):
        if not path.is_file():
            raise FileNotFoundError(f'Missing {path}. Run the data-preparation cell first.')
    x1, y1 = load_svmlight_file(str(train))
    x2, y2 = load_svmlight_file(str(test), n_features=x1.shape[1])
    x = np.vstack((x1.toarray(), x2.toarray())).astype(np.float32)
    raw_y = np.concatenate((y1, y2))
    labels, y = np.unique(raw_y, return_inverse=True)
    return (x, y.astype(np.int64), labels)

def contaminate(x, y, rate, classes, seed):
    if not np.isfinite(rate) or not 0 <= rate <= 1:
        raise ValueError('The contamination rate must lie in [0,1].')
    if classes < 2 or len(x) != len(y) or np.any(y < 0) or np.any(y >= classes):
        raise ValueError('Expected matching features and labels in [0, classes).')
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(y))
    offsets = rng.integers(1, classes, size=len(y))
    count = int(round(len(y) * rate))
    positions = order[:count]
    x_bad, y_bad = (x.copy(), y.copy())
    y_bad[positions] = (y[positions] + offsets[positions]) % classes
    assert np.array_equal(x_bad, x)
    assert np.count_nonzero(y_bad != y) == count
    return (x_bad, y_bad, count, positions)

def fit_scaler(x):
    center = np.median(x, axis=0)
    scale = (np.percentile(x, 75, axis=0) - np.percentile(x, 25, axis=0)) / 1.349
    return (center.astype(np.float32), np.maximum(scale, 1e-06).astype(np.float32))

def transform(x, center, scale):
    return np.clip((x - center) / scale, -6.0, 6.0).astype(np.float32)

def split_data(x, y, classes, rate, config, seed, dataset=None):
    if dataset is None:
        if len(config.datasets) != 1:
            raise ValueError('Specify dataset when configuring multiple datasets.')
        dataset = config.datasets[0]
    if dataset != 'vowel':
        raise ValueError('This standalone experiment is for Vowel only.')
    indices = np.arange(len(y))
    if len(y) > config.sample_size:
        indices = train_test_split(indices, test_size=config.sample_size, stratify=y, random_state=seed)[1]
    train_idx, test_idx = train_test_split(indices, test_size=config.test_fraction, stratify=y[indices], random_state=seed)
    fit_idx, val_idx = train_test_split(np.arange(len(train_idx)), test_size=config.validation_fraction, stratify=y[train_idx], random_state=seed + 1)
    x_bad, y_bad, count, contamination_positions = contaminate(x[train_idx], y[train_idx], rate, classes, seed=seed + 200000)
    mechanism = 'uniform_wrong_label_features_unchanged'
    center, scale = fit_scaler(x_bad[fit_idx])
    result = {'x_fit': transform(x_bad[fit_idx], center, scale), 'y_fit': y_bad[fit_idx], 'x_val': transform(x_bad[val_idx], center, scale), 'y_val': y_bad[val_idx], 'x_test': transform(x[test_idx], center, scale), 'y_test': y[test_idx], 'center': center, 'scale': scale, 'train_indices': train_idx, 'test_indices': test_idx, 'fit_positions': fit_idx, 'validation_positions': val_idx, 'contamination_count': count, 'actual_contamination_rate': count / len(train_idx), 'contamination_mechanism': mechanism, 'contamination_positions': contamination_positions, 'train_labels_clean': y[train_idx], 'train_labels_corrupted': y_bad, 'features_unchanged': bool(np.array_equal(x_bad, x[train_idx]))}
    return result

class RawLogitTransformer(TabularTransformer):

    def forward(self, x):
        tokens = x.unsqueeze(-1) * self.value_weight + self.feature_bias
        tokens = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), tokens), dim=1)
        for block in self.blocks:
            tokens = block(tokens)
        return self.head(torch.cat((tokens[:, 0], tokens[:, 1:].mean(1)), dim=1))

def fit_likelihood(initial, x, y, config, seed):
    seed_all(seed)
    model = copy.deepcopy(initial)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.baseline_epochs, eta_min=config.lr * 0.1)
    first = evaluate(model, x, y)
    best_nll, best_epoch, best_state = (first['nll'], 0, snapshot(model))
    history = [{'epoch': 0, 'train_nll': first['nll'], 'train_error': first['error']}]
    max_gradient_norm = 0.0
    for epoch in range(1, config.baseline_epochs + 1):
        model.train()
        order = torch.randperm(len(y), device=x.device)
        for indices in order.split(config.batch_size):
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x[indices]), y[indices], reduction='mean')
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite unconstrained training NLL; no fallback clipping.')
            loss.backward()
            squared = sum((p.grad.detach().square().sum() for p in model.parameters() if p.grad is not None))
            if not torch.isfinite(squared):
                raise FloatingPointError('Nonfinite gradient; no fallback clipping.')
            max_gradient_norm = max(max_gradient_norm, float(squared.sqrt().item()))
            optimizer.step()
        scheduler.step()
        if epoch % config.eval_every == 0 or epoch == config.baseline_epochs:
            metrics = evaluate(model, x, y)
            if not math.isfinite(metrics['nll']):
                raise FloatingPointError('Nonfinite full training NLL.')
            history.append({'epoch': epoch, 'train_nll': metrics['nll'], 'train_error': metrics['error']})
            if metrics['nll'] < best_nll:
                best_nll, best_epoch, best_state = (metrics['nll'], epoch, snapshot(model))
    model.load_state_dict(best_state)
    return (model, {'objective': 'mean training negative log conditional likelihood', 'selected_epoch': best_epoch, 'epochs_run': config.baseline_epochs, 'best_train_nll': best_nll, 'history': history, 'maximum_observed_gradient_norm': max_gradient_norm, 'checkpoint_rule': 'lowest evaluated full training NLL', 'gradient_clipping': False, 'parameter_projection': False, 'logit_squashing': False, 'weight_decay': 0.0, 'validation_used_for_selection': False, 'random_starts': 1, 'global_optimality_certified': False})

def search_challenger(current, x, y, config, seed, epochs, report=False, warm_candidate=None, optimizer_name='Adam'):
    reference = all_log_prob(current, x, y)
    best_model = copy.deepcopy(current)
    best_value = 0.0
    details = []
    for restart in range(config.rho_restarts):
        restart_seed = seed + 97 * restart
        seed_all(restart_seed)
        noise_multiplier = None
        if config.rho_restarts > 1 and restart == config.rho_restarts - 1:
            if warm_candidate is None:
                challenger = type(current)(current.features, current.classes, config).to(x.device)
                start_type = 'fresh_random'
            else:
                challenger = copy.deepcopy(warm_candidate)
                start_type = 'MLE_weights_in_rho_class'
        else:
            challenger = copy.deepcopy(current)
            noise_multiplier = config.rho_noise * (restart + 1)
            with torch.no_grad():
                for p in challenger.parameters():
                    scale = max(float(p.std(unbiased=False).item()), 0.02)
                    p.add_(torch.randn_like(p) * noise_multiplier * scale)
            start_type = 'perturbed_current'
        project_parameters(challenger)
        with torch.no_grad():
            initial_distance = float(sum(((p - reference_p).square().sum() for p, reference_p in zip(challenger.parameters(), current.parameters()))).sqrt().item())
        optimizer_class = {'Adam': torch.optim.Adam, 'RAdam': torch.optim.RAdam}[optimizer_name]
        optimizer = optimizer_class(challenger.parameters(), lr=config.lr, weight_decay=0)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=config.lr * 0.1)
        local_best = comparison_value(challenger, x, y, reference)
        initial_value = local_best
        local_state = snapshot(challenger)
        best_epoch = 0
        epochs_run = 0
        for epoch in range(1, epochs + 1):
            challenger.train()
            order = torch.randperm(len(y), device=x.device)
            for indices in order.split(config.batch_size):
                optimizer.zero_grad(set_to_none=True)
                log_q = observed_log_prob(challenger, x[indices], y[indices])
                loss = -rho_terms(reference[indices], log_q).mean()
                loss.backward()
                checked_step(challenger, optimizer, config)
            scheduler.step()
            epochs_run = epoch
            if epoch % config.eval_every == 0 or epoch == epochs:
                value = comparison_value(challenger, x, y, reference)
                if not math.isfinite(value):
                    raise FloatingPointError('Non-finite rho comparison')
                if value > local_best + 1e-07:
                    local_best, local_state, best_epoch = (value, snapshot(challenger), epoch)
                if epoch - best_epoch >= config.rho_inner_patience:
                    break
        if local_best > best_value:
            best_value = local_best
            best_model.load_state_dict(local_state)
        details.append({'restart': restart, 'start': start_type, 'seed': restart_seed, 'noise_multiplier': noise_multiplier, 'initial_parameter_distance': initial_distance, 'initial_T_sum': initial_value, 'best_T_sum': local_best, 'epochs': epochs_run})
        if report:
            print(f'      challenger {restart + 1}/{config.rho_restarts}: T={local_best:.3f}, epochs={epochs_run}', flush=True)
    return (best_model, best_value, details)

def fit_rho(initial, x, y, config, seed, verbose=True, warm_candidate=None, optimizer_name='Adam'):
    current = copy.deepcopy(initial)
    path = []
    stop_reason = 'iteration_limit'
    for step in range(config.rho_rounds):
        epochs = config.rho_first_epochs if step == 0 else config.rho_inner_epochs
        challenger, value, details = search_challenger(current, x, y, config, seed + step * 10000, epochs, report=verbose, warm_candidate=warm_candidate, optimizer_name=optimizer_name)
        accepted = value > config.rho_threshold
        path.append({'round': step + 1, 'best_found_T_sum': value, 'accepted': accepted, 'restarts': details})
        if verbose:
            action = 'update entire network' if accepted else 'local search stop' if config.rho_early_stop else 'keep current network; continue fixed-round search'
            print(f'    rho round {step + 1}: best found T={value:.3f}, {action}', flush=True)
        if not accepted:
            if config.rho_early_stop:
                stop_reason = 'no_challenger_found_above_threshold'
                break
        else:
            current = challenger
    return (current, {'stop_reason': stop_reason, 'rho_rounds': len(path), 'fixed_round_budget': not config.rho_early_stop, 'global_rho_certificate': False, 'path': path})

def fit_without_validation(initial, x, y, method, tuning, config, seed):
    seed_all(seed)
    model = copy.deepcopy(initial)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.baseline_epochs, eta_min=config.lr * 0.1)
    if method == 'MOM':
        blocks = torch.tensor_split(torch.randperm(len(y), device=x.device), min(int(tuning), len(y)))
    elif method != 'Huber':
        raise ValueError(method)
    for epoch in range(1, config.baseline_epochs + 1):
        model.train()
        if method == 'MOM':
            mom_step(model, x, y, blocks, optimizer, config)
        else:
            for indices in torch.randperm(len(y), device=x.device).split(config.batch_size):
                xb, yb = (x[indices], y[indices])
                weights = clipped_score_weights(model, xb, yb, float(tuning), config.per_sample_chunk)
                optimizer.zero_grad(set_to_none=True)
                losses = torch.nn.functional.cross_entropy(model(xb), yb, reduction='none')
                (weights * losses).mean().backward()
                checked_step(model, optimizer, config)
        scheduler.step()
    return (model, dict(best_epoch=epoch, epochs_run=epoch, selected_epoch=epoch, stop_reason='fixed_budget_final_iterate', validation_used=False, test_used_for_selection=False, early_stopping=False))

def prepare_seed(data_root, config, seed):
    x, y, labels = load_dataset(data_root, 'vowel', config)
    clean = split_data(x, y, len(labels), 0.0, config, seed)
    if (len(clean['x_fit']), len(clean['x_val']), len(clean['x_test']), clean['x_fit'].shape[1], len(labels)) != (623, 70, 297, 10, 11):
        raise ValueError('Unexpected Vowel split dimensions.')
    return (clean, labels)

def fit(method, tuning, x, y, config, seed):
    active = replace(config, logit_bound=None, parameter_bound=None, global_grad_clip=None) if method == 'MLE' else config
    seed_all(seed)
    initial = (RawLogitTransformer if method == 'MLE' else TabularTransformer)(x.shape[1], 11, active)
    started = time.perf_counter()
    if method == 'MLE':
        model, info = fit_likelihood(initial, x, y, active, seed + 1000)
        assert not info['gradient_clipping'] and (not info['parameter_projection'])
    elif method == 'rho':
        model, info = fit_rho(initial, x, y, active, seed + 1000, verbose=False, optimizer_name='Adam')
        assert info['rho_rounds'] <= 10
    else:
        model, info = fit_without_validation(initial, x, y, method, tuning, active, seed + 1000)
        assert info['epochs_run'] == 120 and (not info['validation_used'])
    elapsed = time.perf_counter() - started
    return (model, info, elapsed)

def array_digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()

def seed_map(seed):
    i = SEEDS.index(seed)
    initialization = 1000000000 + 200000 * i
    split = 1100000000 + 10 * i
    corruption = 1200000000 + 10 * i
    return dict(master=seed, split=split, holdout=split + 1, original_clean_split_unused_corruption=split + 200000, initialization=initialization, trainer=initialization + 1000, order=corruption, signs=corruption + 1, wrong_labels=corruption + 2, rho_restarts=[initialization + 1000 + step * 10000 + restart * 97 for step in range(10) for restart in range(3)])

def case(plan, clean, rate):
    selected = plan['order'][:round(623 * rate)]
    x, y = (clean['x_fit'].copy(), clean['y_fit'].copy())
    x[selected], y[selected] = (plan['replacement_x'][selected], plan['targets'][selected])
    assert np.count_nonzero(y != clean['y_fit']) == len(selected)
    assert np.count_nonzero(np.any(x != clean['x_fit'], axis=1)) == len(selected)
    return (x, y)

def hashes(x, y, clean):
    return {name: array_digest(value) for name, value in dict(x_train=x, y_train=y, x_test=clean['x_test'], y_test=clean['y_test']).items()}

def configuration(seed):
    return replace(Config(), repeats=50, contamination_rates=RATES, seed=seed)

def make_plan(seed, data_root):
    streams = seed_map(seed)
    clean, _ = prepare_seed(data_root, configuration(seed), streams['split'])
    n = len(clean['y_fit'])
    plan = {key: clean[key] for key in ('train_indices', 'test_indices', 'fit_positions', 'validation_positions', 'center', 'scale')}
    plan.update(order=np.random.default_rng(streams['order']).permutation(n), replacement_x=np.random.default_rng(streams['signs']).choice(np.array([-50.0, 50.0], dtype=np.float32), size=(n, 10)), targets=(clean['y_fit'] + np.random.default_rng(streams['wrong_labels']).integers(1, 11, n)) % 11)
    return (plan, clean)

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.part')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)

def frozen_json(path, value):
    normalized = json.loads(json.dumps(value))
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != normalized:
            raise RuntimeError('Existing experiment settings differ; use a new output directory.')
    else:
        save_json(path, normalized)

def prepare_data(data_root):
    directory = data_root / 'vowel'
    directory.mkdir(parents=True, exist_ok=True)
    for name, expected in INPUT_SHA256.items():
        target = directory / name
        if not target.is_file():
            remote = 'vowel.scale' if name == 'vowel.scale.txt' else name
            url = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/' + remote
            print('Downloading ' + remote, flush=True)
            try:
                with urllib.request.urlopen(url, timeout=120) as response:
                    contents = response.read()
            except OSError as exc:
                raise RuntimeError('Download failed. Place vowel.scale.txt and vowel.scale.t in the vowel subdirectory of --data-root, then rerun.') from exc
            if hashlib.sha256(contents).hexdigest() != expected:
                raise RuntimeError('Downloaded dataset checksum differs from the experiment input.')
            target.write_bytes(contents)
        if digest(target) != expected:
            raise RuntimeError('Input checksum mismatch: ' + name)

def record_path(output, seed, rate, method):
    return output / 'fits' / str(seed) / f'rate{rate:g}' / (method + '.json')

def load_record(output, seed, rate, method, expected=None):
    target = record_path(output, seed, rate, method)
    if not target.exists():
        return None
    row = json.loads(target.read_text(encoding='utf-8'))
    if (row['seed'], row['rate'], row['method']) != (seed, rate, method):
        raise RuntimeError('Saved record identity mismatch.')
    if expected is not None and row['case_hashes'] != expected:
        raise RuntimeError('Saved data and contamination differ.')
    for name, sha in row['artifact_sha256'].items():
        if digest(target.with_name(name)) != sha:
            raise RuntimeError('Saved artifact checksum mismatch.')
    if not math.isfinite(row['fit_seconds']) or row['fit_seconds'] < 0:
        raise RuntimeError('Invalid saved fitting time.')
    with np.load(target.with_name(method + '_predictions.npz'), allow_pickle=False) as predictions:
        if len(predictions['true']) != 297:
            raise RuntimeError('Invalid saved test predictions.')
        error = float(np.mean(predictions['predicted'] != predictions['true']))
    if abs(error - row['test_error']) > 1e-12:
        raise RuntimeError('Saved prediction error does not match predictions.')
    return row

def write_csv(path, rows):
    if not rows:
        return
    temporary = path.with_name(path.name + '.part')
    with temporary.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)

def summarize(output):
    rows = []
    for seed in SEEDS:
        for rate in RATES:
            for method in METHODS:
                row = load_record(output, seed, rate, method)
                if row is not None:
                    rows.append(row)
    flattened = [{key: value for key, value in row.items() if key not in ('case_hashes', 'artifact_sha256')} for row in rows]
    write_csv(output / 'raw_results.csv', flattened)
    summary = []
    for rate in RATES:
        for method in METHODS:
            selected = [row for row in rows if row['rate'] == rate and row['method'] == method]
            if not selected:
                continue
            item = dict(rate=rate, method=method, tuning=TUNING[method], repetitions=len(selected))
            for key, label, multiplier in (('test_error', 'prediction_error_percent', 100.0), ('fit_seconds', 'fit_seconds', 1.0)):
                values = np.asarray([row[key] for row in selected]) * multiplier
                item[label + '_mean'] = float(values.mean())
                item[label + '_q25'] = float(np.quantile(values, 0.25, method='linear'))
                item[label + '_q75'] = float(np.quantile(values, 0.75, method='linear'))
            summary.append(item)
    write_csv(output / 'summary.csv', summary)
    index = {(row['seed'], row['rate'], row['method']): row for row in rows}
    paired, changes = ([], [])
    for seed in SEEDS:
        for rate in RATES:
            a, b = (index.get((seed, rate, 'rho')), index.get((seed, rate, 'MLE')))
            if a is not None and b is not None:
                paired.append(dict(seed=seed, rate=rate, rho_minus_mle_error_percentage_points=100.0 * (a['test_error'] - b['test_error']), rho_fit_seconds=a['fit_seconds'], mle_fit_seconds=b['fit_seconds']))
            for method in METHODS:
                row, clean = (index.get((seed, rate, method)), index.get((seed, 0.0, method)))
                if row is not None and clean is not None:
                    changes.append(dict(seed=seed, rate=rate, method=method, error_increase_percentage_points=100.0 * (row['test_error'] - clean['test_error'])))
    write_csv(output / 'paired_rho_minus_mle.csv', paired)
    write_csv(output / 'paired_clean_increases.csv', changes)
    return len(rows)

def run(output, data_root):
    prepare_data(data_root)
    protocol = dict(dataset='Vowel', repetitions=50, seeds=SEEDS, rates=RATES, methods=METHODS, tuning=TUNING, samples=dict(train=623, unused_holdout=70, test=297, features=10, classes=11), config=asdict(configuration(SEEDS[0])), seed_streams={str(seed): seed_map(seed) for seed in SEEDS}, contamination='independent random -50 or +50 features and uniformly sampled incorrect labels', preprocessing='clean fitting-set median and IQR/1.349, clipping to [-6,6] before contamination only', data_resampled_each_repetition=True, contamination_nested_across_rates=True, clean_test=True, input_sha256=INPUT_SHA256, code_sha256=digest(__file__), device='cpu', torch_threads=2, interop_threads=1, runtime=dict(python=platform.python_version(), numpy=np.__version__, torch=torch.__version__), fitting_time_excludes=['network_initialization', 'final_evaluation', 'file_saving'])
    frozen_json(output / 'protocol.json', protocol)
    total = len(SEEDS) * len(RATES) * len(METHODS)
    completed = summarize(output)
    print(f'Vowel: 50 repetitions, {completed}/{total} completed fits, device=cpu', flush=True)
    start = time.perf_counter()
    save_json(output / 'execution.json', dict(status='running', completed=completed, total=total))
    try:
        for repeat, seed in enumerate(SEEDS, 1):
            if (output / 'STOP').exists():
                raise InterruptedError('STOP file found; saved fits can be resumed.')
            plan, clean = make_plan(seed, data_root)
            plan_path = output / 'plans' / (str(seed) + '.npz')
            plan_path.parent.mkdir(parents=True, exist_ok=True)
            if plan_path.exists():
                with np.load(plan_path, allow_pickle=False) as saved:
                    if set(saved.files) != set(plan):
                        raise RuntimeError('Saved contamination plan differs.')
                    for key, value in plan.items():
                        np.testing.assert_array_equal(saved[key], value)
            else:
                plan_temporary = plan_path.with_name(plan_path.stem + '.part.npz')
                np.savez_compressed(plan_temporary, **plan)
                plan_temporary.replace(plan_path)
            config = configuration(seed)
            xt, yt = (torch.from_numpy(clean['x_test']), torch.from_numpy(clean['y_test']))
            for rate in RATES:
                x, y = case(plan, clean, rate)
                expected = hashes(x, y, clean)
                for method in METHODS:
                    if (output / 'STOP').exists():
                        raise InterruptedError('STOP file found; saved fits can be resumed.')
                    if load_record(output, seed, rate, method, expected) is not None:
                        continue
                    print(f'Repeat {repeat}/50, contamination {rate:.0%}, {method}: fitting', flush=True)
                    xb, yb = (torch.from_numpy(x), torch.from_numpy(y))
                    model, info, seconds = fit(method, TUNING[method], xb, yb, config, seed_map(seed)['initialization'])
                    if not all((torch.isfinite(p).all().item() for p in model.parameters())):
                        raise FloatingPointError('Nonfinite fitted parameter.')
                    train, test = (evaluate(model, xb, yb), evaluate(model, xt, yt))
                    with torch.no_grad():
                        predicted = model(xt).argmax(1).cpu().numpy().astype(np.uint8)
                    if abs(float(np.mean(predicted != clean['y_test'])) - test['error']) > 1e-12:
                        raise RuntimeError('Prediction and evaluation disagreement.')
                    model_hash = hashlib.sha256()
                    for name, value in model.state_dict().items():
                        model_hash.update(name.encode())
                        model_hash.update(value.detach().cpu().numpy().tobytes())
                    target = record_path(output, seed, rate, method)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    training_path = target.with_name(method + '_training.json')
                    prediction_path = target.with_name(method + '_predictions.npz')
                    save_json(training_path, info)
                    prediction_temporary = prediction_path.with_name(prediction_path.stem + '.part.npz')
                    np.savez_compressed(prediction_temporary, predicted=predicted, true=clean['y_test'].astype(np.uint8))
                    prediction_temporary.replace(prediction_path)
                    count = round(623 * rate)
                    row = dict(seed=seed, rate=rate, count=count, method=method, tuning=TUNING[method], magnitude=50.0 if rate else 0.0, n_train=623, n_test=297, n_labels_changed=count, test_error=test['error'], fit_seconds=seconds, train_error=train['error'], test_nll=test['nll'], train_nll=train['nll'], rho_rounds=info.get('rho_rounds'), selected_epoch=info.get('selected_epoch'), initialization_seed=seed_map(seed)['initialization'], training_seed=seed_map(seed)['trainer'], case_hashes=expected, model_sha256=model_hash.hexdigest(), artifact_sha256={p.name: digest(p) for p in (prediction_path, training_path)})
                    save_json(target, row)
                    completed += 1
                    save_json(output / 'progress.json', dict(completed=completed, total=total, repeat=repeat, seed=seed, rate=rate, method=method))
                    print(f"  error={test['error']:.2%}, accuracy={1.0 - test['error']:.2%}, fit={seconds:.2f}s; saved ({completed}/{total})", flush=True)
                    del model
                    gc.collect()
            completed = summarize(output)
        if completed != total:
            raise RuntimeError('Unexpected final number of fits.')
        save_json(output / 'execution.json', dict(status='complete', completed=completed, total=total, wall_seconds=time.perf_counter() - start))
        print('Completed. Results: ' + str(output), flush=True)
    except BaseException as exc:
        summarize(output)
        save_json(output / 'execution.json', dict(status='stopped' if isinstance(exc, (InterruptedError, KeyboardInterrupt)) else 'failed', completed=completed, total=total, error_type=type(exc).__name__, wall_seconds=time.perf_counter() - start))
        raise

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, default=Path('data'))
    parser.add_argument('--output-dir', type=Path, default=Path('vowel_append_batch50_results'))
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / 'run.lock').open('a+') as lock:
        if os.name == 'posix':
            import fcntl
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError('Another process is using this output directory.') from exc
        elif os.name == 'nt':
            import msvcrt
            lock.seek(0)
            lock.write('0')
            lock.flush()
            lock.seek(0)
            try:
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError('Another process is using this output directory.') from exc
        run(args.output_dir, args.data_root)
if __name__ == '__main__':
    main()
