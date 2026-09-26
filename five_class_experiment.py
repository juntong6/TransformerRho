import importlib.util
import subprocess
import sys
_required = {'numpy': 'numpy', 'scipy': 'scipy', 'torch': 'torch', 'sklearn': 'scikit-learn', 'pandas': 'pandas', 'matplotlib': 'matplotlib'}
_missing = [pkg for module, pkg in _required.items() if importlib.util.find_spec(module) is None]
if _missing:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', *_missing])
import csv
import math
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import minimize
from scipy.special import logsumexp
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
NUM_EXPERIMENTS = 100
NUM_CLASSES = 5
NUM_HEADS = 2
NUM_FEATURES = 1 + NUM_HEADS
NUM_FREE_CLASSES = NUM_CLASSES - 1
NUM_ATTENTION_PARAMETERS = 2 * NUM_HEADS
NUM_HEAD_PARAMETERS = NUM_FREE_CLASSES * NUM_FEATURES
NUM_PARAMETERS = NUM_ATTENTION_PARAMETERS + NUM_HEAD_PARAMETERS
TRUE_LOGIT_SCALE = 1.5
if not np.isfinite(TRUE_LOGIT_SCALE) or TRUE_LOGIT_SCALE <= 0:
    raise ValueError('TRUE_LOGIT_SCALE must be finite and positive')
TRUTH_MODEL_NAME = f'five_class_level_and_spread_scale_{TRUE_LOGIT_SCALE:g}'
CLASS_NAMES = ['low_level', 'high_level', 'small_spread', 'large_spread', 'middle_state']
NUM_TOKENS = 5
TOKEN_POSITIONS = np.linspace(-1.0, 1.0, NUM_TOKENS)
TRAIN_N = 500
TEST_N = 10000
CONTAMINATION_RATES = [0.0, 0.01, 0.02, 0.03, 0.05, 0.1]
MOM_BLOCK_COUNTS = [5, 10, 20, 50]
HUBER_DELTAS = [0.5, 1.0, 2.0, 4.0]
FIXED_CONTAMINATION_TYPE = 'all_positive'
FIXED_CONTAMINATION_SCALE = 20.0
FIXED_CONTAMINATION_LABEL_MODE = 'least_likely'
ATTENTION_BOUND = 2.5
HEAD_BOUND = 30.0
MLE_MAX_ITERATIONS = 1200
ROBUST_MAX_ITERATIONS = 800
MOM_MAX_FUNCTION_EVALUATIONS = 10000
RHO_MAX_ITERATIONS = 10
RHO_RESTART_NOISE_SCALES = [0.0, 0.1, 0.35]
RHO_MAX_OPTIMIZER_ITERATIONS = 800
RHO_STOP_THRESHOLD = -1.0
BASE_SEED = 1000
SCRIPT_DIR = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
OUTPUT_DIR = SCRIPT_DIR / f'five_class_easier_scale_{TRUE_LOGIT_SCALE:g}_results'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_FILE = OUTPUT_DIR / 'five_class_mle_rho_mom_huber.csv'
SUMMARY_FILE = OUTPUT_DIR / 'five_class_mle_rho_mom_huber_summary.csv'
TRUE_ATTENTION = np.array([1.6, 0.7, -1.3, -0.6])
TRUE_BETA = TRUE_LOGIT_SCALE * np.array([[-1.8, -0.8, -0.8], [-1.8, 0.8, 0.8], [3.0, -3.0, 3.0], [-5.4, 3.0, -3.0]])
TRUE_THETA = np.concatenate([TRUE_ATTENTION, TRUE_BETA.ravel()])
PARAMETER_BOUNDS = [(0.0, ATTENTION_BOUND), (-ATTENTION_BOUND, ATTENTION_BOUND), (-ATTENTION_BOUND, 0.0), (-ATTENTION_BOUND, ATTENTION_BOUND)] + [(-HEAD_BOUND, HEAD_BOUND)] * NUM_HEAD_PARAMETERS
if not all((lower <= value <= upper for value, (lower, upper) in zip(TRUE_THETA, PARAMETER_BOUNDS))):
    raise ValueError('The true model must lie inside the fitted parameter bounds')

class GeneralTinyAttentionTransformer(nn.Module):

    def __init__(self, attention_parameters: np.ndarray | None=None, beta: np.ndarray | None=None):
        super().__init__()
        if attention_parameters is None:
            attention_parameters = TRUE_ATTENTION
        if beta is None:
            beta = TRUE_BETA
        attention_parameters = np.asarray(attention_parameters, dtype=np.float32)
        beta = np.asarray(beta, dtype=np.float32)
        if attention_parameters.shape != (4,):
            raise ValueError('attention_parameters must have shape (4,)')
        if beta.shape != (NUM_FREE_CLASSES, NUM_FEATURES):
            raise ValueError(f'beta must have shape ({NUM_FREE_CLASSES},{NUM_FEATURES})')
        self.attention_parameters = nn.Parameter(torch.from_numpy(attention_parameters.copy()))
        self.register_buffer('token_positions', torch.tensor(TOKEN_POSITIONS, dtype=torch.float32))
        self.beta = nn.Parameter(torch.from_numpy(beta.copy()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != NUM_TOKENS:
            raise ValueError(f'x must have shape [batch_size, {NUM_TOKENS}]')
        summaries = []
        for head_index in range(2):
            content_coefficient = self.attention_parameters[2 * head_index]
            position_coefficient = self.attention_parameters[2 * head_index + 1]
            scores = content_coefficient * x + position_coefficient * self.token_positions[None, :]
            weights = torch.softmax(scores, dim=1)
            summary = torch.sum(weights * x, dim=1)
            summaries.append(summary)
        features = torch.stack([torch.ones_like(summaries[0]), summaries[0], summaries[1]], dim=1)
        logits = features @ self.beta.T
        return torch.cat([logits, torch.zeros_like(logits[:, :1])], dim=1)

def unpack_theta(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    theta = np.asarray(theta, dtype=float)
    if theta.shape != (NUM_PARAMETERS,):
        raise ValueError(f'theta must contain {NUM_PARAMETERS} parameters')
    attention = theta[:4]
    beta = theta[NUM_ATTENTION_PARAMETERS:].reshape(NUM_FREE_CLASSES, NUM_FEATURES)
    return (attention, beta)

def attention_features_and_derivatives(x: np.ndarray, attention: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    attention = np.asarray(attention, dtype=float)
    summaries = []
    derivatives = np.zeros((x.shape[0], 2, 2), dtype=float)
    for head_index in range(2):
        content_coefficient = attention[2 * head_index]
        position_coefficient = attention[2 * head_index + 1]
        scores = content_coefficient * x + position_coefficient * TOKEN_POSITIONS[None, :]
        scores -= np.max(scores, axis=1, keepdims=True)
        weights = np.exp(scores)
        weights /= np.sum(weights, axis=1, keepdims=True)
        summary = np.sum(weights * x, axis=1)
        summaries.append(summary)
        centered_value = x - summary[:, None]
        derivatives[:, head_index, 0] = np.sum(weights * centered_value * x, axis=1)
        derivatives[:, head_index, 1] = np.sum(weights * centered_value * TOKEN_POSITIONS[None, :], axis=1)
    features = np.column_stack([np.ones(x.shape[0], dtype=float), summaries[0], summaries[1]])
    return (features, derivatives)

def class_probabilities(x: np.ndarray, theta: np.ndarray) -> np.ndarray:
    attention, beta = unpack_theta(theta)
    features, _ = attention_features_and_derivatives(x, attention)
    logits = np.column_stack([features @ beta.T, np.zeros(x.shape[0], dtype=float)])
    return np.exp(logits - logsumexp(logits, axis=1, keepdims=True))

def verify_pytorch_numpy_equivalence() -> None:
    rng = np.random.default_rng(123)
    x = rng.normal(size=(64, NUM_TOKENS)).astype(np.float32)
    model = GeneralTinyAttentionTransformer(attention_parameters=TRUE_ATTENTION, beta=TRUE_BETA)
    model.eval()
    with torch.no_grad():
        torch_probability = torch.softmax(model(torch.from_numpy(x)), dim=1).numpy()
    numpy_probability = class_probabilities(x, TRUE_THETA)
    maximum_difference = float(np.max(np.abs(torch_probability - numpy_probability)))
    print('Maximum PyTorch/NumPy probability difference:', maximum_difference)
    if maximum_difference > 2e-06:
        raise RuntimeError('PyTorch/NumPy equivalence check failed.')

def simulate_characteristics(rng: np.random.Generator, n: int) -> np.ndarray:
    common_shift = rng.normal(loc=0.0, scale=0.9, size=(n, 1))
    mixture = rng.random(n)
    common_shift[mixture < 0.25] += 1.5
    common_shift[(mixture >= 0.25) & (mixture < 0.5)] -= 1.5
    x = common_shift + 0.65 * TOKEN_POSITIONS[None, :] + rng.normal(loc=0.0, scale=0.65, size=(n, NUM_TOKENS))
    return x

def simulate_responses(rng: np.random.Generator, x: np.ndarray) -> np.ndarray:
    probability = class_probabilities(x, TRUE_THETA)
    uniforms = rng.random(x.shape[0])
    cumulative = np.cumsum(probability, axis=1)
    cumulative[:, -1] = 1.0
    return np.sum(uniforms[:, None] > cumulative, axis=1).astype(int)

def simulate_candidate_contamination(rng: np.random.Generator, n: int, contamination_type: str, scale: float, label_mode: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if contamination_type == 'clean_like':
        x = simulate_characteristics(rng, n)
    elif contamination_type == 'uniform':
        x = rng.uniform(low=-scale, high=scale, size=(n, NUM_TOKENS))
    elif contamination_type == 'all_positive':
        x = np.full((n, NUM_TOKENS), scale, dtype=float)
    elif contamination_type == 'all_negative':
        x = np.full((n, NUM_TOKENS), -scale, dtype=float)
    else:
        raise ValueError(f'Unknown contamination type: {contamination_type}')
    if contamination_type in {'all_positive', 'all_negative'}:
        for index in range(n):
            x[index, index % NUM_TOKENS] += 1e-07 * scale * index
    true_probability = class_probabilities(x, TRUE_THETA)
    if label_mode == 'least_likely':
        y = np.argmin(true_probability, axis=1).astype(int)
    elif label_mode == 'random':
        y = rng.integers(low=0, high=NUM_CLASSES, size=n)
    else:
        raise ValueError(f'Unknown contamination label mode: {label_mode}')
    label_probabilities = true_probability[np.arange(n), y]
    return (x, y, label_probabilities)

def multinomial_nll_and_gradient(theta, x, y):
    observed_log_probability, gradient = log_observed_probability_and_gradient(theta, x, y)
    return (float(-np.sum(observed_log_probability)), -np.sum(gradient, axis=0))

@dataclass
class MLEFit:
    theta: np.ndarray
    objective: float
    success: bool
    message: str
    gradient_norm: float

def fit_mle(x: np.ndarray, y: np.ndarray, start_theta: np.ndarray | None=None, use_full_multistart: bool=True) -> MLEFit:
    starts = []
    if start_theta is not None:
        starts.append(np.asarray(start_theta, dtype=float).copy())
    if use_full_multistart or not starts:
        attention_starts = [np.zeros(4), np.array([0.5, 0.2, -0.5, -0.2]), np.array([1.5, 0.8, -1.5, -0.8]), np.array([0.2, 0.5, -0.2, -0.5]), np.array([0.5, -0.5, -0.5, 0.5])]
        for attention_start in attention_starts:
            starts.append(np.concatenate([attention_start, np.zeros(NUM_HEAD_PARAMETERS, dtype=float)]))
    best_result = None
    for start in starts:
        result = minimize(fun=lambda value: multinomial_nll_and_gradient(value, x, y)[0], x0=start, jac=lambda value: multinomial_nll_and_gradient(value, x, y)[1], method='L-BFGS-B', bounds=PARAMETER_BOUNDS, options={'maxiter': MLE_MAX_ITERATIONS, 'ftol': 1e-12, 'gtol': 1e-08})
        if best_result is None or result.fun < best_result.fun:
            best_result = result
    if best_result is None:
        raise RuntimeError('MLE optimization produced no result.')
    theta = np.asarray(best_result.x, dtype=float)
    gradient_norm = float(np.linalg.norm(multinomial_nll_and_gradient(theta, x, y)[1]))
    return MLEFit(theta=theta, objective=float(best_result.fun), success=bool(best_result.success), message=str(best_result.message), gradient_norm=gradient_norm)

def hellinger_squared_empirical(x: np.ndarray, estimated_theta: np.ndarray) -> float:
    true_probability = class_probabilities(x, TRUE_THETA)
    estimated_probability = class_probabilities(x, estimated_theta)
    per_sample = 1.0 - np.sum(np.sqrt(true_probability * estimated_probability), axis=1)
    return float(np.mean(per_sample))

def hellinger_distance_from_squared(h_squared: float) -> float:
    return float(math.sqrt(max(h_squared, 0.0)))

def classification_error(x: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:
    prediction = np.argmax(class_probabilities(x, theta), axis=1)
    return float(np.mean(prediction != y))

def log_observed_probability_and_gradient(theta, x, y):
    attention, beta = unpack_theta(theta)
    features, derivatives = attention_features_and_derivatives(x, attention)
    n = x.shape[0]
    logits = np.column_stack([features @ beta.T, np.zeros(n, dtype=float)])
    log_probability = logits - logsumexp(logits, axis=1, keepdims=True)
    probability = np.exp(log_probability)
    observed_log_probability = log_probability[np.arange(n), y]
    gradient = np.zeros((n, NUM_PARAMETERS), dtype=float)
    coefficients = (y[:, None] == np.arange(NUM_FREE_CLASSES)[None, :]).astype(float)
    coefficients -= probability[:, :NUM_FREE_CLASSES]
    gradient[:, NUM_ATTENTION_PARAMETERS:] = (coefficients[:, :, None] * features[:, None, :]).reshape(n, NUM_HEAD_PARAMETERS)
    for head_index in range(NUM_HEADS):
        for local_parameter_index in range(2):
            value = derivatives[:, head_index, local_parameter_index]
            logit_derivative = np.zeros((n, NUM_CLASSES), dtype=float)
            logit_derivative[:, :NUM_FREE_CLASSES] = value[:, None] * beta[:, 1 + head_index][None, :]
            gradient[:, 2 * head_index + local_parameter_index] = logit_derivative[np.arange(n), y] - np.sum(probability * logit_derivative, axis=1)
    return (observed_log_probability, gradient)

def rho_objective_and_gradient(challenger_theta: np.ndarray, current_log_probability: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    challenger_log_probability, challenger_gradient = log_observed_probability_and_gradient(challenger_theta, x, y)
    transformed_difference = (current_log_probability - challenger_log_probability) / 4.0
    contribution = np.tanh(transformed_difference)
    derivative_weight = -0.25 * (1.0 - contribution ** 2)
    gradient = np.sum(derivative_weight[:, None] * challenger_gradient, axis=0)
    return (float(np.sum(contribution)), gradient)

def choose_rho_starter(x, y):
    initial_attention = np.array([0.5, 0.2, -0.5, -0.2], dtype=float)
    features, _ = attention_features_and_derivatives(x, initial_attention)
    y = np.asarray(y, dtype=int)
    if len(y) == 0 or np.any((y < 0) | (y >= NUM_CLASSES)):
        raise ValueError('Labels must be nonempty integers in the configured class range')
    counts = np.bincount(y, minlength=NUM_CLASSES)
    prior = (counts + 0.5) / (len(y) + 0.5 * NUM_CLASSES)
    full_coefficients = np.zeros((NUM_CLASSES, NUM_FEATURES), dtype=float)
    full_coefficients[:, 0] = np.log(prior)
    if len(np.unique(y)) >= 2:
        svm = LinearSVC(C=10.0, tol=1e-06, max_iter=200000, dual=False)
        svm.fit(features[:, 1:], y)
        svm_rows = np.column_stack([svm.intercept_, svm.coef_])
        if len(svm.classes_) == 2:
            svm_rows = np.vstack([-0.5 * svm_rows[0], 0.5 * svm_rows[0]])
        full_coefficients[svm.classes_.astype(int)] = svm_rows
    reference = full_coefficients[NUM_CLASSES - 1]
    beta = full_coefficients[:NUM_FREE_CLASSES] - reference[None, :]
    return np.concatenate([initial_attention, beta.ravel()])

@dataclass
class RhoFit:
    theta: np.ndarray
    iterations: int
    path_values: list[float]

def fit_rho(x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> RhoFit:
    current_theta = choose_rho_starter(x, y)
    current_theta = np.clip(current_theta, np.array([b[0] for b in PARAMETER_BOUNDS]), np.array([b[1] for b in PARAMETER_BOUNDS]))
    path_values: list[float] = []
    lower_bounds = np.array([bound[0] for bound in PARAMETER_BOUNDS], dtype=float)
    upper_bounds = np.array([bound[1] for bound in PARAMETER_BOUNDS], dtype=float)
    for iteration in range(RHO_MAX_ITERATIONS):
        current_log_probability, _ = log_observed_probability_and_gradient(current_theta, x, y)
        starts = [current_theta.copy()]
        for noise_scale in RHO_RESTART_NOISE_SCALES[1:]:
            noisy_start = current_theta + rng.normal(loc=0.0, scale=noise_scale, size=current_theta.shape)
            starts.append(np.clip(noisy_start, lower_bounds, upper_bounds))
        best_result = None
        for start in starts:
            result = minimize(fun=lambda value: rho_objective_and_gradient(value, current_log_probability, x, y)[0], x0=start, jac=lambda value: rho_objective_and_gradient(value, current_log_probability, x, y)[1], method='L-BFGS-B', bounds=PARAMETER_BOUNDS, options={'maxiter': RHO_MAX_OPTIMIZER_ITERATIONS, 'ftol': 1e-12, 'gtol': 1e-08})
            if best_result is None or result.fun < best_result.fun:
                best_result = result
        if best_result is None:
            raise RuntimeError('rho challenger optimization produced no result.')
        current_theta = np.asarray(best_result.x, dtype=float)
        statistic_value = float(best_result.fun)
        path_values.append(statistic_value)
        print(f'rho iteration {iteration + 1}/{RHO_MAX_ITERATIONS}, minimum statistic={statistic_value:.6f}, attention={np.round(current_theta[:4], 4)}')
        if statistic_value > RHO_STOP_THRESHOLD:
            print(f'rho stopped because statistic > {RHO_STOP_THRESHOLD:.1f}')
            break
    return RhoFit(theta=current_theta, iterations=len(path_values), path_values=path_values)

@dataclass
class ComparisonFit:
    theta: np.ndarray
    objective: float
    success: bool
    message: str

def huber_nll_objective_and_gradient(theta, x, y, delta):
    if delta <= 0:
        raise ValueError('Huber delta must be positive')
    logp, logp_gradient = log_observed_probability_and_gradient(theta, x, y)
    nll = np.maximum(-logp, 0.0)
    residual = np.sqrt(2.0 * nll)
    quadratic = residual <= delta
    losses = np.where(quadratic, nll, delta * residual - 0.5 * delta ** 2)
    weights = np.ones_like(residual)
    np.divide(delta, residual, out=weights, where=~quadratic)
    gradient = np.sum(-logp_gradient * weights[:, None], axis=0)
    return (float(np.sum(losses)), gradient)

def make_mom_blocks(n, k, seed):
    if not 1 <= k <= n:
        raise ValueError('MOM requires 1 <= K <= n')
    return np.array_split(np.random.default_rng(seed).permutation(n), k)

def mom_nll_objective(theta, x, y, blocks):
    logp, _ = log_observed_probability_and_gradient(theta, x, y)
    return float(np.median([np.mean(-logp[block]) for block in blocks]))

def comparison_starts(x, y):
    svm_start = choose_rho_starter(x, y)
    neutral = np.concatenate([np.array([0.5, 0.2, -0.5, -0.2]), np.zeros(NUM_HEAD_PARAMETERS)])
    low = np.array([b[0] for b in PARAMETER_BOUNDS])
    high = np.array([b[1] for b in PARAMETER_BOUNDS])
    return [np.clip(svm_start, low, high), neutral]

def fit_comparison(x, y, method, tuning, block_seed):
    candidates = []
    blocks = make_mom_blocks(len(y), int(tuning), block_seed) if method == 'MOM' else None
    for initial in comparison_starts(x, y):
        if method == 'MOM':
            result = minimize(mom_nll_objective, initial, args=(x, y, blocks), method='Powell', bounds=PARAMETER_BOUNDS, options={'maxiter': ROBUST_MAX_ITERATIONS, 'maxfev': MOM_MAX_FUNCTION_EVALUATIONS, 'xtol': 1e-05, 'ftol': 1e-08})
        elif method == 'Huber':
            result = minimize(huber_nll_objective_and_gradient, initial, args=(x, y, float(tuning)), jac=True, method='L-BFGS-B', bounds=PARAMETER_BOUNDS, options={'maxiter': ROBUST_MAX_ITERATIONS, 'ftol': 1e-12, 'gtol': 1e-08})
        else:
            raise ValueError(method)
        if np.isfinite(result.fun) and np.all(np.isfinite(result.x)):
            candidates.append(result)
    if not candidates:
        raise RuntimeError(f'No finite result for {method} {tuning}')
    best = min(candidates, key=lambda r: r.fun)
    return ComparisonFit(best.x, float(best.fun), bool(best.success), str(best.message))

def fit_all(x, y, seed):
    fitted = {}
    print('  Fitting MLE', flush=True)
    mle_candidates = [fit_mle(x, y, start_theta=s, use_full_multistart=False) for s in comparison_starts(x, y)]
    fitted['MLE'] = min(mle_candidates, key=lambda fit: fit.objective)
    print('  Fitting rho', flush=True)
    fitted['rho'] = fit_rho(x, y, np.random.default_rng(seed + 20000))
    for k in MOM_BLOCK_COUNTS:
        name = f'MOM K={k}'
        print(f'  Fitting {name}', flush=True)
        fitted[name] = fit_comparison(x, y, 'MOM', k, seed + 30000 + k)
    for delta in HUBER_DELTAS:
        name = f'Huber delta={delta:g}'
        print(f'  Fitting {name}', flush=True)
        fitted[name] = fit_comparison(x, y, 'Huber', delta, seed)
    return fitted

def evaluate(fit, x_test, y_test):
    return (hellinger_squared_empirical(x_test, fit.theta), classification_error(x_test, y_test, fit.theta))

def make_row(seed, rate, name, clean_fit, bad_fit, x_test, y_test):
    clean_h2, clean_error = evaluate(clean_fit, x_test, y_test)
    bad_h2, bad_error = evaluate(bad_fit, x_test, y_test)
    is_rho = name == 'rho'
    return dict(num_classes=NUM_CLASSES, truth_model=TRUTH_MODEL_NAME, true_logit_scale=TRUE_LOGIT_SCALE, seed=seed, contamination_rate=rate, contamination_count=int(round(TRAIN_N * rate)), method=name, tuning_parameter='K' if name.startswith('MOM') else 'delta' if name.startswith('Huber') else '', tuning_value=name.split('=')[1] if '=' in name else '', Hellinger_squared_clean=clean_h2, Hellinger_squared_contaminated=bad_h2, Hellinger_squared_change=bad_h2 - clean_h2, Hellinger_distance_clean=hellinger_distance_from_squared(clean_h2), Hellinger_distance_contaminated=hellinger_distance_from_squared(bad_h2), error_clean=clean_error, error_contaminated=bad_error, error_change=bad_error - clean_error, optimizer_success_clean='' if is_rho else clean_fit.success, optimizer_success_contaminated='' if is_rho else bad_fit.success, optimizer_message_clean='rho heuristic; no global certificate' if is_rho else clean_fit.message, optimizer_message_contaminated='rho heuristic; no global certificate' if is_rho else bad_fit.message, rho_iterations_clean=clean_fit.iterations if is_rho else '', rho_iterations_contaminated=bad_fit.iterations if is_rho else '', rho_stop_threshold_met_clean=clean_fit.path_values[-1] > RHO_STOP_THRESHOLD if is_rho else '', rho_stop_threshold_met_contaminated=bad_fit.path_values[-1] > RHO_STOP_THRESHOLD if is_rho else '', contamination_type=FIXED_CONTAMINATION_TYPE, contamination_scale=FIXED_CONTAMINATION_SCALE, TRAIN_N=TRAIN_N, TEST_N=TEST_N)

def include_clean_zero_rate(frame):
    required = {'seed', 'method', 'contamination_rate'}
    if not required.issubset(frame.columns):
        return frame
    if 'scenario' in frame.columns and frame['scenario'].nunique() > 1:
        raise ValueError('Select one scenario before plotting; do not pool different true functions')
    keys = ['seed', 'method']
    pairs = frame[keys].drop_duplicates()
    zero_pairs = frame.loc[frame['contamination_rate'].eq(0), keys].drop_duplicates()
    missing = pairs.merge(zero_pairs, on=keys, how='left', indicator=True)
    missing = missing.loc[missing['_merge'].eq('left_only'), keys]
    if missing.empty:
        return frame.copy()
    clean_column = 'Hellinger_squared_clean'
    if clean_column not in frame.columns:
        print('Cannot add missing 0% points: this CSV does not contain clean-fit results.')
        return frame.copy()
    candidates = frame.merge(missing, on=keys, how='inner')
    if not np.isfinite(candidates[clean_column].to_numpy(dtype=float)).all():
        raise ValueError('Non-finite clean values prevent constructing the 0% point')
    spread = candidates.groupby(keys)[clean_column].agg(['min', 'max'])
    if not np.allclose(spread['min'], spread['max'], rtol=1e-10, atol=1e-12):
        raise ValueError('Clean values differ for the same seed/method; cannot deduplicate them as one clean fit')
    clean_columns = [clean_column]
    if 'error_clean' in candidates.columns:
        if not np.isfinite(candidates['error_clean'].to_numpy(dtype=float)).all():
            raise ValueError('Non-finite clean prediction errors prevent constructing the 0% point')
        error_spread = candidates.groupby(keys)['error_clean'].agg(['min', 'max'])
        if not np.allclose(error_spread['min'], error_spread['max'], rtol=1e-10, atol=1e-12):
            raise ValueError('Clean prediction errors differ for the same seed/method')
        clean_columns.append('error_clean')
    zero = candidates[keys + clean_columns].drop_duplicates(keys).copy()
    zero['num_classes'] = NUM_CLASSES
    zero['truth_model'] = TRUTH_MODEL_NAME
    zero['true_logit_scale'] = TRUE_LOGIT_SCALE
    zero['contamination_rate'] = 0.0
    zero['Hellinger_squared_contaminated'] = zero[clean_column]
    if 'error_clean' in zero.columns:
        zero['error_contaminated'] = zero['error_clean']
    print(f'Added {len(zero)} zero-contamination plotting rows from existing clean fits (one per seed/method).')
    return pd.concat([frame, zero], ignore_index=True)

def pointwise_plot_summary(frame):
    required = {'seed', 'method', 'contamination_rate', 'Hellinger_squared_contaminated'}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f'CSV is missing columns: {sorted(missing)}')
    if frame.empty:
        raise ValueError('No experimental results to summarize')
    if 'scenario' in frame.columns and frame['scenario'].nunique() > 1:
        raise ValueError('Select one scenario before plotting; distinct true functions must not be pooled.')
    if frame.duplicated(['seed', 'method', 'contamination_rate']).any():
        raise ValueError('Duplicate seed/method/rate rows would double-count repetitions')
    column = 'Hellinger_squared_contaminated'
    if not np.isfinite(frame[column].to_numpy(dtype=float)).all():
        raise ValueError(f'Non-finite values in {column}; inspect experiment results')
    rates = frame['contamination_rate'].to_numpy(dtype=float)
    if not np.isfinite(rates).all() or np.any((rates < 0) | (rates > 1)):
        raise ValueError('Contamination rates must be finite fractions in [0,1]')
    return frame.groupby(['method', 'contamination_rate'], sort=True)[column].agg(mean='mean', q25=lambda values: values.quantile(0.25), q75=lambda values: values.quantile(0.75), n='count').reset_index()

def draw_mean_and_iqr(ax, stats, color, marker, label):
    stats = stats.sort_values('contamination_rate')
    xs = 100.0 * stats['contamination_rate'].to_numpy(dtype=float)
    ax.plot(xs, stats['mean'].to_numpy(dtype=float), color=color, linestyle='-', marker=marker, markersize=5.5, linewidth=1.6, label=label, zorder=3)
    usable = stats['n'].to_numpy() >= 2
    q25 = stats['q25'].to_numpy(dtype=float)[usable]
    q75 = stats['q75'].to_numpy(dtype=float)[usable]
    ax.vlines(xs[usable], q25, q75, color=color, linestyles='-', linewidth=1.25, alpha=0.55, zorder=2)
    ax.plot(xs[usable], q25, linestyle='none', marker='_', markersize=7, color=color, alpha=0.75)
    ax.plot(xs[usable], q75, linestyle='none', marker='_', markersize=7, color=color, alpha=0.75)

def show_contamination_tables(stats, names, metric_label='Hellinger squared on clean test data', file_prefix='attention_contamination'):
    try:
        from IPython.display import display
    except ImportError:
        display = None
    for rate, group in stats.groupby('contamination_rate', sort=True):
        table = group.set_index('method').reindex(names).dropna(subset=['mean'])[['mean', 'q25', 'q75', 'n']]
        table['n'] = table['n'].astype(int)
        table.index.name = 'Method'
        print(f'\nReplacement contamination = {100 * rate:g}% | {metric_label}')
        print('mean: average; q25/q75: 25th/75th percentiles; n: repetitions')
        if display is not None:
            with pd.option_context('display.float_format', lambda value: f'{value:.6f}'):
                display(table)
        else:
            print(table.to_string(float_format=lambda value: f'{value:.6f}'))
        rate_tag = f'{100 * rate:g}'.replace('.', 'p')
        table.to_csv(OUTPUT_DIR / f'{file_prefix}_{rate_tag}pct_table.csv')

def validate_five_class_results(frame):
    required = {'num_classes', 'truth_model'}
    if not required.issubset(frame.columns):
        raise ValueError("Use this five-class experiment's raw CSV, including num_classes and truth_model columns")
    if not frame['num_classes'].eq(NUM_CLASSES).all() or not frame['truth_model'].eq(TRUTH_MODEL_NAME).all():
        raise ValueError('Result file belongs to a different class count or true model; do not mix it with this experiment')

def select_configured_rates(frame):
    rates = frame['contamination_rate'].to_numpy(dtype=float)
    allowed = np.asarray(CONTAMINATION_RATES, dtype=float)
    selected = np.isclose(rates[:, None], allowed[None, :], rtol=0, atol=1e-12).any(axis=1)
    if not selected.all():
        omitted = sorted(frame.loc[~selected, 'contamination_rate'].unique())
        print('Omitting unconfigured contamination settings: ' + ', '.join((f'{100 * rate:g}%' for rate in omitted)))
    filtered = frame.loc[selected].copy()
    if filtered.empty:
        raise ValueError('No results match CONTAMINATION_RATES')
    return filtered

def true_model_diagnostics(n=100000):
    x = simulate_characteristics(np.random.default_rng(57139), n)
    p = class_probabilities(x, TRUE_THETA)
    best_class = p.argmax(axis=1)
    bayes_error = float(np.mean(1 - p.max(axis=1)))
    table = pd.DataFrame({'class_index': np.arange(NUM_CLASSES), 'class_name': CLASS_NAMES, 'marginal_probability': p.mean(axis=0), 'Bayes_prediction_fraction': [np.mean(best_class == k) for k in range(NUM_CLASSES)]})
    table['Bayes_error_overall'] = bayes_error
    table['true_logit_scale'] = TRUE_LOGIT_SCALE
    table['truth_model'] = TRUTH_MODEL_NAME
    table.to_csv(OUTPUT_DIR / 'five_class_truth_diagnostics.csv', index=False)
    print('Five-class truth diagnostics (not fitted-model results):')
    print(table.to_string(index=False))
    print(f'Clean Bayes classification error is approximately {100 * bayes_error:.2f}% under this stochastic label model.')
    return table

def prediction_error_summary(frame):
    required = {'seed', 'method', 'contamination_rate', 'error_contaminated'}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f'Prediction-error plot requires columns: {sorted(missing)}')
    if frame.empty:
        raise ValueError('No prediction-error results to summarize')
    if 'scenario' in frame.columns and frame['scenario'].nunique() > 1:
        raise ValueError('Select one scenario; do not pool different true functions')
    if frame.duplicated(['seed', 'method', 'contamination_rate']).any():
        raise ValueError('Duplicate seed/method/rate rows would double-count repetitions')
    values = frame['error_contaminated'].to_numpy(dtype=float)
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError('Prediction errors must be finite fractions in [0,1]; check the raw CSV, including 0% values')
    rates = frame['contamination_rate'].to_numpy(dtype=float)
    if not np.isfinite(rates).all() or np.any((rates < 0) | (rates > 1)):
        raise ValueError('Contamination rates must be finite fractions in [0,1]')
    return frame.groupby(['method', 'contamination_rate'], sort=True)['error_contaminated'].agg(mean='mean', q25=lambda values: values.quantile(0.25), q75=lambda values: values.quantile(0.75), n='count').reset_index()

def plot_prediction_error_bars(frame, styles=None):
    validate_five_class_results(frame)
    frame = select_configured_rates(frame)
    frame = include_clean_zero_rate(frame)
    if 'error_contaminated' not in frame.columns:
        print('Prediction-error plot skipped: use the raw experiment CSV containing error_contaminated, not a Hellinger-only summary.')
        return None
    stats = prediction_error_summary(frame)
    names = list(frame['method'].drop_duplicates())
    rates = sorted(stats['contamination_rate'].unique())
    positions = np.arange(len(rates), dtype=float)
    width = 0.84 / len(names)
    palette = plt.get_cmap('tab10')
    fig, ax = plt.subplots(figsize=(16, 7))
    for index, name in enumerate(names):
        group = stats[stats['method'] == name].set_index('contamination_rate').reindex(rates)
        means = group['mean'].to_numpy(dtype=float)
        valid = np.isfinite(means)
        if not valid.all():
            print(f'Missing some contamination settings for {name}; those bars are omitted.')
        xs = positions + (index - (len(names) - 1) / 2.0) * width
        color = styles[name][0] if styles is not None and name in styles else palette(index % 10)
        ax.bar(xs[valid], 100.0 * means[valid], width=width * 0.94, color=color, label=name, alpha=0.9, edgecolor='white', linewidth=0.35, zorder=2)
        usable = valid & (group['n'].to_numpy(dtype=float) >= 2)
        q25 = 100.0 * group['q25'].to_numpy(dtype=float)[usable]
        q75 = 100.0 * group['q75'].to_numpy(dtype=float)[usable]
        ax.vlines(xs[usable], q25, q75, color='#263238', linewidth=0.9, zorder=4)
        ax.plot(xs[usable], q25, linestyle='none', marker='_', markersize=4, color='#263238', zorder=4)
        ax.plot(xs[usable], q75, linestyle='none', marker='_', markersize=4, color='#263238', zorder=4)
    ax.set_xticks(positions)
    ax.set_xticklabels([f'{100 * rate:g}%' for rate in rates])
    ax.set_xlabel('Replacement contamination (grouped settings)')
    ax.set_ylabel('Prediction error on clean test data (%)')
    ax.set_title('Five-class prediction error by method and training contamination')
    ax.set_ylim(bottom=0)
    ax.grid(axis='y', alpha=0.2, zorder=0)
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), frameon=False)
    fig.text(0.5, 0.025, 'Bars: mean classification error. Capped lines: 25th-75th percentiles across repetitions, not confidence intervals.\n0% uses the clean fit. Each equally spaced group is one contamination setting.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    plot_path = OUTPUT_DIR / 'attention_prediction_error_grouped_bars.png'
    fig.savefig(plot_path, dpi=180, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    stats_path = OUTPUT_DIR / 'attention_prediction_error_mean_q25_q75.csv'
    stats.to_csv(stats_path, index=False)
    display_stats = stats.copy()
    display_stats[['mean', 'q25', 'q75']] *= 100.0
    show_contamination_tables(display_stats, names, metric_label='Prediction error (%) on clean test data', file_prefix='attention_prediction_error_percent_contamination')
    if (stats['n'] < 2).any():
        print('Some prediction-error groups have one repetition; quartile bars are omitted for these groups.')
    print(f'Saved prediction-error plot: {plot_path}')
    print(f'Saved prediction-error summary (fractions): {stats_path}')
    return stats

def plot_results(frame):
    validate_five_class_results(frame)
    frame = select_configured_rates(frame)
    frame = include_clean_zero_rate(frame)
    stats = pointwise_plot_summary(frame)
    names = list(frame['method'].drop_duplicates())
    rates = sorted(stats['contamination_rate'].unique())
    expected_rates = CONTAMINATION_RATES
    missing_rates = [rate for rate in expected_rates if not any((np.isclose(rate, existing) for existing in rates))]
    if missing_rates:
        print('Results not present for: ' + ', '.join((f'{100 * rate:g}%' for rate in missing_rates)) + '. Only available measurements are plotted; run those rates to add the missing points.')
    if (stats['n'] < 2).any():
        print('Some groups contain one repetition: mean shown; table quantiles coincide and interval bars are omitted.')
    counts = stats.groupby('contamination_rate')['n'].agg(['min', 'max'])
    if (counts['min'] != counts['max']).any():
        print('Some methods have different repetition counts; inspect n in the tables.')
    stats_path = OUTPUT_DIR / 'attention_contaminated_mean_q25_q75.csv'
    stats.to_csv(stats_path, index=False)
    markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '<', '>']
    palette = plt.get_cmap('tab10')
    styles = {name: (palette(index % 10), markers[index % len(markers)]) for index, name in enumerate(names)}
    groups = [('Five-class: MLE, rho and MOM under replacement contamination', [name for name in names if name in ('MLE', 'rho') or name.startswith('MOM ')], 'attention_mle_rho_mom_replacement_iqr.png'), ('Five-class: rho and Huber under replacement contamination', [name for name in names if name == 'rho' or name.startswith('Huber ')], 'attention_rho_huber_replacement_iqr.png')]
    plot_paths = []
    for title, selected_names, filename in groups:
        if not selected_names:
            continue
        fig, ax = plt.subplots(figsize=(11.5, 6.5))
        for name in selected_names:
            color, marker = styles[name]
            draw_mean_and_iqr(ax, stats[stats['method'] == name], color, marker, name)
        ax.set_xticks([100 * rate for rate in rates])
        ax.set_xlabel('Replacement contamination (%)')
        ax.set_ylabel('Hellinger squared on clean test data')
        ax.set_title(title)
        ax.grid(axis='y', alpha=0.2)
        ax.margins(x=0.035)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, 3))
        ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), frameon=False)
        fig.text(0.5, 0.025, 'Markers: means. Capped vertical bars: 25th-75th percentiles across repetitions (not confidence intervals).', ha='center', fontsize=9)
        fig.tight_layout(rect=(0, 0.065, 1, 1))
        plot_path = OUTPUT_DIR / filename
        fig.savefig(plot_path, dpi=180, bbox_inches='tight')
        plt.show()
        plt.close(fig)
        plot_paths.append(plot_path)
    show_contamination_tables(stats, names)
    for plot_path in plot_paths:
        print(f'Saved plot: {plot_path}')
    print(f'Saved all-rate table: {stats_path}')
    plot_prediction_error_bars(frame, styles=styles)
    return stats

def main():
    if NUM_EXPERIMENTS < 1:
        raise ValueError('NUM_EXPERIMENTS must be >= 1')
    verify_pytorch_numpy_equivalence()
    assert sum((p.numel() for p in GeneralTinyAttentionTransformer().parameters())) == NUM_PARAMETERS
    true_model_diagnostics()
    rows = []
    for repetition in range(NUM_EXPERIMENTS):
        seed = BASE_SEED + repetition
        rng = np.random.default_rng(seed)
        x = simulate_characteristics(rng, TRAIN_N)
        y = simulate_responses(rng, x)
        x_test = simulate_characteristics(rng, TEST_N)
        y_test = simulate_responses(rng, x_test)
        true_test_probability = class_probabilities(x_test, TRUE_THETA)
        bayes_error = float(np.mean(1 - true_test_probability.max(axis=1)))
        bayes_sampled_error = float(np.mean(true_test_probability.argmax(axis=1) != y_test))
        print(f'\nRepetition {repetition + 1}/{NUM_EXPERIMENTS}: clean training', flush=True)
        clean_fits = fit_all(x, y, seed)
        for rate in CONTAMINATION_RATES:
            print(f'\nRepetition {repetition + 1}: contamination {rate:.0%}', flush=True)
            count = int(round(TRAIN_N * rate))
            if count == 0:
                bad_fits = clean_fits
            else:
                cx, cy, _ = simulate_candidate_contamination(rng, count, FIXED_CONTAMINATION_TYPE, FIXED_CONTAMINATION_SCALE, FIXED_CONTAMINATION_LABEL_MODE)
                bad_x = np.vstack([x[:TRAIN_N - count], cx])
                bad_y = np.concatenate([y[:TRAIN_N - count], cy])
                bad_fits = fit_all(bad_x, bad_y, seed)
            for name, bad_fit in bad_fits.items():
                row = make_row(seed, rate, name, clean_fits[name], bad_fit, x_test, y_test)
                row['Bayes_error_test'] = bayes_error
                row['Bayes_sampled_error_test'] = bayes_sampled_error
                rows.append(row)
                print(f"{name:18s} clean H²={row['Hellinger_squared_clean']:.6f}  contaminated H²={row['Hellinger_squared_contaminated']:.6f}")
                if name != 'rho' and (not bad_fit.success or not clean_fits[name].success):
                    print('  WARNING: optimizer did not report success; inspect CSV diagnostics.')
            pd.DataFrame(rows).to_csv(CSV_FILE, index=False)
            print(f'Checkpoint saved: {CSV_FILE}', flush=True)
    frame = pd.DataFrame(rows)
    metrics = ['Hellinger_squared_clean', 'Hellinger_squared_contaminated', 'Hellinger_squared_change', 'error_clean', 'error_contaminated', 'error_change']

    def q25(series):
        return series.quantile(0.25)

    def q75(series):
        return series.quantile(0.75)
    summary = frame.groupby(['contamination_rate', 'method'])[metrics].agg(['mean', 'std', q25, q75, 'count'])
    summary.columns = [f'{metric}_{stat}' for metric, stat in summary.columns]
    summary.to_csv(SUMMARY_FILE)
    plot_results(frame)
    print(f'Detailed results: {CSV_FILE}\nSummary: {SUMMARY_FILE}')
    return frame
if __name__ == '__main__':
    print(f'Five-class CPU experiment: {NUM_PARAMETERS} fitted parameters, {2 + len(MOM_BLOCK_COUNTS) + len(HUBER_DELTAS)} methods, {len(CONTAMINATION_RATES)} contamination rates; MOM optimization can be slow.')
    print('For an initial run, set NUM_EXPERIMENTS = 1.')
    results = main()

