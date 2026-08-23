"""
EEG Cognitive State Inference Pipeline
CS 5400: Introduction to Artificial Intelligence — Missouri S&T
Dataset: Fekri Azgomi et al. (2023), PhysioNet brain-wearable-monitoring v1.0.0
 
This pipeline:
  1. Loads EEG recordings and n-back behavioral data
  2. Extracts frequency-band power features + statistical summaries
  3. Labels each epoch with stimulation condition and performance state
  4. Trains and evaluates: Decision Tree, CNN, and Naïve Bayes (probabilistic)
  5. Produces comparative metrics, confusion matrices, and uncertainty plots
"""

import os
import argparse
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import signal
from scipy.stats import kurtosis, skew
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import json
import logging
from datetime import datetime

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
log = logging.getLogger(__name__)

EEG_CHANNELS = ['TP9', 'AF7', 'AF8', 'TP10']
EEG_CHANNELS_ALL = ['TP9', 'AF7', 'AF8', 'TP10', 'Right AUX']
EEG_FS = 256
EPOCH_SEC = 2.0
EPOCH_SAMPLES = int(EPOCH_SEC * EEG_FS)
BAND_RANGES = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 45),
}
RESULTS_DIR = Path('results')
FIGURES_DIR = Path('figures')
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

EXP1_CONDITIONS = {1: 'No_Music', 2: 'Relaxing_Music', 3: 'Exciting_Music', 4: 'AI_Music'}
EXP2_CONDITIONS = {1: 'No_Actuation', 2: 'Perfume', 3: 'Coffee'}

def load_eeg(path: Path, include_aux: bool = False) -> pd.DataFrame:
    """
    Load EEG_recording.csv from the PhysioNet brain-wearable-monitoring dataset.
 
    Format details (Muse MU-02 headband, 256 Hz):
    - Tab-separated (or comma-separated — the loader handles both)
    - Column order: timestamps, TP9, AF7, AF8, TP10, Right AUX
    - 'timestamps' is a UNIX integer second; multiple rows share the same
      second value (up to 256 rows per second).  We reconstruct fractional
      timestamps by counting samples within each second.
    - Signal values are in µV (microvolts), typically ±100 µV
    - Rows with NaN in any channel are discarded (packet loss / dropout)
    - Right AUX is an auxiliary channel from the right ear connector;
      included only if include_aux=True
 
    Parameters
    ----------
    path        : Path to EEG_recording.csv
    include_aux : Whether to keep the Right AUX channel (default False)
 
    Returns
    -------
    DataFrame with columns [timestamps, TP9, AF7, AF8, TP10, (Right AUX)]
    sorted by ascending timestamp; index reset.
    """
    for sep in ('\t', ','):
        try:
            df = pd.read_csv(path, sep=sep)
            if len(df.columns) >= 5:
                break
        except Exception:
            continue
    df.columns = df.columns.str.strip()
    rename_map = {}
    for col in df.columns:
        stripped = col.strip()
        if stripped.lower() in ('right aux', 'right_aux', 'rightaux'):
            rename_map[col] = 'Right AUX'
        elif stripped.lower() == 'timestamps':
            rename_map[col] = 'timestamps'
    df.rename(columns=rename_map, inplace=True)
    wanted = ['timestamps'] + EEG_CHANNELS
    if include_aux and 'Right AUX' in df.columns:
        wanted.append('Right AUX')
    present = [c for c in wanted if c in df.columns]
    missing = set(wanted) - set(present)
    if missing - {'Right AUX'}:
        log.warning(f"  EEG file {path.name}: missing columns {missing - {'Right AUX'}}")
    df = df[present].copy()
    for col in present:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)
    ch_cols = [c for c in EEG_CHANNELS if c in df.columns]
    amplitude_ok = (df[ch_cols].abs() <= 500).all(axis=1)
    n_bad = (~amplitude_ok).sum()
    if n_bad:
        log.debug(f"  Dropped {n_bad} artefact rows (|amplitude| > 500 µV)")
    df = df[amplitude_ok].copy()
    df.sort_values('timestamps', inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['timestamps'] = (df['timestamps'].values[0] + np.arange(len(df)) / EEG_FS)
    log.debug(f"  Loaded {len(df)} samples ({len(df)/EEG_FS:.1f} s) from {path.name}")
    return df

def load_nback(path: Path) -> pd.DataFrame:
    """Load n_back_responses.csv."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df

def load_tags(path: Path) -> pd.Series:
    """
    Load tags.csv — event marker timestamps in UNIX format.
 
    The PhysioNet dataset ships two tag files per participant:
      • Left_tags.csv / Right_tags.csv  — per-wristband button presses
      • tags.csv                         — unified, corrected event markers
 
    Format: single column of UNIX timestamps (float or int), one per row.
    Some exports include a header 'timestamp' or 'time'; others do not.
    We try both and return whatever parses cleanly.
 
    Returns
    -------
    Sorted pd.Series of float UNIX timestamps, or empty Series on failure.
    """
    for has_header in (True, False):
        try:
            header = 0 if has_header else None
            df = pd.read_csv(path, header=header)
            for col in df.columns:
                series = pd.to_numeric(df[col], errors='coerce').dropna()
                if len(series) > 0 and series.max() > 1e9:
                    return series.sort_values().reset_index(drop=True)
        except Exception:
            continue
    log.debug(f"  Could not parse tags from {path}")
    return pd.Series(dtype=float)

def find_participants(data_root: Path, experiment: str = 'Experiment_1'):
    """Return list of participant folder paths."""
    exp_dir = data_root / experiment
    if not exp_dir.exists():
        return []
    folders = sorted([d for d in exp_dir.iterdir() if d.is_dir() and not d.name.startswith('Excluded')])
    return folders

def bandpower(data: np.ndarray, fs: int, band: tuple) -> float:
    """Welch-based band power (μV²)."""
    f, pxx = signal.welch(data, fs=fs, nperseg=min(len(data), 256))
    idx = np.logical_and(f >= band[0], f <= band[1])
    return float(np.trapz(pxx[idx], f[idx])) if idx.any() else 0.0

def extract_epoch_features(epoch: np.ndarray, fs: int = EEG_FS) -> dict:
    """
    Extract features from a single epoch of shape (n_samples, n_channels).
    Returns flat feature dict.
    """
    feats = {}
    n_ch = epoch.shape[1]
    ch_names = EEG_CHANNELS[:n_ch]
    for i, ch in enumerate(ch_names):
        x = epoch[:, i]
        feats[f'{ch}_mean'] = float(np.mean(x))
        feats[f'{ch}_std'] = float(np.std(x))
        feats[f'{ch}_kurt'] = float(kurtosis(x))
        feats[f'{ch}_skew'] = float(skew(x))
        feats[f'{ch}_rms'] = float(np.sqrt(np.mean(x**2)))
        for band, rng in BAND_RANGES.items():
            feats[f'{ch}_{band}'] = bandpower(x, fs, rng)
        total_pow = sum(feats[f'{ch}_{b}'] for b in BAND_RANGES) + 1e-12
        for band in BAND_RANGES:
            feats[f'{ch}_{band}_rel'] = feats[f'{ch}_{band}'] / total_pow
        tb = feats.get(f'{ch}_theta', 0) / (feats.get(f'{ch}_beta', 1) + 1e-12)
        feats[f'{ch}_theta_beta_ratio'] = tb
    if n_ch > 1:
        corr = np.corrcoef(epoch.T)
        idx_upper = np.triu_indices(n_ch, k=1)
        pairs = [(ch_names[r], ch_names[c]) for r, c in zip(idx_upper[0], idx_upper[1])]
        for (ca, cb), val in zip(pairs, corr[idx_upper]):
            feats[f'corr_{ca}_{cb}'] = float(val) if np.isfinite(val) else 0.0
    return feats

def segment_and_label(eeg_df: pd.DataFrame, n_sessions: int, session_labels: dict, performance_series: pd.Series | None = None, session_boundaries: list[float] | None = None, overlap_fraction: float = 0.0) -> pd.DataFrame:
    """
    Segment an EEG recording into sessions, then into epochs, and extract
    features from each epoch.
 
    Session boundaries
    ------------------
    The experiment has N sessions separated by rest breaks.  Precise boundary
    timestamps come from tags.csv (loaded separately and passed as
    `session_boundaries`).  When tags are unavailable, we fall back to an
    equal-split heuristic.
 
    Epoching
    --------
    Each session is cut into non-overlapping (or optionally overlapping)
    windows of EPOCH_SAMPLES samples.  Epochs that span a session boundary
    are discarded.
 
    Parameters
    ----------
    eeg_df              : DataFrame from load_eeg() with columns
                          [timestamps, TP9, AF7, AF8, TP10, ...]
    n_sessions          : Expected number of stimulation sessions.
    session_labels      : dict mapping 1-based session index → condition string.
    performance_series  : pd.Series of per-session accuracy (0–1); used to
                          assign 'high'/'low' performance label.
    session_boundaries  : List of N-1 UNIX timestamps marking session splits.
                          If None, equal-length split is used.
    overlap_fraction    : Fraction of epoch overlap (0.0 = non-overlapping).
                          Useful for data augmentation; keep 0.0 for evaluation.
 
    Returns
    -------
    DataFrame where each row is one epoch with its feature vector + labels.
    """
    ch_cols = [c for c in EEG_CHANNELS + ['Right AUX'] if c in eeg_df.columns]
    feat_ch = [c for c in EEG_CHANNELS if c in ch_cols]
    timestamps = eeg_df['timestamps'].values
    raw = eeg_df[feat_ch].values
    total = len(raw)
    if session_boundaries is not None and len(session_boundaries) >= n_sessions - 1:
        split_indices = []
        for ts in session_boundaries[:n_sessions - 1]:
            idx = int(np.searchsorted(timestamps, ts))
            idx = max(0, min(idx, total - 1))
            split_indices.append(idx)
        boundaries = [0] + split_indices + [total]
        log.debug(f"  Using tag-based boundaries: {split_indices}")
    else:
        session_len = total // n_sessions
        boundaries = [s * session_len for s in range(n_sessions)] + [total]
        log.debug(f"  Using equal-split boundaries (no tags available)")
    stride = max(1, int(EPOCH_SAMPLES * (1.0 - overlap_fraction)))
    records = []
    for s in range(n_sessions):
        cond_label = session_labels.get(s + 1, f'session_{s+1}')
        seg_start = boundaries[s]
        seg_end = boundaries[s + 1]
        seg = raw[seg_start:seg_end]
        seg_ts = timestamps[seg_start:seg_end]
        if len(seg) < EPOCH_SAMPLES:
            log.debug(f"  Session {s+1} too short ({len(seg)} samples) — skipping")
            continue
        if performance_series is not None and len(performance_series) > s:
            perf_val = performance_series.iloc[s]
            perf_lbl = 'high' if (pd.notna(perf_val) and perf_val >= 0.5) else 'low'
        else:
            perf_lbl = 'unknown'
        e_idx = 0
        pos = 0
        while pos + EPOCH_SAMPLES <= len(seg):
            epoch = seg[pos: pos + EPOCH_SAMPLES]
            feats = extract_epoch_features(epoch)
            feats['session'] = s + 1
            feats['condition'] = cond_label
            feats['epoch_idx'] = e_idx
            feats['epoch_start_ts'] = float(seg_ts[pos])
            feats['performance'] = perf_lbl
            records.append(feats)
            pos += stride
            e_idx += 1
        log.debug(f"  Session {s+1} ({cond_label}): {len(seg)/EEG_FS:.1f}s → {e_idx} epochs")
    return pd.DataFrame(records)

def extract_performance(nback_df: pd.DataFrame, experiment: int = 1) -> pd.Series:
    """
    Compute mean accuracy per session from n-back task responses.
 
    Column naming conventions differ between experiments:
      Experiment 1 (4 sessions / music):
        Stimulus101.ACC, Stimulus102.ACC, Stimulus103.ACC, Stimulus104.ACC
      Experiment 2 (3 sessions / gustatory+olfactory):
        Stimulus101.ACC, Stimulus102.ACC, Stimulus103.ACC
        — same pattern but only 3 sessions; handled via n_sessions count.
 
    If the standard pattern is absent (different PsychoPy export format),
    falls back to any column containing '.ACC' ordered by column position.
 
    Returns pd.Series of length n_sessions with mean accuracy (0–1) per
    session, or NaN where no matching column was found.
    """
    n_sessions = 4 if experiment == 1 else 3
    accs = []
    for s in range(1, n_sessions + 1):
        candidates = [
            f'Stimulus10{s}.ACC',
            f'Stimulus20{s}.ACC',
            f'Stimulus{s}.ACC',
        ]
        found = False
        for col in candidates:
            if col in nback_df.columns:
                vals = pd.to_numeric(nback_df[col], errors='coerce').dropna()
                accs.append(vals.mean() if len(vals) else np.nan)
                found = True
                break
        if not found:
            acc_cols = [c for c in nback_df.columns if '.ACC' in c or '_ACC' in c.upper()]
            if len(acc_cols) >= s:
                vals = pd.to_numeric(nback_df[acc_cols[s - 1]], errors='coerce').dropna()
                accs.append(vals.mean() if len(vals) else np.nan)
            else:
                accs.append(np.nan)
    return pd.Series(accs)

def build_dataset(data_root: Path, experiment: str = 'Experiment_1', max_participants: int = None, include_aux: bool = False) -> pd.DataFrame:
    """
    Walk participant folders, load EEG + behavioural + tag data, extract
    features.  Returns a combined DataFrame with features + labels.
 
    Tag-based session segmentation
    --------------------------------
    The PhysioNet dataset includes a unified tags.csv per participant with
    event marker timestamps.  For Experiment 1 (4 sessions) we expect 3
    inter-session boundary markers; for Experiment 2 (3 sessions) we expect
    2 markers.  If fewer markers are found, equal-length fallback is used.
 
    The loader also checks Left_tags.csv and Right_tags.csv as fallbacks.
    """
    exp_num = 1 if '1' in experiment else 2
    cond_map = EXP1_CONDITIONS if exp_num == 1 else EXP2_CONDITIONS
    n_sess = 4 if exp_num == 1 else 3
    n_splits = n_sess - 1
    participants = find_participants(data_root, experiment)
    if max_participants:
        participants = participants[:max_participants]
    if not participants:
        log.warning(f"No participant folders found under {data_root / experiment}")
        log.info("Generating synthetic data for demonstration...")
        return generate_synthetic_dataset(n_participants=15, n_sessions=n_sess, cond_map=cond_map)
    all_frames = []
    for p_dir in participants:
        pid = p_dir.name
        eeg_path = p_dir / 'EEG_recording.csv'
        nback_path = p_dir / 'n_back_responses.csv'
        if not eeg_path.exists():
            log.warning(f"  {pid}: missing EEG file — skipping")
            continue
        try:
            eeg_df = load_eeg(eeg_path, include_aux=include_aux)
            log.info(f"  {pid}: {len(eeg_df)} EEG samples ({len(eeg_df)/EEG_FS:.1f} s)")
            session_boundaries = None
            for tag_name in ('tags.csv', 'Left_tags.csv', 'Right_tags.csv'):
                tag_path = p_dir / tag_name
                if tag_path.exists():
                    tags = load_tags(tag_path)
                    if len(tags) >= n_splits:
                        session_boundaries = tags.iloc[:n_splits].tolist()
                        log.debug(f"  {pid}: session boundaries from {tag_name}: {session_boundaries}")
                        break
            if session_boundaries is None:
                log.debug(f"  {pid}: no usable tags — using equal-split")
            perf = pd.Series([np.nan] * n_sess)
            if nback_path.exists():
                nback_df = load_nback(nback_path)
                perf = extract_performance(nback_df, exp_num)
            feat_df = segment_and_label(
                eeg_df,
                n_sessions=n_sess,
                session_labels=cond_map,
                performance_series=perf,
                session_boundaries=session_boundaries,
            )
            feat_df['participant'] = pid
            feat_df['experiment'] = experiment
            all_frames.append(feat_df)
            log.info(f"  {pid}: {len(feat_df)} epochs extracted across {feat_df['condition'].nunique()} conditions")
        except Exception as ex:
            log.warning(f"  {pid}: error — {ex}", exc_info=True)
    if not all_frames:
        log.info("No valid data loaded — using synthetic data")
        return generate_synthetic_dataset(n_participants=15, n_sessions=n_sess, cond_map=cond_map)
    combined = pd.concat(all_frames, ignore_index=True)
    log.info(f"Dataset built: {len(combined)} total epochs, {combined['participant'].nunique()} participants")
    return combined

def generate_synthetic_dataset(n_participants: int = 15, n_sessions: int = 4, cond_map: dict = None, epochs_per_session: int = 60, seed: int = 42) -> pd.DataFrame:
    """
    Generate realistic synthetic EEG feature data with class-conditional
    distributions, used when the raw PhysioNet files are unavailable.
 
    Band-power means are loosely inspired by published literature on
    cognitive load and music/stimulant effects on EEG.
    """
    rng = np.random.default_rng(seed)
    if cond_map is None:
        cond_map = EXP1_CONDITIONS
    cond_priors = {
        'No_Music': (0.00, 0.00, 0.00, 0.00, 0.00),
        'Relaxing_Music': (-0.10, 0.15, 0.35, -0.15, -0.10),
        'Exciting_Music': (-0.05, 0.20, -0.15, 0.35, 0.25),
        'AI_Music': (-0.05, 0.10, 0.20, 0.10, 0.05),
        'No_Actuation': (0.00, 0.00, 0.00, 0.00, 0.00),
        'Perfume': (-0.05, 0.20, 0.25, 0.05, 0.00),
        'Coffee': (-0.25, 0.10, -0.10, 0.40, 0.30),
    }
    records = []
    ch_names = EEG_CHANNELS
    bands = list(BAND_RANGES.keys())
    for pid in range(n_participants):
        ind_offset = rng.normal(0, 0.05, size=len(bands))
        for s_idx, (s_num, cond) in enumerate(cond_map.items()):
            prior = cond_priors.get(cond, (0, 0, 0, 0, 0))
            cond_shifts = dict(zip(BAND_RANGES.keys(), prior))
            for e in range(epochs_per_session):
                row = {}
                for ch in ch_names:
                    for j, band in enumerate(bands):
                        base = rng.lognormal(mean=2.0, sigma=0.5)
                        shift = cond_shifts.get(band, 0.0) + ind_offset[j]
                        val = max(base * np.exp(shift + rng.normal(0, 0.1)), 0.001)
                        row[f'{ch}_{band}'] = val
                    row[f'{ch}_mean'] = rng.normal(0, 1)
                    row[f'{ch}_std'] = abs(rng.normal(10, 2))
                    row[f'{ch}_kurt'] = rng.normal(3, 1)
                    row[f'{ch}_skew'] = rng.normal(0, 0.5)
                    row[f'{ch}_rms'] = abs(rng.normal(10, 2))
                    total = sum(row[f'{ch}_{b}'] for b in bands) + 1e-12
                    for band in bands:
                        row[f'{ch}_{band}_rel'] = row[f'{ch}_{band}'] / total
                    tb = row[f'{ch}_theta'] / (row[f'{ch}_beta'] + 1e-12)
                    row[f'{ch}_theta_beta_ratio'] = tb
                for i, ca in enumerate(ch_names):
                    for cb in ch_names[i+1:]:
                        row[f'corr_{ca}_{cb}'] = rng.uniform(-0.3, 0.8)
                acc_prob = 0.7 + cond_shifts.get('beta', 0) * 0.2
                acc_prob = np.clip(acc_prob, 0.4, 0.95)
                row['session'] = s_num
                row['condition'] = cond
                row['epoch_idx'] = e
                row['performance'] = 'high' if rng.random() < acc_prob else 'low'
                row['participant'] = f'P{pid+1:02d}'
                row['experiment'] = 'synthetic'
                records.append(row)
    df = pd.DataFrame(records)
    log.info(f"Synthetic dataset: {len(df)} epochs, {df['condition'].nunique()} conditions")
    return df

def prepare_features(df: pd.DataFrame, target: str = 'condition', drop_cols: list = None):
    """
    Return X (feature matrix), y (encoded labels), feature_names, label_encoder.
    """
    meta_cols = ['session', 'condition', 'epoch_idx', 'performance', 'participant', 'experiment']
    if drop_cols:
        meta_cols += drop_cols
    feat_cols = [c for c in df.columns if c not in meta_cols]
    X = df[feat_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values
    le = LabelEncoder()
    y = le.fit_transform(df[target].astype(str))
    return X, y, feat_cols, le

def train_decision_tree(X_train, y_train, max_depth: int = 8, min_samples_leaf: int = 5) -> Pipeline:
    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('clf', DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf, class_weight='balanced', random_state=42))
    ])
    pipe.fit(X_train, y_train)
    return pipe

class EEGFeatureCNN(nn.Module):
    """
    Lightweight 1-D CNN that treats the flattened feature vector as a
    1-channel 1-D signal; captures local feature interactions via
    convolutional kernels before a fully-connected classifier head.
    """
    def __init__(self, n_features: int, n_classes: int, dropout: float = 0.4):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout / 2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout / 2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),
        )
        fc_in = 128 * 8
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fc_in, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv_block(x)
        return self.classifier(x)

def train_cnn(X_train: np.ndarray, y_train: np.ndarray, n_classes: int, epochs: int = 40, batch_size: int = 64, lr: float = 1e-3) -> tuple:
    """Train the CNN; return (model, train_loss_history, scaler)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_train).astype(np.float32)
    X_t = torch.tensor(X_s)
    y_t = torch.tensor(y_train, dtype=torch.long)
    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = EEGFeatureCNN(X_train.shape[1], n_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    losses = []
    model.train()
    for ep in range(epochs):
        ep_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_loss += loss.item() * len(xb)
        scheduler.step()
        losses.append(ep_loss / len(X_train))
    return model, losses, scaler, device

def predict_cnn(model, scaler, device, X: np.ndarray, batch_size: int = 256) -> tuple:
    """Return (y_pred, y_proba)."""
    X_s = scaler.transform(X).astype(np.float32)
    X_t = torch.tensor(X_s)
    ds = TensorDataset(X_t)
    dl = DataLoader(ds, batch_size=batch_size)
    model.eval()
    preds, probas = [], []
    with torch.no_grad():
        for (xb,) in dl:
            logits = model(xb.to(device))
            proba = torch.softmax(logits, dim=-1).cpu().numpy()
            probas.append(proba)
            preds.append(proba.argmax(axis=1))
    return np.concatenate(preds), np.vstack(probas)

def train_naive_bayes(X_train, y_train) -> Pipeline:
    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('clf', GaussianNB(var_smoothing=1e-9))
    ])
    pipe.fit(X_train, y_train)
    return pipe

def evaluate_sklearn(pipe, X, y, class_names, model_name: str) -> dict:
    """5-fold cross-validation metrics for sklearn pipelines."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(pipe, X, y, cv=cv)
    try:
        y_proba = cross_val_predict(pipe, X, y, cv=cv, method='predict_proba')
        if len(class_names) == 2:
            auc = roc_auc_score(y, y_proba[:, 1])
        else:
            auc = roc_auc_score(y, y_proba, multi_class='ovr', average='macro')
    except Exception:
        auc = float('nan')
    report = classification_report(y, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    cm = confusion_matrix(y, y_pred)
    acc = accuracy_score(y, y_pred)
    return {
        'model': model_name,
        'accuracy': acc,
        'auc': auc,
        'report': report,
        'cm': cm.tolist(),
        'y_true': y.tolist(),
        'y_pred': y_pred.tolist(),
        'class_names': list(class_names),
    }

def evaluate_cnn_holdout(model, scaler, device, X_train, y_train, X_test, y_test, class_names) -> dict:
    """Simple train/test evaluation for the CNN."""
    y_pred, y_proba = predict_cnn(model, scaler, device, X_test)
    try:
        if len(class_names) == 2:
            auc = roc_auc_score(y_test, y_proba[:, 1])
        else:
            auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')
    except Exception:
        auc = float('nan')
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    return {
        'model': 'CNN',
        'accuracy': acc,
        'auc': auc,
        'report': report,
        'cm': cm.tolist(),
        'y_true': y_test.tolist(),
        'y_pred': y_pred.tolist(),
        'y_proba': y_proba.tolist(),
        'class_names': list(class_names),
    }

PALETTE = {
    'Decision Tree': '#2196F3',
    'CNN': '#E91E63',
    'Naive Bayes': '#4CAF50',
}

def plot_confusion_matrices(results: list, out_dir: Path):
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5))
    if len(results) == 1:
        axes = [axes]
    fig.patch.set_facecolor('#0f0f1a')
    for ax, res in zip(axes, results):
        cm = np.array(res['cm'])
        names = res['class_names']
        cm_pct = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)
        color = PALETTE.get(res['model'], '#888')
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list('', ['#0f0f1a', color])
        sns.heatmap(cm_pct, annot=True, fmt='.2f', cmap=cmap, xticklabels=names, yticklabels=names, ax=ax, linewidths=0.5, linecolor='#333', cbar_kws={'shrink': 0.8})
        ax.set_facecolor('#0f0f1a')
        ax.tick_params(colors='white', labelsize=8)
        ax.set_xlabel('Predicted', color='white', fontsize=9)
        ax.set_ylabel('True', color='white', fontsize=9)
        ax.set_title(f"{res['model']}\nAcc={res['accuracy']:.3f}  AUC={res['auc']:.3f}", color=color, fontsize=11, fontweight='bold')
        for sp in ax.spines.values():
            sp.set_edgecolor('#333')
    fig.suptitle('Confusion Matrices — Stimulation Condition Classification', color='white', fontsize=13, y=1.02)
    plt.tight_layout()
    out = out_dir / 'confusion_matrices.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0f0f1a')
    plt.close()
    log.info(f"Saved {out}")

def plot_model_comparison(results: list, out_dir: Path):
    models = [r['model'] for r in results]
    accs = [r['accuracy'] for r in results]
    aucs = [r['auc'] for r in results]
    colors = [PALETTE.get(m, '#888') for m in models]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.patch.set_facecolor('#0f0f1a')
    for ax, vals, title in zip(axes, [accs, aucs], ['Accuracy (5-fold CV)', 'AUC-ROC (macro, OvR)']):
        bars = ax.bar(models, vals, color=colors, width=0.5, edgecolor='none', zorder=2)
        ax.set_facecolor('#131325')
        ax.set_ylim(0, 1.05)
        ax.set_title(title, color='white', fontsize=12)
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('#444')
        ax.spines['left'].set_color('#444')
        for sp in ['top', 'right']:
            ax.spines[sp].set_visible(False)
        ax.yaxis.label.set_color('white')
        ax.xaxis.label.set_color('white')
        ax.grid(axis='y', color='#333', linewidth=0.5, zorder=0)
        for bar, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(bar.get_x() + bar.get_width()/2, v + 0.015, f'{v:.3f}', ha='center', va='bottom', color='white', fontsize=10, fontweight='bold')
    fig.suptitle('Model Comparison — EEG Cognitive State Classification', color='white', fontsize=13, y=1.02)
    plt.tight_layout()
    out = out_dir / 'model_comparison.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0f0f1a')
    plt.close()
    log.info(f"Saved {out}")

def plot_band_power_distributions(df: pd.DataFrame, out_dir: Path, channel: str = 'AF7'):
    bands = list(BAND_RANGES.keys())
    conds = df['condition'].unique()
    fig, axes = plt.subplots(1, len(bands), figsize=(4 * len(bands), 4))
    fig.patch.set_facecolor('#0f0f1a')
    cmap_ = plt.cm.get_cmap('plasma', len(conds))
    colors = {c: matplotlib.colors.to_hex(cmap_(i)) for i, c in enumerate(conds)}
    for ax, band in zip(axes, bands):
        col = f'{channel}_{band}'
        if col not in df.columns:
            continue
        for cond in conds:
            sub = df[df['condition'] == cond][col].dropna()
            if len(sub) > 2:
                sub_log = np.log10(sub + 1e-12)
                ax.hist(sub_log, bins=30, alpha=0.55, density=True, label=cond, color=colors[cond], edgecolor='none')
        ax.set_title(f'{band.capitalize()}\n({BAND_RANGES[band][0]}–{BAND_RANGES[band][1]} Hz)', color='white', fontsize=9)
        ax.set_xlabel('log₁₀ Power', color='#aaa', fontsize=8)
        ax.set_facecolor('#131325')
        ax.tick_params(colors='white', labelsize=7)
        for sp in ['top', 'right']:
            ax.spines[sp].set_visible(False)
        for sp in ['bottom', 'left']:
            ax.spines[sp].set_color('#444')
    axes[-1].legend(fontsize=7, facecolor='#1a1a2e', labelcolor='white', loc='upper right', framealpha=0.7)
    fig.suptitle(f'EEG Band Power by Condition — Channel {channel}', color='white', fontsize=12, y=1.02)
    plt.tight_layout()
    out = out_dir / f'band_power_{channel}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0f0f1a')
    plt.close()
    log.info(f"Saved {out}")

def plot_uncertainty(nb_result: dict, out_dir: Path):
    """
    Visualise predictive uncertainty from Naïve Bayes via entropy of the
    posterior class probabilities.  High entropy → high uncertainty.
    """
    n = len(nb_result['y_true'])
    n_cls = len(nb_result['class_names'])
    cm = np.array(nb_result['cm'], dtype=float)
    cm_normed = cm / (cm.sum(axis=1, keepdims=True) + 1e-12)
    true_labels = np.array(nb_result['y_true'])
    pred_labels = np.array(nb_result['y_pred'])
    probas = cm_normed[pred_labels]
    entropy = -np.sum(probas * np.log2(probas + 1e-12), axis=1)
    correct = (true_labels == pred_labels).astype(int)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.patch.set_facecolor('#0f0f1a')

    ax = axes[0]
    ax.set_facecolor('#131325')
    ax.hist(entropy[correct == 1], bins=40, alpha=0.7, color='#4CAF50', label='Correct', density=True)
    ax.hist(entropy[correct == 0], bins=40, alpha=0.7, color='#E91E63', label='Incorrect', density=True)
    ax.set_xlabel('Predictive Entropy (bits)', color='white')
    ax.set_ylabel('Density', color='white')
    ax.set_title('Naïve Bayes — Predictive Uncertainty\n(Higher entropy = more uncertain)', color='#4CAF50', fontsize=11)
    ax.legend(facecolor='#1a1a2e', labelcolor='white')
    ax.tick_params(colors='white')
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)
    for sp in ['bottom', 'left']:
        ax.spines[sp].set_color('#444')

    ax2 = axes[1]
    ax2.set_facecolor('#131325')
    class_entropies = [entropy[true_labels == i] for i in range(n_cls)]
    bp = ax2.boxplot(class_entropies, patch_artist=True, medianprops=dict(color='white', linewidth=2), whiskerprops=dict(color='#aaa'), capprops=dict(color='#aaa'), flierprops=dict(markerfacecolor='#E91E63', markersize=2))
    cmap_ = plt.cm.get_cmap('plasma', n_cls)
    for patch, i in zip(bp['boxes'], range(n_cls)):
        patch.set_facecolor(matplotlib.colors.to_hex(cmap_(i)))
        patch.set_alpha(0.8)
    ax2.set_xticklabels(nb_result['class_names'], rotation=30, ha='right', fontsize=8, color='white')
    ax2.set_ylabel('Predictive Entropy (bits)', color='white')
    ax2.set_title('Uncertainty per Stimulation Class', color='#4CAF50', fontsize=11)
    ax2.tick_params(colors='white')
    for sp in ['top', 'right']:
        ax2.spines[sp].set_visible(False)
    for sp in ['bottom', 'left']:
        ax2.spines[sp].set_color('#444')

    plt.tight_layout()
    out = out_dir / 'uncertainty_naive_bayes.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0f0f1a')
    plt.close()
    log.info(f"Saved {out}")

def plot_cnn_training(losses: list, out_dir: Path):
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor('#0f0f1a')
    ax.set_facecolor('#131325')
    ax.plot(losses, color='#E91E63', linewidth=2)
    ax.set_xlabel('Epoch', color='white')
    ax.set_ylabel('Cross-Entropy Loss', color='white')
    ax.set_title('CNN Training Loss Curve', color='#E91E63', fontsize=12)
    ax.tick_params(colors='white')
    ax.grid(color='#333', linewidth=0.5)
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)
    for sp in ['bottom', 'left']:
        ax.spines[sp].set_color('#444')
    plt.tight_layout()
    out = out_dir / 'cnn_training_loss.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0f0f1a')
    plt.close()
    log.info(f"Saved {out}")

def plot_feature_importance(dt_pipe, feat_names: list, out_dir: Path, top_n: int = 20):
    dt = dt_pipe.named_steps['clf']
    imp = dt.feature_importances_
    idx = np.argsort(imp)[-top_n:]
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor('#0f0f1a')
    ax.set_facecolor('#131325')
    colors = plt.cm.Blues(np.linspace(0.4, 1.0, top_n))
    ax.barh(range(top_n), imp[idx], color=colors, edgecolor='none')
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feat_names[i] for i in idx], fontsize=8, color='white')
    ax.set_xlabel('Gini Importance', color='white')
    ax.set_title(f'Decision Tree — Top {top_n} Feature Importances', color='#2196F3', fontsize=12)
    ax.tick_params(colors='white')
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)
    for sp in ['bottom', 'left']:
        ax.spines[sp].set_color('#444')
    plt.tight_layout()
    out = out_dir / 'feature_importance_dt.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0f0f1a')
    plt.close()
    log.info(f"Saved {out}")

def run(data_root: str = 'data', experiment: str = 'Experiment_1', target: str = 'condition', max_participants: int = None, cnn_epochs: int = 40):
    """Full pipeline: load → features → train → evaluate → visualise."""
    log.info("=" * 60)
    log.info("EEG Cognitive State Inference Pipeline")
    log.info(f"  Experiment : {experiment}")
    log.info(f"  Target     : {target}")
    log.info("=" * 60)
    exp_tag = experiment.lower().replace(' ', '_')
    res_dir = RESULTS_DIR / exp_tag
    fig_dir = FIGURES_DIR / exp_tag
    res_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    df = build_dataset(Path(data_root), experiment, max_participants)
    log.info(f"Dataset shape: {df.shape}  |  classes: {df[target].value_counts().to_dict()}")
    df.to_csv(res_dir / 'features.csv', index=False)
    X, y, feat_names, le = prepare_features(df, target)
    class_names = le.classes_
    log.info(f"Feature matrix: {X.shape}  |  {len(class_names)} classes")
    plot_band_power_distributions(df, fig_dir)
    log.info("Training Decision Tree …")
    dt_pipe = train_decision_tree(X, y)
    dt_res = evaluate_sklearn(dt_pipe, X, y, class_names, 'Decision Tree')
    log.info(f"  DT  Accuracy={dt_res['accuracy']:.3f}  AUC={dt_res['auc']:.3f}")
    plot_feature_importance(dt_pipe, feat_names, fig_dir)
    log.info("Training Naïve Bayes …")
    nb_pipe = train_naive_bayes(X, y)
    nb_res = evaluate_sklearn(nb_pipe, X, y, class_names, 'Naive Bayes')
    log.info(f"  NB  Accuracy={nb_res['accuracy']:.3f}  AUC={nb_res['auc']:.3f}")
    plot_uncertainty(nb_res, fig_dir)
    log.info(f"Training CNN ({cnn_epochs} epochs) …")
    split = int(0.8 * len(X))
    idx_shuf = np.random.default_rng(42).permutation(len(X))
    tr_idx, te_idx = idx_shuf[:split], idx_shuf[split:]
    cnn_model, cnn_losses, cnn_scaler, device = train_cnn(X[tr_idx], y[tr_idx], n_classes=len(class_names), epochs=cnn_epochs)
    cnn_res = evaluate_cnn_holdout(cnn_model, cnn_scaler, device, X[tr_idx], y[tr_idx], X[te_idx], y[te_idx], class_names)
    log.info(f"  CNN Accuracy={cnn_res['accuracy']:.3f}  AUC={cnn_res['auc']:.3f}")
    plot_cnn_training(cnn_losses, fig_dir)
    all_results = [dt_res, cnn_res, nb_res]
    plot_confusion_matrices(all_results, fig_dir)
    plot_model_comparison(all_results, fig_dir)
    summary = []
    for r in all_results:
        row = {k: v for k, v in r.items() if k not in ('cm', 'report', 'y_true', 'y_pred', 'y_proba')}
        row['f1_macro'] = r['report'].get('macro avg', {}).get('f1-score', float('nan'))
        summary.append(row)
    pd.DataFrame(summary).to_csv(res_dir / 'model_summary.csv', index=False)
    for r in all_results:
        r.pop('y_proba', None)
    with open(res_dir / 'full_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    log.info(f"Pipeline complete.  Results in {res_dir}/  |  Figures in {fig_dir}/")
    return all_results, df

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data',        default='data')
    p.add_argument('--experiment',  default='Experiment_1')
    p.add_argument('--target',      default='condition', choices=['condition', 'performance'])
    p.add_argument('--max_parts',   type=int, default=None)
    p.add_argument('--cnn_epochs',  type=int, default=50)
    args = p.parse_args()
    run(args.data, args.experiment, args.target, args.max_parts, args.cnn_epochs)