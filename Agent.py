"""
agent.py — Intelligent Agent Formulation + Probabilistic Reasoning
CS 5400: Introduction to Artificial Intelligence — Missouri S&T
 
Formalises the EEG classification system as an intelligent agent (Task 1)
and implements HMM-based temporal reasoning over EEG state sequences (Task 3).
 
Agent PEAS:
  Performance : Macro F1-score on stimulation-condition or cognitive-state labels
  Environment : EEG recordings collected during n-back memory tasks
  Actuators   : Classification output / alert / recommendation
  Sensors     : 4-channel Muse EEG (TP9, AF7, AF8, TP10) at 256 Hz
"""
 
from __future__ import annotations
from collections import Counter
from sklearn.metrics import f1_score, accuracy_score
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
from dataclasses import dataclass, field
from typing import Optional
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')
 
@dataclass
class AgentPercept:
    """
    A single percept delivered to the cognitive-state inference agent.
 
    In AIMA terminology a percept is the agent's perceptual input at one
    moment in time.  Here each percept is a feature vector computed from a
    2-second EEG epoch.
    """
    timestamp: float   
    features: np.ndarray             
    feature_names: list[str] = field(default_factory=list)
    raw_channels: Optional[np.ndarray] = None 
 
 
@dataclass
class AgentAction:
    """
    The action emitted by the agent after processing a percept.
 
    Actions include:
      - classify    : predict the stimulation condition label
      - alert       : flag a high-cognitive-load epoch
      - recommend   : suggest a regulatory intervention (closed-loop)
    """
    action_type: str 
    predicted_label: str  
    confidence: float 
    uncertainty: float
    recommendation: Optional[str] = None
 
 
class CognitiveStateAgent:
    """
    Simple reflex + model-based agent for EEG cognitive-state inference.
 
    Holds an internal model (fitted classifier) and a short history of
    recent predictions to support temporal smoothing via majority vote.
    """
 
    def __init__(self, classifier, class_names: list[str], history_len: int = 5, high_load_classes: Optional[list[str]] = None):
        """
        Parameters
        ----------
        classifier     : Any sklearn-compatible Pipeline with predict_proba.
        class_names    : Ordered list of class labels matching the encoder.
        history_len    : Number of recent predictions to smooth over.
        high_load_classes : Classes that represent high cognitive load;
                           used for 'alert' actions.
        """
        self.clf = classifier
        self.class_names = list(class_names)
        self.history_len = history_len
        self.high_load = set(high_load_classes or [])
        self._history: list[str] = []

    @staticmethod
    def performance_measure(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """
        Compute the agent's performance measure.
        Primary metric  : macro-averaged F1-score (handles class imbalance).
        Secondary metric: accuracy, useful for balanced datasets.
        """
        return {
            'macro_f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'accuracy': accuracy_score(y_true, y_pred),
        }
  
    def act(self, percept: AgentPercept) -> AgentAction:
        """Process one percept and return an action."""
        X = percept.features.reshape(1, -1)
        proba = self.clf.predict_proba(X)[0]
        pred_i = int(np.argmax(proba))
        label = self.class_names[pred_i]
        conf = float(proba[pred_i])
        ent = float(-np.sum(proba * np.log2(proba + 1e-12)))
 
        self._history.append(label)
        if len(self._history) > self.history_len:
            self._history.pop(0)
        smooth_label = Counter(self._history).most_common(1)[0][0]
        if smooth_label in self.high_load:
            a_type = 'alert'
            rec = self._intervention(smooth_label)
        else:
            a_type = 'classify'
            rec = None
        return AgentAction(action_type=a_type, predicted_label=smooth_label, confidence=conf, uncertainty=ent, recommendation=rec)
 
    def _intervention(self, label: str) -> str:
        """
        Rule-based closed-loop recommendation keyed on condition label.
        Covers both Experiment 1 (auditory) and Experiment 2 (gustatory/olfactory).
        """
        mapping = {
            'Exciting_Music':  'Switch to relaxing music to reduce arousal.',
            'No_Music':        'Introduce calming auditory stimulus.',
            'Relaxing_Music':  'Current stimulus is working — maintain relaxing music.',
            'AI_Music':        'AI-generated music detected; monitor for fatigue onset.',
            'Coffee':          'High beta activity detected; consider reducing caffeine '
                               'intake and taking a short movement break.',
            'Perfume':         'Olfactory stimulus active; watch for habituation '
                               '(diminishing effect after ~10 minutes).',
            'No_Actuation':    'No active stimulus — consider introducing a '
                               'calming olfactory or auditory cue.',
        }
        return mapping.get(label, f'Elevated cognitive load detected ({label}); ' f'consider a short break.')
  
    def batch_act(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (predicted_labels, entropies) for a full feature matrix."""
        proba = self.clf.predict_proba(X)
        pred_i = proba.argmax(axis=1)
        labels = np.array([self.class_names[i] for i in pred_i])
        ent = -np.sum(proba * np.log2(proba + 1e-12), axis=1)
        return labels, ent
 
 
# 2. PROBABILISTIC REASONING — Hidden Markov Model
class EEGCognitiveHMM:
    """
    Gaussian HMM for modelling temporal dynamics of EEG cognitive states.
 
    Treats EEG feature sequences as observations emitted by a latent Markov
    chain whose hidden states correspond to cognitive arousal levels.
 
    Use-case A : Unsupervised discovery of recurring brain micro-states.
    Use-case B : State decoding — align learned states to condition labels.
    Use-case C : Online smoothing — decode the most likely hidden state
                 sequence with the Viterbi algorithm.
    """
 
    def __init__(self, n_components: int = 4, n_iter: int = 100, covariance_type: str = 'diag', random_state: int = 42):
        self.n_components = n_components
        self.n_iter = n_iter
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.model: Optional[hmm.GaussianHMM] = None
        self.scaler = StandardScaler()
        self._fitted = False

    def fit(self, sequences: list[np.ndarray]) -> 'EEGCognitiveHMM':
        """
        Fit the HMM to a list of EEG feature sequences.
 
        Parameters
        ----------
        sequences : list of 2-D arrays, each (T_i, n_features).
                    Each array is one participant's session.
        """
        all_data = np.vstack(sequences)
        self.scaler.fit(all_data)
        scaled = [self.scaler.transform(s) for s in sequences]
        X_concat = np.vstack(scaled)
        lengths = [len(s) for s in scaled]
 
        self.model = hmm.GaussianHMM(
            n_components = self.n_components,
            covariance_type = self.covariance_type,
            n_iter = self.n_iter,
            random_state = self.random_state,
            verbose = False,
        )
        self.model.fit(X_concat, lengths)
        self._fitted = True
        return self
  
    def decode(self, sequence: np.ndarray) -> np.ndarray:
        """
        Viterbi decoding: return most likely hidden state sequence.
 
        Parameters
        ----------
        sequence : (T, n_features)
 
        Returns
        -------
        states : (T,) integer array of hidden states 0..n_components-1
        """
        self._check_fitted()
        X_s = self.scaler.transform(sequence)
        _, states = self.model.decode(X_s, algorithm='viterbi')
        return states
 
    def score(self, sequence: np.ndarray) -> float:
        """Log-likelihood of a sequence under the model (per sample)."""
        self._check_fitted()
        X_s = self.scaler.transform(sequence)
        return self.model.score(X_s) / len(X_s)
  
    def posterior(self, sequence: np.ndarray) -> np.ndarray:
        """
        Return (T, n_components) array of posterior state probabilities
        (forward–backward algorithm).
        """
        self._check_fitted()
        X_s = self.scaler.transform(sequence)
        posteriors = self.model.predict_proba(X_s)
        return posteriors
  
    def align_states_to_labels(self, sequences: list[np.ndarray], labels:    list[np.ndarray]) -> dict:
        """
        For each hidden state, find the most common true label among epochs
        where that state is the dominant decoded state.
 
        Returns dict mapping hidden-state-index → most-frequent-label.
        """
        self._check_fitted()
        state_label_counts: dict[int, dict] = {i: {} for i in range(self.n_components)}
 
        for seq, lbl in zip(sequences, labels):
            if len(seq) != len(lbl):
                min_len = min(len(seq), len(lbl))
                seq, lbl = seq[:min_len], lbl[:min_len]
            states = self.decode(seq)
            for s, l in zip(states, lbl):
                state_label_counts[s][l] = state_label_counts[s].get(l, 0) + 1
 
        alignment = {}
        for state, counts in state_label_counts.items():
            if counts:
                alignment[state] = max(counts, key=counts.get)
            else:
                alignment[state] = 'unknown'
        return alignment
  
    def transition_summary(self) -> pd.DataFrame:
        """Return the learned transition probability matrix as a DataFrame."""
        self._check_fitted()
        cols = [f'state_{i}' for i in range(self.n_components)]
        return pd.DataFrame(self.model.transmat_, index=cols, columns=cols)
 
    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("HMM must be fitted before calling this method.")
 
def run_hmm_analysis(df: pd.DataFrame, feature_cols: list[str], n_states: int = 4, figures_dir=None) -> dict:
    """
    Build per-participant EEG feature sequences; fit the HMM; decode states;
    evaluate alignment to ground-truth condition labels.
 
    Returns dict with model, alignment, log-likelihoods, decoded sequences.
    """
    matplotlib.use('Agg')
 
    label_encoder = {v: i for i, v in enumerate(df['condition'].unique())}
    sequences, label_seqs = [], []
 
    for pid, grp in df.groupby('participant'):
        grp_sorted = grp.sort_values(['session', 'epoch_idx'])
        feats = grp_sorted[feature_cols].apply(
            pd.to_numeric, errors='coerce').fillna(0).values
        lbls = np.array([label_encoder.get(c, 0) for c in grp_sorted['condition']])
        if len(feats) > 10:
            sequences.append(feats)
            label_seqs.append(lbls)
 
    if not sequences:
        return {'error': 'no sequences built'}
 
    hmm_model = EEGCognitiveHMM(n_components=n_states)
    hmm_model.fit(sequences)
 
    alignment = hmm_model.align_states_to_labels(sequences, label_seqs)
    log_likes = [hmm_model.score(s) for s in sequences]
    trans_df = hmm_model.transition_summary()
 
    sample_states = hmm_model.decode(sequences[0])
    sample_posterior = hmm_model.posterior(sequences[0])
 
    if figures_dir is not None:
        figures_dir = Path(figures_dir)
        fig, axes = plt.subplots(3, 1, figsize=(14, 9))
        fig.patch.set_facecolor('#0f0f1a')
        ax = axes[0]
        ax.set_facecolor('#131325')
        cmap_ = plt.cm.get_cmap('plasma', n_states)
        colors = [matplotlib.colors.to_hex(cmap_(i)) for i in range(n_states)]
        for i in range(n_states):
            mask = (sample_states == i)
            ax.fill_between(np.arange(len(sample_states)), 0, 1, where=mask, alpha=0.6, color=colors[i], label=f'State {i} → {alignment.get(i, "?")}')
        ax.set_yticks([])
        ax.set_xlim(0, len(sample_states))
        ax.set_title('Viterbi-Decoded Hidden State Sequence (Participant 1)', color='white', fontsize=11)
        ax.legend(loc='upper right', fontsize=8, facecolor='#1a1a2e', labelcolor='white', framealpha=0.8)
        ax.tick_params(colors='white')
        for sp in ['top', 'right']:
            ax.spines[sp].set_visible(False)
        for sp in ['bottom', 'left']:
            ax.spines[sp].set_color('#444')
        ax2 = axes[1]
        ax2.set_facecolor('#131325')
        for i in range(n_states):
            ax2.plot(sample_posterior[:, i], color=colors[i], linewidth=0.8, alpha=0.85, label=f'P(state {i})')
        ax2.set_ylabel('P(state | obs)', color='white', fontsize=9)
        ax2.set_xlim(0, len(sample_posterior))
        ax2.set_ylim(0, 1)
        ax2.set_title('Forward–Backward Posterior State Probabilities', color='white', fontsize=11)
        ax2.legend(loc='upper right', fontsize=8, facecolor='#1a1a2e', labelcolor='white', framealpha=0.8)
        ax2.tick_params(colors='white')
        for sp in ['top', 'right']:
            ax2.spines[sp].set_visible(False)
        for sp in ['bottom', 'left']:
            ax2.spines[sp].set_color('#444')
        ax3 = axes[2]
        sns.heatmap(trans_df, annot=True, fmt='.3f', cmap='Blues', ax=ax3, linewidths=0.5, linecolor='#333', cbar_kws={'shrink': 0.8})
        ax3.set_facecolor('#131325')
        ax3.set_title('Learned HMM Transition Matrix', color='white', fontsize=11)
        ax3.tick_params(colors='white', labelsize=8)
        plt.tight_layout()
        out = figures_dir / 'hmm_analysis.png'
        plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0f0f1a')
        plt.close()
 
    return {
        'model': hmm_model,
        'alignment': alignment,
        'log_likelihoods': log_likes,
        'mean_ll': float(np.mean(log_likes)),
        'transition': trans_df.to_dict(),
        'n_sequences': len(sequences),
    }