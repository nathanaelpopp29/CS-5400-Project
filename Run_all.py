"""
run_all.py — Master orchestration script
CS 5400: Introduction to Artificial Intelligence — Missouri S&T
 
Executes the full EEG cognitive-state inference pipeline:
  1. Data loading / synthetic generation
  2. Feature extraction
  3. Decision Tree, CNN, Naïve Bayes training + evaluation
  4. HMM temporal reasoning
  5. All figures and tables saved
 
Usage
-----
  # With PhysioNet data already downloaded to ./data/
  python run_all.py --data ./data
 
  # Without real data (uses realistic synthetic EEG features)
  python run_all.py
"""
 
import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
 
sys.path.insert(0, str(Path(__file__).parent / 'src'))
 
from Pipeline import run, prepare_features, FIGURES_DIR, RESULTS_DIR
from Agent    import CognitiveStateAgent, AgentPercept, run_hmm_analysis
 
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
log = logging.getLogger(__name__)
 
# High-load condition labels per experiment — used to trigger agent 'alert' actions
HIGH_LOAD_CONDITIONS = {'Experiment_1': ['Exciting_Music', 'No_Music'], 'Experiment_2': ['Coffee', 'No_Actuation']}
 
 
def main():
    import argparse
    p = argparse.ArgumentParser( description='EEG Cognitive State Inference — Full Pipeline')
    p.add_argument('--data', default='data', help='Root directory of PhysioNet dataset (default: data/ — uses synthetic if absent)')
    p.add_argument('--experiment', default='Experiment_1', choices=['Experiment_1', 'Experiment_2'])
    p.add_argument('--target', default='condition', choices=['condition', 'performance'], help='Classification target variable')
    p.add_argument('--cnn_epochs', type=int, default=50)
    p.add_argument('--n_hmm_states', type=int, default=4)
    args = p.parse_args()
 
    log.info("━" * 64)
    log.info("  EEG Cognitive State Inference — CS 5400 Final Project")
    log.info("  Missouri University of Science and Technology · 2026")
    log.info("━" * 64)   
 
    # Derive the same experiment-scoped subdirectory that run() uses internally,
    # so every output from this file lands in the same place.
    exp_tag = args.experiment.lower().replace(' ', '_')   # e.g. 'experiment_1'
    res_dir = RESULTS_DIR / exp_tag
    fig_dir = FIGURES_DIR / exp_tag
    res_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
 
    results, df = run(data_root = args.data, experiment = args.experiment, target = args.target, cnn_epochs = args.cnn_epochs)
 
    log.info("Running HMM temporal analysis …")
    X_all, y_all, feat_names, le = prepare_features(df, args.target)
    hmm_results = run_hmm_analysis(
        df           = df,
        feature_cols = feat_names,
        n_states     = args.n_hmm_states,
        figures_dir  = fig_dir,          # ← experiment-scoped figures dir
    )
    if 'error' not in hmm_results:
        log.info(f"  HMM mean log-likelihood/sample: {hmm_results['mean_ll']:.4f}")
        log.info(f"  HMM hidden-state → label alignment: {hmm_results['alignment']}")
        pd.DataFrame({'state': list(hmm_results['alignment'].keys()), 'label': list(hmm_results['alignment'].values())}).to_csv(res_dir / 'hmm_alignment.csv', index=False)
 
    log.info("Demonstrating intelligent agent on 10 sample percepts …")
    from Pipeline import train_naive_bayes
    nb_pipe = train_naive_bayes(X_all, y_all)
    agent   = CognitiveStateAgent(classifier = nb_pipe, class_names = le.classes_.tolist(), history_len= 5, high_load_classes = HIGH_LOAD_CONDITIONS[args.experiment])
    agent_log = []
    rng = np.random.default_rng(99)
    for i in range(10):
        feat_vec = X_all[rng.integers(0, len(X_all))]
        percept = AgentPercept(timestamp=float(i * 2), features=feat_vec, feature_names=feat_names)
        action = agent.act(percept)
        agent_log.append({
            'percept_t': percept.timestamp,
            'pred_label': action.predicted_label,
            'confidence': round(action.confidence, 4),
            'uncertainty': round(action.uncertainty, 4),
            'action_type': action.action_type,
            'recommendation': action.recommendation or '',
        })
        log.info(f"  t={i*2:4.0f}s  {action.action_type:10s}  "
                 f"{action.predicted_label:20s}  "
                 f"conf={action.confidence:.3f}  H={action.uncertainty:.3f}  "
                 f"{action.recommendation or ''}")
    pd.DataFrame(agent_log).to_csv(res_dir / 'agent_demo.csv', index=False)  # ← scoped
 
    # ── STEP 8: Print final summary ──────────────────────────────────────────
    log.info("")
    log.info("━" * 64)
    log.info("  FINAL RESULTS SUMMARY")
    log.info("━" * 64)
    log.info(f"  {'Model':<20}  {'Accuracy':>9}  {'AUC-ROC':>9}  {'F1-Macro':>9}")
    log.info(f"  {'─'*20}  {'─'*9}  {'─'*9}  {'─'*9}")
    for r in results:
        f1 = r['report'].get('macro avg', {}).get('f1-score', float('nan'))
        log.info(f"  {r['model']:<20}  {r['accuracy']:>9.4f}  "
                 f"{r['auc']:>9.4f}  {f1:>9.4f}")
    log.info("━" * 64)
    log.info(f"  All figures  → {fig_dir}/")
    log.info(f"  All results  → {res_dir}/")
    log.info("━" * 64)
 
 
if __name__ == '__main__':
    main()