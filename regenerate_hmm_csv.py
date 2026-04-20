"""
Regenerates HMM_rollingpredict_results2.csv with fixed align_regime_labels.

New convention (ascending by return):
    0 = Bear Market   (lowest mean return)
    1 = Sideways      (middle mean return)
    2 = Bull Market   (highest mean return)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, SpectralClustering
from hmmlearn import hmm
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# ── Load data ────────────────────────────────────────────────────────────────
df = pd.read_csv('sp500_master_dataset.csv', parse_dates=['Date'], index_col='Date')
df = df.sort_index().dropna(subset=['Rolling_Return_21', 'Rolling_Volatility_21'])
print(f"Loaded {len(df):,} rows: {df.index[0].date()} → {df.index[-1].date()}")

# ── Fixed alignment function ──────────────────────────────────────────────────
def align_regime_labels(df_slice, raw_label_col, ret_col='Rolling_Return_21'):
    """
    Maps raw cluster integers to economic regime labels sorted by mean return (ascending).
        0 = lowest return  → Bear Market
        1 = middle return  → Sideways
        2 = highest return → Bull Market
    """
    stats_ret = df_slice.groupby(raw_label_col)[ret_col].mean().sort_values()
    dynamic_mapping = {
        stats_ret.index[0]: 0,  # Lowest  → Bear Market
        stats_ret.index[1]: 1,  # Middle  → Sideways
        stats_ret.index[2]: 2,  # Highest → Bull Market
    }
    return df_slice[raw_label_col].map(dynamic_mapping)

# ── Clustering functions ──────────────────────────────────────────────────────
def run_kmeans_clustering(X_scaled, n_clusters=3, random_state=42):
    return KMeans(n_clusters=n_clusters, n_init=20, random_state=random_state).fit_predict(X_scaled)

def run_spectral_clustering(X_scaled, n_clusters=3, n_neighbors=35, random_state=42):
    model = SpectralClustering(
        n_clusters=n_clusters, affinity='nearest_neighbors',
        n_neighbors=n_neighbors, assign_labels='kmeans',
        random_state=random_state, n_jobs=-1
    )
    return model.fit_predict(X_scaled)

# ── Rolling window ────────────────────────────────────────────────────────────
def process_rolling(df, track='KMeans', window_size=1260, step=21):
    n_samples = len(df)
    final_hmm_states = np.full(n_samples, np.nan)
    feature_cols = ['Rolling_Return_21', 'Rolling_Volatility_21']

    for t in tqdm(range(window_size, n_samples, step), desc=f"Rolling ({track})"):
        train_df = df.iloc[t - window_size: t].copy()
        forecast_end = min(t + step, n_samples)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(train_df[feature_cols])

        if track == 'KMeans':
            train_raw_labels = run_kmeans_clustering(X_train_scaled)
        else:
            train_raw_labels = run_spectral_clustering(X_train_scaled)

        train_df['raw_cluster'] = train_raw_labels
        try:
            train_df['aligned_cluster'] = align_regime_labels(
                train_df, raw_label_col='raw_cluster', ret_col='Rolling_Return_21')
        except Exception:
            continue

        aligned_labels = train_df['aligned_cluster'].values
        hmm_input = aligned_labels.reshape(-1, 1)
        model_hmm = hmm.CategoricalHMM(n_components=3, n_features=3, n_iter=200, random_state=42)
        try:
            model_hmm.fit(hmm_input)
        except Exception:
            continue

        if np.any(np.isnan(model_hmm.transmat_)):
            continue

        train_hidden = model_hmm.predict(hmm_input)
        temp_df = train_df.copy()
        temp_df['hmm_hidden'] = train_hidden
        try:
            temp_df['hmm_aligned'] = align_regime_labels(
                temp_df, raw_label_col='hmm_hidden', ret_col='Rolling_Return_21')
        except Exception:
            continue

        hmm_to_aligned = {}
        for h in range(3):
            mask = (train_hidden == h)
            if np.any(mask):
                hmm_to_aligned[h] = temp_df.loc[mask, 'hmm_aligned'].mode()[0]
            else:
                hmm_to_aligned[h] = h

        last_hidden_state = train_hidden[-1]
        next_state_probs = model_hmm.transmat_[last_hidden_state]
        predicted_hidden = np.argmax(next_state_probs)
        predicted_aligned = hmm_to_aligned.get(predicted_hidden, predicted_hidden)
        final_hmm_states[t: forecast_end] = predicted_aligned

    return final_hmm_states

# ── Run both tracks ───────────────────────────────────────────────────────────
df['HMM_States_KMeans']   = process_rolling(df, track='KMeans')
df['HMM_States_Spectral'] = process_rolling(df, track='Spectral')

output_df = df.dropna(subset=['HMM_States_KMeans'])

label_map = {0: 'Bear Market', 1: 'Sideways', 2: 'Bull Market'}
output_df['HMM_KMeans']   = output_df['HMM_States_KMeans'].map(label_map)
output_df['HMM_Spectral'] = output_df['HMM_States_Spectral'].map(label_map)

# ── Verify alignment ──────────────────────────────────────────────────────────
print("\n=== Verification: mean Log_Return by state ===")
print("HMM_States_KMeans:")
print(output_df.groupby('HMM_States_KMeans')['Log_Return'].mean().sort_index())
print("\nHMM_States_Spectral:")
print(output_df.groupby('HMM_States_Spectral')['Log_Return'].mean().sort_index())
print("\nState 2 must have the highest mean return in both tracks.")

# ── Save ──────────────────────────────────────────────────────────────────────
out_path = 'Phase 3 HMM/HMM_rollingpredict_results2.csv'
output_df.to_csv(out_path)
print(f"\nSaved {len(output_df):,} rows → {out_path}")
