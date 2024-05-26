import pandas as pd
import sportmodules as sm
import features as feat
import matplotlib.pyplot as plt

good_polar_recovery_ids = ["HH270", "HH276", "HH279", "HH282"]

config = {
    "fs": 130,
    "order_low": 5,
    "order": 10,
    "cutoff_high": 12,
    "cutoff_low": 0.3,
    "nyq": 65.0}

cpet_lens = sm.get_cpet_lens(good_polar_recovery_ids)
features = []


features = [None] * len(cpet_lens)
# %%
for i in range(len(cpet_lens)):
    # Read and slice the signal in one step to avoid intermediate steps
    polar_ppg = sm.read_signal_raw(good_polar_recovery_ids[i], 'polar').iloc[1000:20000, 3]

    # Process the polar signal
    filtered_signal = sm.preprocess_polar_signal(polar_ppg, config)
    # get relative BP
    rel_bp = sm.get_rel_bp(filtered_signal)
    # Calculate features from the filtered signal
    ppg_features = feat.calculate_ppg_features(filtered_signal, good_polar_recovery_ids[i], config)
    # information_features =
    # causal_features =
    # Store the result in the pre-allocated list
    features[i] = ppg_features


features = pd.DataFrame(features)
features['cpet_lens'] = cpet_lens
features.to_csv("data/data.csv", index=False)
# %%
