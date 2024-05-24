import sportmodules as sm
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.mlab import psd
from scipy.integrate import simpson
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests
import pandas as pd

def calculate_granger_causality(rel_bp,RR):
    data = list(zip(rel_bp,RR))
    data = pd.DataFrame(data)
    return grangercausalitytests(data, 4)
def calculate_ppg_features(filtered_signal, name, config):

    def get_lin_model_coefs(HR,RR):
        model = LinearRegression()
        model.fit(RR[1:].reshape(-1, 1),HR.reshape(-1, 1))
        return model.coef_[0][0]

    
    RR, heights = sm.get_RR_peaks(filtered_signal)
    
    RMSSD = sm.calculate_hrv_from_rest(RR)['RMSSD']
    
    HR = sm.get_HR_from_RR(RR, config)
    plt.plot(HR)
    
    HR_ENTROPY = sm.entropy(HR)
    
    HR_LIN = get_lin_model_coefs(HR, RR)
    
    (S, f) = psd(filtered_signal, Fs = config['fs'] )
    
    PSD_FS_MAX = f[S.argmax()]
    
    PSD_CURVE = simpson(S)

    time_features = {
        "RMSSD": RMSSD,
        "MEAN_HR": np.mean(HR),
        "HR_LIN": HR_LIN}
    
    frequency_features = {
        "PSD_FS_MAX": PSD_FS_MAX,
        "PSD_CURVE":PSD_CURVE}
    
    information_features = {"HR_ENTROPY": HR_ENTROPY,}
    
    causality_features = {}
    
    features = {}
    for f in (time_features, frequency_features, information_features,causality_features): features.update(f)
    
    return features