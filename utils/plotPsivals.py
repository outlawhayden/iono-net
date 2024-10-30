import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd


def coeffsToComplex(coeffs):
    n = len(coeffs)
    complex_coeffs = np.zeros(n//2, dtype=complex)
    for i in range(n//2):
        complex_coeffs[i] = complex(coeffs[i], coeffs[2*i+1])
    return complex_coeffs


def buildPsiVals(kPsi, compl_ampls, x)
    psiVals = np.zeros(len(x), dtype=complex)
    for i in range(len(compl_ampls)):
        psiVals += np.real(compl_ampls[i] * np.exp(1j * kPsi[i] * x))
    return psiVals

x_range = np.arange(0, 375, 0.01)

kpsi_path = "/home/houtlaw/iono-net/data/SAR_AF_ML_toyDataset_etc/radar_coeffs_csv_small/kPsi_20241029_193914.csv"
kpsi_df = pd.read_csv(kpsi_path)

coeffs_path = "/home/houtlaw/iono-net/data/SAR_AF_ML_toyDataset_etc/radar_coeffs_csv_small/compl_ampls_20241029_193914.csv"
coeffs_df = pd.read_csv(coeffs_path)



