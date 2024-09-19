import numpy as np
import jax as jnp
import scipy.io
import pandas as pd
import h5py



def mat_to_dict(mat_file, savepath):
    
    mat_data = scipy.io.loadmat(mat_file)['dataset']
    return mat_data


data_dict = mat_to_dict("data/SAR_AF_ML_toyDataset_etc/radarSeries.mat")


