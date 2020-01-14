import datetime
import sys

argvs = sys.argv

chrom = argvs[1] # chrx

print(datetime.datetime.now())
print(f"Generate BlindTest for {chrom}")

import pandas as pd
import numpy as np
import os

predon = "B" # or "B"
version = "guacamole6"

filepath = "./data/processed_data/"
predpath = filepath + "prediction/" + version

## create folder is does not exist
os.makedirs(predpath, exist_ok=True)

import pickle

f_avg  = filepath + "blind_avg_" + chrom + "_arcsinh.pickle"
f_var  = filepath + "blind_var_" + chrom + "_arcsinh.pickle"

with open(f_avg, "rb" ) as f1:
    data_avg = pickle.load(f1)
    
with open(f_var, "rb" ) as f2:
    data_var = pickle.load(f2)

df_meta = pd.read_csv("./data/Encode_meta.tsv", sep="\t")
df_meta.rename(columns={"Training(T),Validation(V),Blind-test(B)":"DataType"}, 
               inplace=True)

## select those assays that only exist in Training
a = df_meta.groupby(["Assay"])
b = a.DataType.unique()
c = b.apply(lambda x: len(x)==1 and x[0] == "T")

assay2drop = c[c==True].index.values
#print(assay2drop)

idx_drop = df_meta.Assay.isin(assay2drop)
df_meta = df_meta[~idx_drop]

## remove assay with only single cell type
idx_pred = (df_meta.DataType == predon) 
key_pred = list(zip(df_meta.Cell_ID[idx_pred], 
                    df_meta.Mark_ID[idx_pred]))

print(f" {len(key_pred)} sets of {predon}")

celltypes = list(df_meta.Cell_ID.unique())
assays    = list(df_meta.Mark_ID.unique())

print(len(celltypes), "celltypes")
print(len(assays), "assays")

import os
from keras.models import load_model

model_path = filepath + f"keras_model/model_v4_{version}_{chrom}.h5"
model_path = os.path.abspath(model_path)

model = load_model(model_path)

def data_generator(celltypes, assays, 
                   data_all, 
                   data_avg,
                   data_var,
                   batch_size, 
                   shuffle=False):
    """Generate training data and label
    
    Params:
    ========
    celltypes: list. 
        list of unique cell types
    assays: list. 
        list of unique assay types
    data_all: dict. 
        (CellID, AsssayID): np.array(shape=(n_bins,))
    data_avg: dict. 
        (CellID, AsssayID): np.array(shape=(n_bins,))
        This is the average values of given assay type
        exclude the current cell type
    data_motif: np.array
        shape of (n_bins, 599). 
        columns are one-hot encoded motif type
    batch_size: int.
    shuffle: bool. whether to read from the start
    """
    
    start = 0
    
    # indices for looking up celltype and assay name
    # shape  = (2, len(data_all))
    # values = [0:n_celltypes, 0:n_assays]
    indices = np.array([[celltypes.index(celltype) for celltype, _ in data_all.keys()],
                        [assays.index(assay) for _, assay in data_all.keys()]])

    # [file1, file2, file3, ... fileN], N = len(data_all)
    # each file size = len of chrN (25 bp bin)
    tracks     = list(data_all.values())
    tracks_avg = list(data_avg.values())
    tracks_var = list(data_var.values())
    
    n_positions = len(tracks[0])
    
    while True:
        
        celltype_idxs      = np.zeros(batch_size, dtype='int32')
        assay_idxs         = np.zeros(batch_size, dtype='int32')
        
        if shuffle:
            genomic_25bp_idxs  = np.random.randint(n_positions, size=batch_size)
        else:
            genomic_25bp_idxs  = np.arange(start, start+batch_size) % n_positions
        
        genomic_250bp_idxs = genomic_25bp_idxs // 10
        genomic_5kbp_idxs  = genomic_25bp_idxs // 200
        value              = np.zeros(batch_size, dtype='float32')
        value_avg          = np.zeros(batch_size, dtype='float32')
        value_var          = np.zeros(batch_size, dtype='float32')
        
        
        # len(data): number of (celltype, assay) samples 
        # randomly sample the (celltype, assay) batch_size times
        # batch_size ~ 10,000
        idxs = np.random.randint(len(data_all), size=batch_size)
        
        for i, idx in enumerate(idxs):
            celltype_idx     = indices[0, idx]
            assay_idx        = indices[1, idx]

            celltype_idxs[i] = celltype_idx
            assay_idxs[i]    = assay_idx

            value[i]      = tracks[idx][genomic_25bp_idxs[i]]
            value_avg[i]  = tracks_avg[idx][genomic_25bp_idxs[i]]
            value_var[i]  = tracks_var[idx][genomic_25bp_idxs[i]]

        d = {
            'celltype_input': celltype_idxs, 
            'assay_input': assay_idxs, 
            'genome_25bp_input': genomic_25bp_idxs, 
            'genome_250bp_input': genomic_250bp_idxs,
            'genome_5kbp_input': genomic_5kbp_idxs,
            'average_value_input': value_avg,
            'variance_value_input':value_var
            }

        yield d, value

        start += batch_size

def predict(model, key, 
            batch_size = 10000, 
            arcsinh_to_orig = False,
            plot_prediction = False,
            start = 25750, end=28000):
    
    d_avg  = {key: data_avg[key]}
    d_var  = {key: data_var[key]}
    avg    = data_avg[key]
    
    pred_generator = data_generator(celltypes, assays, 
                                    d_avg, d_avg, d_var,
                                    #data_motif.values,
                                    batch_size,
                                    shuffle=False)
    n_positions = len(avg)
    len_pred = n_positions/batch_size
    len_pred = int(np.ceil(len_pred))
    
    pred = model.predict_generator(
        generator = pred_generator,
        use_multiprocessing = False,
        steps = len_pred,
        verbose = 1)
    pred = pred.squeeze()
    pred = pred[:n_positions]
    
    ## if prediction values is < 0, make it 0
    pred[pred<0] = 0

    # transform the arcsinh back to original scale
    if arcsinh_to_orig:
        pred = np.sinh(pred)
    
    ## plotting
    if plot_prediction:
        idx = (df_meta.Cell_ID == key[0]) & (df_meta.Mark_ID == key[1])
        name_cell = df_meta.CellType.values[idx]
        name_assay= df_meta.Assay.values[idx]

        x = np.arange(start*25/1000., end*25/1000., 25/1000.)

        plt.figure(figsize=(14, 4))
        plt.fill_between(x, 0, pred[start:end], color='g', label="Prediction")
        plt.legend(fontsize=14)
        plt.ylabel("Signal Value", fontsize=14)
        plt.xlabel("Genomic Coordinate (kb)", fontsize=14)
        plt.ylim(0, 7)
        plt.xlim(start*25/1000., end*25/1000.)
        plt.show()
    
    return pred

from tqdm import tqdm

for k in tqdm(key_pred):
    pred = predict(model, k, arcsinh_to_orig=True)
    
    ## save to file
    fname = "".join(x for x in k) + "_" + chrom + "_pred.npy"
    np.save(predpath+"/"+fname, pred)

print(datetime.datetime.now())
print("DONE Generating " + chrom)