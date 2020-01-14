import datetime
import sys

argvs = sys.argv

chrom = argvs[1] # chrx
batch_size = int(argvs[2])
n_epoch = int(argvs[3])

print(datetime.datetime.now())
print("Train " + chrom + " using bs " + argvs[2] + " for " + argvs[3] + " epoch.")

import pandas as pd
import numpy as np

from keras.layers import Input, Dense, concatenate, add
from keras.layers import Embedding, Flatten, PReLU
from keras.layers import Dropout, Activation, BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
import keras.backend as K

import seaborn as sns
import matplotlib.pylab as plt

filepath_v3 = "./data/processed_data/"
filepath_v4 = "./data/processed_data/"

import pickle

f_all  = filepath_v3 + "dict_all_" + chrom + "_arcsinh.pickle"
f_avg  = filepath_v3 + "dict_avg_" + chrom + "_arcsinh.pickle"
f_var  = filepath_v3 + "dict_var_" + chrom + "_arcsinh.pickle"

with open(f_all, "rb" ) as f0:
    data_all = pickle.load(f0)

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

idx_train = (df_meta.DataType != "B") 

key_train = list(zip(df_meta.Cell_ID[idx_train], 
                     df_meta.Mark_ID[idx_train]))

data_all_train = dict((k, data_all[k]) for k in key_train)
data_avg_train = dict((k, data_avg[k]) for k in key_train)
data_var_train = dict((k, data_var[k]) for k in key_train)

## load pre-trained motif model
from keras.models import load_model

model_file = filepath_v4 + f"keras_model/model_v4_pre_gucamole6_{chrom}.h5"
model_pre = load_model(model_file)

# remove the last densde layer
model_pre.layers.pop()
model_pre.layers.pop()
model_pre.layers.pop()
model_pre.layers.pop()
model_pre.summary()

model_pre_last = model_pre.layers[-1].output

model_pre.inputs[5] = Input(shape=(1,), name="average_value_input")
variance_value      = Input(shape=(1,), name="variance_value_input")
model_pre.inputs.append(variance_value)

def Guacamole():  
    
    inputs = model_pre.inputs
    average_value = inputs[5]
    variance_value= inputs[6]
    
    x = model_pre_last
    x = concatenate([variance_value, average_value, x], name="concat_last")
    
    x = Dense(2048, name="dense_3")(x)
    x = PReLU(name="dense_3_ac")(x)
    x = BatchNormalization(name="dense_3_bn")(x)
    x = Dropout(0.7, name="dense_3_dp")(x)    

    y = Dense(1, name="y_pred")(x)
    y = add([y, average_value])

    model = Model(inputs=inputs, outputs=y)
    model.compile(optimizer = Adam(lr=0.001), 
                  loss='mse', metrics=['mse'])

    return model

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

celltypes = list(df_meta.Cell_ID.unique())
assays    = list(df_meta.Mark_ID.unique())

print(len(celltypes), "celltypes")
print(len(assays), "assays")

train_generator = data_generator(celltypes, assays, 
                                 data_all_train, 
                                 data_avg_train,
                                 data_var_train,
                                 batch_size,
                                 shuffle=False)

n_positions = len(data_all[list(data_all.keys())[0]])
len_train = n_positions/batch_size
len_train = int(np.ceil(len_train))

import os
from keras.callbacks import ModelCheckpoint

model_path = filepath_v4 + "keras_model/model_v4_guacamole6_" + chrom + ".h5"
model_path = os.path.abspath(model_path)
checkpoint = ModelCheckpoint(filepath=model_path,
                             monitor='loss',
                             verbose=1,
                             save_best_only=True)

model = Guacamole()
model.summary()

history = model.fit_generator(
    train_generator,
    steps_per_epoch = len_train,
    epochs = n_epoch,
    use_multiprocessing = True,
    callbacks = [checkpoint])

fig_path = filepath_v4 + "training_fig/train/"
os.makedirs(fig_path, exist_ok=True)


f_hist  = fig_path + f"training_hist_{chrom}.pickle"

with open(f_hist, 'wb') as fhist:
    pickle.dump(history, fhist)

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.title('Model loss ('+ chrom +' training, bs = ' + str(batch_size) + ', n_epoch = ' + str(n_epoch) + ')')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper left')
plt.savefig(fig_path+f"loss_{chrom}.png")
plt.clf()

from keras.models import load_model
model = load_model(model_path)

embed_cell = model.get_layer("celltype_embedding")
embed_cell = embed_cell.get_weights()[0]

cell_map = df_meta.groupby("Cell_ID")["Cell_ID", "CellType"].first()
cellnames = []
for c in celltypes:
    name = cell_map.loc[cell_map.Cell_ID == c, "CellType"][0]
    cellnames.append(name)

df_celltypes = pd.DataFrame(embed_cell, index=cellnames)
#df_celltypes.head()
g = sns.clustermap(df_celltypes, figsize=(8, 13))
#g_fig = g.get_figure() 
plt.savefig(fig_path+f"cell_heatmap_{chrom}.png")

from umap import UMAP
tsne_cell = UMAP(n_components = 2, 
                 n_neighbors  = 5, 
                 #learning_rate= 5, 
                 random_state = 77).fit_transform(df_celltypes)
tsne_cell = pd.DataFrame(tsne_cell)

plt.figure(figsize=(8, 8))
for i,type in enumerate(df_celltypes.index.values):
    x = tsne_cell.iloc[i, 0]
    y = tsne_cell.iloc[i, 1]
    plt.scatter(x, y, marker='x', color='red')
    plt.text(x+0.03, y+0.03, type, fontsize=10)
plt.title('Cell Embedding ('+ chrom +' training, bs = ' + str(batch_size) + ', n_epoch = ' + str(n_epoch) + ')')
plt.savefig(fig_path+f"cell_embed_{chrom}.png")


embed_assay = model.get_layer("assay_embedding")
embed_assay = embed_assay.get_weights()[0]
assay_map = df_meta.groupby("Mark_ID")["Mark_ID", "Assay"].first()
assaynames = []
for a in assays:
    name = assay_map.loc[assay_map.Mark_ID == a, "Assay"][0]
    assaynames.append(name)
df_assaytypes = pd.DataFrame(embed_assay, index=assaynames)

g = sns.clustermap(df_assaytypes, figsize=(8, 13))
#g_fig = g.get_figure() 
plt.savefig(fig_path+f"assay_heatmap_{chrom}.png")

tsne_assay = UMAP(n_components = 2, 
                  n_neighbors  = 4, 
                  #learning_rate= 5, 
                  random_state = 77).fit_transform(df_assaytypes)
tsne_assay = pd.DataFrame(tsne_assay)
plt.figure(figsize=(8, 8))
for i,type in enumerate(df_assaytypes.index.values):
    x = tsne_assay.iloc[i, 0]
    y = tsne_assay.iloc[i, 1]
    plt.scatter(x, y, marker='x', color='red')
    plt.text(x+0.03, y+0.03, type, fontsize=10)
plt.title('Assay Embedding ('+ chrom +' training, bs = ' + str(batch_size) + ', n_epoch = ' + str(n_epoch) + ')')
plt.savefig(fig_path+f"assay_embed_{chrom}.png")

print(datetime.datetime.now())
print("DONE Training " + chrom)