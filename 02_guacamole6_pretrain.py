import datetime
import sys

argvs = sys.argv

chrom = argvs[1] # chrx
batch_size = int(argvs[2])
n_epoch = int(argvs[3])

n_25bp_factors  = int(argvs[4])
n_250bp_factors = int(argvs[5])
n_5kbp_factors  = int(argvs[6])

print(datetime.datetime.now())
print("Pretrain " + chrom + " using bs " + argvs[2] + " for " + argvs[3] + " epoch.")
print(f"25bp:{n_25bp_factors}, 250bp:{n_250bp_factors}, 5kb:{n_5kbp_factors}")

import pandas as pd
import numpy as np

from keras.layers import Input, Dense, concatenate 
from keras.layers import Embedding, Flatten, PReLU, add
from keras.layers import Dropout, Activation, BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.metrics import categorical_accuracy

from tqdm import tqdm

import pickle

filepath_v3 = "./data/processed_data/"
filepath_v4 = "./data/processed_data/"

f_all  = filepath_v3 + "dict_3cat_" + chrom + "_arcsinh.pickle"
f_avg  = filepath_v3 + "dict_avg3cat_" + chrom + "_arcsinh.pickle"

with open(f_avg, "rb" ) as f1:
    data_avg = pickle.load(f1)

with open(f_all, "rb" ) as f0:
    data_all = pickle.load(f0)

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
idx_train = (df_meta.DataType != "B")


key_train = list(zip(df_meta.Cell_ID[idx_train], 
                     df_meta.Mark_ID[idx_train]))


data_all_train = dict((k, data_all[k]) for k in key_train)
data_avg_train = dict((k, data_avg[k]) for k in key_train)

def Precamole(n_celltypes, 
              n_assays, 
              n_positions,
              n_celltype_factors = 45,
              n_assay_factors    = 65,
              n_25bp_factors     = 25, 
              n_250bp_factors    = 30, 
              n_5kbp_factors     = 60):
    
    # Embeddings for celltypes, assays, and locations
    celltype_input = Input(shape=(1,), name="celltype_input")
    celltype_embedding = Embedding(n_celltypes, 
                                   n_celltype_factors, 
                                   input_length=1, 
                                   name="celltype_embedding")
    celltype = Flatten()(celltype_embedding(celltype_input))

    assay_input = Input(shape=(1,), name="assay_input")
    assay_embedding = Embedding(n_assays, 
                                n_assay_factors,
                                input_length=1, 
                                name="assay_embedding")
    assay = Flatten()(assay_embedding(assay_input))

    genome_25bp_input = Input(shape=(1,), name="genome_25bp_input")
    genome_25bp_embedding = Embedding(n_positions, 
                                      n_25bp_factors, 
                                      input_length=1, 
                                      name="genome_25bp_embedding")
    genome_25bp = Flatten()(genome_25bp_embedding(genome_25bp_input))

    genome_250bp_input = Input(shape=(1,), name="genome_250bp_input")
    genome_250bp_embedding = Embedding(int(n_positions / 10) + 1,
                                       n_250bp_factors, 
                                       input_length=1, 
                                       name="genome_250bp_embedding")
    genome_250bp = Flatten()(genome_250bp_embedding(genome_250bp_input))

    genome_5kbp_input = Input(shape=(1,), name="genome_5kbp_input")
    genome_5kbp_embedding = Embedding(int(n_positions / 200) + 1, 
                                      n_5kbp_factors, 
                                      input_length=1, 
                                      name="genome_5kbp_embedding")
    genome_5kbp = Flatten()(genome_5kbp_embedding(genome_5kbp_input))

    # average input
    average_value = Input(shape=(3,), name="average_value_input")
    
    # concatenate all layers
    inputs = [celltype_input, 
              assay_input, 
              genome_25bp_input, 
              genome_250bp_input, 
              genome_5kbp_input,
              average_value]
    
    layers = [celltype, 
              assay, 
              genome_25bp, 
              genome_250bp, 
              genome_5kbp]
    
    x = concatenate(layers) # motif layer
    
    
    x = Dense(2048, name="dense_1")(x)
    x = PReLU(name="dense_1_ac")(x)
    x = BatchNormalization(name="dense_1_bn")(x)
    x = Dropout(0.5, name="dense_1_dp")(x)

    x = Dense(2048, name="dense_2")(x)
    x = PReLU(name="dense_2_ac")(x)
    x = BatchNormalization(name="dense_2_bn")(x)
    x = Dropout(0.5, name="dense_2_dp")(x)

    y = Dense(3, name="y_pred_dense")(x)
    y = add([y, average_value])
    y = Activation("softmax", name="y_pred_act")(y)

    model = Model(inputs=inputs, outputs=y)
    model.compile(optimizer=Adam(lr=0.0005), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])

    return model

def data_generator(celltypes, assays, 
                   d_cat, d_avg, 
                   batch_size, 
                   shuffle=False):
    """Generate training data and label
    
    Params:
    ========
    celltypes: list. 
        list of unique cell types
    assays: list. 
        list of unique assay types
    data_cat: dict. 
        (CellID, AsssayID): np.array(shape=(n_bins,))
    data_avg: dict. 
        (CellID, AsssayID): np.array(shape=(n_bins,))
        This is the average values of given assay type
        exclude the current cell type
    batch_size: int.
    shuffle: bool. whether to read from the start
    """
    
    start = 0
    
    # indices for looking up celltype and assay name
    # shape  = (2, len(data_cat))
    # values = [0:n_celltypes, 0:n_assays]
    indices = np.array([[celltypes.index(celltype) for celltype, _ in d_cat.keys()],
                        [assays.index(assay) for _, assay in d_cat.keys()]])

    # [file1, file2, file3, ... fileN], N = len(data_cat)
    # each file size = len of chrN (25 bp bin)
    tracks     = list(d_cat.values())
    tracks_avg = list(d_avg.values())
    
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
        value              = np.zeros((batch_size, 3), dtype='float32')
        value_avg          = np.zeros((batch_size, 3), dtype='float32')
        
        # len(data): number of (celltype, assay) samples 
        # randomly sample the (celltype, assay) batch_size times
        # batch_size ~ 10,000
        idxs = np.random.randint(len(d_cat), size=batch_size)

        for i, idx in enumerate(idxs):
            celltype_idx     = indices[0, idx]
            assay_idx        = indices[1, idx]

            celltype_idxs[i] = celltype_idx
            assay_idxs[i]    = assay_idx
            
            # convert cat to onehot
            val_cat = tracks[idx][genomic_25bp_idxs[i]]
            avg_cat = tracks_avg[idx][genomic_25bp_idxs[i]] 
            value[i]     = to_categorical(val_cat, num_classes=3)
            value_avg[i] = to_categorical(avg_cat, num_classes=3)     

        d = {
            'celltype_input': celltype_idxs, 
            'assay_input': assay_idxs, 
            'genome_25bp_input': genomic_25bp_idxs, 
            'genome_250bp_input': genomic_250bp_idxs,
            'genome_5kbp_input': genomic_5kbp_idxs,
            'average_value_input': value_avg
            }

        yield d, value

        start += batch_size

celltypes = list(df_meta.Cell_ID.unique())
assays    = list(df_meta.Mark_ID.unique())

#print(len(celltypes), "celltypes")
#print(len(assays), "assays")

train_generator = data_generator(celltypes, assays, 
                                 data_all_train, 
                                 data_avg_train,
                                 batch_size,
                                 shuffle=False)

n_positions = len(data_all[list(data_all.keys())[0]])
len_train = n_positions/batch_size
len_train = int(np.ceil(len_train))

import os
from keras.callbacks import ModelCheckpoint

model_path = filepath_v4 + "keras_model/"
model_path = f"{model_path}model_v4_pre_gucamole6_{chrom}.h5"
model_path = os.path.abspath(model_path)
checkpoint = ModelCheckpoint(filepath=model_path,
                             monitor='loss',
                             verbose=1,
                             save_best_only=True)

model = Precamole(n_celltypes = len(celltypes), 
                  n_assays = len(assays), 
                  n_positions = n_positions,
                  n_25bp_factors = n_25bp_factors,
                  n_250bp_factors= n_250bp_factors,
                  n_5kbp_factors = n_5kbp_factors)
model.summary()

history = model.fit_generator(
    train_generator,
    steps_per_epoch = len_train,
    epochs = n_epoch,
    use_multiprocessing = True,
    callbacks = [checkpoint])

import matplotlib.pylab as plt
import seaborn as sns

fig_path = filepath_v4 + "training_fig/pretrain/"
os.makedirs(fig_path, exist_ok=True)

f_hist  = fig_path + f"pretrain_hist_{chrom}.pickle"

with open(f_hist, 'wb') as fhist:
    pickle.dump(history, fhist)

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.title('Model loss ('+ chrom +' pretraining, bs = ' + str(batch_size) + ', n_epoch = ' + str(n_epoch) + ')')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper left')
plt.savefig(fig_path+f"loss_{chrom}.png")
plt.clf()

# Plot training & validation loss values
plt.plot(history.history['acc'])
plt.title('Model acc ('+ chrom +' pretraining, bs = ' + str(batch_size) + ', n_epoch = ' + str(n_epoch) + ')')
plt.ylabel('acc')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper left')
plt.savefig(fig_path+f"acc_{chrom}.png")
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
plt.title('Cell Embedding ('+ chrom +' pretraining, bs = ' + str(batch_size) + ', n_epoch = ' + str(n_epoch) + ')')
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
plt.title('Assay Embedding ('+ chrom +' pretraining, bs = ' + str(batch_size) + ', n_epoch = ' + str(n_epoch) + ')')
plt.savefig(fig_path+f"assay_embed_{chrom}.png")

print(datetime.datetime.now())
print("DONE Pretrain " + chrom)