import pyBigWig
import gc
import sys
import pickle
import pandas as pd
import numpy as np
import datetime

from multiprocess import Pool


argvs = sys.argv
chrom = argvs[1] # chrx
n_cpu = int(argvs[2]) # int

print(datetime.datetime.now())
print(f"Process bigwig files for {chrom} using {n_cpu} cpu.")

## define folder paths =========================================================
filepath = "./data/"

df_meta = pd.read_csv(filepath + "Encode_meta.tsv", sep="\t")
df_meta.rename(columns={"Training(T),Validation(V),Blind-test(B)":"DataType"}, 
               inplace=True)

filepath_train = filepath + "training_data/"
filepath_valid = filepath + "validation_data/"

outpath = filepath + "processed_data/"

## define folder paths =========================================================

def get_binvals(filepath, chrom):
    """Get 25 bin values of given chrom
    Take arcsinh first then bin (mean)
    
    Params:
    ========
    filepath: str. path to the bigwig file
    chrom: str. specifiy which chromosome to process
    
    Return:
    ========
    vals: np.array('float32') with length of nbins
    """
    bw = pyBigWig.open(filepath)

    end = bw.chroms(chrom)
    nbins= (-(-end // 25)) ## round-up

    vals = bw.values(chrom, 0, end)
    vals = np.array(vals)
    vals[np.isnan(vals)] = 0

    # calc arcsinh
    vals = np.arcsinh(vals)

    # pad 0 in order to match nbin*25
    n_extra = nbins*25 - end
    vals = np.append(vals, np.zeros(n_extra))

    # calc mean
    vals = vals.reshape(nbins, -1).mean(axis=1)
    vals = vals.astype('float32')

    bw.close()
    
    return(vals)

def preproc(chrom, filepath):
    
    vals = get_binvals(filepath, chrom)
    
    fname = filepath.split("/")[-1]
    Cell_ID = fname[0:3]
    Mark_ID = fname[3:6]
    
    return {(Cell_ID, Mark_ID): vals}

## convert all chrom_N values into .pickle file ================================
print("convert bigwig file to dictionary and pickle it")

fnames = []
for i, f in df_meta.iterrows():
    
    if f.DataType == "T":
        fpth = filepath_train + f.fileName
    
    elif f.DataType == "V":
        fpth = filepath_valid + f.fileName
 
    else:
        continue
        
    fnames.append(fpth)


pool = Pool(n_cpu) # use 5 for large genomes

result = pool.map(lambda x: preproc(chrom, x), fnames)

pool.close()
pool.join()

data = {}
for d in result:
    data.update(d)

outname = "dict_all_" + chrom + "_arcsinh.pickle"
with open(outpath + outname, 'wb') as handle:
    pickle.dump(data, handle, 
                protocol=pickle.HIGHEST_PROTOCOL)
    
del result, data, pool, d
gc.collect()

## calculate avg value =========================================================
print("calculate average")

outname = "dict_all_" + chrom + "_arcsinh.pickle"

with open(outpath + outname, 'rb') as handle:
    data = pickle.load(handle)
# get data_avg for same assay (for each cell type, the avg assay is the same)
indices = np.array([[celltype for celltype, _ in data.keys()],
                    [assay    for _, assay    in data.keys()]])
tracks  = list(data.values())

def vals2cat(v):
    
    vals = pd.Series(v)
    vals = vals.rank(method='first') 
    vals = pd.qcut(vals, 3, labels=[0, 1, 2])
    vals = np.array(vals)
    vals.astype('int8')
    
    return vals

def get_avg(k, v):
      
    c_id = k[0]
    a_id = k[1]
    
    # average same assays together
    track_avg_idxs = (indices[1,:] == a_id) 
    track_avg      = np.zeros(len(tracks[0]), dtype=np.float32)
    n_other_cells  = 0
    
    for j, b in enumerate(track_avg_idxs):
        if b:
            track_avg    += tracks[j]
            n_other_cells += 1
    
    #print(f"processing {k}, n_other_cells: {n_other_cells}")
    
    if n_other_cells > 1:
        track_avg       = track_avg/n_other_cells
        #data_avg[k]     = track_avg
        return {k: track_avg}
    return None

def get_var(k, v):
      
    c_id = k[0]
    a_id = k[1]
    
    # Get assays variance across cells
    track_var_idxs = (indices[1,:] == a_id) 
    track_var      = np.zeros(len(tracks[0]), dtype=np.float32)
    track_avg      = np.zeros(len(tracks[0]), dtype=np.float32)
    n_other_cells  = 0
    
    for j, b in enumerate(track_var_idxs):
        if b:
            track_avg    += tracks[j]
            track_var    += tracks[j]**2
            n_other_cells += 1
    
    #print(f"processing {k}, n_other_cells: {n_other_cells}")
    
    if n_other_cells > 1:
        track_var       = track_var/n_other_cells - (track_avg/n_other_cells)**2
        #data_avg[k]     = track_avg
        return {k: track_var}
    return None

def get_avg_cat(k, v):
      
    c_id = k[0]
    a_id = k[1]
    
    # average same assays together
    track_avg_idxs = (indices[1,:] == a_id) 
    track_avg      = np.zeros(len(tracks[0]), dtype=np.float32)
    n_other_cells  = 0
    
    for j, b in enumerate(track_avg_idxs):
        if b:
            track_avg    += tracks[j]
            n_other_cells += 1
    
    #print(f"processing {k}, n_other_cells: {n_other_cells}")
    
    if n_other_cells > 1:
        track_avg       = track_avg/n_other_cells
        #data_avg[k]     = track_avg
        return {k: vals2cat(track_avg)}
    return None

def get_all_cat(k, v):
    return {k: vals2cat(v)}

pool = Pool(n_cpu) # use 5 for large genomes

result = pool.map(lambda x: get_avg(x[0], x[1]), data.items())

pool.close()
pool.join()

data_avg = {}
for d in result:
    if d is not None:
        data_avg.update(d)

outname_avg     = "dict_avg_" + chrom + "_arcsinh.pickle"

with open(outpath + outname_avg, 'wb') as handle1:
    pickle.dump(data_avg, handle1, 
                protocol=pickle.HIGHEST_PROTOCOL)

del result, data_avg, pool, d
gc.collect()

## calculate var value =========================================================
print("calculate variance")
pool = Pool(n_cpu) # use 5 for large genomes

result = pool.map(lambda x: get_var(x[0], x[1]), data.items())

pool.close()
pool.join()

data_var = {}
for d in result:
    if d is not None:
        data_var.update(d)

outname_var     = "dict_var_" + chrom + "_arcsinh.pickle"

with open(outpath + outname_var, 'wb') as handle1:
    pickle.dump(data_var, handle1, 
                protocol=pickle.HIGHEST_PROTOCOL)

del result, data_var, pool, d
gc.collect()

## calculate avg-3-cat value ===================================================
print("convert all average values to 3 categories")
pool = Pool(n_cpu) # use 5 for large genomes

result = pool.map(lambda x: get_avg_cat(x[0], x[1]), data.items())

pool.close()
pool.join()

data_avg_cat = {}
for d in result:
    if d is not None:
        data_avg_cat.update(d)

outname_avg_cat = "dict_avg3cat_" + chrom + "_arcsinh.pickle"
    
with open(outpath + outname_avg_cat, 'wb') as handle2:
    pickle.dump(data_avg_cat, handle2, 
                protocol=pickle.HIGHEST_PROTOCOL)

del result, data_avg_cat, pool, d
gc.collect()

## convert all to 3-cat value ==================================================
print("convert all assays to 3-categories")
pool = Pool(n_cpu) # use 5 for large genomes

result = pool.map(lambda x: get_all_cat(x[0], x[1]), data.items())

pool.close()
pool.join()

data_all_cat = {}
for d in result:
    if d is not None:
        data_all_cat.update(d)


outname_all_cat = "dict_3cat_" + chrom + "_arcsinh.pickle"
    
with open(outpath + outname_all_cat, 'wb') as handle3:
    pickle.dump(data_all_cat, handle3, 
                protocol=pickle.HIGHEST_PROTOCOL)

del result, data_all_cat, pool, d
gc.collect()

print(datetime.datetime.now())
print(f"Finish data generation for {chrom}!")