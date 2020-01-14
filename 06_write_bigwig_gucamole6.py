import pandas as pd
import numpy as np
import pyBigWig
import glob

import shutil
import datetime
import sys
import os

argvs = sys.argv
vname = argvs[1] # bigwig filename

DataType = "B"
version = "guacamole6"
n_cpu = 1

filepath = f"./data/processed_data/prediction/{version}/"
filepath_out = filepath + "submission/"
filepath_temp= "./data/submission_template/"

## create folder is does not exist
os.makedirs(filepath_out, exist_ok=True)

df_meta = pd.read_csv("./data/Encode_meta.tsv", sep="\t")
df_meta.rename(columns={"Training(T),Validation(V),Blind-test(B)":"DataType"}, 
               inplace=True)
idx = df_meta.DataType == DataType
filenames = df_meta.loc[idx, 'fileName'].values
## create dummy bigwig files for read/write
for f in filenames:
    shutil.copyfile(filepath_temp+"submission_template (copy).bigwig",
                    filepath_out+f) 

df_temp = pyBigWig.open(filepath_temp+"submission_template.bigwig")
dict_chrom = df_temp.chroms()
df_temp.close()

def write_bigwig(vname):
    
    vname = vname.strip(".bigwig")

    df_bw = pyBigWig.open(filepath_out+vname+".bigwig", "w")
    df_bw.addHeader(list(dict_chrom.items()))
    
    fnames = glob.glob(filepath + f"*{vname}*.npy")
    
    chroms = [x.split(f"{vname}_")[1] for x in fnames]
    chroms = [x.strip("_pred.npy") for x in chroms]
    
    ## sort files by chrom order 1, 2, 3, ...21, 22,X
    chroms_num = [int(x.strip("chr")) if x != "chrX" else 23 for x in chroms]
    chroms_num = np.array(chroms_num)
    fnames = [fnames[x] for x in chroms_num.argsort()]
    chroms = [chroms[x] for x in chroms_num.argsort()]
    
    ## load all chroms for a given validation
    ## loop through chrom
    sta_list = []
    end_list = []
    chr_list = []
    val_list = []
    
    for i, f in enumerate(fnames):
        chrom = chroms[i]
        vals  = np.load(f)
        vals  = vals.astype('float') # pybigwig only takes float64

        end    = dict_chrom[chrom]
        chrom_ = [chrom]*len(vals)
        ranges = np.arange(0, end, 25)
        ranges = list(ranges)

        starts = ranges[:-1]
        ends   = ranges[1: ]

        delta = end - ends[-1]

        assert delta < 25

        if delta > 0:
            starts.append(ends[-1])
            ends.append(end)

        assert len(ends) == len(starts) == len(vals)
        
        sta_list.extend(starts)
        end_list.extend(ends)
        chr_list.extend(chrom_)
        val_list.extend(vals)

    
    df_bw.addEntries(chr_list, sta_list, ends=end_list, 
                     values=val_list, validate=True)
    df_bw.close()
    print(f"Done {vname}.")

print(f"processing {vname}")
print(datetime.datetime.now())
write_bigwig(vname)