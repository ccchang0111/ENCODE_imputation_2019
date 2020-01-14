import pandas as pd
import datetime
import sys

argvs = sys.argv
chrom = argvs[1] # chrx

print(datetime.datetime.now())
print(f"Generate Blind Avg and Var for {chrom}.")

filepath = "./data/"

df_meta = pd.read_csv(filepath + "Encode_meta.tsv", sep="\t")
df_meta.rename(columns={"Training(T),Validation(V),Blind-test(B)":"DataType"}, 
               inplace=True)

import pickle

outpath_v3 = "./data/processed_data/"
v3_avg = "dict_avg_" + chrom + "_arcsinh.pickle"
v3_var = "dict_var_" + chrom + "_arcsinh.pickle"

with open(outpath_v3 + v3_avg, 'rb') as handle1:
    data_avg = pickle.load(handle1)
    
with open(outpath_v3 + v3_var, 'rb') as handle2:
    data_var = pickle.load(handle2)

df_blind = df_meta[df_meta.DataType == "B"]

## get one sample for each assay for lookup
idx_blind_assay = df_meta.Mark_ID.isin(df_blind.Mark_ID.values)
idx_test_data   = df_meta.DataType == "T" 

df_lookup = df_meta[idx_blind_assay&idx_test_data]
df_lookup = df_lookup.groupby("Mark_ID").first()
df_lookup = df_lookup.reset_index()

blind_avg = {}
blind_var = {}

for k in zip(df_blind.Cell_ID, df_blind.Mark_ID):
    
    mark = k[1]
    cell = df_lookup.Cell_ID.values[df_lookup.Mark_ID == mark][0]
    
    blind_avg[k] = data_avg[(cell, mark)]
    blind_var[k] = data_var[(cell, mark)]

outpath = "./data/processed_data/"
outname_avg     = "blind_avg_" + chrom + "_arcsinh.pickle"
outname_var     = "blind_var_" + chrom + "_arcsinh.pickle"

with open(outpath + outname_avg, 'wb') as handle1:
    pickle.dump(blind_avg, handle1, 
                protocol=pickle.HIGHEST_PROTOCOL)

with open(outpath + outname_var, 'wb') as handle2:
    pickle.dump(blind_var, handle2, 
                protocol=pickle.HIGHEST_PROTOCOL)

print(datetime.datetime.now())
print("DONE preprocessing " + chrom)