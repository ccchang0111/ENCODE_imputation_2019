#!/bin/bash

# preprocess data
for vname in 'chr1' 'chr2' 'chr3' 'chr4' 'chr5' 'chr6' 'chr7' 'chr8' 'chr9' 'chr10' \
'chr11' 'chr12' 'chr13' 'chr14' 'chr15' 'chr16' 'chr17' 'chr18' 'chr19' 'chr20' \
'chr21' 'chr22' 'chrX'

do 
    python 00_data_generation.py $vname 5 # using 5 CPU
    python 01_data_preproc.py $vname    
done

# train each chrom
python 02_guacamole6_pretrain.py chr1 21000 150 10 10 45
python 03_guacamole6_train.py chr1 21000 400

python 02_guacamole6_pretrain.py chr2 20000 150 10 10 45
python 03_guacamole6_train.py chr2 20000 400

python 02_guacamole6_pretrain.py chr3 17000 150 25 30 45
python 03_guacamole6_train.py chr3 17000 400

python 02_guacamole6_pretrain.py chr4 16000 150 25 30 45
python 03_guacamole6_train.py chr4 16000 400

python 02_guacamole6_pretrain.py chr5 15000 150 25 30 45
python 03_guacamole6_train.py chr5 15000 400

python 02_guacamole6_pretrain.py chr6 14000 150 25 30 45
python 03_guacamole6_train.py chr6 14000 400

python 02_guacamole6_pretrain.py chr7 10000 200 25 30 45
python 03_guacamole6_train.py chr7 10000 400

python 02_guacamole6_pretrain.py chr8 10000 200 25 30 45
python 03_guacamole6_train.py chr8 10000 450

python 02_guacamole6_pretrain.py chr9 10000 200 25 30 45
python 03_guacamole6_train.py chr9 10000 450

python 02_guacamole6_pretrain.py chr10 10000 150 25 30 45
python 03_guacamole6_train.py chr10 10000 500

python 02_guacamole6_pretrain.py chr11 10000 200 25 30 45
python 03_guacamole6_train.py chr11 10000 450

python 02_guacamole6_pretrain.py chr12 10000 200 25 30 45
python 03_guacamole6_train.py chr12 10000 450

python 02_guacamole6_pretrain.py chr13 10000 200 25 30 45
python 03_guacamole6_train.py chr13 10000 450

python 02_guacamole6_pretrain.py chr14 10000 200 25 30 45
python 03_guacamole6_train.py chr14 10000 450

python 02_guacamole6_pretrain.py chr15 10000 200 25 30 45
python 03_guacamole6_train.py chr15 10000 450

python 02_guacamole6_pretrain.py chr16 10000 200 25 30 45
python 03_guacamole6_train.py chr16 10000 450

python 02_guacamole6_pretrain.py chr17 10000 200 25 30 45
python 03_guacamole6_train.py chr17 10000 450

python 02_guacamole6_pretrain.py chr18 10000 200 25 30 45
python 03_guacamole6_train.py chr18 10000 450

python 02_guacamole6_pretrain.py chr19 10000 200 25 30 60
python 03_guacamole6_train.py chr19 10000 800

python 02_guacamole6_pretrain.py chr20 10000 200 25 30 60
python 03_guacamole6_train.py chr20 10000 800

python 02_guacamole6_pretrain.py chr21 10000 200 25 30 60
python 03_guacamole6_train.py chr21 10000 800

python 02_guacamole6_pretrain.py chr22 10000 200 25 30 60
python 03_guacamole6_train.py chr22 10000 800

python 02_guacamole6_pretrain.py chrX 10000 150 25 30 45
python 03_guacamole6_train.py chrX 10000 400

# make prediction
for vname in 'chr1' 'chr2' 'chr3' 'chr4' 'chr5' 'chr6' 'chr7' 'chr8' 'chr9' 'chr10' \
'chr11' 'chr12' 'chr13' 'chr14' 'chr15' 'chr16' 'chr17' 'chr18' 'chr19' 'chr20' \
'chr21' 'chr22' 'chrX'

do 
    python 04_guacamole6_generate.py $vname   
done

# write bigwig files
for vname in 'C05M17.bigwig' 'C05M18.bigwig' 'C05M20.bigwig' 'C05M29.bigwig' \
'C06M16.bigwig' 'C06M17.bigwig' 'C06M18.bigwig' 'C07M20.bigwig' \
'C07M29.bigwig' 'C12M01.bigwig' 'C12M02.bigwig' 'C14M01.bigwig' \
'C14M02.bigwig' 'C14M16.bigwig' 'C14M17.bigwig' 'C14M22.bigwig' \
'C19M16.bigwig' 'C19M17.bigwig' 'C19M18.bigwig' 'C19M20.bigwig' \
'C19M22.bigwig' 'C19M29.bigwig' 'C22M16.bigwig' 'C22M17.bigwig' \
'C28M17.bigwig' 'C28M18.bigwig' 'C28M22.bigwig' 'C28M29.bigwig' \
'C31M01.bigwig' 'C31M02.bigwig' 'C31M16.bigwig' 'C31M29.bigwig' \
'C38M01.bigwig' 'C38M02.bigwig' 'C38M17.bigwig' 'C38M18.bigwig' \
'C38M20.bigwig' 'C38M22.bigwig' 'C38M29.bigwig' 'C39M16.bigwig' \
'C39M17.bigwig' 'C39M18.bigwig' 'C39M20.bigwig' 'C39M22.bigwig' \
'C39M29.bigwig' 'C40M16.bigwig' 'C40M17.bigwig' 'C40M18.bigwig' \
'C40M20.bigwig' 'C40M22.bigwig' 'C40M29.bigwig' 'C51M16.bigwig' \
'C51M17.bigwig' 'C51M18.bigwig' 'C51M20.bigwig' 'C51M29.bigwig'

do 
    python 06_write_bigwig_gucamole6.py $vname
done