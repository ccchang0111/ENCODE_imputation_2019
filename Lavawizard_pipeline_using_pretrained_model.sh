#!/bin/bash

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
