Reproducible codes for 2019 ENCODE Imputation Challenge

https://www.synapse.org/#!Synapse:syn17083203/wiki/587192

## How TO

1. ```conda env create -f env.yml```
2. ```conda activate encode```
3. please put all original bigwig files into `/data/training_data/` and `/data/validation_data/` respectively
4. please download `submission_template.bigwig` file from the competion website into `/data/submission_template/` folder
5. run `Gucamole_pipeline.sh` from the command line