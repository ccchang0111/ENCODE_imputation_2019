Codes for reproducing the submission for **Team Lavawizard** in [2019 ENCODE Imputation Challenge](https://www.synapse.org/#!Synapse:syn17083203/wiki/587192)

## Installation
1. `git clone https://github.com/ccchang0111/ENCODE_imputation_2019.git`
2. ```conda env create -f env.yml```

## How to train the model & generate predictions

1. Download all original [training](https://www.synapse.org/#!Synapse:syn18143306) & [validation](https://www.synapse.org/#!Synapse:syn18143307) data (.bigwig) into `/data/training_data/` and `/data/validation_data/` respectively
2. Download [`submission_template.bigwig`](https://www.synapse.org/#!Synapse:syn18145351) file to the folder `/data/submission_template/`
3. Run `Lavawizard_pipeline.sh` from the command line

or alternatively,

## How to generate predictions from pretrained models

1. Download all [pretrained models](https://www.synapse.org/#!Synapse:syn21519009) (23 files) to the folder `/data/processed_data/keras_model/`
2. Run `Lavawizard_pipeline_using_pretrained_model.sh`
