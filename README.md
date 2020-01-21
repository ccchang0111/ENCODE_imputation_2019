Codes for reproducing the submission for Team Lavawizard in [2019 ENCODE Imputation Challenge](https://www.synapse.org/#!Synapse:syn17083203/wiki/587192)

## How to train the model & generate predictions

1. ```conda env create -f env.yml```
2. ```conda activate encode```
3. Please put all original bigwig files into `/data/training_data/` and `/data/validation_data/` respectively
4. Please download `submission_template.bigwig` file from the competion website into `/data/submission_template/` folder
5. Run `Lavawizard_pipeline.sh` from the command line

## How to generate predictions from trained models

1. Download all the [trained model](https://www.synapse.org/#!Synapse:syn21519009) (23 files) to the folder `/data/processed_data/keras_model/`
2. Run `Lavawizard_pipeline_using_pretrained_model.sh`
