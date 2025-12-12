SHELL := /bin/bash

.PHONY: all step_read_data step_preprocess step_data_validation step_EDA step_evaluation step_report clean

all: step_read_data step_preprocess step_data_validation step_EDA step_evaluation step_report

# Ensure directories exist
prepare_dirs:
	mkdir -p data/raw data/processed results/models results/figures results/tables report

# 1. Read Data
step_read_data: scripts/data_load.py prepare_dirs
	python scripts/data_load.py --dataset_id 222 --output-path data/raw --output-name bank_marketing.csv

# 2. Preprocess the data
step_preprocess: scripts/split_n_preprocess.py step_read_data
	python scripts/split_n_preprocess.py --raw-data data/raw/bank_marketing.csv --data-to data/processed --preprocessor-to results/models

# 3. Validation
step_data_validation: scripts/validate_model.py step_preprocess
	python scripts/validate_model.py --input-path data/processed/bank_train.csv

# 4. EDA
step_EDA: scripts/eda.py step_preprocess
	python scripts/eda.py --data data/processed/bank_train.csv --plot-to results/figures

# 5. Fit and Predict
step_evaluation: scripts/fit_and_predict.py step_EDA
	python scripts/fit_and_predict.py --save_location results/figures/ \
		--preprocessor_pickle results/models/bank_preprocessor.pickle \
		--train_dataset_path data/processed/bank_train.csv \
		--test_dataset_path data/processed/bank_test.csv

# 6. Report
step_report: report/marketing_campain_predictor.qmd step_evaluation
	quarto render report/marketing_campain_predictor.qmd --to html

clean:
	rm -f data/raw/*
	rm -f data/processed/*
	rm -f results/figures/*
	rm -f results/models/*
	rm -f results/tables/*
	rm -f report/marketing_campain_predictor.html
