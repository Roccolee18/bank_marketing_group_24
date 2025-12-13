SHELL := /bin/bash

.PHONY: all clean cl env build run up stop docker-build-push docker-build-local

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
	rm -rf report/marketing_campain_predictor_files
	rm -rf src/__pycache__

cl: ## create conda lock for multiple platforms
	# the linux-aarch64 is used for ARM Macs using linux docker container
	conda-lock lock \
		--file environment.yml \
		-p linux-64 \
		-p osx-64 \
		-p osx-arm64 \
		-p win-64 \
		-p linux-aarch64

env: ## remove previous and create environment from lock file
	# remove the existing env, and ignore if missing
	conda env remove dockerlock || true
	conda-lock install -n dockerlock conda-lock.yml

build: ## build the docker image from the Dockerfile
	docker build -t dockerlock --file Dockerfile .

run: ## alias for the up target
	make up

up: ## stop and start docker-compose services
	# by default stop everything before re-creating
	make stop
	docker-compose up -d

stop: ## stop docker-compose services
	docker-compose stop

# docker multi architecture build rules (from Claude) -----

docker-build-push: ## Build and push multi-arch image to Docker Hub (amd64 + arm64)
	docker buildx build \
		--platform linux/amd64,linux/arm64 \
		--tag chendaniely/docker-condalock-jupyterlab:latest \
		--tag chendaniely/docker-condalock-jupyterlab:local-$(shell git rev-parse --short HEAD) \
		--push \
		.

docker-build-local: ## Build single-arch image for local testing (current platform only)
	docker build \
		--tag chendaniely/docker-condalock-jupyterlab:local \
		.