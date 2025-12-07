# Predicting Marketing Campaign Response Using Logistic Regression

Authors: *Rabindranath Duran Pons, Valeria Siciliano, Rocco Lee, Yasaman Eftekharypur*

Canada, Vancouver

Demo of a data analysis project for DSCI 522 (Data Science workflows); a course in the Master of Data Science program at the University of British Columbia. 2025-2026.

## **About**

The project builds a machine-learning pipeline to predict whether a customer will subscribe to a marketing campaign. We use Logistic regression with a preprocessing pipeline (StandardScaler and OneHotEncoder) to handle numerical and categorical features. Since the dataset is highly imbalanced, we enabled the class_weight = "balanced" to help the model detect the minority class more efficiently.

Our final classifier performed reasonably well. After applying class-weights, the model achieved an overall accuracy of approximately 85%, with a ROC-AUC of \~0.91, indicating strong ability to distinguish between the two classes. Most importantly, the recall for the “yes” class reached 0.81, meaning the model correctly identified a large proportion of potential subscribers. However, the model still produced some false negatives—cases where it predicted “no” even though the customer would have subscribed. In a real marketing context, such errors could result in missed business opportunities. Because of this, future improvements (e.g., exploring different models or additional resampling techniques) could further enhance the model’s ability to capture rare positive cases.

The dataset used in this project is the **Bank Marketing Dataset (**[Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing)), created by researchers at the University of Minho in Portugal (Moro, Cortez, & Rita, 2014). It contains information collected from marketing phone calls conducted by a Portuguese banking institution and is widely used for teaching and research in binary classification. The dataset was sourced from the UCI Machine Learning Repository and can be accessed online. Each row represents a single customer and includes attributes such as employment type, marital status, loan status, previous campaign contacts, call duration, and the final subscription outcome.

## Usage in a fully configured Conda environment (Recommended)

## **Dependencies**

- [Docker](https://www.docker.com/) 

Follow the instructions below to reproduce the analysis in a fully configured Conda environment:

### Setup

> If you are using Windows or Mac, make sure Docker Desktop is running.

1.  Clone this GitHub repository by running the following command and navigate to the root of the project:

```bash
git pull https://github.com/Roccolee18/bank_marketing_group_24
```

2. Pull the Docker image

Download the latest version of the image from Docker Hub by running this command:

```bash
docker pull roccolee18/docker-condalock-jupyterlab:latest
```

3. Run the container with the following command:

```bash
docker run -it --rm \
    -p 8888:8888 \
    -v $(pwd):/workplace \
    roccolee18/docker-condalock-jupyterlab
```

When the container starts, you’ll see a URL in the terminal like:

```bash
http://127.0.0.1:8888/lab
```

Open the link in a new browser window to ensure the Docker image is set up properly.


4. Run the analysis scripts

**These next few steps should be completed in the terminal of the newly opened browser after the docker container is set up (http://127.0.0.1:8888/lab)**

The analysis has been split into several python scripts for ease of execution.

To begin running the scripts, run the following commands in a new terminal, again, in the environment created by the docker container in the previous step. The order of execution ideally should be the following:

1.  Run data loading script:

```bash
python scripts/data_load.py --dataset_id 222 --output-path data/raw --output-name bank_marketing.csv
```

2. Run data validation script:

```bash
python scripts/validate_model.py --input-path data/processed/bank_train.csv
```

3. Run split and preprocessing script:

```bash
python scripts/split_n_preprocess.py --raw-data data/raw/bank_marketing.csv --data-to data/processed --preprocessor-to results/models
```

4. Run eda script:

```bash
python scripts/eda.py --data data/processed/bank_train.csv --plot-to results/figures
```

5. Run fit and predict script:

```bash
python scripts/fit_and_predict.py --save_location results/figures/ --preprocessor_pickle results/models/bank_preprocessor.pickle --train_dataset_path data/processed/bank_train.csv --test_dataset_path data/processed/bank_test.csv
```

The required arguments for each script can be found with the following command:

```bash
python <script_name.py> --help
```

5. Generate the report and view it on your browser (Google Chrome is recommended):

After all scripts have been run, the .qmd report under the /report directory should be populated with the analysis results from the latest run of the scripts. To generate the final html report, run the following command:

```bash
quarto render report/marketing_campain_predictor.qmd --to html
```

After that you can open the report ([report/marketing_campain_predictor.html](https://github.com/Roccolee18/bank_marketing_group_24/blob/main/report/marketing_campain_predictor.html)) on your browser to visualize it (Google Chrome is recommended).

6. Stop the container

When finished, press Ctrl + C in the terminal to stop the Jupyter server and exit the container.

## Usage in a local environment setup (Alternative to Docker setup)

## **Dependencies**

This project relies on a Python environment defined in the included environment.yml file ([Dependencies file](https://github.com/Roccolee18/bank_marketing_group_24/blob/main/environment.yml)). It contains all the necessary libraries for data preprocessing, visualization, and building the logistic regression model. To recreate the environment, users simply need to create the Conda environment using this file before running the analysis.

Follow the instructions below to reproduce the analysis:

### Setup

1.  Install Conda on your computer

2.  Clone this GitHub repository

3.  From the project root, create the environment using the provided environment.yml file:

```bash
conda env create -f environment.yml
```

4.  Activate the environment:

```bash
conda activate 522
```

5.  Launch VS Code or JupyterLab selecting the activated environment:

```bash
code .
```

OR

```bash
jl
```

### Running the analysis

1.  Run the scripts

The analysis has been split into several python scripts for ease of execution.

To begin running the scripts, run the following commands in a new terminal in the git directory. **Ensure that the 522 environment created from the environment.yml file in the preivous step is active**
The order of execution ideally should be the following:

1.  Run data loading script:

```bash
python scripts/data_load.py --dataset_id 222 --output-path data/raw --output-name bank_marketing.csv
```

2. Run data validation script:

```bash
python scripts/validate_model.py --input-path data/processed/bank_train.csv
```

3. Run split and preprocessing script:

```bash
python scripts/split_n_preprocess.py --raw-data data/raw/bank_marketing.csv --data-to data/processed --preprocessor-to results/models
```

4. Run eda script:

```bash
python scripts/eda.py --data data/processed/bank_train.csv --plot-to results/figures
```

5. Run fit and predict script:

```bash
python scripts/fit_and_predict.py --save_location results/figures/ --preprocessor_pickle results/models/bank_preprocessor.pickle --train_dataset_path data/processed/bank_train.csv --test_dataset_path data/processed/bank_test.csv
```

The required arguments for each script can be found with the following command:

```bash
python <script_name.py> --help
```

2. Generate the report and view it on your browser (Google Chrome is recommended):

After all scripts have been run, the .qmd report under the /report directory should be populated with the analysis results from the latest run of the scripts. To generate the final html report, run the following command:

```bash
quarto render report/marketing_campain_predictor.qmd --to html
```

After that you can open the report ([report/marketing_campain_predictor.html](https://github.com/Roccolee18/bank_marketing_group_24/blob/main/report/marketing_campain_predictor.html)) on your browser to visualize it (Google Chrome is recommended).

3. Deactivate the environment

To return your environment back to its default state, run the following command:

```bash
conda deactivate
```

## **License**

The “Predicting Marketing Campaign Response Using Logistic Regression” report contained in this repository is licensed under the MIT License, see [LICENSE](https://github.com/Roccolee18/bank_marketing_group_24/blob/writing-and-editing-code/LICENSE) for more information.

## **References**

A data modeling approach for classification problems: application to bank telemarketing prediction Stéphane Cédric KOUMETIO TEKOUABOU, Walid Cherif, H. Silkan Published in International Conferences on… 27 March 2019 Computer Science, Business <https://www.semanticscholar.org/paper/A-data-modeling-approach-for-classification-to-bank-TEKOUABOU-Cherif/241d6ca92c4bc65ac3ee903e4732f70bff5c5e9f>

Predicting the Accuracy for Telemarketing Process in Banks Using Data Mining F. Alsolami, Farrukh Saleem, A. Al-malaise, AL-Ghamdi, Published 2020 Business, Computer Science <https://www.semanticscholar.org/paper/Predicting-the-Accuracy-for-Telemarketing-Process-Alsolami-Saleem/6391b7edcdd3c443bb57624b153bf9a8cca027db>

Using Logistic Regression Model to Predict the Success of Bank Telemarketing Y. Jiang, Published 21 June 2018 Business, Computer Science, Journal of data science <https://www.semanticscholar.org/paper/Using-Logistic-Regression-Model-to-Predict-the-of-Jiang/11ea58c843d0e745716d624b03067235dc285c30>

Prediction of Term Deposit in Bank: Using Logistic Model Enjing Jiang, Zihao Wang, Jiaying Zhao, Published in BCP Business & Management 14 December 2022 Business, Computer Science <https://www.semanticscholar.org/paper/Prediction-of-Term-Deposit-in-Bank%3A-Using-Logistic-Jiang-Wang/e36cafceaad636e9b2b558166c16be31a913ad0d>

## 
