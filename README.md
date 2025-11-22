# Predicting Marketing Campaign Response Using Logistic Regression

Authors: *Rabindranatah Duran Pons, Valeria Siciliano, Rocco Lee, Yasaman Eftekharypur*

Canada, Vancouver

^University\ of\ British\ Columbia,\ MDS,\ 2025-2026^

## **About**

The project builds a machine-learning pipeline to predict whether a customer will subscribe to a marketing campaign. We use Logistic regression with a preprocessing pipeline (StandardScaler and OneHotEncoder) to handle numerical and categorical features. Since the dataset is highly imbalanced, we enabled the class_weight = "balanced" to help the model detect the minority class more efficiently.

Our final classifier performed reasonably well. After applying class-weights, the model achieved an overall accuracy of approximately 85%, with a ROC-AUC of \~0.91, indicating strong ability to distinguish between the two classes. Most importantly, the recall for the “yes” class reached 0.81, meaning the model correctly identified a large proportion of potential subscribers. However, the model still produced some false negatives—cases where it predicted “no” even though the customer would have subscribed. In a real marketing context, such errors could result in missed business opportunities. Because of this, future improvements (e.g., exploring different models or additional resampling techniques) could further enhance the model’s ability to capture rare positive cases.

The dataset used in this project is the **Bank Marketing Dataset (**[Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing)), created by researchers at the University of Minho in Portugal (Moro, Cortez, & Rita, 2014). It contains information collected from marketing phone calls conducted by a Portuguese banking institution and is widely used for teaching and research in binary classification. The dataset was sourced from the UCI Machine Learning Repository and can be accessed online. Each row represents a single customer and includes attributes such as employment type, marital status, loan status, previous campaign contacts, call duration, and the final subscription outcome.

## 

## **Dependencies**

This project relies on a Python environment defined in the included environment.yml file ([Dependencies file](https://github.com/Roccolee18/bank_marketing_group_24/blob/writing-and-editing-code/environment.yml)). It contains all the necessary libraries for data preprocessing, visualization, and building the logistic regression model. To recreate the environment, users simply need to create the Conda environment using this file before running the analysis.

## Usage

Follow the instructions below to reproduce the analysis:

### Setup

1.  Install Conda on your computer

2.  Clone this GitHub repository

3.  From the project root, create the environment using the provided environment.yml file:

    conda env create -f environment.yml

4.  Activate the environment:

    conda activate 522

5.  Launch VS Code or JupyterLab selecting the activated environment:

    code .

    OR

    jl

### Running the analysis

1.  Open the file [analysis.ipynb](https://github.com/Roccolee18/bank_marketing_group_24/blob/main/analysis.ipynb)
2.  Select the 522 kernel
3.  Run all cells to preprocess the dataset, train the regression model, evaluate the accuracy, visualize the results

## **License**

The “Predicting Marketing Campaign Response Using Logistic Regression” report contained in this repository is licensed under the MIT License, wee [LICENSE](https://github.com/Roccolee18/bank_marketing_group_24/blob/writing-and-editing-code/LICENSE) for more information.

## **References**

A data modeling approach for classification problems: application to bank telemarketing prediction Stéphane Cédric KOUMETIO TEKOUABOU, Walid Cherif, H. Silkan Published in International Conferences on… 27 March 2019 Computer Science, Business <https://www.semanticscholar.org/paper/A-data-modeling-approach-for-classification-to-bank-TEKOUABOU-Cherif/241d6ca92c4bc65ac3ee903e4732f70bff5c5e9f>

Predicting the Accuracy for Telemarketing Process in Banks Using Data Mining F. Alsolami, Farrukh Saleem, A. Al-malaise, AL-Ghamdi, Published 2020 Business, Computer Science <https://www.semanticscholar.org/paper/Predicting-the-Accuracy-for-Telemarketing-Process-Alsolami-Saleem/6391b7edcdd3c443bb57624b153bf9a8cca027db>

Using Logistic Regression Model to Predict the Success of Bank Telemarketing Y. Jiang, Published 21 June 2018 Business, Computer Science, Journal of data science <https://www.semanticscholar.org/paper/Using-Logistic-Regression-Model-to-Predict-the-of-Jiang/11ea58c843d0e745716d624b03067235dc285c30>

Prediction of Term Deposit in Bank: Using Logistic Model Enjing Jiang, Zihao Wang, Jiaying Zhao, Published in BCP Business & Management 14 December 2022 Business, Computer Science <https://www.semanticscholar.org/paper/Prediction-of-Term-Deposit-in-Bank%3A-Using-Logistic-Jiang-Wang/e36cafceaad636e9b2b558166c16be31a913ad0d>

## 
