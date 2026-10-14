# Python Intro for Data Science 
This is a repository to upload files related to CodeSpace DataScience + IA Master. 
We will cover from data analysis to model creation&evaluation using MLFlow and model deployment using Docker. 
We will employ a datataset from a challenge in Kaggle: Loan Prediction Challenge! (https://www.kaggle.com/competitions/playground-series-s4e10/submissions)

**Languages and Data Science tools**

![Python](https://img.shields.io/badge/Python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![mlflow](https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)


# Outline of the analysis: 

## 0. Github codespace

We will perform everything using a codespace from github. To do so, we need to go to our github repository, create a branch named as "mlops" and then go to the green badge named as "<> Code" and then, within the "Codespaces" tab, click in the "..." to "Open in Visual Studio Code". Esentially, we are using the virual machine in Github but we are running it in our local installed VSCode as it is easier to manage

## 1. Model Prediction
Perform data analysis and model prediction optimization and evaluation using a dataset from Kaggle: Loan Prediction Challenge (https://www.kaggle.com/competitions/playground-series-s4e10/submissions)

### 1.1. Setup

a) Download Loan Predictions challenge dataset from https://www.kaggle.com/competitions/playground-series-s4e10/submissions
b) Create a conda environment. To create a new conda environment use:

```bash
conda create -n myenv python=3.14
```

c) Install required libraries. To install all required libraries at once, use:

```bash
pip install -r requirements.txt
```
d) Install MLFlow

```bash
pip install mlflow
```

### 1.2. Exploratory Data Analysis (EDA)

We will perform encode categorical variables and scale numerical variables, then we will simply evaluate the included variables in the analysis using a correlation matrix with the predicted/target variable

### 1.3. Model

We will split the data into test and training data and test several models, such as RandomForesClassifier, CatBoostClassifier, HistGradientBoostingClassifier, xgboost and lightgbm. We will store all those models in MLFlow to track them. 
Once we have selected that model, we will optimize it by testing different training dataset splitting proportions and several thresholds to select the minimize the profit sacrificed (FP) while maximizing the loss avoided (TP): objective = Maximize(Amount of money we lose by TP - Amount of money we win by FP)

### 1.4. Submission file

Once we have tested several models parameters, we prepare the submission file to upload it to MLFlow run


## 2. MLFlow 
Track created models and obtained predictions from the same training dataset using MLFlow

### 2.1. Connect to MLflow

To start the MLflow server, in the terminal, run:

```bash
mlflow server
```

### 2.2. Submit to MLflow

To connect to MLflow server, in python notebook chunks run:

```python
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("loan_prediction")
```
To create runs and store them in MLflow server, in python run:

```python
with mlflow.start_run():

    mlflow.set_tag("model", model.__class__.__name__)

    mlflow.log_metric("auc", auc_metric)
    mlflow.log_metric("optimal_threshold", optimal_threshold)
    mlflow.log_metric("fpr_optimal", fpr_optimal)
    mlflow.log_metric("tpr_optimal", tpr_optimal)
    mlflow.log_metric("average_threshold", average_threshold)
    mlflow.log_metric("fpr_average", fpr_average)
    mlflow.log_metric("tpr_average", tpr_average)

    trusted_types = [
    "catboost.core.CatBoostClassifier",
    "sklearn.metrics._ranking.roc_auc_score",
    "sklearn.metrics._scorer._CurveScorer",
    "sklearn.metrics._scorer._Scorer",
    "sklearn.utils._metadata_requests.MetadataRequest",
    "sklearn.utils._metadata_requests.MethodMetadataRequest",
    ]

    mlflow.sklearn.log_model(
    model,
    "model",
    skops_trusted_types=trusted_types,
    )

  #  mlflow.sklearn.log_model(model, "model")
    mlflow.log_artifact(SUBMISSION_CSV_PATH)
```

## 3. Model deployment with Docker

## 3.1. Option A: Create a dockerfile within the main branch, build docker image and run it

To build dockerfile image

```bash
docker build -t my_first_image -f ./Dockerfile .
```

To run Docker file

```bash
docker run --rm my_first_image
```
## 3.2. Deploy the model using MLFlow and Docker.

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
mlflow models build-docker \
  --model-uri "models:/loan_prediction_model/1" \
  --name "mymodel/loan_prediction_model"
docker run <my_docker_image_name>
```

In our case it was:

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
mlflow models build-docker --model-uri models:/<my_model>/Mmy_version> -n <my_docker_image_name>
docker run mymodel/loan_prediction_model
```

Optional: To check available registered models:

```python
import mlflow
mlflow.set_tracking_uri("http://localhost:5000")

client = mlflow.tracking.MlflowClient()

# List registered model versions
for mv in client.search_model_versions("name='loan_prediction_model'"):
    print(mv.name, mv.version, mv.aliases, mv.source)

# Or inspect a specific model
print(client.get_registered_model("loan_prediction_model"))
```


## 4. Kaggle challenge result (Spoiler alert!)

This is the final result obtained with Kaggle, employing CatBoostClassifier and optimizing the thresold to minimize the number of false positives (FP) (possible profit loss) and maximizing true positives (TP) or loss avoided:

![description](https://github.com/Irampardo/Python-Intro-for-Data-Science/blob/main/Kaggle_LoanPrediction_Challenge.png?raw=true)





