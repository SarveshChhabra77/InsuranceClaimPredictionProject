# 🚑 Insurance Claim Prediction Project

## Overview
InsuranceClaimPredictionProject trains and serves a binary classifier that predicts whether an insurance claim is fraudulent, including a Streamlit UI for interactive predictions and a modular training pipeline with data ingestion, validation, transformation, post-validation checks, model training, and MLflow tracking. The repository includes raw data, notebooks for cleaning/EDA/modeling, saved final artifacts (model and preprocessor), and timestamped pipeline artifacts and logs for reproducibility and auditing.

## Features
- End-to-end training pipeline (ingestion → validation → transformation → post-validation → training) orchestrated by main.py with timestamped artifact folders and structured logs.
- Streamlit UI (app.py) that loads the trained model and preprocessor, infers feature inputs from the dataset, applies preprocessing, and returns Fraud/Not Fraud predictions.
- MLflow experiment outputs stored under mlruns (metrics, model artifacts metadata), with CatBoost/XGBoost/ensemble options and recall/ROC-AUC evaluation emphasis.

## Tech stack
- Language: Python with modular package InsuranceClaimPredictionProject and CLI entry scripts app.py and main.py.
- ML/DL: scikit-learn, CatBoost, XGBoost, imbalanced-learn, category-encoders, NumPy, pandas for modeling, encoding, and sampling.
- App/Tracking/Visualization: Streamlit for UI; MLflow for experiment tracking; seaborn/matplotlib/plotly for analysis; Jupyter notebooks for EDA and modeling.

## Key dependencies
- Core: pandas, numpy, scikit-learn, scipy, joblib, category-encoders, imbalanced-learn, xgboost, catboost.
- Serving/UX: streamlit, uvicorn (optional dev server), Flask and FastAPI present but UI entrypoint uses Streamlit (app.py).
- Tracking/Utilities: mlflow, matplotlib, seaborn.
- See requirements.txt for the complete, version-pinned list used by this codebase.

## Project structure
```
InsuranceClaimPredictionProject-main
|
|
├── data
│   └── raw
│       └── insurance_claims.csv
├── data_schema
│   └── __init__.py
├── final_model
│   ├── model.pkl
│   └── preprocessor.pkl
├── InsuranceClaimPredictionProject```
|   ├── components
│   │   ├── __init__.py
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   ├── data_validation.py
│   │   ├── model_trainer.py
│   │   └── post_validation.py
│   ├── constants
│   │   └── __init__.py
│   ├── entity
│   │   ├── __init__.py
│   │   ├── artifacts_config.py
│   │   └── config_entity.py
│   ├── exceptions
│   │   ├── __init__.py
│   │   └── exception.py
│   ├── logging
│   │   ├── __init__.py
│   │   └── logger.py
│   └── utils
│       ├── __init__.py
│       └── main_utils.py
├── mlruns
│   └── 0
│       ├── <many run folders with metrics```gs/outputs/meta.yaml>
│       └── meta.yaml
├── notebook
│   ├── catboost_info
│   │   ├── learn
│   │   │   └── events.out.tfevents```  │   ├── tmp
│   │   │   ├── cat_feature_index.*```p
│   │   │   └── ...
│   │   ├── catboost_training.json
│   │   ├── learn_error.tsv
│   │   └── time_left.tsv
│   ├── data
│   │   └── processed
│   │       └── processed_data.csv
│   ├── 1_data_cleaning.ipynb
│   ├── 2_Eda.ipynb
│   ├── 3_modeling.ipynb
│   ├── __init__.py
│   ├── model
│   ├── model.pkl
│   ├── preprocessor.pkl
│   └── smote.pkl
├── .gitignore
├── app.py
├── main.py
├── README.md
├── requirements.txt
└── setup.py
```


## Data
- Raw dataset lives at data/raw/insurance_claims.csv and is used to derive Streamlit UI feature lists and numeric ranges for inputs at inference time in app.py.
- The internal DataIngestion class currently reads from an absolute Windows path in export_data_as_dataframe and should be updated to the repository path (data/raw/insurance_claims.csv) for portability before running the pipeline.

## Training pipeline
- Entry point: python main.py builds a timestamped Artifacts directory containing data_ingestion, data_validation, data_transformation, post_data_validation, and model_trainer outputs, with logs emitted to ./logs.
- Configuration classes (TrainingPipelineConfig, DataIngestionConfig, DataValidationConfig, DataTransformationConfig, PostDataValidationConfig, ModelTrainingConfig) define folder layout and file names for artifacts and intermediate data.
- DataIngestion exports a feature store copy and splits train/test; DataValidation enforces schema, drops extras, checks missingness and drift; DataTransformation encodes categorical features (OneHot for low-cardinality, TargetEncoder for high-cardinality), scales numerics, applies SMOTE, and serializes the preprocessor and NumPy arrays.
- PostDataValidation validates transformed arrays (shape consistency, NaN checks, binary target values) prior to training; ModelTrainer evaluates multiple models with hyperparameter search, prioritizing recall, logs metrics/models to MLflow, and persists the best model.

## Serving (Streamlit UI)
- Run the interactive app with: streamlit run app.py, which loads final_model/model.pkl and uses transform_dates_and_csl to align inputs with training-time preprocessing before prediction.
- The app builds input widgets dynamically from the dataset columns (categorical select boxes from unique values, numeric sliders from min/max/mean), then returns “Fraud” or “Not Fraud” based on the model prediction.

## Getting started

### Prerequisites
- Python environment with virtualenv or conda is recommended; install dependencies from requirements.txt at the repo root for exact versions.[1]
- The project includes setup.py to package the InsuranceClaimPredictionProject module; primary execution happens via main.py (training) and app.py (inference UI).[1]

### Installation
```bash
# create and activate a virtual environment```xample with venv)
python -m venv .venv
source .venv/bin/activate  # on Windows```venv\\Scripts\\activate

# install exact dependencies
pip install --upgrade pip
pip install -r requirements.txt
```


### Configuration notes
- Update paths in InsuranceClaimPredictionProject/components/data_ingestion.py and InsuranceClaimPredictionProject/constants/__init__.py to use repository-relative paths instead of absolute Windows drives before running the training pipeline.[1]
- The schema module is referenced via constants.Schema_file_path; ensure it points to data_schema/__init__.py inside this repo for consistent validation behavior.

## How to run

### Train the model
```bash
python main.py
```
This executes the pipeline stages sequentially and writes artifacts to Artifacts/<timestamp>, with logs in ./logs and MLflow runs under ./mlruns/0.

### Launch the UI
```bash
streamlit run app.py
```
This loads final_model/model.pkl and preprocessor.pkl to make predictions from user inputs derived from the dataset columns in data/raw/insurance_claims.csv.

## Artifacts and outputs
- final_model/model.pkl and final_model/preprocessor.pkl contain the trained estimator and preprocessing pipeline for inference outside the notebook context.
- The mlruns directory holds MLflow metrics (Recall Score, Roc Auc Score), run metadata, and logged models for experiment comparison and reproducibility.

## Notebooks
- 1_data_cleaning.ipynb, 2_Eda.ipynb, and 3_modeling.ipynb provide iterative development steps for cleaning, exploratory analysis, and model experimentation under the notebook/ folder.
- Additional notebook artifacts include intermediate processed data and serialized objects for quick re-use during exploration.

## Logging and monitoring
- Logging is configured to write timestamped log files to a logs directory with a consistent format for line number, module, and level, aiding debugging of pipeline runs.
- MLflow metrics and model registries in mlruns allow comparing runs, tracking configurations, and auditing changes in model performance over time.

## Testing and quality
- No automated test suite or pytest configuration was found; consider adding unit tests for components and data-contract checks in future iterations.
- Type/lint tools appear in the dependency set (e.g., mypy extensions), but no lint/format configs were detected; adding pyproject.toml or tool configs would help standardize quality gates.

## Packaging
- setup.py defines the package metadata and installs InsuranceClaimPredictionProject via find_packages(), enabling absolute imports across pipeline modules.
- Requirements are pinned in requirements.txt to ensure deterministic environments across machines and CI systems.

## Known caveats
- Absolute Windows paths in data_ingestion.py and constants.Schema_file_path must be made relative for cross-platform portability; adjust to repository-local paths before running training.
- The Streamlit app depends on data/raw/insurance_claims.csv at runtime to construct input widgets; ensure this file exists and matches the training schema.



## Acknowledgments
- The project integrates CatBoost, XGBoost, scikit-learn, imbalanced-learn, and category-encoders, with MLflow for experiment tracking and Streamlit for UI, as reflected in code and requirements.
- Notebooks and serialized artifacts under notebook/ and final_model/ capture exploration outputs and the final deployable pipeline components.

## Full requirements (pinned)
- The complete dependency list is available in requirements.txt at the repository root; install with pip install -r requirements.txt for an identical environment.
- Selected highlights include: streamlit, mlflow, scikit-learn, xgboost, catboost, imbalanced-learn, category-encoders, pandas, numpy, seaborn, matplotlib, and plotly for the end-to-end workflow.

## Quickstart
- Install dependencies: pip install -r requirements.txt from the repo root in an activated virtual environment.
- Train the pipeline: python main.py to generate Artifacts/<timestamp> and MLflow runs.
- Serve predictions: streamlit run app.py to open the UI in a browser and make Fraud/Not Fraud predictions.

=======
