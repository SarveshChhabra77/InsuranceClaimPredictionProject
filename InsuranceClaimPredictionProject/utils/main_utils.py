from InsuranceClaimPredictionProject.logging.logger import logging
from InsuranceClaimPredictionProject.exceptions.exception import ClaimPredictionException
from InsuranceClaimPredictionProject.entity.artifacts_config import ClassificationMetricArtifact
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import recall_score,roc_auc_score
import os
import sys
import yaml
import pandas as pd
import numpy as np
import pickle

def write_yaml_file(file_path:str,content:object,replace:bool=False)->None:
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
        dir_name = os.path.dirname(file_path)
        os.makedirs(dir_name,exist_ok=True)

        with open(file_path,'w') as file_obj:
            yaml.dump(content,file_obj)
    except Exception as e:
        raise ClaimPredictionException(e,sys)

def save_numpy_array_data(file_path:str,array:np.array):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'wb') as file:
            np.save(file,array)
            
    except Exception as e:
        raise ClaimPredictionException(e,sys)
    
    
def save_obj(file_path:str,obj:object)->None:
    try:
        logging.info('Entered the save-object method of main_utils ')
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'wb') as file_obj:
            pickle.dump(obj,file_obj)
        logging.info('Exited the save_object method of the main_utils')
    except Exception as e:
        raise ClaimPredictionException(e,sys)

def load_numpy_array_data(file_path:str)->np.array:
    try:
        if not os.path.exists(file_path):
            raise Exception(f'File path : {file_path} not exist')
        with open(file_path,'rb') as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise ClaimPredictionException(e,sys)
    
def load_obj(file_path:str)->object:
    try:
        if not os.path.exists(file_path):
            raise Exception(f'The file path : {file_path} is not exist')
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise ClaimPredictionException(e,sys)
    
    




def evaluate_models(
    x_train, 
    y_train, 
    x_test, 
    y_test, 
    models, 
    params, 
    recall_weight=0.6, 
    roc_auc_weight=0.4):

    try:
        report = {}
        tuned_models = {}

        for model_name, model in models.items():
            para = params.get(model_name, {})
            rcv = RandomizedSearchCV(
                estimator=model,
                param_distributions=para,
                cv=5,
                n_jobs=-1,
                n_iter=20,
                random_state=42,
                scoring='recall'  # prioritize recall during tuning
            )

            rcv.fit(x_train, y_train)
            best_model = rcv.best_estimator_

            # Get predictions & probabilities
            y_pred = best_model.predict(x_test)
            if hasattr(best_model, "predict_proba"):
                y_prob = best_model.predict_proba(x_test)[:, 1]
            elif hasattr(best_model, "decision_function"):
                # fallback for models without predict_proba (like SVM without probability=True)
                y_prob = best_model.decision_function(x_test)
            else:
                y_prob = y_pred  # last resort (not ideal, but avoids crash)

            # Calculate metrics
            test_recall = recall_score(y_test, y_pred)
            test_roc_auc = roc_auc_score(y_test, y_prob)

            final_score = (recall_weight * test_recall) + (roc_auc_weight * test_roc_auc)

            report[model_name] = {
                "recall": round(test_recall, 4),
                "roc_auc": round(test_roc_auc, 4),
                "final_score": round(final_score, 4)
            }
            tuned_models[model_name] = best_model

        return report, tuned_models

    except Exception as e:
        raise ClaimPredictionException(e, sys)
    
def get_classification_score(y_true,y_pred)->ClassificationMetricArtifact:
    try:
        model_recall_score=recall_score(y_true,y_pred)
        model_roc_auc_score=roc_auc_score(y_true,y_pred)

        regression_metric=ClassificationMetricArtifact(
            recall_score=model_recall_score,
            roc_auc_score=model_roc_auc_score
        )
        return regression_metric
    except Exception as e:
        raise ClaimPredictionException(e,sys)


class ClaimPredictionModel:
    
    def __init__(self,preprocessor,model):
        try:
            self.preprocessor=preprocessor
            self.model=model
        except Exception as e:
            raise ClaimPredictionException(e,sys)
    def predict(self,x):
        try:
            x_transform=self.preprocessor.transform(x)
            y_pred=self.model.predict(x_transform)
            return y_pred
        except Exception as e:
            raise ClaimPredictionException(e,sys)
        


def transform_dates_and_csl(df: pd.DataFrame) -> pd.DataFrame:
        try:
            logging.info("Starting date and CSL transformations.")
            
            # Convert policy_bind_date
            if 'policy_bind_date' in df.columns:
                df['policy_bind_date'] = pd.to_datetime(df['policy_bind_date'])
                df['policy_bind_year'] = df['policy_bind_date'].dt.year
                df['policy_bind_month'] = df['policy_bind_date'].dt.month
                df['policy_bind_day'] = df['policy_bind_date'].dt.day
                df.drop('policy_bind_date', axis=1, inplace=True)

            # Convert incident_date
            if 'incident_date' in df.columns:
                df['incident_date'] = pd.to_datetime(df['incident_date'])
                df['incident_date_year'] = df['incident_date'].dt.year
                df['incident_date_month'] = df['incident_date'].dt.month
                df['incident_date_day'] = df['incident_date'].dt.day
                df.drop('incident_date', axis=1, inplace=True)

            # Handle policy_csl (like "250/500")
            if 'policy_csl' in df.columns:
                df['csl_per_person'] = df['policy_csl'].str.split('/').str[0].astype(int)
                df.drop('policy_csl', axis=1, inplace=True)
            
            # Drop columns not needed
            drop_cols = [
                "policy_number", "insured_zip", "incident_state", "incident_city",
                "incident_location", "policy_csl", "_c39"
            ]
            df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

            logging.info("Date and CSL transformations completed.")
            return df
        except Exception as e:
            raise ClaimPredictionException(e, sys)