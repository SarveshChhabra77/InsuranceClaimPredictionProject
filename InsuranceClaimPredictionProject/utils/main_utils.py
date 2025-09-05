from InsuranceClaimPredictionProject.logging.logger import logging
from InsuranceClaimPredictionProject.exceptions.exception import ClaimPredictionException
from InsuranceClaimPredictionProject.entity.artifacts_config import ClassificationMetricArtifact
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import recall_score,roc_auc_score
import os
import sys
import yaml
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
    
    


# def evaluate_models(x_train, y_train, x_test, y_test, models, params):
#     try:
#         report = {}
#         tunned_Models = {}

#         for model_name, model in models.items():
#             para = params[model_name]

#             rcv = RandomizedSearchCV(model, para, cv=5, n_jobs=-1, n_iter=20, random_state=42)
#             rcv.fit(x_train, y_train)

#             best_model = rcv.best_estimator_
#             y_pred = best_model.predict(x_test)

#             # ✅ Calculate multiple metrics
#             test_recall = recall_score(y_test, y_pred)
#             test_roc_auc = roc_auc_score(y_test, y_pred)

#             # ✅ Weighted score (you can tune weights based on business need)
#             final_score = (0.6 * test_recall) + (0.4 * test_roc_auc)

#             report[model_name] = final_score
#             tunned_Models[model_name] = best_model

#         return report, tunned_Models

#     except Exception as e:
#         raise ClaimPredictionException(e, sys)



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