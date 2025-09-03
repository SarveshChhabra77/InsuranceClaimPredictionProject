from InsuranceClaimPredictionProject.logging.logger import logging
from InsuranceClaimPredictionProject.exceptions.exception import ClaimPredictionException
from InsuranceClaimPredictionProject.entity.artifacts_config import ModelTrainerArtifact,PostDataValidationArtifact,ClassificationMetricArtifact
from InsuranceClaimPredictionProject.utils.main_utils import save_obj,load_numpy_array_data,evaluate_models,get_classification_score,load_obj,ClaimPredictionModel
from InsuranceClaimPredictionProject.entity.config_entity import ModelTrainingConfig
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier,ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import r2_score,recall_score
from sklearn.model_selection import RandomizedSearchCV
import mlflow
import sys
import os


class ModelTrainer:
    def __init__(self,model_trainer_config:ModelTrainingConfig,post_data_validation_artifact:PostDataValidationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.post_data_validation_artifact = post_data_validation_artifact
        except Exception as e:
            raise ClaimPredictionException(e,sys)
        
    def track_mlflow(self,best_model,classification_metrics:ClassificationMetricArtifact):
        try:
            with mlflow.start_run():
                model_recall_score = classification_metrics.recall_score
                model_roc_auc_score = classification_metrics.roc_auc_score
                
                mlflow.log_metric('Recall Score',model_recall_score)
                mlflow.log_metric('Roc Auc Score',model_roc_auc_score)
                
                mlflow.sklearn.log_model(best_model,'Classification_Model')
                
        except Exception as e:
            raise ClaimPredictionException(e,sys)    
        
        
    def model_trainer(self,X_train,y_train,X_test,y_test):
        try:
            classification_models = {
                    "Logistic Regression": LogisticRegression(),
                    "K-Nearest Neighbors": KNeighborsClassifier(),
                    "Decision Tree": DecisionTreeClassifier(),
                    "Random Forest": RandomForestClassifier(),
                    "Extra Trees": ExtraTreesClassifier(),
                    "AdaBoost": AdaBoostClassifier(),
                    "Gradient Boosting": GradientBoostingClassifier(),
                    "XGBoost": XGBClassifier(),
                    "CatBoost": CatBoostClassifier(verbose=False),
                    "Naive Bayes (Gaussian)": GaussianNB()
                }
            param_grids = {
                    "Logistic Regression": {
                        "penalty": ["l1", "l2", "elasticnet", "none"],
                        "C": [0.01, 0.1, 1, 10, 100],
                        "solver": ["lbfgs", "liblinear", "saga", "newton-cg"],
                        "max_iter": [100, 200, 500],
                        "class_weight": [None, "balanced"]
                    },
                    
                    "K-Nearest Neighbors": {
                        "n_neighbors": [3, 5, 7, 9, 11],
                        "weights": ["uniform", "distance"],
                        "metric": ["euclidean", "manhattan", "minkowski"],
                        "p": [1, 2]
                    },
                    
                    "Decision Tree": {
                        "criterion": ["gini", "entropy"],
                        "max_depth": [None, 5, 10, 20, 30],
                        "min_samples_split": [2, 5, 10],
                        "min_samples_leaf": [1, 2, 4],
                        "max_features": [None, "sqrt", "log2"],
                        "class_weight": [None, "balanced"]
                    },
                    
                    "Random Forest": {
                        "n_estimators": [100, 200, 300],
                        "criterion": ["gini", "entropy"],
                        "max_depth": [None, 10, 20, 30],
                        "min_samples_split": [2, 5, 10],
                        "min_samples_leaf": [1, 2, 4],
                        "max_features": ["sqrt", "log2", None],
                        "bootstrap": [True, False],
                        "class_weight": [None, "balanced"]
                    },
                    
                    "Extra Trees": {
                        "n_estimators": [100, 200, 300],
                        "criterion": ["gini", "entropy"],
                        "max_depth": [None, 10, 20, 30],
                        "min_samples_split": [2, 5, 10],
                        "min_samples_leaf": [1, 2, 4],
                        "max_features": ["sqrt", "log2", None],
                        "bootstrap": [True, False],
                        "class_weight": [None, "balanced"]
                    },
                    
                    "AdaBoost": {
                        "n_estimators": [50, 100, 200, 300],
                        "learning_rate": [0.001, 0.01, 0.1, 0.5, 1],
                        "algorithm": ["SAMME", "SAMME.R"]
                    },
                    
                    "Gradient Boosting": {
                        "n_estimators": [100, 200, 300],
                        "learning_rate": [0.001, 0.01, 0.1, 0.2],
                        "max_depth": [3, 5, 7],
                        "min_samples_split": [2, 5, 10],
                        "min_samples_leaf": [1, 2, 4],
                        "subsample": [0.6, 0.8, 1.0],
                        "max_features": [None, "sqrt", "log2"]
                    },
                    
                    "XGBoost": {
                        "n_estimators": [100, 200, 300],
                        "learning_rate": [0.001, 0.01, 0.1, 0.2],
                        "max_depth": [3, 5, 7, 10],
                        "subsample": [0.6, 0.8, 1.0],
                        "colsample_bytree": [0.6, 0.8, 1.0],
                        "gamma": [0, 0.1, 0.3, 0.5],
                        "reg_alpha": [0, 0.01, 0.1, 1],
                        "reg_lambda": [1, 1.5, 2]
                    },
                    
                    "CatBoost": {
                        "depth": [4, 6, 8, 10, 12],
                        "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
                        "iterations": [200, 500, 800, 1000],
                        "l2_leaf_reg": [1, 3, 5, 7, 9],
                        "bagging_temperature": [0, 0.5, 1, 2, 5],
                        "random_strength": [0, 1, 2, 5, 10],
                        "rsm": [0.6, 0.8, 1.0],
                        "grow_policy": ["SymmetricTree", "Depthwise", "Lossguide"],
                        "min_data_in_leaf": [1, 5, 10, 20, 50],
                        "max_bin": [128, 254, 512]
                    },
                    
                    "Naive Bayes (Gaussian)": {
                        "var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6]
                    }
                }
            
            model_report,tunned_model = evaluate_models(X_train,y_train,X_test,y_test,classification_models,param_grids)
            
            best_model_score = max(list(model_report.values))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            
            best_model = tunned_model[best_model_name]
            
            y_train_predict = best_model.predict(X_train)
            
            classification_train_metrics = get_classification_score(y_train,y_train_predict)
            
            self.track_mlflow(best_model=best_model,classification_metrics=classification_train_metrics)
            
            y_test_pred = best_model.predict(X_test)
            
            classification_test_metrics = get_classification_score(y_test,y_test_pred)
            
            self.track_mlflow(best_model=best_model,classification_metrics=classification_test_metrics)

            preprocessor = load_obj(self.post_data_validation_artifact.valid_object_file_path)
            
            model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir_path)
            
            model = ClaimPredictionModel(preprocessor=preprocessor,model= best_model)
            
            save_obj(self.model_trainer_config.trained_model_file_path)

            save_obj('final_model/model.pkl',model)

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                model_name = best_model_name,
                train_metric_artifact = classification_train_metrics,
                test_metric_artifact = classification_test_metrics
            )
            
            logging.info('Model Trainer Artifact ', model_trainer_artifact)

            return model_trainer_artifact
            
        except Exception as e:
            raise ClaimPredictionException(e,sys)
        
        
    def initiate_model_trainer(self)->ModelTrainerArtifact:
        try:
            tarin_file_path = self.post_data_validation_artifact.valid_train_file_path
            test_file_path = self.post_data_validation_artifact.valid_test_file_path
            
            train_arr = load_numpy_array_data(tarin_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            X_train,X_test,y_train,y_test = (
                train_arr[:,:-1],
                test_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,-1]
            )
            
            model_trainer_artifact = self.model_trainer(X_train,y_train,X_test,y_test)
            return model_trainer_artifact
        except Exception as e:
            raise ClaimPredictionException(e,sys)
        