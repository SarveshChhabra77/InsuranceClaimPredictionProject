from InsuranceClaimPredictionProject.exceptions.exception import ClaimPredictionException
from InsuranceClaimPredictionProject.logging.logger import logging
from InsuranceClaimPredictionProject.entity.config_entity import PostDataValidationConfig
from InsuranceClaimPredictionProject.entity.artifacts_config import DataTransformationArtifacts, PostDataValidationArtifact
from InsuranceClaimPredictionProject.utils.main_utils import load_numpy_array_data, load_obj,save_numpy_array_data,save_obj,write_yaml_file
from sklearn.base import BaseEstimator
import sys
import numpy as np
import os
from scipy.stats import ks_2samp

class PostDataValidation:
    def __init__(self, data_transformation_artifacts: DataTransformationArtifacts, post_data_validation_config: PostDataValidationConfig):
        try:
            self.data_transformation_artifacts = data_transformation_artifacts
            self.post_data_validation_config = post_data_validation_config
        except Exception as e:
            raise ClaimPredictionException(e, sys)

    def validate_shapes(self,train_array:np.array,test_array:np.array) -> bool:
        try:
            if train_array.size == 0 or test_array.size == 0:
                logging.error("Train or test array is empty!")
                return False
                    
            if train_array.shape[1] != test_array.shape[1]:
                logging.error("Train and test dataset column count mismatch!")
                return False

            logging.info(f"Train shape: {train_array.shape}, Test shape: {test_array.shape}")
            return True
        except Exception as e:
            raise ClaimPredictionException(e, sys)

    def validate_no_missing(self,train_array:np.array,test_array:np.array) -> bool:
        try:
            logging.info("Checking for missing values in transformed datasets...")

            if np.isnan(train_array).any():
                logging.error("NaN values found in transformed train data!")
                return False
            if np.isnan(test_array).any():
                logging.error("NaN values found in transformed test data!")
                return False

            logging.info("No missing values found in train and test datasets.")
            return True
        except Exception as e:
            raise ClaimPredictionException(e, sys)

    def validate_target(self,train_array:np.array,test_array:np.array) -> bool:
        try:
            logging.info("Validating target column in train dataset...")
            target_train = train_array[:, -1]  # Last column is assumed to be target

            if not set(np.unique(target_train)).issubset({0, 1}):
                logging.error("Target column values are not 0/1 in train dataset!")
                return False
            
            target_test = test_array[:,-1]
            if not set(np.unique(target_test)).issubset({0, 1}):
                logging.error("Target column values are not 0/1 in test dataset!")
                return False

            logging.info("Target column validated successfully.")
            return True
        except Exception as e:
            raise ClaimPredictionException(e, sys)

    def validate_preprocessor(self,preprocessor:object) -> bool:
        try:
            logging.info("Validating preprocessor object...")
            if not isinstance(preprocessor, BaseEstimator):
                logging.error("Preprocessor is not a valid sklearn object!")
                return False
            logging.info("Preprocessor object loaded successfully.")
            return True
        except Exception as e:
            raise ClaimPredictionException(e, sys)

    def detect_dataset_drift_array(self, base_array: np.ndarray, current_array: np.ndarray, threshold:float) -> bool:
        try:
            status = True
            report = {}

            n_features = base_array.shape[1]

            for i in range(n_features):
                base_col = base_array[:, i]
                current_col = current_array[:, i]

                # Ensure columns are numeric for KS test
                base_col = base_col.astype(float)
                current_col = current_col.astype(float)

                # ks_2samp works on 1D arrays
                test_result = ks_2samp(base_col, current_col)

                # Drift exists if p-value < threshold
                drift_found = float(test_result.pvalue) < threshold
                if drift_found:
                    status = False

                report[f"feature_{i}"] = {
                    "p_value": float(test_result.pvalue),
                    "drift_status": drift_found
                }

            drift_report_file_path = self.post_data_validation_config.drift_report_file_name
            write_yaml_file(drift_report_file_path, report)

            return status

        except Exception as e:
            raise ClaimPredictionException(e, sys)

    def initiate_post_validation(self)->PostDataValidationArtifact:
        try:
            train_array = load_numpy_array_data(self.data_transformation_artifacts.transformed_train_file_path)
            test_array = load_numpy_array_data(self.data_transformation_artifacts.transformed_test_file_path)
            preprocessor = load_obj(self.data_transformation_artifacts.transformed_object_file_path)
            
            logging.info("Starting post-validation checks...")
            
            status_shape = self.validate_shapes(train_array,test_array)
            status_missing = self.validate_no_missing(train_array,test_array)
            status_target = self.validate_target(train_array,test_array)
            status_preprocessor = self.validate_preprocessor(preprocessor)
            status_drift = self.detect_dataset_drift_array(train_array,test_array,self.post_data_validation_config.threshold)
            
            overall_status = status_shape and status_missing and status_target and status_preprocessor and status_drift
            
            dir_path = os.path.dirname(self.post_data_validation_config.post_valid_train_file_path)
            os.makedirs(dir_path, exist_ok=True)
            save_numpy_array_data(self.post_data_validation_config.post_valid_train_file_path,train_array)
            save_numpy_array_data(self.post_data_validation_config.post_valid_test_file_path,test_array)
            
            obj_dir = os.path.dirname(self.post_data_validation_config.valid_object_file_path)
            os.makedirs(obj_dir,exist_ok=True)
            save_obj(self.post_data_validation_config.valid_object_file_path,preprocessor)
            
            
            

            # Save post-validation report or artifacts
            artifact = PostDataValidationArtifact(
                validation_status=overall_status,
                valid_train_file_path=self.post_data_validation_config.post_valid_train_file_path,
                valid_test_file_path=self.post_data_validation_config.post_valid_test_file_path,
                invalid_train_file_path=None,
                invalid_test_file_path=None,
                valid_object_file_path=self.post_data_validation_config.valid_object_file_path,
                drift_report_file_path=self.post_data_validation_config.drift_report_file_name
            )

            logging.info("Post-validation completed successfully.")
            return artifact

        except Exception as e:
            raise ClaimPredictionException(e, sys)
