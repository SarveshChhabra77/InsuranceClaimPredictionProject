from InsuranceClaimPredictionProject.exceptions.exception import ClaimPredictionException
from InsuranceClaimPredictionProject.logging.logger import logging
from InsuranceClaimPredictionProject.entity.config_entity import DataValidationConfig
from InsuranceClaimPredictionProject.entity.artifacts_config import DataValidationArtifact,DataIngestionArtifact
from InsuranceClaimPredictionProject.utils.main_utils import write_yaml_file
import os
import pandas as pd
import sys
from typing import Dict,Union,List,Any
from data_schema import processed_schema
from scipy.stats import ks_2samp


class DataValidation:
    def __init__(self,data_validation_config:DataValidationConfig,data_ingestion_artifact:DataIngestionArtifact):
        try:
            self.data_validation_config = data_validation_config
            self.data_ingestion_artifacts = data_ingestion_artifact
        except Exception as e:
            raise ClaimPredictionException(e,sys)
    
    @staticmethod
    def read_data(file:str)->pd.DataFrame:
       try:
            return pd.read_csv(file)
       except Exception as e:
           raise ClaimPredictionException(e,sys)
       
    def validate_dataframe_schema(self,dataframe: pd.DataFrame, schema: Dict[str, Union[Any, List[Any]]]) -> bool:
        try:
            logging.info('Start Data Schema Validation')

            expected_columns = set(schema.keys())
            actual_columns = set(dataframe.keys())

            if expected_columns!=actual_columns:
                missing_columns = expected_columns - actual_columns
                extra_columns = actual_columns - expected_columns
                
                if missing_columns:
                    logging.info(f"Validation FAILED: Missing columns: {missing_columns}")                
                    return False
                if extra_columns:
                    logging.info(f"Extra columns found: {extra_columns}. Dropping Them")
                    dataframe.drop(columns=list(extra_columns),inplace=True) 

            for column,expected_dtype in schema.items():
                if not isinstance(expected_dtype,list):
                    expected_dtypes = [expected_dtype]
                    
                actual_dtype = dataframe[column].dtype
                
                if not any(pd.api.types.is_dtype_equal(actual_dtype,dt)for dt in expected_dtypes):
                    logging.info(
                        f"Validation FAILED: Column '{column}' has dtype '{actual_dtype}', "
                        f"expected one of {expected_dtypes}"
                    )
                    return False
            logging.info("Validation PASSED: DataFrame schema is correct.")
            return True
                
        except Exception as e:
            raise ClaimPredictionException(e,sys)
        
    def check_missing_values(self,dataframe: pd.DataFrame, threshold: float = 0.5):
        try:
            missing_report = {}

            for cols in dataframe.columns:
                missing_value = dataframe[cols].isna().mean()
                if missing_value>0:
                    missing_report[cols]=missing_value
            if any(value > threshold for value in missing_report.values()):
                logging.info(f"Validation FAILED: Columns with more than {threshold*100}% missing values found.")
                logging.info(missing_report)
                return False
            else:
                logging.info("Validation PASSED: Missing value check successful.")
                return True
                    
        except Exception as e:
            raise ClaimPredictionException(e,sys)
        
    def detect_dataset_drift(self, base_df: pd.DataFrame, current_df: pd.DataFrame, threshold=0.05) -> bool:
        try:
            status = True
            report = {}
            for col in base_df.columns:
                d1 = base_df[col]
                d2 = current_df[col]
                
                is_sample_dist = ks_2samp(d1,d2)
                
                if threshold <= is_sample_dist.pvalue:
                    is_found = False # no drift
                else:
                    is_found = True # drift
                    status = False
                    # Save results for the column
                report.update({
                    col : {
                        'p_value': float(is_sample_dist.pvalue),
                        'drift_status': is_found
                        }                   
                    })
                
                drift_report_file_path = self.data_validation_config.drift_report_file_name
                
                write_yaml_file(drift_report_file_path,report)
                
                return status
            
        except Exception as e:
            raise ClaimPredictionException(e,sys)
        
    
    def initiate_data_validation(self)->DataValidationArtifact:
        try:
            train_file_path = self.data_ingestion_artifacts.train_file_path
            test_file_path = self.data_ingestion_artifacts.test_file_path

            train_df = self.read_data(train_file_path)
            test_df = self.read_data(test_file_path)

            # --- Step 2: Schema validation ---
            status = self.validate_dataframe_schema(train_df,processed_schema)
            if not status:
                logging.info('Train dataframe does not contain all required columns.')

            status = self.validate_dataframe_schema(test_df,processed_schema)
            if not status:
                logging.info('Test dataframe does not contain all required columns.')

            # --- Step 3: Missing values check ---
            status = self.check_missing_values(train_df)
            if not status:
                logging.info('Train dataframe contains excessive missing data.')

            status = self.check_missing_values(test_df)
            if not status:
                logging.info('Test dataframe contains excessive missing data.')

            # --- Step 4: Drift detection ---
            status = self.detect_dataset_drift(base_df=train_df, current_df=test_df)

            # --- Step 5: Save valid datasets ---
            dir_path = os.path.dirname(self.data_validation_config.valid_train_file_path)
            os.makedirs(dir_path, exist_ok=True)

            train_df.to_csv(self.data_validation_config.valid_train_file_path, index=False, header=True)
            test_df.to_csv(self.data_validation_config.valid_test_file_path, index=False, header=True)

            # --- Step 6: Create artifact object ---
            data_validation_artifacts = DataValidationArtifact(
                validation_status=status,
                valid_train_file_path=self.data_validation_config.valid_train_file_path,
                valid_test_file_path=self.data_validation_config.valid_test_file_path,
                invalid_train_file_path=None,
                invalid_test_file_path=None,
                drift_report_file_path=self.data_validation_config.drift_report_file_name
            )
            return data_validation_artifacts
        except Exception as e:
            raise ClaimPredictionException(e,sys)