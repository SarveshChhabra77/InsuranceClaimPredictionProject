from InsuranceClaimPredictionProject.logging.logger import logging
from InsuranceClaimPredictionProject.exceptions.exception import ClaimPredictionException
from InsuranceClaimPredictionProject.entity.config_entity import DataIngestionConfig
from InsuranceClaimPredictionProject.entity.artifacts_config import DataIngestionArtifact
from sklearn.model_selection import train_test_split
import os
import sys
import pandas as pd

class DataIngestion:
    def __init__(self,data_ingestion_config:DataIngestionConfig):
        try:
            self.data_ingestion_config=data_ingestion_config
        except Exception as e:
            raise ClaimPredictionException(e,sys)
    def export_data_as_dataframe(self):
        try:
            df = pd.read_csv(r'D:\Insurance Claim Prediction Project\data\raw\insurance_claims.csv')
            return df
        except Exception as e:
            raise ClaimPredictionException(e,sys)
        
    def export_data_into_feature_stored(self,dataframe:pd.DataFrame):
        try:
            feature_stored_file_path = self.data_ingestion_config.feature_stored_file_path
            feature_stored_dir = os.path.dirname(feature_stored_file_path)
            os.makedirs(feature_stored_dir,exist_ok=True)
            df = self.export_data_as_dataframe()
            df.to_csv(feature_stored_file_path,header=True,index=False)
        except Exception as e:
            raise ClaimPredictionException(e,sys)
        
    def split_data_as_train_test(self,dataframe:pd.DataFrame):
        try:
            train_test_split_ratio=self.data_ingestion_config.train_test_split_ratio
            logging.info('Performing Train Test Split on the Dataframe')
            train_set,test_set=train_test_split(dataframe,test_size=train_test_split_ratio,random_state=42)
            logging.info('Train Test Split is Completed')
            train_file_path = self.data_ingestion_config.train_file_path
            test_file_path = self.data_ingestion_config.test_file_path
            ingested_dir = os.path.dirname(train_file_path)
            
            os.makedirs(ingested_dir,exist_ok=True)
            logging.info('Exporting Dataframe as Train and Test file')
            train_set.to_csv(train_file_path)
            test_set.to_csv(test_file_path)
            logging.info('Exported Train and Test File')
            
        except Exception as e:
            raise ClaimPredictionException(e,sys)
        
    def initiate_data_ingestion(self):
        try:
            df = self.export_data_as_dataframe()
            self.export_data_into_feature_stored(df)
            self.split_data_as_train_test(df)
            
            data_ingsestion_artifacts = DataIngestionArtifact(
                train_file_path=self.data_ingestion_config.train_file_path,
                test_file_path=self.data_ingestion_config.test_file_path
            )
            
            return data_ingsestion_artifacts
            
        except Exception as e:
            raise ClaimPredictionException(e,sys)