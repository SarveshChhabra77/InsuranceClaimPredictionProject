import os
import sys
from datetime import datetime
from InsuranceClaimPredictionProject import constants

class TrainingPipelineConfig:
    def __init__(self,timestamp=datetime.now()):
        timestamp=timestamp.strftime('%m_%d_%Y_%H_%M_%S')
        self.artifacts_name:str = constants.Artifact_Dir
        self.artifacts_dir:str=os.path.join(self.artifacts_name,timestamp)
        self.timestamp:str=timestamp
        
class DataIngestion:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.training_pipeline_config=training_pipeline_config
        self.data_ingestion_dir:str = os.path.join(self.training_pipeline_config.artifacts_dir,constants.Data_Ingestion_Dir_name)
        self.feature_stored_file_path:str = os.path.join(self.data_ingestion_dir,constants.Data_Ingestion_Feature_Store_Dir,constants.File_Name)
        self.train_file_path:str = os.path.join(self.data_ingestion_dir,constants.Data_Ingestion_Ingested_Dir,constants.Train_File_Name)
        self.test_file_path:str = os.path.join(self.data_ingestion_dir,constants.Data_Ingestion_Ingested_Dir,constants.Test_File_Name)
        self.train_test_split_ratio:float = constants.Data_Ingestion_Train_Test_Split_Ratio

class DataValidation:
    pass
class DataTransformation:
    pass
class ModelTraining:
    pass
