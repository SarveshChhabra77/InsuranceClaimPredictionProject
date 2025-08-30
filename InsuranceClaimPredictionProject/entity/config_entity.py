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
        
class DataIngestionConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.training_pipeline_config=training_pipeline_config
        self.data_ingestion_dir:str = os.path.join(self.training_pipeline_config.artifacts_dir,constants.Data_Ingestion_Dir_name)
        self.feature_stored_file_path:str = os.path.join(self.data_ingestion_dir,constants.Data_Ingestion_Feature_Store_Dir,constants.File_Name)
        self.train_file_path:str = os.path.join(self.data_ingestion_dir,constants.Data_Ingestion_Ingested_Dir,constants.Train_File_Name)
        self.test_file_path:str = os.path.join(self.data_ingestion_dir,constants.Data_Ingestion_Ingested_Dir,constants.Test_File_Name)
        self.train_test_split_ratio:float = constants.Data_Ingestion_Train_Test_Split_Ratio

class DataValidationConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.training_pipeline_config = training_pipeline_config
        
        self.data_validation_dir:str =  os.path.join(self.training_pipeline_config.artifacts_dir,constants.Data_Validation_Dir_Name)
        self.valid_train_dir:str = os.path.join(self.data_validation_dir,constants.Data_Validation_Valid_Data_Dir)
        self.invalid_train_dir:str = os.path.join(self.data_validation_dir,constants.Data_Validation_Invalid_Data_Dir)
        
        self.valid_train_file_path:str = os.path.join(self.valid_train_dir,constants.Train_File_Name)
        self.valid_test_file_path:str = os.path.join(self.valid_train_dir,constants.Test_File_Name)
        
        self.invalid_train_file_path:str = os.path.join(self.invalid_train_dir,constants.Train_File_Name)
        self.invalid_test_file_path:str = os.path.join(self.invalid_train_dir,constants.Test_File_Name)
        
        self.drift_report_file_name:str = os.path.join(self.data_validation_dir,constants.Data_Validation_Drift_Report_Dir,constants.Data_Validation_Drift_Report_File_Name)
        
        
class DataTransformation:
    pass
class ModelTraining:
    pass
