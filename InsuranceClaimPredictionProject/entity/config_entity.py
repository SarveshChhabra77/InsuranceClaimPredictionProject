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
        
        
class DataTransformationConfig:
    def __init__(self,training_pipelien_config:TrainingPipelineConfig):
        self.training_pipeline_config = training_pipelien_config
        
        self.data_transformation_dir:str = os.path.join(self.training_pipeline_config.artifacts_dir,constants.DATA_TRANSFORMATION_DIR_NAME)
        
        self.transformed_train_file_path:str = os.path.join(self.data_transformation_dir,constants.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,constants.Train_File_Name.replace('csv','npy'))
        
        self.transformed_test_file_path:str = os.path.join(self.data_transformation_dir,constants.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,constants.Test_File_Name.replace('csv','npy'))
        
        self.transformed_object_file:str = os.path.join(self.data_transformation_dir,constants.DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,constants.PREPROCESSING_OBJECT_FILE_NAME)
        
class PostDataValidationConfig:
    def __init__(self,traning_pipeline_config:TrainingPipelineConfig):
        self.training_pipelien_config = traning_pipeline_config
        
        self.post_data_validation_dir:str = os.path.join(self.training_pipelien_config.artifacts_dir,constants.Post_Data_Validation_Dir_Name)
        
        self.post_data_validation_valid_dir:str = os.path.join(self.post_data_validation_dir,constants.Post_Data_Validation_Valid_Data_Dir)
        
        self.post_data_validation_invalid_dir:str = os.path.join(self.post_data_validation_dir,constants.Post_Data_Validation_Invalid_Data_Dir)
        
        self.post_valid_train_file_path:str = os.path.join(self.post_data_validation_valid_dir,constants.Train_File_Name.replace('csv','npy'))
        
        self.post_valid_test_file_path:str = os.path.join(self.post_data_validation_valid_dir,constants.Test_File_Name.replace('csv','npy'))

        self.post_invalid_train_file_path:str = os.path.join(self.post_data_validation_invalid_dir,constants.Train_File_Name.replace('csv','npy'))
        
        self.post_invalid_test_file_path:str = os.path.join(self.post_data_validation_invalid_dir,constants.Test_File_Name.replace('csv','npy'))
        
        self.valid_object_file_path:str  = os.path.join(self.post_data_validation_dir,constants.Post_Data_Validation_Validated_OBJECT_DIR,constants.PREPROCESSING_OBJECT_FILE_NAME)
        
        self.drift_report_file_name:str = os.path.join(self.post_data_validation_dir,constants.Post_Data_Validation_Drift_Report_Dir,constants.Post_Data_Validation_Drift_Report_File_Name)
        
        self.threshold:float = 0.05

class ModelTrainingConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        
        self.model_training_dir:str = os.path.join(training_pipeline_config.artifacts_dir,constants.MODEL_TRAINER_DIR_NAME)

        self.trained_model_file_path:str = os.path.join(self.model_training_dir,constants.MODEL_TRAINER_TRAINED_MODEL_NAME)
        
        self.expected_r2_score:float = constants.MODEL_TRAINER_EXPECTED_SCORE
        
        self.overfittting_underfitting_threshold:float = constants.MODEL_TRAINER_OVER_FITTING_UNDER_FITTING_THRESHOLD
