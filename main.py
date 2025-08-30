import sys
from InsuranceClaimPredictionProject.entity.config_entity import TrainingPipelineConfig,DataIngestionConfig,DataValidationConfig,DataTransformation,ModelTraining
from InsuranceClaimPredictionProject.components.data_ingestion import DataIngestion
from InsuranceClaimPredictionProject.components.data_validation import DataValidation
from InsuranceClaimPredictionProject.logging.logger import logging
from InsuranceClaimPredictionProject.exceptions.exception import ClaimPredictionException



if __name__ == '__main__':
    try:
        training_pipeline_config = TrainingPipelineConfig()
        
        data_ingestion_config = DataIngestionConfig(training_pipeline_config=training_pipeline_config)
        
        data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
        
        logging.info('Initiating Data Ingestion')
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        logging.info('Data Ingestion Complted')
        
        print(data_ingestion_artifact)
        
        data_validation_config = DataValidationConfig(training_pipeline_config)
        
        data_validation = DataValidation(data_validation_config=data_validation_config,data_ingestion_artifact=data_ingestion_artifact)
        
        logging.info('Initiating Data Validation')
        data_validation_artifact = data_validation.initiate_data_validation()
        logging.info('Data Validation Config')

        print(data_validation_artifact)
    
    
    
    except Exception as e:
        raise ClaimPredictionException(e,sys)