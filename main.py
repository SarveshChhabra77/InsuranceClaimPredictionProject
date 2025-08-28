import sys
from InsuranceClaimPredictionProject.entity.config_entity import TrainingPipelineConfig,DataIngestionConfig
from InsuranceClaimPredictionProject.components.data_ingestion import DataIngestion
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
        
    
    
    
    except Exception as e:
        raise ClaimPredictionException(e,sys)