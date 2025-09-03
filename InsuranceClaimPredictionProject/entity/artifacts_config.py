from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    train_file_path:str
    test_file_path:str
    
@dataclass
class DataValidationArtifact:
    validation_status:bool
    valid_train_file_path:str
    valid_test_file_path:str
    invalid_train_file_path:str
    invalid_test_file_path:str
    drift_report_file_path:str
    
    
@dataclass
class DataTransformationArtifacts:
    transformed_object_file_path:str
    transformed_train_file_path:str
    transformed_test_file_path:str
    
@dataclass
class PostDataValidationArtifact:
    validation_status:bool
    valid_train_file_path:str
    valid_test_file_path:str
    invalid_train_file_path:str
    invalid_test_file_path:str
    valid_object_file_path:str
    drift_report_file_path:str
    
@dataclass
class ClassificationMetricArtifact:
    roc_auc_score:float
    recall_score:float
        
    
    
@dataclass
class ModelTrainerArtifact:
    trained_model_file_path:str
    model_name:str
    train_metric_artifact:ClassificationMetricArtifact
    test_metric_artifact:ClassificationMetricArtifact