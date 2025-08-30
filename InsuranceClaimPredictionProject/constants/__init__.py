import os


"Defining the training pipeline constants"

Target_Column:str="fraud_reported"
Pipeline_Name:str=''
Artifact_Dir:str='Artifacts'
File_Name:str='insurance_claims.csv'
Train_File_Name:str='Train.csv'
Test_File_Name:str='Test.csv'
Saved_Model_Dir:str=os.path.join('saved_models')
Model_File_Name:str='model.pkl'
Schema_file_path:str='D:\Insurance Claim Prediction Project\data_schema\__init__.py'



"Defining the data ingestion constant names starting with data ingestion"
# Data_Ingestion_Collection_Name:str=''
Data_Ingestion_Dir_name:str='data_ingestion'
Data_Ingestion_Feature_Store_Dir:str='feature_stored'
Data_Ingestion_Ingested_Dir:str='ingested'
Data_Ingestion_Train_Test_Split_Ratio:float=0.2

"Defining the data transformation constant names starting with data Validation"
Data_Validation_Dir_Name:str='data_validation'
Data_Validation_Valid_Data_Dir:str='validated'
Data_Validation_Invalid_Data_Dir:str='invalid'
Data_Validation_Drift_Report_Dir:str='drift_report'
Data_Validation_Drift_Report_File_Name:str='report.yaml'