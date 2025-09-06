from InsuranceClaimPredictionProject.entity.artifacts_config import DataTransformationArtifacts,DataValidationArtifact
from InsuranceClaimPredictionProject.entity.config_entity import DataTransformationConfig
from InsuranceClaimPredictionProject.exceptions.exception import ClaimPredictionException
from InsuranceClaimPredictionProject.logging.logger import logging
from InsuranceClaimPredictionProject.constants import Target_Column
from InsuranceClaimPredictionProject.utils.main_utils import save_numpy_array_data,save_obj,CustomTransformer
import sys
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
import category_encoders as ce
from imblearn.over_sampling import SMOTE
import numpy as np

class DataTransformation:
    def __init__(self,data_validation_artifact:DataValidationArtifact,data_tranformation_config:DataTransformationConfig):
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_tranformation_config
        except Exception as e:
            raise ClaimPredictionException(e,sys)

    @staticmethod
    def read_data(file_path:str)->pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise ClaimPredictionException(e,sys)
        
    
    def build_preprocessor(self,dataframe:pd.DataFrame)->ColumnTransformer:
        try: 
            
            low_cat_columns = [col for col in dataframe.columns if dataframe[col].dtype == 'O' and dataframe[col].nunique() <= 5]
            high_cat_columns = [col for col in dataframe.columns if dataframe[col].dtype == 'O' and dataframe[col].nunique() > 5]
            numerical_columns = [col for col in dataframe.columns if dataframe[col].dtype != 'O']           
                        
            low_cat_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('onehot',OneHotEncoder(handle_unknown='ignore',drop='first'))
                ]
            )
            high_cat_pipeline =Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('target_encoder',ce.TargetEncoder())
                ]
            )
            numerical_columns_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='mean')),
                    ('scaler' , StandardScaler())
                ]
            )
            feature_transformer = ColumnTransformer(
                transformers=[
                    ('low_cat_col', low_cat_pipeline, low_cat_columns),
                    ('high_cat_col', high_cat_pipeline, high_cat_columns),
                    ('numerical-columns', numerical_columns_pipeline, numerical_columns)
                ]
            )
            
            preprocessor = Pipeline([
            ('custom_transform', CustomTransformer()),
            ('feature_transform', feature_transformer)
            ])
            
            return preprocessor
        except Exception as e:
            raise ClaimPredictionException(e,sys)
        
    def initiate_data_transformation(self):
        try:
            logging.info('Enter inittiate_data_transformation method of DataTransformation class')
            train_df = self.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df = self.read_data(self.data_validation_artifact.valid_test_file_path)
            
            
            
            input_feature_train_df=train_df.drop(columns=[Target_Column],axis=1)
            target_feature_train_df=train_df[Target_Column]
            
            input_feature_test_df=test_df.drop(columns=[Target_Column],axis=1)
            target_feature_test_df=test_df[Target_Column]
            
            
            target_feature_train_df = target_feature_train_df.map({'Y': 1 , 'N' : 0}).astype(int)
            target_feature_test_df = target_feature_test_df.map({'Y': 1 , 'N' : 0}).astype(int)
            
            preprocessor = self.build_preprocessor(dataframe=input_feature_train_df)
             
            preprocessor_obj = preprocessor.fit(input_feature_train_df,target_feature_train_df)
            
            transformed_input_train_feature = preprocessor_obj.transform(input_feature_train_df)
            transformed_input_test_feature = preprocessor_obj.transform(input_feature_test_df)
            
            ## Handling the imbalance data
            smote = SMOTE(random_state=42)
            transformed_input_train_feature, target_feature_train_df = smote.fit_resample(transformed_input_train_feature, target_feature_train_df)
            
            final_train_df = np.c_[transformed_input_train_feature, target_feature_train_df.to_numpy()]
            final_test_df = np.c_[transformed_input_test_feature, target_feature_test_df.to_numpy()]
            
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path,final_train_df)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path,final_test_df)
            
            save_obj(self.data_transformation_config.transformed_object_file,preprocessor_obj)
            
            save_obj('final_model/preprocessor.pkl',preprocessor_obj)
            
            data_transformation_artifact = DataTransformationArtifacts(
                transformed_train_file_path = self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path = self.data_transformation_config.transformed_test_file_path,
                transformed_object_file_path = self.data_transformation_config.transformed_object_file
            )
            
            return data_transformation_artifact
        
        except Exception as e:
            raise ClaimPredictionException(e,sys)