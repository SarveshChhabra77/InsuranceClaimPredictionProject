from InsuranceClaimPredictionProject.logging.logger import logging
from InsuranceClaimPredictionProject.exceptions.exception import ClaimPredictionException
import os
import sys
import yaml
import numpy as np
import pickle

def write_yaml_file(file_path:str,content:object,replace:bool=False)->None:
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
        dir_name = os.path.dirname(file_path)
        os.makedirs(dir_name,exist_ok=True)

        with open(file_path,'w') as file_obj:
            yaml.dump(content,file_obj)
    except Exception as e:
        raise ClaimPredictionException(e,sys)

def save_numpy_array_data(file_path:str,array:np.array):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'wb') as file:
            np.save(file,array)
            
    except Exception as e:
        raise ClaimPredictionException(e,sys)
    
    
def save_obj(file_path:str,obj:object)->None:
    try:
        logging.info('Entered the save-object method of main_utils ')
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'wb') as file_obj:
            pickle.dump(obj,file_obj)
        logging.info('Exited the save_object method of the main_utils')
    except Exception as e:
        raise ClaimPredictionException(e,sys)

def load_numpy_array_data(file_path:str)->np.array:
    try:
        if not os.path.exists(file_path):
            raise Exception(f'File path : {file_path} not exist')
        with open(file_path,'rb') as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise ClaimPredictionException(e,sys)
    
def load_obj(file_path:str)->object:
    try:
        if not os.path.exists(file_path):
            raise Exception(f'The file path : {file_path} is not exist')
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise ClaimPredictionException(e,sys)