from InsuranceClaimPredictionProject.logging.logger import logging
from InsuranceClaimPredictionProject.exceptions.exception import ClaimPredictionException
import os
import sys
import yaml


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