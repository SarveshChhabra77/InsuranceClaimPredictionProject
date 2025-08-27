from InsuranceClaimPredictionProject.logging.logger import logging
import os
import sys

class ClaimPredictionException(Exception):
    def __init__(self,message:str,message_details:sys):
        self.error_message = message
        _,_,exc_db = message_details.exc_info()
        self.file_name=exc_db.tb_frame.f_code.co_filename
        self.file_line=exc_db.tb_lineno
        
    def __str__(self):
        return "Execution failed in file '{0}' on line {1}. Details: {2}".format(self.file_name, self.file_line, self.error_message)
    
# try:
#     a=1/0
# except Exception as e:
#     logging.info('Raising custom error')
#     raise ClaimPredictionException(e,sys)