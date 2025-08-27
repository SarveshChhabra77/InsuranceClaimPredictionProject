import logging
import os
from datetime import datetime


file_name = f"{datetime.now().strftime('%m_%d_%Y_%S_%M_%H')}.log"

log_dir=os.path.join(os.getcwd(),'logs')

os.makedirs(log_dir,exist_ok=True)

file_path=os.path.join(log_dir,file_name)

logging.basicConfig(
    filename=file_path,
    level=logging.INFO,
    format='[%(asctime)s]-%(lineno)d-%(name)s-%(levelname)s-%(message)s'
)

logging.info("Logging file has made")