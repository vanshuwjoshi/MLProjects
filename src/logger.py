import logging
import os
from datetime import datetime

LOG_FILE=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log" #naming convention for our log file (current datetime with given format)
log_path=os.path.join(os.getcwd(),"logs",LOG_FILE) #path where our log file will get created, os.getcwd() returns current working directory
os.makedirs(log_path,exist_ok=True)

LOG_FILE_PATH = os.path.join(log_path,LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s", #best practice
    level=logging.INFO
)