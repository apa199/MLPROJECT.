import os
import sys
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformationConfig, DataTransformation
from src.components.model_trainer import ModelTrainerConfig,ModelTrainer
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

#add dataclass so that we can create constructor with it
@dataclass
class DataIngestionConfig():
    #here we are defining a folder artifact in which train test and raw will be stored
    train_data_path:str=os.path.join('artifact','train.csv')
    test_data_path:str=os.path.join('artifact','test.csv')
    raw_data_path:str=os.path.join('artifact','data.csv')

class DataIngestion:
    #we create a class for ingestion where constructor is defined and then we add config so that we can add the paths 
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("data ingestion component")


        try:
            df=pd.read_csv("/Users/aparnamallik/Documents/MLproject/notebook/data/stud.csv")
            logging.info("loaded the dataset")
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("train test split inititaed")
            train_set, test_set=train_test_split(df,test_size=0.2,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False,header=True)
            return (self.ingestion_config.train_data_path,
                    self.ingestion_config.test_data_path)

        except:
            raise CustomException(e,sys)
        

if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()

    datatransformation=DataTransformation()
    train_arr,test_arr,_=datatransformation.initiate_data_transformation(train_data,test_data)
    model_trainer=ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_arr,test_arr))