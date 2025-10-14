import os
import sys
from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
@dataclass
class DataTransformationConfig:
    data_transform_path=os.path.join('artifact','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_transformer_object(self):
        try:
            num_features=['writing_score','reading_score']
            cat_features=['gender','race_ethnicity','parental_level_of_education','lunch','test_preparation_course']

            num_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ])
            cat_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("ohe",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))
            ])
            logging.info("cat data encoding completed and num data scaling")

            preprocessor=ColumnTransformer(
                [
                    ("numerical_pipeline",num_pipeline,num_features),
                    ("categorical_pipeline",cat_pipeline,cat_features)
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info("train and test data")
            preprocessing_obj=self.get_transformer_object()

            target="math_score"
            input_feature_train_df=train_df.drop(columns=[target],axis=1) #x_train
            input_feature_test_df=test_df.drop(columns=[target],axis=1) #y_train
            target_feature_train_df=train_df[target] #x_test
            target_feature_test_df=test_df[target] #y_test

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(
                file_path=self.data_transformation_config.data_transform_path,
                obj=preprocessing_obj
)
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.data_transform_path)
        except Exception as e:
            raise CustomException(e, sys)