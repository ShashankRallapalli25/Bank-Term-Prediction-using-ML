import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
#from src.exception import CustomException
import os
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path= os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):

        '''
        This function is responsible for data transformation
        '''
            
        numerical_features = ["reading_score","writing_score"]
        categorical_features = ["gender", "race_ethnicity", "parental_level_of_education","lunch", "test_preparation_course"]

        num_pipeline = Pipeline (
                steps = [
                    ("imputer", SimpleImputer( strategy= "median" )),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )
        cat_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer (strategy= "most_frequent")),
                    ("encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

        preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_features),
                    ("cat_pipeline", cat_pipeline,categorical_features)
                ]
            )

        return preprocessor