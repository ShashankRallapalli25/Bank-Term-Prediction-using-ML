import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sys
#from src.exception import CustomException
from dataclasses import dataclass

@dataclass
class DataLoadConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "data.csv")

class DataLoad:
    def __init__(self):
        self.ingestion_config = DataLoadConfig()

    def initiate_data_ingestion(self):
            df=pd.read_csv("notebook/data/bank-full.csv", delimiter= ";")
            print(df.columns)
            #remove the outliers
            m_bal = np.mean(df['balance'])
            st_bal = np.std(df['balance'])

            thresh = 3 * st_bal
            outliers = df[(df['balance'] > m_bal + thresh) | (df['balance'] < m_bal - thresh)]
            df = df.drop(outliers.index)
            
            #Handling the unknown cases and education statuses based on the basic understanding
            df.loc[(df['age'] > 60) & (df['job'] == 'unknown'), 'job'] = 'retired'
            df.loc[(df['education'] == 'unknown') & (df['job'] == 'management'), 'education'] = 'tertiary'

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False, header= True)
            
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False, header= True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False, header= True)
       
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path, 
            )

if __name__ == "__main__":
    obj = DataLoad()
    train_data, test_data = obj.initiate_data_ingestion()

    