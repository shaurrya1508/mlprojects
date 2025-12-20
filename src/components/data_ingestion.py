import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation, DataTransformationConfig

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            # Get the project root (assuming script is in project root or adjust as needed)
            # Get the project root (two levels up from the script's directory)
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_file_path = os.path.join(project_root, 'notebook', 'data', 'stud.csv')
            
            print(f"Current working directory: {os.getcwd()}")
            print(f"Attempting to read data from: {data_file_path}")
            
            if not os.path.exists(data_file_path):
                raise FileNotFoundError(f"Data file not found at {data_file_path}. Please check the path.")
            
            df = pd.read_csv(data_file_path)
            logging.info('Read the dataset as dataframe')
            print(f"Dataset shape: {df.shape}")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            print(f"Created artifacts directory at: {os.path.dirname(self.ingestion_config.train_data_path)}")

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            print(f"Saved raw data to: {self.ingestion_config.raw_data_path}")

            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            print(f"Saved train data to: {self.ingestion_config.train_data_path}")
            print(f"Saved test data to: {self.ingestion_config.test_data_path}")

            logging.info("Ingestion of the data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            logging.error(f"Error in data ingestion: {str(e)}")
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
    print(f"Train data path: {train_data}")
    print(f"Test data path: {test_data}")

    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)
    print("Data transformation completed.")