import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

class Preprocessor(): 

    def __init__(self, df):
        self.df = df

    def split(self, df):

        X = df[df.columns[1:-1]]

        y = df[df.columns[-1]]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

        y_train = np.array(y_train)

        y_test = np.array(y_test)

        return X_train, X_test, y_train, y_test
    
    def scale(self, X_train, X_test):

        scale = StandardScaler()

        X_train_processed = scale.fit_transform(X_train)

        X_test_processed = scale.transform(X_test)

        return X_train_processed, X_test_processed
    
    def main(self):
         
         print("Starting preprocessing pipeline ...")

         print("Splitting data")

         X_train, X_test, y_train, y_test = self.split(self.df) 

         print("Splitting successful, starting scaling process")

         X_train_processed, X_test_processed = self.scale(X_train, X_test)

         print("Scaling successful!")

         print("Processing pipeline completed!")

         return X_train_processed, X_test_processed, y_train, y_test
    
if __name__ == "__main__": 

    print("Start!")

    FILE_PATH = '/Users/thananpornsethjinda/Desktop/pulsar-pred/data/pulsar_data.csv'

    df = pd.read_csv(FILE_PATH)

    X_train_processed, X_test_processed, y_train, y_test = Preprocessor(df).main()

    print(len(X_test_processed[0]))

         




