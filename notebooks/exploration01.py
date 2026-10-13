import pandas as pd

df = pd.read_csv("data/raw/financial_fraud_detection_dataset.csv")
print(df.shape)                 #Number of rows/columns
df.info()                       #Data types and non-null values
print(df.isnull().sum())        #Missing values per column
print(df.duplicated().sum())    #Duplicates
print(df["is_fraud"].value_counts(normalize=True))   #Fraud percentage distribution
print(df.head())