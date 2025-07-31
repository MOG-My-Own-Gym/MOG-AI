import requests
import zipfile
import pandas as pd
import os
import kagglehub
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans

categorical_cols = ['Type', 'BodyPart', 'Equipment', 'Level']


def prepare_model():
    # 데이터 다운로드, 로드, 전처리
    path = kagglehub.dataset_download("niharika41298/gym-exercise-data")
    csv_path = os.path.join(path,'megaGymDataset.csv')
    df = pd.read_csv(csv_path)
    df.rename(columns={'Unnamed: 0': 'gymId'}, inplace=True)

    for col in categorical_cols:
        df[col] = df[col].fillna('Unknown')

    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    k = 30
    kmeans = KMeans(n_clusters=k, random_state=42)
    df['cluster'] = kmeans.fit_predict(df[categorical_cols])

    return df, encoders, kmeans