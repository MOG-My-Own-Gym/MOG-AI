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
    csv_path = os.path.join(path, 'megaGymDataset.csv')
    df = pd.read_csv(csv_path)
    df.rename(columns={'Unnamed: 0': 'gymId'}, inplace=True)

    for col in categorical_cols:
        df[col] = df[col].fillna('Unknown')
        # 각 문자열 컬럼에 normalize_string 적용
        df[col] = df[col].astype(str).apply(normalize_string)
  
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    k = 30
    kmeans = KMeans(n_clusters=k, random_state=42)
    df['cluster'] = kmeans.fit_predict(df[categorical_cols])

    return df, encoders, kmeans



import re

def normalize_string(s: str) -> str:
    s = s.lower()  # 소문자 변환
    s = re.sub(r"[-\/,.]", " ", s)  # -, /, , . → 공백
    s = re.sub(r"\s+", " ", s)  # 여러 공백 → 한 칸 공백
    s = s.strip()  # 앞뒤 공백 제거
    return s
