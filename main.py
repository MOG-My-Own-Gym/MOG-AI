from typing import List
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, model_validator
import numpy as np
from itertools import product
from kmeans_recommendation import prepare_model, categorical_cols
import requests
import os

app = FastAPI()


#OLLAMA URL
OLLAMA_URL = f"http://127.0.0.1:11434/api/generate"

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://myowngym.kro.kr"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 모델 준비
df, encoders, kmeans = prepare_model()


# ✅ 모델 정의
class Preferences(BaseModel):
    Type: List[str] = []
    BodyPart: List[str] = []
    Equipment: List[str] = []
    Level: List[str] = []

    @model_validator(mode='after')
    def at_least_one_keyword(cls, values):
        if not (values.Type or values.BodyPart or values.Equipment or values.Level):
            raise ValueError("At least one keyword must be specified in preferences.")
        return values


class RecommendationRequest(BaseModel):
    usersId: int
    preferences: Preferences


# 추천 클러스터 예측 함수
def user_pref_to_cluster_combinations(user_pref):
    values = {k: v if v else ['Unknown'] for k, v in user_pref.items()}
    combinations = list(product(*[values[col] for col in categorical_cols]))

    input_vectors = []
    for combo in combinations:
        vec = []
        for col, val in zip(categorical_cols, combo):
            le = encoders[col]
            if val in le.classes_:
                vec.append(le.transform([val])[0])
            else:
                vec.append(-1)
        input_vectors.append(vec)

    input_array = np.array(input_vectors)
    cluster_labels = kmeans.predict(input_array)
    return list(set(cluster_labels))


@app.post("/suggest/exercises")
def suggest_exercises(request: RecommendationRequest):
    user_pref = request.preferences.model_dump(mode="python")
    cluster_labels = user_pref_to_cluster_combinations(user_pref)

    recs = df[df['cluster'].isin(cluster_labels)].copy()

    # 숫자 인코딩 된 컬럼을 원본 문자열로 변환
    for col in categorical_cols:
        le = encoders[col]
        recs[col] = le.inverse_transform(recs[col])

    top10 = recs.sort_values(by='Rating', ascending=False)[['Title', 'Rating']].head(20).to_dict(orient='records')
    return {"recommendations": top10}


# 요청 데이터 구조
class ChatRequest(BaseModel):
  prompt: str
  model: str = "llama3"

@app.post("/suggest/chat")
def chat(req: ChatRequest):
  payload ={
    "model" : req.model,
    "prompt" : req.prompt,
    "stream" : False
  }
  response = requests.post(OLLAMA_URL, json=payload)
  print(response)
  return response.json()

@app.get("/suggest/")
def root():
  return {"message": "Ollama FastAPI proxy is Running"}