from enum import Enum # 미리 정의 가능한 고정 변수 값
from typing import Union
from fastapi import FastAPI
from fastapi.responses import JSONResponse # json 형식으로 데이터 전달 
from pydantic import BaseModel
import numpy as np
from kmeans_recommendation import prepare_model,categorical_cols



app = FastAPI()

df, encoders, kmeans = prepare_model()


def user_pref_to_cluster(user_pref ):
    pref_vec = []
    for col in categorical_cols:
        val = user_pref.get(col, 'Unknown')
        le = encoders[col]
        if val in le.classes_:
            pref_vec.append(le.transform([val])[0])
        else:
            pref_vec.append(-1)
    pref_vec = np.array(pref_vec).reshape(1, -1)
    cluster_label = kmeans.predict(pref_vec)[0]
    return cluster_label




class Preferences(BaseModel):
    Type:str
    BodyPart:str
    Equipment:str
    Level:str
class RecommendationRequest(BaseModel):
    usersId: int
    preferences: Preferences

@app.post("/api/v1/suggest/exercises")
def read_root(request: RecommendationRequest):
    user_pref = request.preferences.model_dump()
    print(request.preferences.model_dump())
    cluster_label = user_pref_to_cluster(user_pref)
    recs = df[df['cluster'] == cluster_label].sort_values(by='Rating', ascending=False)
    return {"recommendations": recs[['Title', 'Rating']].head(10).to_dict(orient='records')}

