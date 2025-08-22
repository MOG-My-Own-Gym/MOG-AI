import datetime
from enum import Enum
from typing import Dict, List, Optional, Union
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field, model_validator
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




# 로그 분석 json 모델
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict


class EventType(str, Enum):
    AUTH_FAILURE = "AUTH_FAILURE"
    AUTH_SUCCESS = "AUTH_SUCCESS"
    SESSION_EVENT = "SESSION_EVENT"
    NETWORK_CONNECTION = "NETWORK_CONNECTION"
    SUDO_USAGE = "SUDO_USAGE"
    CRON_JOB = "CRON_JOB"
    SYSTEM_EVENT = "SYSTEM_EVENT"
    USER_MANAGEMENT = "USER_MANAGEMENT"
    ANOMALY = "ANOMALY"
    UNKNOWN = "UNKNOWN"


class SeverityLevel(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


class Location(BaseModel):
    lat: float
    lon: float


class SourceIP(BaseModel):
    ip: str
    country_code: Optional[str] = None
    country_name: Optional[str] = None
    city: Optional[str] = None
    region: Optional[str] = None
    region_code: Optional[str] = None
    location: Optional[Location] = None


class HostInfo(BaseModel):
    hostname: str
    ip_addresses: List[str]


class SecurityEvent(BaseModel):
    model_config = ConfigDict(use_enum_values=True, extra='allow')
    
    event_type: EventType
    severity: SeverityLevel
    related_logs: List[str] = Field(..., min_length=1, description="Original log lines that triggered this event")
    description: str = ""  # Allow empty string
    confidence_score: Union[float, int] = Field(..., ge=0.0, le=1.0, description="Confidence level (0.0-1.0)")
    source_ips: Optional[Union[List[str], List[SourceIP]]] = Field(default=None, description="Source IP address list - can be strings or detailed objects")
    username: Optional[str] = None
    process: Optional[str] = None
    service: Optional[str] = None
    recommended_actions: List[str] = []  # Allow empty list
    requires_human_review: bool


class Statistics(BaseModel):
    total_events: int
    auth_failures: int
    unique_ips: int
    unique_users: int
    event_by_type: Optional[Dict[str, int]] = None


class LogAnalysis(BaseModel):
    model_config = ConfigDict(extra='allow')  # Allow additional fields
    
    summary: str = ""  # Allow empty string
    events: List[SecurityEvent] = Field(..., min_length=1, description="List of events - MUST NEVER BE EMPTY")
    statistics: Statistics
    highest_severity: Optional[SeverityLevel] = None
    requires_immediate_attention: bool
    
    # Optional metadata fields that might be present
    chunk_analysis_start_utc: Optional[str] = Field(default=None, alias="@chunk_analysis_start_utc")
    chunk_analysis_end_utc: Optional[str] = Field(default=None, alias="@chunk_analysis_end_utc")
    chunk_analysis_elapsed_time: Optional[int] = Field(default=None, alias="@chunk_analysis_elapsed_time")
    processing_result: Optional[str] = Field(default=None, alias="@processing_result")
    log_count: Optional[int] = Field(default=None, alias="@log_count")
    processing_mode: Optional[str] = Field(default=None, alias="@processing_mode")
    access_mode: Optional[str] = Field(default=None, alias="@access_mode")
    llm_provider: Optional[str] = Field(default=None, alias="@llm_provider")
    llm_model: Optional[str] = Field(default=None, alias="@llm_model")
    log_path: Optional[str] = Field(default=None, alias="@log_path")
    token_size_input: Optional[int] = Field(default=None, alias="@token_size_input")
    token_size_output: Optional[int] = Field(default=None, alias="@token_size_output")
    timestamp: Optional[str] = Field(default=None, alias="@timestamp")
    log_type: Optional[str] = Field(default=None, alias="@log_type")
    document_id: Optional[str] = Field(default=None, alias="@document_id")
    host: Optional[HostInfo] = Field(default=None, alias="@host")

@app.post("/suggest/test")
def test(req:LogAnalysis):
    print(req.model_dump_json())
    return req.model_dump_json()