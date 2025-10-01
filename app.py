# 라이브러리
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from diffusers import StableDiffusionPipeline
import torch

# 앱 생성
app = FastAPI(title="기계 설명 서버")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 입력 형식
class RequestData(BaseModel):
    device: str
    action: str

# 모델 로드 (초경량 sd-tiny)
model_id = "Shakker-Labs/sd-tiny"
pipe = StableDiffusionPipeline.from_pretrained(model_id, safety_checker=None)
pipe.to("cpu")

# 텍스트 설명
def make_text(device: str, action: str) -> str:
    return f"{device}에서 '{action}' 하는 법이에요. 버튼을 누르면 불이 들어와요!"

# API
@app.post("/generate/")
def generate(data: RequestData):
    text = make_text(data.device, data.action)

    images = []
    for i in range(1, 4):  # 3장만 만들어도 충분
        prompt = f"simple cartoon, child drawing, step {i}, {data.device}, {data.action}"
        image = pipe(prompt, num_inference_steps=8).images[0]
        filename = f"step_{i}.png"
        image.save(filename)
        images.append(filename)

    return {"text": text, "images": images}
