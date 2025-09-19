from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import os
import json

# Импортируем GemmaClient напрямую
from src.gemma_client import GemmaClient
import os
import json
import base64
from fastapi import UploadFile, File

app = FastAPI(title="CDEK Packaging & Tariff API", description="API для определения упаковки и расчёта тарифов СДЭК")

# Модели данных
class AdditionalService(BaseModel):
    alias: str
    params: List[dict]

class Package(BaseModel):
    width: float
    length: float
    height: float
    weight: float

class TariffRequest(BaseModel):
    serviceId: str
    mode: str
    payerType: str
    currencyMark: str
    senderCityId: str
    receiverCityId: str
    packages: List[Package]
    additionalServices: List[AdditionalService]

class TextRequest(BaseModel):
    description: str

# Эндпоинт для обработки изображения товара
@app.post("/api/package/from-image")
async def package_from_image(file: UploadFile = File(...)):
    try:
        # Сохраняем загруженное изображение во временный файл
        contents = await file.read()
        temp_path = f"/tmp/{file.filename}"
        with open(temp_path, "wb") as f:
            f.write(contents)
        
        # Инициализируем клиент Gemma
        client = GemmaClient(
            api_key=os.getenv("GEMMA_API_KEY"),
            base_url=os.getenv("GEMMA_BASE_URL")
        )
        
        # Читаем промт из файла
        with open("src/promts/prompt_photo.txt", "r", encoding="utf-8") as f:
            photo_prompt = f.read().strip()
        
        # Получаем параметры упаковки от Gemma
        response = client.execute_with_image(prompt=photo_prompt, image_path=temp_path)
        
        # Удаляем временный файл
        os.remove(temp_path)
        
        return JSONResponse(content=json.loads(response))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка обработки изображения: {str(e)}")

# Эндпоинт для обработки текстового описания товара
@app.post("/api/package/from-text")
async def package_from_text(request: TextRequest):
    try:
        # Инициализируем клиент Gemma
        client = GemmaClient(
            api_key=os.getenv("GEMMA_API_KEY"),
            base_url=os.getenv("GEMMA_BASE_URL")
        )
        
        # Читаем промт из файла
        with open("src/promts/prompt_from_text.txt", "r", encoding="utf-8") as f:
            text_prompt = f.read().strip()
        
        # Формируем полный промт, объединяя шаблон и ввод пользователя
        full_prompt = f"{text_prompt}\n\nПользовательское описание: {request.description}"
        
        # Получаем параметры упаковки от Gemma
        response = client.execute_without_image(prompt=full_prompt)
        
        return JSONResponse(content=json.loads(response))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка обработки текста: {str(e)}")

# Эндпоинт для расчёта тарифа CDEK (заглушка — требует реальной интеграции)
@app.post("/api/calculate-tariff")
async def calculate_tariff(request: TariffRequest):
    # Здесь должен быть вызов внешнего API CDEK
    # Пока возвращаем заглушку
    return {
        "data": {
            "totalSum": 990.50,
            "deliveryPeriodMin": 2,
            "deliveryPeriodMax": 4,
            "tariffId": "3e0900c7-18f1-4128-9d85-545143235849",
            "deliveryDateMin": "2025-09-21T00:00:00+07:00",
            "deliveryDateMax": "2025-09-23T00:00:00+07:00"
        }
    }

@app.get("/")
async def root():
    return {"message": "CDEK Packaging & Tariff API is running. Use /api/package/from-image, /api/package/from-text, or /api/calculate-tariff"}
