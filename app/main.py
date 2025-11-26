# app/main.py
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
import os
import shutil
import numpy as np

from app.utils.feature_extractor import get_ecapa_embedding_from_file
from app.utils.storage import save_embedding, load_all_embeddings
from app.utils.compare import match_embedding, THRESHOLD

app = FastAPI()

TEMP_DIR = "app/data/temp_audio"
os.makedirs(TEMP_DIR, exist_ok=True)


@app.get("/")
def home():
    return {"message": "Backend running offline!"}


@app.get("/test_connection")
def test_connection():
    return {"status": "ok"}


@app.get("/list_speakers")
def list_speakers():
    users = load_all_embeddings()
    return {"speakers": list(users.keys())}


@app.delete("/delete_speaker/{name}")
def delete_speaker(name: str):
    p = os.path.join("app/data/familiar_embeddings", name)
    if not os.path.exists(p):
        raise HTTPException(404, "Speaker not found")
    shutil.rmtree(p)
    return {"status": "deleted", "name": name}


@app.post("/clear_temp_audio")
def clear_temp_audio():
    shutil.rmtree(TEMP_DIR, ignore_errors=True)
    os.makedirs(TEMP_DIR, exist_ok=True)
    return {"status": "cleared"}


@app.post("/enroll")
async def enroll(name: str = Form(...), file: UploadFile = File(...)):
    temp = os.path.join(TEMP_DIR, file.filename)

    with open(temp, "wb") as f:
        shutil.copyfileobj(file.file, f)

    emb = get_ecapa_embedding_from_file(temp)
    save_embedding(name, emb)

    return {"status": "enrolled", "name": name}


@app.post("/verify")
async def verify(file: UploadFile = File(...)):
    temp = os.path.join(TEMP_DIR, file.filename)

    with open(temp, "wb") as f:
        shutil.copyfileobj(file.file, f)

    query = get_ecapa_embedding_from_file(temp)
    users = load_all_embeddings()

    best = None
    best_score = 0

    for user, stored in users.items():
        score = match_embedding(query, stored)
        if score > best_score:
            best = user
            best_score = score

    if best_score >= THRESHOLD:
        return {"result": "familiar", "name": best, "similarity": float(best_score)}

    return {"result": "stranger", "similarity": float(best_score)}


@app.post("/recognize")
async def recognize(file: UploadFile = File(...)):
    return await verify(file)