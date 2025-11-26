# app/main.py
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
import os
import shutil

from app.utils.feature_extractor import get_ecapa_embedding_from_file
from app.utils.storage import save_embedding, load_all_embeddings
from app.utils.compare import match_embedding, THRESHOLD

app = FastAPI()

TEMP_DIR = "app/data/temp_audio"
os.makedirs(TEMP_DIR, exist_ok=True)


@app.get("/test_connection")
def test_connection():
    return {"status": "ok"}


@app.get("/list_speakers")
def list_speakers():
    users = load_all_embeddings()
    return {"speakers": list(users.keys())}


@app.delete("/delete_speaker/{name}")
def delete_speaker(name: str):
    speaker_dir = os.path.join("app/data/familiar_embeddings", name)
    if not os.path.exists(speaker_dir):
        raise HTTPException(status_code=404, detail="Speaker not found")

    shutil.rmtree(speaker_dir)
    return {"status": "deleted", "speaker": name}


@app.post("/clear_temp_audio")
def clear_temp_audio():
    try:
        shutil.rmtree(TEMP_DIR)
        os.makedirs(TEMP_DIR, exist_ok=True)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(500, f"Failed: {e}")


@app.post("/enroll")
async def enroll(name: str = Form(...), file: UploadFile = File(...)):
    temp_path = os.path.join(TEMP_DIR, file.filename)

    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    emb = get_ecapa_embedding_from_file(temp_path)
    save_embedding(name, emb)

    return {"status": "enrolled", "name": name, "dim": len(emb)}


@app.post("/verify")
async def verify(file: UploadFile = File(...)):

    temp_path = os.path.join(TEMP_DIR, file.filename)
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    query_emb = get_ecapa_embedding_from_file(temp_path)

    users = load_all_embeddings()

    best_match, best_score = None, 0.0
    for user, stored_emb in users.items():
        score = match_embedding(query_emb, stored_emb)
        if score > best_score:
            best_score, best_match = score, user

    if best_score >= THRESHOLD:
        return {"result": "familiar", "name": best_match, "similarity": float(best_score)}

    return {"result": "stranger", "similarity": float(best_score)}


@app.post("/recognize")
async def recognize(file: UploadFile = File(...)):
    return await verify(file)


@app.get("/")
def home():
    return {"message": "Speaker Recognition Backend Running"}
