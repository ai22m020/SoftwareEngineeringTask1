from typing import Optional

from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def predictImage():
    return {"message": "Hallo Welt!"}
