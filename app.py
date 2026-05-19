from fastapi import FastAPI

app=FastAPI()

@app.get("/")
async def teste():
    return {"status":"ML online"}