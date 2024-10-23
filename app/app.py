from fastapi import FastAPI

app = FastAPI(
    title="Anomaly Detection",
    # description="Finding Anomaly", 
    version="0.1")


# @app.get("/")
# async def root():
#     return {"message": "Hello World"}


@app.get("/")
async def health_check():
    return {'health':'OK'}


@app.get("/detect/")
async def predict():
    return {'Anomaly':'TRUE'}