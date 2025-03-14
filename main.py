from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
from v1.routers.places import router as places_router


app = FastAPI(
    title="Quratr API",
    description="API for Quratr",
)

# origins = [
#     "**",
# ]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

@app.get("/")
def read_root():
    return {"Hello": "World"}

app.include_router(places_router, prefix="/v1")