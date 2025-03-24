from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from v1.routers.places import router as places_router

app = FastAPI(
    title="Quratr API",
    description="API for Quratr AI",
)

origins = [
    "**",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {
        "name": "Quratr Agentic API",
        "about": "Quratr is an AI-powered platform designed to provide intelligent solutions for various applications.",
        "version": "0.1.0",
    }

app.include_router(places_router, prefix="/v1")