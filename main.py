from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from v1.routers.places import router as places_router
from fastapi.responses import FileResponse
from v1.routers.auth.verify import router as auth_router


app = FastAPI(
    title="Quratr API",
    description="API for Quratr AI",
)

origins = [
    "http://localhost:3000",  # Explicitly allow local development server
    "http://localhost:3001",
    "https://quratr.com",
    "https://www.quratr.com",
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
        "version": "1.0",
    }

favicon_path = 'favicon.ico'

@app.get('/favicon.ico', include_in_schema=False)
async def favicon():
    return FileResponse(favicon_path)

app.include_router(places_router, prefix="/v1")
app.include_router(auth_router, prefix="/v1/auth")