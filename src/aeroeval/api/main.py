"""
AeroEval FastAPI Main Application.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from aeroeval.api.routes import evaluate_router, models_router, results_router

app = FastAPI(
    title="AeroEval API",
    description="Real-Time UAV Vision & AI Evaluation REST API Platform",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for cross-origin dashboard / frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(models_router)
app.include_router(evaluate_router)
app.include_router(results_router)


@app.get("/", tags=["Health"])
def root():
    """Health check and platform metadata."""
    return {
        "platform": "AeroEval — UAV Vision & AI Evaluation Platform",
        "status": "healthy",
        "version": "0.1.0",
        "documentation": "/docs"
    }
