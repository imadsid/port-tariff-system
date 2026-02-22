"""
api/app.py
FastAPI application entry point.
Run with:  uvicorn api.app:app --port 8000
"""
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse


def create_app() -> FastAPI:
    app = FastAPI(
        title="Port Tariff Calculation API",
        version="1.0.0",
        description=(
            "Automated port dues calculation for Transnet National Ports Authority (TNPA). "
            "Calculates Light Dues, Port Dues, VTS Dues, Pilotage Dues, "
            "Towage Dues, and Berthing/Running Lines Dues from vessel data."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(Exception)
    async def _global_handler(request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": "Internal server error", "detail": str(exc)},
        )

    from api.routes import router
    app.include_router(router, prefix="/api/v1", tags=["Port Tariff"])

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, log_level="info")
