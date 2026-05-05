from fastapi import FastAPI

from src.app.api.routes.flows import router as flows_router
from src.app.api.routes.health import router as health_router
from src.app.api.routes.predictions import router as predictions_router
from src.app.api.routes.tickets import router as tickets_router

app = FastAPI(
    title="Signal Conso API",
    version="0.2.0",
    description=(
        "API Signal Conso — prédictions ML, gestion des tickets "
        "et orchestration des flows Prefect KPI."
    ),
)

app.include_router(health_router)
app.include_router(tickets_router, prefix="/tickets", tags=["tickets"])
app.include_router(predictions_router, prefix="/predictions", tags=["predictions"])
app.include_router(flows_router, prefix="/flows", tags=["flows prefect"])


@app.get("/", tags=["root"])
def root() -> dict:
    return {
        "message": "Bienvenue sur la plateforme intelligente de Signal Conso !",
        "docs": "/docs",
        "flows": "/flows",
    }