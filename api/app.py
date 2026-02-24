from api.main import app
from api.rag_routes import router as rag_router

app.include_router(rag_router, prefix="/rag", tags=["rag"])
