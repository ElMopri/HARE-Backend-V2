from fastapi import FastAPI
from app.models import create_tables
from app.routers.UsuarioRoute import router as usuario_router
from app.routers.EstudianteRoute import router as estudiante_router
from app.routers.CatalogoRoute import router as catalogo_router

app = FastAPI(
    title="HARE Backend",
    description="Backend para el sistema educativo HARE",
    version="2.0.0"
)

# Incluir los routers
app.include_router(usuario_router)
app.include_router(estudiante_router)
app.include_router(catalogo_router)

# Crear las tablas al iniciar la aplicación
@app.on_event("startup")
async def startup_event():
    await create_tables()

@app.get("/")
def read_root():
    return {"message": "¡Bienvenido al Backend de HARE!"}