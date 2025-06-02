from fastapi import FastAPI
from app.models import create_tables
from app.routers.UsuarioRoute import router as usuario_router
from app.routers.EstudianteRoute import router as estudiante_router
from app.routers.CatalogoRoute import router as catalogo_router
from app.auth.authRoutes import router as auth_router
from fastapi.middleware.cors import CORSMiddleware

from fastapi import Request

app = FastAPI(
    title="HARE Backend",
    description="Backend para el sistema educativo HARE",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://app93.proyectos.fireploy.online", "http://localhost:10001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Incluir los routers
app.include_router(auth_router)  # Primero incluimos el router de autenticación
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