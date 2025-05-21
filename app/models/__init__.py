from app.config.database import Base, engine
from app.models.UsuarioModel import UsuarioModel
from app.models.EstudianteModel import EstudianteModel
from app.models.TipoDocumentoModel import TipoDocumentoModel
from app.models.EstadoMatriculaModel import EstadoMatriculaModel
from app.models.ColegioEgresadoModel import ColegioEgresadoModel
from app.models.MunicipioNacimientoModel import MunicipioNacimientoModel
from app.models.UsuarioEstudianteModel import UsuarioEstudianteModel
from app.models.MetricaEvaluacionModel import MetricaEvaluacionModel
import asyncio

# Lista de todos los modelos para asegurar que están registrados
models = [
    UsuarioModel,
    EstudianteModel,
    TipoDocumentoModel,
    EstadoMatriculaModel,
    ColegioEgresadoModel,
    MunicipioNacimientoModel,
    UsuarioEstudianteModel,
    MetricaEvaluacionModel
]

# Función para crear todas las tablas
async def create_tables():
    async with engine.begin() as conn:
        # Primero, eliminar todas las tablas existentes
        await conn.run_sync(Base.metadata.drop_all)
        # Luego, crear todas las tablas
        await conn.run_sync(Base.metadata.create_all)
        print("¡Todas las tablas han sido creadas!") 