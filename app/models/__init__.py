from app.config.database import Base, engine
from app.models.UsuarioModel import UsuarioModel
from app.models.EstudianteModel import EstudianteModel
from app.models.TipoDocumentoModel import TipoDocumentoModel
from app.models.EstadoMatriculaModel import EstadoMatriculaModel
from app.models.ColegioEgresadoModel import ColegioEgresadoModel
from app.models.MunicipioNacimientoModel import MunicipioNacimientoModel
from app.models.UsuarioEstudianteModel import UsuarioEstudianteModel
from app.models.MetricaEvaluacionModel import MetricaEvaluacionModel
from sqlalchemy import inspect, text

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

async def create_tables():
    """Crea las tablas en la base de datos si no existen."""
    async with engine.begin() as conn:
        # Verificar qué tablas ya existen usando una consulta SQL directa
        result = await conn.execute(text("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
        """))
        existing_tables = {row[0] for row in result}
        
        # Obtener todas las tablas que necesitamos crear
        tables_to_create = {
            table.__tablename__: table
            for table in models
        }
        
        # Filtrar solo las tablas que no existen
        missing_tables = {
            name: table
            for name, table in tables_to_create.items()
            if name not in existing_tables
        }
        
        if missing_tables:
            print("Creando tablas faltantes:", ", ".join(missing_tables.keys()))
            # Crear las tablas que faltan
            await conn.run_sync(lambda sync_conn: Base.metadata.create_all(
                bind=sync_conn,
                tables=[table.__table__ for table in missing_tables.values()]
            ))
        else:
            print("Todas las tablas ya existen en la base de datos") 