from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
import os
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Configuración de la base de datos
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"postgresql+asyncpg://{os.getenv('DATABASE_USER', 'postgres')}:{os.getenv('DATABASE_PASSWORD', 'tu_contraseña')}@{os.getenv('DATABASE_HOST', 'localhost')}:{os.getenv('DATABASE_PORT', '5432')}/{os.getenv('DATABASE_NAME', 'hare_db')}"
)

# Crear el motor de la base de datos
engine = create_async_engine(
    DATABASE_URL,
    echo=True,  # Para ver las consultas SQL en la consola
)

# Crear una clase de sesión asíncrona
AsyncSessionLocal = sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)

# Base declarativa para los modelos
Base = declarative_base()

# Función para obtener la base de datos
async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close() 