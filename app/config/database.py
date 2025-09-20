from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
import os

# Usar directamente DATABASE_URL de las variables de entorno de Render
DATABASE_URL = os.getenv("DATABASE_URL")

# Validación
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is not set")

print(f"DATABASE_URL: {DATABASE_URL}")  # Para debug (remover después)

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