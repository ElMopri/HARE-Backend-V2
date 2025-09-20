from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
import os
from dotenv import load_dotenv

# Cargar variables de entorno (solo funciona en local)
load_dotenv()

# Intentar usar DATABASE_URL primero (para Render)
DATABASE_URL = os.getenv("DATABASE_URL")

# Si no existe DATABASE_URL, construirla desde variables individuales (para local)
if not DATABASE_URL:
    PROTOCOLE = "postgresql+asyncpg"
    DB_USER = os.getenv("DB_USER")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    DB_HOST = os.getenv("DB_HOST")
    DB_PORT = os.getenv("DB_PORT")
    DB_DATABASE = os.getenv("DB_DATABASE")
    
    # Validar que todas las variables existan
    if not all([DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_DATABASE]):
        raise ValueError("Missing database configuration. Set either DATABASE_URL or individual DB_* variables")
    
    # Construir la URL
    DATABASE_URL = f"{PROTOCOLE}://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_DATABASE}"

print(f"DATABASE_URL configured: {DATABASE_URL[:50]}...")  # Para debug (solo primeros 50 chars)

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