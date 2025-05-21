import asyncio
from passlib.context import CryptContext
import psycopg2
from psycopg2.extras import DictCursor
from dotenv import load_dotenv
import os

# Cargar variables de entorno
load_dotenv()

# Configuración de hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def test_password_hashing():
    """Prueba la generación y verificación de hash de contraseña"""
    print("\n=== Prueba de Hashing de Contraseña ===")
    
    # Contraseña de prueba
    password = "Admin123*"
    
    # Generar hash
    hashed_password = pwd_context.hash(password)
    print(f"Contraseña original: {password}")
    print(f"Hash generado: {hashed_password}")
    
    # Verificar contraseña
    is_valid = pwd_context.verify(password, hashed_password)
    print(f"Verificación de contraseña correcta: {is_valid}")
    
    # Probar con contraseña incorrecta
    wrong_password = "WrongPassword123"
    is_invalid = pwd_context.verify(wrong_password, hashed_password)
    print(f"Verificación de contraseña incorrecta: {is_invalid}")

def test_database_connection():
    """Prueba la conexión a la base de datos y verifica el usuario admin"""
    print("\n=== Prueba de Conexión a Base de Datos ===")
    
    try:
        # Obtener credenciales de la base de datos
        db_host = os.getenv("DATABASE_HOST", "localhost")
        db_port = os.getenv("DATABASE_PORT", "5432")
        db_name = os.getenv("DATABASE_NAME", "hare_db")
        db_user = os.getenv("DATABASE_USER", "postgres")
        db_password = os.getenv("DATABASE_PASSWORD", "tu_contraseña")
        
        # Conectar a la base de datos
        conn = psycopg2.connect(
            host=db_host,
            port=db_port,
            dbname=db_name,
            user=db_user,
            password=db_password
        )
        
        print("Conexión a la base de datos exitosa")
        
        # Buscar usuario admin
        with conn.cursor(cursor_factory=DictCursor) as cur:
            cur.execute("SELECT * FROM usuario WHERE correo = 'admin@hare.com'")
            user = cur.fetchone()
            
            if user:
                print("\nUsuario admin encontrado:")
                print(f"ID: {user['id']}")
                print(f"Correo: {user['correo']}")
                print(f"Hash de contraseña almacenado: {user['contraseña']}")
                
                # Probar verificación de contraseña
                test_password = "Admin123*"
                is_valid = pwd_context.verify(test_password, user['contraseña'])
                print(f"\nVerificación de contraseña de admin: {is_valid}")
            else:
                print("Usuario admin no encontrado en la base de datos")
        
    except Exception as e:
        print(f"Error al conectar a la base de datos: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    print("=== Iniciando Pruebas de Autenticación ===")
    
    # Ejecutar pruebas
    test_password_hashing()
    test_database_connection()
    
    print("\n=== Pruebas Completadas ===") 