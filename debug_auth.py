import psycopg2
from psycopg2.extras import DictCursor
from passlib.context import CryptContext
from dotenv import load_dotenv
import os

# Cargar variables de entorno
load_dotenv()

# Configuración
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Credenciales de prueba
test_username = "admin@hare.com"
test_password = "Admin123*"

print("\n=== Diagnóstico de Autenticación ===")

try:
    # Conectar a la base de datos
    print("\n1. Conectando a la base de datos...")
    conn = psycopg2.connect(
        host=os.getenv("DATABASE_HOST", "localhost"),
        port=os.getenv("DATABASE_PORT", "5432"),
        dbname=os.getenv("DATABASE_NAME", "hare_db"),
        user=os.getenv("DATABASE_USER", "postgres"),
        password=os.getenv("DATABASE_PASSWORD", "root")
    )
    print("✓ Conexión exitosa")

    with conn.cursor(cursor_factory=DictCursor) as cur:
        # Verificar si el usuario existe
        print("\n2. Buscando usuario en la base de datos...")
        cur.execute("""
            SELECT id, correo, contraseña, rol 
            FROM usuario 
            WHERE correo = %s
        """, (test_username,))
        user = cur.fetchone()

        if user:
            print("✓ Usuario encontrado:")
            print(f"  ID: {user['id']}")
            print(f"  Correo: {user['correo']}")
            print(f"  Rol: {user['rol']}")
            print(f"  Hash almacenado: {user['contraseña']}")

            # Verificar la contraseña
            print("\n3. Verificando contraseña...")
            is_valid = pwd_context.verify(test_password, user['contraseña'])
            print(f"✓ Contraseña válida: {is_valid}")

            if not is_valid:
                # Generar un nuevo hash para comparar
                print("\n4. Generando nuevo hash para comparación...")
                new_hash = pwd_context.hash(test_password)
                print(f"  Nuevo hash generado: {new_hash}")
                print(f"  Hash almacenado: {user['contraseña']}")
        else:
            print("✗ Usuario no encontrado en la base de datos")

except Exception as e:
    print(f"\n✗ Error: {str(e)}")
finally:
    if 'conn' in locals():
        conn.close()

print("\n=== Fin del diagnóstico ===") 