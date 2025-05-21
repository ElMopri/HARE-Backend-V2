from passlib.context import CryptContext
import psycopg2
from psycopg2.extras import DictCursor
from dotenv import load_dotenv
import os

# Cargar variables de entorno
load_dotenv()

# Usar la misma configuración que en la aplicación
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Generar el hash
password = "Admin123*"
hashed_password = pwd_context.hash(password)

# Obtener credenciales de la base de datos
db_host = os.getenv("DATABASE_HOST", "localhost")
db_port = os.getenv("DATABASE_PORT", "5432")
db_name = os.getenv("DATABASE_NAME", "hare_db")
db_user = os.getenv("DATABASE_USER", "postgres")
db_password = os.getenv("DATABASE_PASSWORD", "root")

print("\n=== Configuración de la base de datos ===")
print(f"Host: {db_host}")
print(f"Puerto: {db_port}")
print(f"Base de datos: {db_name}")
print(f"Usuario: {db_user}")
print(f"Contraseña: {'*' * len(db_password)}")

try:
    # Conectar a la base de datos
    print("\n=== Intentando conexión a la base de datos ===")
    conn = psycopg2.connect(
        host=db_host,
        port=db_port,
        dbname=db_name,
        user=db_user,
        password=db_password
    )
    print("Conexión exitosa!")
    
    with conn.cursor(cursor_factory=DictCursor) as cur:
        # Verificar si la tabla existe
        print("\n=== Verificando tabla usuario ===")
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'usuario'
            );
        """)
        table_exists = cur.fetchone()[0]
        
        if not table_exists:
            print("¡ERROR: La tabla 'usuario' no existe!")
            raise Exception("La tabla 'usuario' no existe en la base de datos")
            
        print("Tabla usuario encontrada")
        
        # Verificar si el usuario admin ya existe
        print("\n=== Verificando usuario admin ===")
        cur.execute("SELECT * FROM usuario WHERE correo = 'admin@hare.com'")
        existing_user = cur.fetchone()
        
        if existing_user:
            print("\n--- Usuario admin ya existe ---")
            print(f"ID: {existing_user['id']}")
            print(f"Correo: {existing_user['correo']}")
            
            # Actualizar la contraseña si es necesario
            print("\n=== Actualizando contraseña ===")
            update_query = """
            UPDATE usuario 
            SET contraseña = %s 
            WHERE correo = 'admin@hare.com';
            """
            cur.execute(update_query, (hashed_password,))
            conn.commit()
            print("Contraseña actualizada exitosamente")
        else:
            # Crear nuevo usuario admin
            print("\n=== Creando nuevo usuario admin ===")
            insert_query = """
            INSERT INTO usuario (nombres, apellido, correo, telefono, contraseña, rol)
            VALUES (%s, %s, %s, %s, %s, %s);
            """
            values = ('Admin', 'Sistema', 'admin@hare.com', '3001234567', hashed_password, 'admin')
            cur.execute(insert_query, values)
            conn.commit()
            print("Usuario admin creado exitosamente")
        
        # Verificar que el usuario existe
        print("\n=== Verificación final ===")
        cur.execute("SELECT * FROM usuario WHERE correo = 'admin@hare.com'")
        final_check = cur.fetchone()
        if final_check:
            print("Usuario encontrado en la base de datos:")
            print(f"ID: {final_check['id']}")
            print(f"Correo: {final_check['correo']}")
            print(f"Rol: {final_check['rol']}")
        else:
            print("¡ERROR: No se pudo encontrar el usuario después de crearlo!")
            
except Exception as e:
    print(f"\n¡ERROR: {str(e)}!")
finally:
    if 'conn' in locals():
        conn.close()

print("\nPara iniciar sesión como admin:")
print("Correo: admin@hare.com")
print("Contraseña: Admin123*") 