from sqlalchemy import Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class UserModel(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    nombres = Column(String, nullable=False)
    apellido = Column(String, nullable=False)
    telefono = Column(String, nullable=True)
    correo = Column(String, unique=True, index=True, nullable=False)
    contraseña = Column(String, nullable=False)
