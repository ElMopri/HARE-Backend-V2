from sqlalchemy import Column, Integer, String, UniqueConstraint
from sqlalchemy.orm import relationship
from app.config.database import Base

class UsuarioModel(Base):
    __tablename__ = "usuario"
    
    __table_args__ = (
        UniqueConstraint('nombres', 'apellido', name='uq_nombres_apellido'),
    )

    id = Column(Integer, primary_key=True, index=True)
    nombres = Column(String, nullable=False)
    apellido = Column(String, nullable=False)
    telefono = Column(String, nullable=True)
    correo = Column(String, unique=True, index=True, nullable=False)
    contraseña = Column(String, nullable=False)
    rol = Column(String, nullable=False)

    # Relaciones
    estudiantes_asociados = relationship("UsuarioEstudianteModel", back_populates="usuario")
