from sqlalchemy import Column, Integer, String, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from app.config.database import Base
from datetime import datetime

class UsuarioEstudianteModel(Base):
    __tablename__ = "usuario_estudiante"

    id = Column(Integer, primary_key=True, index=True)
    usuario_id = Column(Integer, ForeignKey('usuario.id'), nullable=False)
    estudiante_id = Column(Integer, ForeignKey('estudiante.id'), nullable=False)
    fecha_indexacion = Column(DateTime, default=datetime.utcnow)

    # Relaciones
    usuario = relationship("UsuarioModel", back_populates="estudiantes_asociados")
    estudiante = relationship("EstudianteModel", back_populates="usuarios_asociados") 