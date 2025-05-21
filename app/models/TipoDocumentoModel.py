from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import relationship
from app.config.database import Base

class TipoDocumentoModel(Base):
    __tablename__ = "tipo_documento"

    id = Column(Integer, primary_key=True, index=True)
    nombre = Column(String, nullable=False, unique=True)

    # Relaciones
    estudiantes = relationship("EstudianteModel", back_populates="tipo_documento") 