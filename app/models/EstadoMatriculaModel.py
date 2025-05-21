from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import relationship
from app.config.database import Base

class EstadoMatriculaModel(Base):
    __tablename__ = "estado_matricula"

    id = Column(Integer, primary_key=True, index=True)
    nombre = Column(String, nullable=False, unique=True)

    # Relaciones
    estudiantes = relationship("EstudianteModel", back_populates="estado_matricula") 