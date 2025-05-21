from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import relationship
from app.config.database import Base

class ColegioEgresadoModel(Base):
    __tablename__ = "colegio_egresado"

    id = Column(Integer, primary_key=True, index=True)
    nombre = Column(String, nullable=False, unique=True)

    # Relaciones
    estudiantes = relationship("EstudianteModel", back_populates="colegio_egresado") 