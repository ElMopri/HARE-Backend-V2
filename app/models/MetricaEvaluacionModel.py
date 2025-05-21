from sqlalchemy import Column, Integer, Float, ForeignKey
from sqlalchemy.orm import relationship
from app.config.database import Base

class MetricaEvaluacionModel(Base):
    __tablename__ = "metrica_evaluacion"

    id = Column(Integer, primary_key=True, index=True)
    estudiante_id = Column(Integer, ForeignKey('estudiante.id'), nullable=False)
    promedio = Column(Float, nullable=False)

    # Relaciones
    estudiante = relationship("EstudianteModel", back_populates="metricas_evaluacion") 