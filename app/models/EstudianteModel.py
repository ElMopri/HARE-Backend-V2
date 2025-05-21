from sqlalchemy import Column, Integer, String, ForeignKey, UniqueConstraint
from sqlalchemy.orm import relationship
from app.config.database import Base

class EstudianteModel(Base):
    __tablename__ = "estudiante"

    __table_args__ = (
        UniqueConstraint('codigo', 'documento', name='uq_codigo_documento'),
    )

    id = Column(Integer, primary_key=True, index=True)
    codigo = Column(String, nullable=False)
    nombre = Column(String, nullable=False)
    tipo_documento_id = Column(Integer, ForeignKey('tipo_documento.id'), nullable=False)
    documento = Column(String, nullable=False)
    semestre = Column(String, nullable=False)
    pensum = Column(String, nullable=False)
    ingreso = Column(String, nullable=False)
    estado_matricula_id = Column(Integer, ForeignKey('estado_matricula.id'), nullable=False)
    celular = Column(String, nullable=True)
    email_personal = Column(String, nullable=True)
    email_institucional = Column(String, nullable=False)
    colegio_egresado_id = Column(Integer, ForeignKey('colegio_egresado.id'), nullable=False)
    municipio_nacimiento_id = Column(Integer, ForeignKey('municipio_nacimiento.id'), nullable=False)

    # Relaciones
    tipo_documento = relationship("TipoDocumentoModel", back_populates="estudiantes")
    estado_matricula = relationship("EstadoMatriculaModel", back_populates="estudiantes")
    colegio_egresado = relationship("ColegioEgresadoModel", back_populates="estudiantes")
    municipio_nacimiento = relationship("MunicipioNacimientoModel", back_populates="estudiantes")
    usuarios_asociados = relationship("UsuarioEstudianteModel", back_populates="estudiante")
    metricas_evaluacion = relationship("MetricaEvaluacionModel", back_populates="estudiante") 