from pydantic import BaseModel, EmailStr
from typing import Optional

class EstudianteBase(BaseModel):
    codigo: str
    nombre: str
    tipo_documento_id: int
    documento: str
    semestre: str
    pensum: str
    ingreso: str
    estado_matricula_id: int
    celular: Optional[str] = None
    email_personal: Optional[str] = None
    email_institucional: EmailStr
    colegio_egresado_id: int
    municipio_nacimiento_id: int

class EstudianteCreate(EstudianteBase):
    pass

class EstudianteUpdate(BaseModel):
    codigo: Optional[str] = None
    nombre: Optional[str] = None
    tipo_documento_id: Optional[int] = None
    documento: Optional[str] = None
    semestre: Optional[str] = None
    pensum: Optional[str] = None
    ingreso: Optional[str] = None
    estado_matricula_id: Optional[int] = None
    celular: Optional[str] = None
    email_personal: Optional[str] = None
    email_institucional: Optional[EmailStr] = None
    colegio_egresado_id: Optional[int] = None
    municipio_nacimiento_id: Optional[int] = None

class Estudiante(EstudianteBase):
    id: int

    class Config:
        from_attributes = True 