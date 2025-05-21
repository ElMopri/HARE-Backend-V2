from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class UsuarioEstudianteBase(BaseModel):
    usuario_id: int
    estudiante_id: int

class UsuarioEstudianteCreate(UsuarioEstudianteBase):
    pass

class UsuarioEstudiante(UsuarioEstudianteBase):
    id: int
    fecha_indexacion: datetime

    class Config:
        from_attributes = True 