from pydantic import BaseModel, EmailStr
from typing import Optional

class UsuarioBase(BaseModel):
    nombres: str
    apellido: str
    correo: EmailStr
    telefono: Optional[str] = None
    rol: str

class UsuarioCreate(UsuarioBase):
    contraseña: str

class UsuarioUpdate(BaseModel):
    nombres: Optional[str] = None
    apellido: Optional[str] = None
    correo: Optional[EmailStr] = None
    telefono: Optional[str] = None
    rol: Optional[str] = None
    contraseña: Optional[str] = None

class Usuario(UsuarioBase):
    id: int

    class Config:
        from_attributes = True 