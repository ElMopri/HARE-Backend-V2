from pydantic import BaseModel, EmailStr, Field, model_validator, ConfigDict
from typing import Optional

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: str | None = None
    rol: str | None = None

class LoginData(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra='allow')
    
    correo: Optional[EmailStr] = None
    contraseña: Optional[str] = None
    username: Optional[EmailStr] = None
    password: Optional[str] = None
    email: Optional[EmailStr] = None
    
    @model_validator(mode='after')
    def validate_fields(self):
        # Aceptar correo, username o email
        email_value = self.correo or self.username or self.email
        if not email_value:
            raise ValueError("Se requiere 'correo', 'username' o 'email'")
        
        # Aceptar contraseña o password
        password_value = self.contraseña or self.password
        if not password_value:
            raise ValueError("Se requiere 'contraseña' o 'password'")
        
        # Normalizar a correo y contraseña para uso interno
        self.correo = email_value
        self.contraseña = password_value
            
        return self
