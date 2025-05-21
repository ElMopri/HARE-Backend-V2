from pydantic import BaseModel, EmailStr

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    correo: str | None = None
    rol: str | None = None

class LoginData(BaseModel):
    username: EmailStr
    password: str
