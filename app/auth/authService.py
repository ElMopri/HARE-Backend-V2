from datetime import timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.models.UsuarioModel import UsuarioModel
from .authUtils import verify_password, create_access_token, ACCESS_TOKEN_EXPIRE_MINUTES
from fastapi import HTTPException, status

async def authenticate_user(username: str, password: str, db: AsyncSession) -> UsuarioModel:
    """Autentica un usuario por correo/username y contraseña."""
    query = select(UsuarioModel).where(UsuarioModel.correo == username)
    result = await db.execute(query)
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Correo o contraseña incorrectos",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not verify_password(password, user.contraseña):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Correo o contraseña incorrectos",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return user

async def login_for_access_token(user: UsuarioModel) -> dict:
    """Genera un token de acceso para un usuario autenticado."""
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.correo, "rol": user.rol},
        expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}
