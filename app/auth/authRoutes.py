from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
from app.config.database import get_db
from .authSchemas import Token, LoginData
from .authService import authenticate_user, login_for_access_token
from .authUtils import get_current_user
from app.models.UsuarioModel import UsuarioModel

router = APIRouter(prefix="/auth", tags=["autenticación"])

@router.post("/login", response_model=Token)
async def login(
    login_data: LoginData,
    db: AsyncSession = Depends(get_db)
):
    """Endpoint para iniciar sesión y obtener el token JWT."""
    user = await authenticate_user(login_data.username, login_data.password, db)
    return await login_for_access_token(user)

@router.post("/login/token", response_model=Token)
async def login_for_token(
    form_data: OAuth2PasswordRequestForm = Depends(OAuth2PasswordRequestForm),
    db: AsyncSession = Depends(get_db)
):
    """Endpoint para iniciar sesión usando form-data (útil para Swagger UI)."""
    user = await authenticate_user(form_data.username, form_data.password, db)
    return await login_for_access_token(user)

@router.get("/me", response_model=dict)
async def read_users_me(
    current_user: UsuarioModel = Depends(get_current_user)
):
    """Endpoint para obtener la información del usuario actual."""
    return {
        "id": current_user.id,
        "correo": current_user.correo,
        "nombres": current_user.nombres,
        "apellido": current_user.apellido,
        "rol": current_user.rol
    }