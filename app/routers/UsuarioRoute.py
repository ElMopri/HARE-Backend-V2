from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List
from app.config.database import get_db
from app.models.UsuarioModel import UsuarioModel
from app.schemas.usuario import UsuarioCreate, Usuario, UsuarioUpdate
from sqlalchemy import select, update, delete
from passlib.context import CryptContext
from app.auth.authUtils import get_current_user

router = APIRouter(prefix="/usuarios", tags=["usuarios"])
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

@router.post("/", response_model=Usuario)
async def create_usuario(
    usuario: UsuarioCreate, 
    db: AsyncSession = Depends(get_db),
    current_user: UsuarioModel = Depends(get_current_user)
):
    hashed_password = pwd_context.hash(usuario.contraseña)
    db_usuario = UsuarioModel(
        nombres=usuario.nombres,
        apellido=usuario.apellido,
        correo=usuario.correo,
        telefono=usuario.telefono,
        contraseña=hashed_password,
        rol="admin"  # Por defecto, todos los usuarios son admin
    )
    db.add(db_usuario)
    await db.commit()
    await db.refresh(db_usuario)
    return db_usuario

@router.get("/", response_model=List[Usuario])
async def read_usuarios(
    skip: int = 0, 
    limit: int = 100, 
    db: AsyncSession = Depends(get_db),
    current_user: UsuarioModel = Depends(get_current_user)
):
    query = select(UsuarioModel).offset(skip).limit(limit)
    result = await db.execute(query)
    usuarios = result.scalars().all()
    return usuarios

@router.get("/{usuario_id}", response_model=Usuario)
async def read_usuario(
    usuario_id: int, 
    db: AsyncSession = Depends(get_db),
    current_user: UsuarioModel = Depends(get_current_user)
):
    query = select(UsuarioModel).where(UsuarioModel.id == usuario_id)
    result = await db.execute(query)
    usuario = result.scalar_one_or_none()
    if usuario is None:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")
    return usuario

@router.put("/{usuario_id}", response_model=Usuario)
async def update_usuario(
    usuario_id: int, 
    usuario: UsuarioUpdate, 
    db: AsyncSession = Depends(get_db),
    current_user: UsuarioModel = Depends(get_current_user)
):
    query = select(UsuarioModel).where(UsuarioModel.id == usuario_id)
    result = await db.execute(query)
    db_usuario = result.scalar_one_or_none()
    
    if db_usuario is None:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")

    update_data = usuario.dict(exclude_unset=True)
    if "contraseña" in update_data:
        update_data["contraseña"] = pwd_context.hash(update_data["contraseña"])

    # Asegurarse de que el rol siempre sea admin
    if "rol" in update_data:
        update_data["rol"] = "admin"

    for key, value in update_data.items():
        setattr(db_usuario, key, value)

    await db.commit()
    await db.refresh(db_usuario)
    return db_usuario

@router.delete("/{usuario_id}")
async def delete_usuario(
    usuario_id: int, 
    db: AsyncSession = Depends(get_db),
    current_user: UsuarioModel = Depends(get_current_user)
):
    query = delete(UsuarioModel).where(UsuarioModel.id == usuario_id)
    result = await db.execute(query)
    await db.commit()
    if result.rowcount == 0:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")
    return {"message": "Usuario eliminado exitosamente"} 