from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List
from app.config.database import get_db
from app.models.EstudianteModel import EstudianteModel
from app.schemas.estudiante import EstudianteCreate, Estudiante, EstudianteUpdate
from sqlalchemy import select, update, delete

router = APIRouter(prefix="/estudiantes", tags=["estudiantes"])

@router.post("/", response_model=Estudiante)
async def create_estudiante(estudiante: EstudianteCreate, db: AsyncSession = Depends(get_db)):
    db_estudiante = EstudianteModel(**estudiante.dict())
    db.add(db_estudiante)
    await db.commit()
    await db.refresh(db_estudiante)
    return db_estudiante

@router.get("/", response_model=List[Estudiante])
async def read_estudiantes(skip: int = 0, limit: int = 100, db: AsyncSession = Depends(get_db)):
    query = select(EstudianteModel).offset(skip).limit(limit)
    result = await db.execute(query)
    estudiantes = result.scalars().all()
    return estudiantes

@router.get("/{estudiante_id}", response_model=Estudiante)
async def read_estudiante(estudiante_id: int, db: AsyncSession = Depends(get_db)):
    query = select(EstudianteModel).where(EstudianteModel.id == estudiante_id)
    result = await db.execute(query)
    estudiante = result.scalar_one_or_none()
    if estudiante is None:
        raise HTTPException(status_code=404, detail="Estudiante no encontrado")
    return estudiante

@router.put("/{estudiante_id}", response_model=Estudiante)
async def update_estudiante(estudiante_id: int, estudiante: EstudianteUpdate, db: AsyncSession = Depends(get_db)):
    query = select(EstudianteModel).where(EstudianteModel.id == estudiante_id)
    result = await db.execute(query)
    db_estudiante = result.scalar_one_or_none()
    
    if db_estudiante is None:
        raise HTTPException(status_code=404, detail="Estudiante no encontrado")

    update_data = estudiante.dict(exclude_unset=True)
    for key, value in update_data.items():
        setattr(db_estudiante, key, value)

    await db.commit()
    await db.refresh(db_estudiante)
    return db_estudiante

@router.delete("/{estudiante_id}")
async def delete_estudiante(estudiante_id: int, db: AsyncSession = Depends(get_db)):
    query = delete(EstudianteModel).where(EstudianteModel.id == estudiante_id)
    result = await db.execute(query)
    await db.commit()
    if result.rowcount == 0:
        raise HTTPException(status_code=404, detail="Estudiante no encontrado")
    return {"message": "Estudiante eliminado exitosamente"} 