from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List
from app.config.database import get_db
from app.models.TipoDocumentoModel import TipoDocumentoModel
from app.models.EstadoMatriculaModel import EstadoMatriculaModel
from app.models.ColegioEgresadoModel import ColegioEgresadoModel
from app.models.MunicipioNacimientoModel import MunicipioNacimientoModel
from app.schemas.catalogs import (
    TipoDocumento, TipoDocumentoCreate,
    EstadoMatricula, EstadoMatriculaCreate,
    ColegioEgresado, ColegioEgresadoCreate,
    MunicipioNacimiento, MunicipioNacimientoCreate
)
from sqlalchemy import select, delete

router = APIRouter(tags=["catálogos"])

# Funciones genéricas para CRUD de catálogos
async def create_catalog_item(item_create, model_class, db: AsyncSession):
    db_item = model_class(nombre=item_create.nombre)
    db.add(db_item)
    await db.commit()
    await db.refresh(db_item)
    return db_item

async def get_catalog_items(model_class, db: AsyncSession):
    query = select(model_class)
    result = await db.execute(query)
    return result.scalars().all()

async def get_catalog_item(item_id: int, model_class, db: AsyncSession):
    query = select(model_class).where(model_class.id == item_id)
    result = await db.execute(query)
    item = result.scalar_one_or_none()
    if item is None:
        raise HTTPException(status_code=404, detail="Item no encontrado")
    return item

async def delete_catalog_item(item_id: int, model_class, db: AsyncSession):
    query = delete(model_class).where(model_class.id == item_id)
    result = await db.execute(query)
    await db.commit()
    if result.rowcount == 0:
        raise HTTPException(status_code=404, detail="Item no encontrado")
    return {"message": "Item eliminado exitosamente"}

# Rutas para Tipos de Documento
@router.post("/tipos-documento/", response_model=TipoDocumento)
async def create_tipo_documento(tipo: TipoDocumentoCreate, db: AsyncSession = Depends(get_db)):
    return await create_catalog_item(tipo, TipoDocumentoModel, db)

@router.get("/tipos-documento/", response_model=List[TipoDocumento])
async def read_tipos_documento(db: AsyncSession = Depends(get_db)):
    return await get_catalog_items(TipoDocumentoModel, db)

@router.get("/tipos-documento/{item_id}", response_model=TipoDocumento)
async def read_tipo_documento(item_id: int, db: AsyncSession = Depends(get_db)):
    return await get_catalog_item(item_id, TipoDocumentoModel, db)

@router.delete("/tipos-documento/{item_id}")
async def delete_tipo_documento(item_id: int, db: AsyncSession = Depends(get_db)):
    return await delete_catalog_item(item_id, TipoDocumentoModel, db)

# Rutas para Estados de Matrícula
@router.post("/estados-matricula/", response_model=EstadoMatricula)
async def create_estado_matricula(estado: EstadoMatriculaCreate, db: AsyncSession = Depends(get_db)):
    return await create_catalog_item(estado, EstadoMatriculaModel, db)

@router.get("/estados-matricula/", response_model=List[EstadoMatricula])
async def read_estados_matricula(db: AsyncSession = Depends(get_db)):
    return await get_catalog_items(EstadoMatriculaModel, db)

@router.get("/estados-matricula/{item_id}", response_model=EstadoMatricula)
async def read_estado_matricula(item_id: int, db: AsyncSession = Depends(get_db)):
    return await get_catalog_item(item_id, EstadoMatriculaModel, db)

@router.delete("/estados-matricula/{item_id}")
async def delete_estado_matricula(item_id: int, db: AsyncSession = Depends(get_db)):
    return await delete_catalog_item(item_id, EstadoMatriculaModel, db)

# Rutas para Colegios Egresados
@router.post("/colegios/", response_model=ColegioEgresado)
async def create_colegio(colegio: ColegioEgresadoCreate, db: AsyncSession = Depends(get_db)):
    return await create_catalog_item(colegio, ColegioEgresadoModel, db)

@router.get("/colegios/", response_model=List[ColegioEgresado])
async def read_colegios(db: AsyncSession = Depends(get_db)):
    return await get_catalog_items(ColegioEgresadoModel, db)

@router.get("/colegios/{item_id}", response_model=ColegioEgresado)
async def read_colegio(item_id: int, db: AsyncSession = Depends(get_db)):
    return await get_catalog_item(item_id, ColegioEgresadoModel, db)

@router.delete("/colegios/{item_id}")
async def delete_colegio(item_id: int, db: AsyncSession = Depends(get_db)):
    return await delete_catalog_item(item_id, ColegioEgresadoModel, db)

# Rutas para Municipios de Nacimiento
@router.post("/municipios/", response_model=MunicipioNacimiento)
async def create_municipio(municipio: MunicipioNacimientoCreate, db: AsyncSession = Depends(get_db)):
    return await create_catalog_item(municipio, MunicipioNacimientoModel, db)

@router.get("/municipios/", response_model=List[MunicipioNacimiento])
async def read_municipios(db: AsyncSession = Depends(get_db)):
    return await get_catalog_items(MunicipioNacimientoModel, db)

@router.get("/municipios/{item_id}", response_model=MunicipioNacimiento)
async def read_municipio(item_id: int, db: AsyncSession = Depends(get_db)):
    return await get_catalog_item(item_id, MunicipioNacimientoModel, db)

@router.delete("/municipios/{item_id}")
async def delete_municipio(item_id: int, db: AsyncSession = Depends(get_db)):
    return await delete_catalog_item(item_id, MunicipioNacimientoModel, db) 