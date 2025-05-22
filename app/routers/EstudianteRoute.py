from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List
from app.config.database import get_db
from app.models.EstudianteModel import EstudianteModel
from app.schemas.estudiante import EstudianteCreate, Estudiante, EstudianteUpdate
from sqlalchemy import select, update, delete
from app.auth.authUtils import get_current_user
from app.models.UsuarioModel import UsuarioModel
from app.models.MetricaEvaluacionModel import MetricaEvaluacionModel
from app.models.UsuarioEstudianteModel import UsuarioEstudianteModel
from app.models.TipoDocumentoModel import TipoDocumentoModel
from app.models.EstadoMatriculaModel import EstadoMatriculaModel
from app.models.ColegioEgresadoModel import ColegioEgresadoModel
from app.models.MunicipioNacimientoModel import MunicipioNacimientoModel
import pandas as pd
import io

router = APIRouter(prefix="/estudiantes", tags=["estudiantes"])

@router.post("/", response_model=Estudiante)
async def create_estudiante(
    estudiante: EstudianteCreate, 
    db: AsyncSession = Depends(get_db),
    current_user: UsuarioModel = Depends(get_current_user)
):
    db_estudiante = EstudianteModel(**estudiante.dict())
    db.add(db_estudiante)
    await db.commit()
    await db.refresh(db_estudiante)
    return db_estudiante

@router.get("/", response_model=List[Estudiante])
async def read_estudiantes(
    skip: int = 0, 
    limit: int = 100, 
    db: AsyncSession = Depends(get_db),
    current_user: UsuarioModel = Depends(get_current_user)
):
    query = select(EstudianteModel).offset(skip).limit(limit)
    result = await db.execute(query)
    estudiantes = result.scalars().all()
    return estudiantes

@router.get("/{estudiante_id}", response_model=Estudiante)
async def read_estudiante(
    estudiante_id: int, 
    db: AsyncSession = Depends(get_db),
    current_user: UsuarioModel = Depends(get_current_user)
):
    query = select(EstudianteModel).where(EstudianteModel.id == estudiante_id)
    result = await db.execute(query)
    estudiante = result.scalar_one_or_none()
    if estudiante is None:
        raise HTTPException(status_code=404, detail="Estudiante no encontrado")
    return estudiante

@router.put("/{estudiante_id}", response_model=Estudiante)
async def update_estudiante(
    estudiante_id: int, 
    estudiante: EstudianteUpdate, 
    db: AsyncSession = Depends(get_db),
    current_user: UsuarioModel = Depends(get_current_user)
):
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
async def delete_estudiante(
    estudiante_id: int, 
    db: AsyncSession = Depends(get_db),
    current_user: UsuarioModel = Depends(get_current_user)
):
    query = delete(EstudianteModel).where(EstudianteModel.id == estudiante_id)
    result = await db.execute(query)
    await db.commit()
    if result.rowcount == 0:
        raise HTTPException(status_code=404, detail="Estudiante no encontrado")
    return {"message": "Estudiante eliminado exitosamente"}

@router.post("/cargar-excel/")
async def cargar_estudiantes_excel(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    current_user: UsuarioModel = Depends(get_current_user)
):
    if not file.filename.endswith('.xlsx'):
        raise HTTPException(
            status_code=400,
            detail="El archivo debe ser un Excel (.xlsx)"
        )

    try:
        # Leer el archivo Excel
        contents = await file.read()
        df = pd.read_excel(io.BytesIO(contents))
        
        # Verificar las columnas requeridas
        columnas_requeridas = [
            'Codigo Alumno', 'Nombre Alumno', 'Tipo Doc', 'Documento',
            'Semestre', 'Pensum', 'Ingreso', 'Promedio', 'Estado Matricula',
            'Celular', 'Email', 'Email Institucional', 'Colegio Egresado',
            'Municipio Nacimiento'
        ]
        
        columnas_faltantes = [col for col in columnas_requeridas if col not in df.columns]
        if columnas_faltantes:
            raise ValueError(f"Faltan las siguientes columnas en el Excel: {', '.join(columnas_faltantes)}")
        
        # Obtener catálogos existentes
        tipo_docs = await db.execute(select(TipoDocumentoModel))
        tipo_docs_dict = {td.nombre: td.id for td in tipo_docs.scalars().all()}
        
        estados = await db.execute(select(EstadoMatriculaModel))
        estados_dict = {em.nombre: em.id for em in estados.scalars().all()}
        
        colegios = await db.execute(select(ColegioEgresadoModel))
        colegios_dict = {ce.nombre: ce.id for ce in colegios.scalars().all()}
        
        municipios = await db.execute(select(MunicipioNacimientoModel))
        municipios_dict = {mn.nombre: mn.id for mn in municipios.scalars().all()}

        estudiantes_creados = []
        
        # Procesar cada fila del Excel
        for index, row in df.iterrows():
            try:
                # Validar que los catálogos existan
                tipo_doc = tipo_docs_dict.get(row['Tipo Doc'])
                if not tipo_doc:
                    raise ValueError(f"Tipo de documento no encontrado: {row['Tipo Doc']}")
                
                estado = estados_dict.get(row['Estado Matricula'])
                if not estado:
                    raise ValueError(f"Estado de matrícula no encontrado: {row['Estado Matricula']}")
                
                colegio = colegios_dict.get(row['Colegio Egresado'])
                if not colegio:
                    raise ValueError(f"Colegio no encontrado: {row['Colegio Egresado']}")
                
                municipio = municipios_dict.get(row['Municipio Nacimiento'])
                if not municipio:
                    raise ValueError(f"Municipio no encontrado: {row['Municipio Nacimiento']}")

                # Crear estudiante
                estudiante = EstudianteModel(
                    codigo=str(row['Codigo Alumno']),
                    nombre=str(row['Nombre Alumno']),
                    tipo_documento_id=tipo_doc,
                    documento=str(row['Documento']),
                    semestre=str(row['Semestre']),
                    pensum=str(row['Pensum']),
                    ingreso=str(row['Ingreso']),
                    estado_matricula_id=estado,
                    celular=str(row['Celular']) if pd.notna(row['Celular']) else None,
                    email_personal=str(row['Email']) if pd.notna(row['Email']) else None,
                    email_institucional=str(row['Email Institucional']),
                    colegio_egresado_id=colegio,
                    municipio_nacimiento_id=municipio
                )
                db.add(estudiante)
                await db.flush()

                # Crear métrica de evaluación
                try:
                    promedio = float(row['Promedio'])
                    # Redondear a 2 decimales
                    promedio = round(promedio, 2)
                except (ValueError, TypeError):
                    raise ValueError(f"El promedio debe ser un número válido. Valor recibido: {row['Promedio']}")

                metrica = MetricaEvaluacionModel(
                    estudiante_id=estudiante.id,
                    promedio=promedio
                )
                db.add(metrica)

                # Crear relación usuario-estudiante
                usuario_estudiante = UsuarioEstudianteModel(
                    usuario_id=current_user.id,
                    estudiante_id=estudiante.id
                )
                db.add(usuario_estudiante)
                
                estudiantes_creados.append(estudiante)

            except Exception as row_error:
                raise ValueError(f"Error en la fila {index + 2}: {str(row_error)}")

        # Guardar todos los cambios
        await db.commit()

        return {"message": f"Se han cargado {len(estudiantes_creados)} estudiantes exitosamente"}

    except ValueError as ve:
        await db.rollback()
        raise HTTPException(
            status_code=400,
            detail=f"Error al procesar el archivo: {str(ve)}"
        )
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Error interno al procesar el archivo: {str(e)}"
        ) 