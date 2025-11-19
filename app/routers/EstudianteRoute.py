from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Query, Response
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional
from app.config.database import get_db
from app.models.EstudianteModel import EstudianteModel
from app.schemas.estudiante import (
    EstudianteCreate, Estudiante, EstudianteUpdate, 
    EstudianteConRiesgo, ListaEstudiantesResponse, 
    NivelRiesgo, TipoEstadistica, EstadisticasResponse,
    EstadisticaPromedio, EstadisticaGeneral, EstadisticaItem
)
from sqlalchemy import select, update, delete, func, case
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
import matplotlib.pyplot as plt
import base64
import textwrap
import math
import os
import asyncio
try:
    from google import genai
except Exception:
    genai = None
import traceback
import logging
from enum import Enum
from matplotlib.backends.backend_pdf import PdfPages
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/estudiantes", tags=["estudiantes"])

class TipoDiagrama(str, Enum):
    BARRAS = "barras"
    TORTA = "torta"
    LINEAS = "lineas"


async def generar_feedback(labels, values, tipo_est, tipo_diag):
    """Generador independiente de retroalimentación.
    Puede usarse desde otros módulos; devuelve (texto, usado_externo: bool).
    Intenta llamar a Google Gemini si `GEMINI_API_KEY` está presente y el SDK `genai`
    está cargado; en caso de fallo devuelve ("No se pudo acceder a la IA", False).
    """
    total = sum(values) if values else 0

    # Mensaje corto si no hay datos
    if total == 0:
        return (f"Reporte: {tipo_est.value} — Gráfico: {tipo_diag.value}. No hay datos disponibles para este gráfico.", False)

    gemini_key = os.getenv("GEMINI_API_KEY")
    print(f"[GEN_FEED] generar_feedback | GEMINI_KEY present: {bool(gemini_key)} | genai loaded: {genai is not None}", flush=True)

    if gemini_key and genai is not None:
        try:
            print("[GEN_FEED] creando genai.Client con la clave proporcionada", flush=True)
            client = genai.Client(api_key=gemini_key)
            logger.info("Se creó genai.Client usando la clave proporcionada (no se muestra aquí por seguridad).")

            items_text = "\n".join([f"- {lab}: {val}" for lab, val in zip(labels, values)])
            prompt = f"""Eres un asistente que genera retroalimentación concisa y accionable en español para un gráfico estadístico.
Datos:
{items_text}

Tipo de estadística: {tipo_est.value}. Tipo de diagrama: {tipo_diag.value}.

Por favor, entrega en 3-5 frases: 1) resumen de los hallazgos principales, 2) observación sobre tendencias o distribución si aplica, 3) recomendación práctica breve.
"""

            print("[GEN_FEED] llamando a client.models.generate_content...", flush=True)
            resp = await asyncio.to_thread(
                lambda: client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
            )
            print("[GEN_FEED] client.models.generate_content devolvió respuesta", flush=True)

            # Extraer el texto de la respuesta
            text = None
            try:
                if hasattr(resp, 'text'):
                    text = resp.text
                elif isinstance(resp, dict):
                    text = resp.get('text') or str(resp)
            except Exception:
                text = str(resp)

            text = (text or "").strip()
            print(f"[GEN_FEED] texto extraído longitud: {len(text)}", flush=True)
            logger.info("Gemini respondió correctamente (longitud %d).", len(text))
            return (text, True)
        except Exception as e:
            try:
                traceback.print_exc()
            except Exception:
                pass
            try:
                logger.exception("Error llamando a Gemini con la clave proporcionada: %s", str(e))
            except Exception:
                print(f"[Gemini] Error al llamar la API con GEMINI_API_KEY: {e}", flush=True)
            print("[GENMI_ERROR] Exception al llamar a Gemini:", e, flush=True)

    logger.info("No se obtuvo respuesta válida de Gemini; devolviendo mensaje de error.")
    return ("No se pudo acceder a la IA", False)


def _describe_chart_header(tipo_est: TipoEstadistica, tipo_diag: TipoDiagrama) -> str:
    """Return a more descriptive header string explaining the chart content."""
    diag_name = tipo_diag.value.capitalize()
    if tipo_est == TipoEstadistica.PROMEDIO:
        return f"Distribución de promedios de los estudiantes — {diag_name}"
    if tipo_est == TipoEstadistica.COLEGIO:
        return f"Número de estudiantes por colegio — {diag_name}"
    if tipo_est == TipoEstadistica.MUNICIPIO:
        return f"Distribución de estudiantes por municipio de nacimiento — {diag_name}"
    if tipo_est == TipoEstadistica.SEMESTRE:
        return f"Cantidad de estudiantes por semestre académico — {diag_name}"
    if tipo_est == TipoEstadistica.NIVEL_RIESGO:
        return f"Distribución por nivel de riesgo académico — {diag_name}"
    # Fallback
    return f"{tipo_est.value} — {diag_name}"


class FeedbackRequest(BaseModel):
    labels: List[str]
    values: List[int]
    tipo_est: TipoEstadistica
    tipo_diag: TipoDiagrama


@router.post("/feedback")
async def generar_feedback_endpoint(
    payload: FeedbackRequest,
    current_user: UsuarioModel = Depends(get_current_user)
):
    """Endpoint que expone el generador de retroalimentación.
    Requiere autenticación y devuelve JSON con el texto de retroalimentación
    y si provino de la IA externa.
    """
    labels = payload.labels
    values = payload.values
    tipo_est = payload.tipo_est
    tipo_diag = payload.tipo_diag

    feedback_text, used_ai = await generar_feedback(labels, values, tipo_est, tipo_diag)
    return {"feedback": feedback_text, "used_ai": used_ai}

@router.post("/", response_model=Estudiante)
async def create_estudiante(
    estudiante: EstudianteCreate, 
    db: AsyncSession = Depends(get_db),
    current_user: UsuarioModel = Depends(get_current_user)
):
    # Verificar que el código no exista
    codigo_existente = await db.execute(
        select(EstudianteModel).where(EstudianteModel.codigo == estudiante.codigo)
    )
    if codigo_existente.scalar_one_or_none():
        raise HTTPException(
            status_code=400, 
            detail=f"Ya existe un estudiante con el código {estudiante.codigo}"
        )
    
    # Extraer el promedio del esquema
    promedio = estudiante.promedio
    estudiante_data = estudiante.dict()
    estudiante_data.pop('promedio')  # Remover promedio del diccionario
    
    # Crear el estudiante
    db_estudiante = EstudianteModel(**estudiante_data)
    db.add(db_estudiante)
    await db.commit()
    await db.refresh(db_estudiante)
    
    # Crear la relación usuario-estudiante
    relacion_usuario_estudiante = UsuarioEstudianteModel(
        usuario_id=current_user.id,
        estudiante_id=db_estudiante.id
    )
    db.add(relacion_usuario_estudiante)
    
    # Crear la métrica de evaluación con el promedio
    metrica_evaluacion = MetricaEvaluacionModel(
        estudiante_id=db_estudiante.id,
        promedio=round(promedio, 2)
    )
    db.add(metrica_evaluacion)
    
    await db.commit()
    
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

@router.get("/{codigo_estudiante}", response_model=Estudiante)
async def read_estudiante(
    codigo_estudiante: str, 
    db: AsyncSession = Depends(get_db),
    current_user: UsuarioModel = Depends(get_current_user)
):
    # Verificar que el estudiante existe y pertenece al usuario actual
    query = select(EstudianteModel).join(
        UsuarioEstudianteModel,
        EstudianteModel.id == UsuarioEstudianteModel.estudiante_id
    ).where(
        EstudianteModel.codigo == codigo_estudiante,
        UsuarioEstudianteModel.usuario_id == current_user.id
    )
    result = await db.execute(query)
    estudiante = result.scalar_one_or_none()
    
    if estudiante is None:
        raise HTTPException(status_code=404, detail="Estudiante no encontrado o no tienes permisos para verlo")
    
    return estudiante

@router.put("/{codigo_estudiante}", response_model=Estudiante)
async def update_estudiante(
    codigo_estudiante: str, 
    estudiante: EstudianteUpdate, 
    db: AsyncSession = Depends(get_db),
    current_user: UsuarioModel = Depends(get_current_user)
):
    # Verificar que el estudiante existe y pertenece al usuario actual
    query = select(EstudianteModel).join(
        UsuarioEstudianteModel,
        EstudianteModel.id == UsuarioEstudianteModel.estudiante_id
    ).where(
        EstudianteModel.codigo == codigo_estudiante,
        UsuarioEstudianteModel.usuario_id == current_user.id
    )
    result = await db.execute(query)
    db_estudiante = result.scalar_one_or_none()
    
    if db_estudiante is None:
        raise HTTPException(status_code=404, detail="Estudiante no encontrado o no tienes permisos para actualizarlo")

    # Extraer el promedio si está presente
    promedio = estudiante.promedio
    update_data = estudiante.dict(exclude_unset=True)
    update_data.pop('promedio', None)  # Remover promedio del diccionario de actualización
    
    # Verificar si se está cambiando el código y si ya existe
    if 'codigo' in update_data and update_data['codigo'] != codigo_estudiante:
        codigo_existente = await db.execute(
            select(EstudianteModel).where(
                EstudianteModel.codigo == update_data['codigo'],
                EstudianteModel.id != db_estudiante.id
            )
        )
        if codigo_existente.scalar_one_or_none():
            raise HTTPException(
                status_code=400, 
                detail=f"Ya existe un estudiante con el código {update_data['codigo']}"
            )
    
    # Actualizar los campos del estudiante
    for key, value in update_data.items():
        setattr(db_estudiante, key, value)

    # Actualizar o crear métrica de evaluación si se proporciona promedio
    if promedio is not None:
        # Buscar métrica existente
        metrica_query = select(MetricaEvaluacionModel).where(
            MetricaEvaluacionModel.estudiante_id == db_estudiante.id
        )
        metrica_result = await db.execute(metrica_query)
        metrica = metrica_result.scalar_one_or_none()
        
        if metrica:
            # Actualizar métrica existente
            metrica.promedio = round(promedio, 2)
        else:
            # Crear nueva métrica
            metrica = MetricaEvaluacionModel(
                estudiante_id=db_estudiante.id,
                promedio=round(promedio, 2)
            )
            db.add(metrica)

    await db.commit()
    await db.refresh(db_estudiante)
    return db_estudiante

@router.delete("/{codigo_estudiante}")
async def delete_estudiante(
    codigo_estudiante: str, 
    db: AsyncSession = Depends(get_db),
    current_user: UsuarioModel = Depends(get_current_user)
):
    # Primero verificar que el estudiante existe y pertenece al usuario actual
    query = select(EstudianteModel).join(
        UsuarioEstudianteModel,
        EstudianteModel.id == UsuarioEstudianteModel.estudiante_id
    ).where(
        EstudianteModel.codigo == codigo_estudiante,
        UsuarioEstudianteModel.usuario_id == current_user.id
    )
    result = await db.execute(query)
    estudiante = result.scalar_one_or_none()
    if estudiante is None:
        raise HTTPException(status_code=404, detail="Estudiante no encontrado o no tienes permisos para eliminarlo")
    # Obtener el ID del estudiante para las eliminaciones
    estudiante_id = estudiante.id
    
    # Eliminar las métricas de evaluación primero
    delete_metricas = delete(MetricaEvaluacionModel).where(
        MetricaEvaluacionModel.estudiante_id == estudiante_id
    )
    await db.execute(delete_metricas)
    
    # Eliminar la relación usuario-estudiante
    delete_relacion = delete(UsuarioEstudianteModel).where(
        UsuarioEstudianteModel.estudiante_id == estudiante_id,
        UsuarioEstudianteModel.usuario_id == current_user.id
    )
    await db.execute(delete_relacion)
    
    # Finalmente eliminar el estudiante
    delete_estudiante_query = delete(EstudianteModel).where(EstudianteModel.id == estudiante_id)
    await db.execute(delete_estudiante_query)
    await db.commit()
    
    return {"message": f"Estudiante con código {codigo_estudiante} eliminado exitosamente"}

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
        estudiantes_actualizados = []
        
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

                # Verificar si el estudiante ya existe
                estudiante_existente = await db.execute(
                    select(EstudianteModel).where(
                        (EstudianteModel.codigo == str(row['Codigo Alumno'])) &
                        (EstudianteModel.documento == str(row['Documento']))
                    )
                )
                estudiante = estudiante_existente.scalar_one_or_none()

                if estudiante:
                    # Actualizar estudiante existente
                    estudiante.nombre = str(row['Nombre Alumno'])
                    estudiante.tipo_documento_id = tipo_doc
                    estudiante.semestre = str(row['Semestre'])
                    estudiante.pensum = str(row['Pensum'])
                    estudiante.ingreso = str(row['Ingreso'])
                    estudiante.estado_matricula_id = estado
                    estudiante.celular = str(row['Celular']) if pd.notna(row['Celular']) else None
                    estudiante.email_personal = str(row['Email']) if pd.notna(row['Email']) else None
                    estudiante.email_institucional = str(row['Email Institucional'])
                    estudiante.colegio_egresado_id = colegio
                    estudiante.municipio_nacimiento_id = municipio
                    estudiantes_actualizados.append(estudiante)
                else:
                    # Crear nuevo estudiante
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
                    estudiantes_creados.append(estudiante)

                await db.flush()

                # Actualizar o crear métrica de evaluación
                try:
                    promedio = float(row['Promedio'])
                    promedio = round(promedio, 2)
                except (ValueError, TypeError):
                    raise ValueError(f"El promedio debe ser un número válido. Valor recibido: {row['Promedio']}")

                # Buscar métrica existente
                metrica_existente = await db.execute(
                    select(MetricaEvaluacionModel).where(
                        MetricaEvaluacionModel.estudiante_id == estudiante.id
                    )
                )
                metrica = metrica_existente.scalar_one_or_none()

                if metrica:
                    metrica.promedio = promedio
                else:
                    metrica = MetricaEvaluacionModel(
                        estudiante_id=estudiante.id,
                        promedio=promedio
                    )
                    db.add(metrica)

                # Verificar y crear relación usuario-estudiante si no existe
                relacion_existente = await db.execute(
                    select(UsuarioEstudianteModel).where(
                        (UsuarioEstudianteModel.usuario_id == current_user.id) &
                        (UsuarioEstudianteModel.estudiante_id == estudiante.id)
                    )
                )
                if not relacion_existente.scalar_one_or_none():
                    usuario_estudiante = UsuarioEstudianteModel(
                        usuario_id=current_user.id,
                        estudiante_id=estudiante.id
                    )
                    db.add(usuario_estudiante)

            except Exception as row_error:
                raise ValueError(f"Error en la fila {index + 2}: {str(row_error)}")

        # Guardar todos los cambios
        await db.commit()

        return {
            "message": f"Proceso completado exitosamente",
            "estudiantes_creados": len(estudiantes_creados),
            "estudiantes_actualizados": len(estudiantes_actualizados)
        }

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

def calcular_nivel_riesgo(promedio: float) -> NivelRiesgo:
    if 0.0 <= promedio <= 1.5:
        return NivelRiesgo.ALTO
    elif 1.6 <= promedio <= 2.9:
        return NivelRiesgo.MEDIO
    else:
        return NivelRiesgo.BAJO

@router.get("/mis-estudiantes/", response_model=ListaEstudiantesResponse)
async def listar_estudiantes_usuario(
    db: AsyncSession = Depends(get_db),
    current_user: UsuarioModel = Depends(get_current_user),
    skip: int = 0,
    limit: int = 100,
    nombre: Optional[str] = None,
    semestre: Optional[str] = None,
    ingreso: Optional[str] = None,
    colegio_egresado_id: Optional[int] = None,
    municipio_nacimiento_id: Optional[int] = None,
    promedio_min: Optional[float] = None,
    promedio_max: Optional[float] = None,
    nivel_riesgo: Optional[NivelRiesgo] = None
):
    # Construir la consulta base
    query = select(
        EstudianteModel.codigo,
        EstudianteModel.nombre,
        EstudianteModel.semestre,
        EstudianteModel.email_institucional,
        MetricaEvaluacionModel.promedio
    ).join(
        UsuarioEstudianteModel,
        EstudianteModel.id == UsuarioEstudianteModel.estudiante_id
    ).outerjoin(
        MetricaEvaluacionModel,
        EstudianteModel.id == MetricaEvaluacionModel.estudiante_id
    ).where(
        UsuarioEstudianteModel.usuario_id == current_user.id
    )

    # Aplicar filtros
    if nombre:
        query = query.where(EstudianteModel.nombre.ilike(f"%{nombre}%"))
    if semestre:
        query = query.where(EstudianteModel.semestre == semestre)
    if ingreso:
        query = query.where(EstudianteModel.ingreso == ingreso)
    if colegio_egresado_id:
        query = query.where(EstudianteModel.colegio_egresado_id == colegio_egresado_id)
    if municipio_nacimiento_id:
        query = query.where(EstudianteModel.municipio_nacimiento_id == municipio_nacimiento_id)
    if promedio_min is not None:
        query = query.where(MetricaEvaluacionModel.promedio >= promedio_min)
    if promedio_max is not None:
        query = query.where(MetricaEvaluacionModel.promedio <= promedio_max)
    if nivel_riesgo:
        if nivel_riesgo == NivelRiesgo.ALTO:
            query = query.where(MetricaEvaluacionModel.promedio <= 1.0)
        elif nivel_riesgo == NivelRiesgo.MEDIO:
            query = query.where(
                (MetricaEvaluacionModel.promedio > 1.0) & 
                (MetricaEvaluacionModel.promedio <= 2.9)
            )
        elif nivel_riesgo == NivelRiesgo.BAJO:
            query = query.where(MetricaEvaluacionModel.promedio > 2.9)

    # Aplicar paginación
    query = query.offset(skip).limit(limit)

    result = await db.execute(query)
    estudiantes_raw = result.all()

    # Contar el total de estudiantes sin el límite
    query_count = select(
        EstudianteModel
    ).join(
        UsuarioEstudianteModel,
        EstudianteModel.id == UsuarioEstudianteModel.estudiante_id
    ).outerjoin(
        MetricaEvaluacionModel,
        EstudianteModel.id == MetricaEvaluacionModel.estudiante_id
    ).where(
        UsuarioEstudianteModel.usuario_id == current_user.id
    )

    # Aplicar los mismos filtros a la consulta de conteo
    if nombre:
        query_count = query_count.where(EstudianteModel.nombre.ilike(f"%{nombre}%"))
    if semestre:
        query_count = query_count.where(EstudianteModel.semestre == semestre)
    if ingreso:
        query_count = query_count.where(EstudianteModel.ingreso == ingreso)
    if colegio_egresado_id:
        query_count = query_count.where(EstudianteModel.colegio_egresado_id == colegio_egresado_id)
    if municipio_nacimiento_id:
        query_count = query_count.where(EstudianteModel.municipio_nacimiento_id == municipio_nacimiento_id)
    if promedio_min is not None:
        query_count = query_count.where(MetricaEvaluacionModel.promedio >= promedio_min)
    if promedio_max is not None:
        query_count = query_count.where(MetricaEvaluacionModel.promedio <= promedio_max)
    if nivel_riesgo:
        if nivel_riesgo == NivelRiesgo.ALTO:
            query_count = query_count.where(MetricaEvaluacionModel.promedio <= 1.0)
        elif nivel_riesgo == NivelRiesgo.MEDIO:
            query_count = query_count.where(
                (MetricaEvaluacionModel.promedio > 1.0) & 
                (MetricaEvaluacionModel.promedio <= 2.9)
            )
        elif nivel_riesgo == NivelRiesgo.BAJO:
            query_count = query_count.where(MetricaEvaluacionModel.promedio > 2.9)

    result_count = await db.execute(query_count)
    total = len(result_count.scalars().all())

    # Convertir los resultados al formato requerido
    estudiantes = []
    for estudiante in estudiantes_raw:
        promedio = estudiante.promedio if estudiante.promedio is not None else 0.0
        estudiantes.append(
            EstudianteConRiesgo(
                codigo=estudiante.codigo,
                nombre=estudiante.nombre,
                semestre=estudiante.semestre,
                email_institucional=estudiante.email_institucional,
                nivel_riesgo=calcular_nivel_riesgo(promedio),
                promedio=round(promedio, 2)
            )
        )

    return ListaEstudiantesResponse(
        estudiantes=estudiantes,
        total=total
    )

@router.get("/estadisticas/", response_model=EstadisticasResponse)
async def obtener_estadisticas(
    tipo: TipoEstadistica,
    db: AsyncSession = Depends(get_db),
    current_user: UsuarioModel = Depends(get_current_user)
):
    # Base query para obtener estudiantes del usuario actual
    base_query = select(EstudianteModel).join(
        UsuarioEstudianteModel,
        EstudianteModel.id == UsuarioEstudianteModel.estudiante_id
    ).where(
        UsuarioEstudianteModel.usuario_id == current_user.id
    )

    if tipo == TipoEstadistica.PROMEDIO:
        # Obtener estadísticas de promedio
        query = select(
            func.avg(MetricaEvaluacionModel.promedio).label('promedio_general'),
            func.count().label('total'),
            func.sum(
                case(
                    (MetricaEvaluacionModel.promedio <= 1.0, 1),
                    else_=0
                )
            ).label('rango_0_1'),
            func.sum(
                case(
                    ((MetricaEvaluacionModel.promedio > 1.0) & (MetricaEvaluacionModel.promedio <= 2.0), 1),
                    else_=0
                )
            ).label('rango_1_2'),
            func.sum(
                case(
                    ((MetricaEvaluacionModel.promedio > 2.0) & (MetricaEvaluacionModel.promedio <= 3.0), 1),
                    else_=0
                )
            ).label('rango_2_3'),
            func.sum(
                case(
                    ((MetricaEvaluacionModel.promedio > 3.0) & (MetricaEvaluacionModel.promedio <= 4.0), 1),
                    else_=0
                )
            ).label('rango_3_4'),
            func.sum(
                case(
                    (MetricaEvaluacionModel.promedio > 4.0, 1),
                    else_=0
                )
            ).label('rango_4_5')
        ).select_from(
            EstudianteModel
        ).join(
            UsuarioEstudianteModel,
            EstudianteModel.id == UsuarioEstudianteModel.estudiante_id
        ).join(
            MetricaEvaluacionModel,
            EstudianteModel.id == MetricaEvaluacionModel.estudiante_id
        ).where(
            UsuarioEstudianteModel.usuario_id == current_user.id
        )

        result = await db.execute(query)
        stats = result.first()
        
        # Calcular distribución de niveles de riesgo
        niveles = [
            {"rango": (0.0, 1.0), "nivel": NivelRiesgo.ALTO},
            {"rango": (1.1, 2.9), "nivel": NivelRiesgo.MEDIO},
            {"rango": (3.0, 5.0), "nivel": NivelRiesgo.BAJO}
        ]
        
        distribucion_niveles = []
        total = stats.total if stats.total > 0 else 1

        for nivel in niveles:
            query_nivel = select(func.count()).select_from(
                EstudianteModel
            ).join(
                UsuarioEstudianteModel,
                EstudianteModel.id == UsuarioEstudianteModel.estudiante_id
            ).join(
                MetricaEvaluacionModel,
                EstudianteModel.id == MetricaEvaluacionModel.estudiante_id
            ).where(
                UsuarioEstudianteModel.usuario_id == current_user.id,
                MetricaEvaluacionModel.promedio >= nivel["rango"][0],
                MetricaEvaluacionModel.promedio <= nivel["rango"][1]
            )
            
            result_nivel = await db.execute(query_nivel)
            cantidad = result_nivel.scalar()
            
            distribucion_niveles.append(
                EstadisticaItem(
                    etiqueta=nivel["nivel"].value,
                    cantidad=cantidad,
                    porcentaje=round((cantidad / total) * 100, 2)
                )
            )

        return EstadisticasResponse(
            tipo=tipo,
            datos=EstadisticaPromedio(
                promedio_general=round(stats.promedio_general, 2) if stats.promedio_general else 0.0,
                distribucion_niveles=distribucion_niveles,
                rango_promedios={
                    "0-1": stats.rango_0_1,
                    "1-2": stats.rango_1_2,
                    "2-3": stats.rango_2_3,
                    "3-4": stats.rango_3_4,
                    "4-5": stats.rango_4_5
                }
            )
        )

    else:
        # Para otros tipos de estadísticas (colegio, municipio, semestre)
        group_by_column = None
        join_model = None
        
        if tipo == TipoEstadistica.COLEGIO:
            group_by_column = ColegioEgresadoModel.nombre
            join_model = ColegioEgresadoModel
            join_condition = (EstudianteModel.colegio_egresado_id == ColegioEgresadoModel.id)
        elif tipo == TipoEstadistica.MUNICIPIO:
            group_by_column = MunicipioNacimientoModel.nombre
            join_model = MunicipioNacimientoModel
            join_condition = (EstudianteModel.municipio_nacimiento_id == MunicipioNacimientoModel.id)
        elif tipo == TipoEstadistica.SEMESTRE:
            group_by_column = EstudianteModel.semestre
        elif tipo == TipoEstadistica.NIVEL_RIESGO:
            # Estadísticas por nivel de riesgo
            query = select(
                case(
                    (MetricaEvaluacionModel.promedio <= 1.0, "ALTO"),
                    (MetricaEvaluacionModel.promedio <= 2.9, "MEDIO"),
                    else_="BAJO"
                ).label('nivel'),
                func.count().label('cantidad')
            ).select_from(
                EstudianteModel
            ).join(
                UsuarioEstudianteModel,
                EstudianteModel.id == UsuarioEstudianteModel.estudiante_id
            ).join(
                MetricaEvaluacionModel,
                EstudianteModel.id == MetricaEvaluacionModel.estudiante_id
            ).where(
                UsuarioEstudianteModel.usuario_id == current_user.id
            ).group_by('nivel')
            
            result = await db.execute(query)
            stats_raw = result.all()
            
            total = sum(item.cantidad for item in stats_raw)
            items = [
                EstadisticaItem(
                    etiqueta=item.nivel,
                    cantidad=item.cantidad,
                    porcentaje=round((item.cantidad / total) * 100, 2)
                )
                for item in stats_raw
            ]
            
            return EstadisticasResponse(
                tipo=tipo,
                datos=EstadisticaGeneral(
                    total_estudiantes=total,
                    items=items
                )
            )

        # Consulta para otros tipos de estadísticas
        if join_model:
            query = select(
                group_by_column.label('grupo'),
                func.count().label('cantidad')
            ).select_from(
                EstudianteModel
            ).join(
                UsuarioEstudianteModel,
                EstudianteModel.id == UsuarioEstudianteModel.estudiante_id
            ).join(
                join_model,
                join_condition
            ).where(
                UsuarioEstudianteModel.usuario_id == current_user.id
            ).group_by(group_by_column)
        else:
            query = select(
                group_by_column.label('grupo'),
                func.count().label('cantidad')
            ).select_from(
                EstudianteModel
            ).join(
                UsuarioEstudianteModel,
                EstudianteModel.id == UsuarioEstudianteModel.estudiante_id
            ).where(
                UsuarioEstudianteModel.usuario_id == current_user.id
            ).group_by(group_by_column)

        result = await db.execute(query)
        stats_raw = result.all()
        
        total = sum(item.cantidad for item in stats_raw)
        items = [
            EstadisticaItem(
                etiqueta=str(item.grupo),
                cantidad=item.cantidad,
                porcentaje=round((item.cantidad / total) * 100, 2)
            )
            for item in stats_raw
        ]
        
        return EstadisticasResponse(
            tipo=tipo,
            datos=EstadisticaGeneral(
                total_estudiantes=total,
                items=items
            )
        )

@router.get("/diagramas/")
async def generar_diagrama(
    tipo_estadistica: TipoEstadistica,
    tipo_diagrama: TipoDiagrama,
    db: AsyncSession = Depends(get_db),
    current_user: UsuarioModel = Depends(get_current_user)
):
    # Obtener los datos de las estadísticas
    estadisticas = await obtener_estadisticas(tipo_estadistica, db, current_user)
    
    # Crear figura de matplotlib
    plt.figure(figsize=(10, 6))
    plt.clf()  # Limpiar la figura actual
    
    if tipo_estadistica == TipoEstadistica.PROMEDIO:
        datos = estadisticas.datos.rango_promedios
        labels = list(datos.keys())
        values = list(datos.values())
    else:
        items = estadisticas.datos.items
        labels = [item.etiqueta for item in items]
        values = [item.cantidad for item in items]
    
    if tipo_diagrama == TipoDiagrama.BARRAS:
        plt.bar(labels, values)
        plt.xticks(rotation=45)
        plt.ylabel('Cantidad')
    elif tipo_diagrama == TipoDiagrama.TORTA:
        plt.pie(values, labels=labels, autopct='%1.1f%%')
    elif tipo_diagrama == TipoDiagrama.LINEAS:
        plt.plot(labels, values, marker='o')
        plt.xticks(rotation=45)
        plt.ylabel('Cantidad')
    
    plt.title(f'Estadísticas por {tipo_estadistica}')
    
    # Guardar el gráfico en un buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    
    # Codificar la imagen en base64
    imagen_base64 = base64.b64encode(buf.getvalue()).decode()
    
    return {
        "tipo_estadistica": tipo_estadistica,
        "tipo_diagrama": tipo_diagrama,
        "imagen_base64": imagen_base64
    }


@router.get("/report/pdf")
async def exportar_reporte_pdf(
    db: AsyncSession = Depends(get_db),
    current_user: UsuarioModel = Depends(get_current_user)
):
    """Genera un PDF que contiene todos los tipos de gráficos (barras, torta, líneas)
    para cada tipo de estadística disponible. Cada gráfico incluye un título descriptivo.
    """
    # Buffer en memoria para el PDF
    pdf_buffer = io.BytesIO()

    # Parámetros de página: tamaño carta en pulgadas
    LETTER_WIDTH_IN = 8.5
    LETTER_HEIGHT_IN = 11.0
    # Márgenes en mm -> pulgadas (usar margen más amplio para impresión)
    # 12.7 mm = 0.5 in
    MARGIN_MM = 12.7
    margin_in = MARGIN_MM / 25.4

    # Usamos la función de nivel de módulo `generar_feedback` definida arriba.
    # Esto evita duplicar la lógica y permite usar el generador desde otros módulos.

    # Crear el PDF y agregar una página por cada gráfico
    with PdfPages(pdf_buffer) as pdf:
        # Página de portada: título y descripción
        fig_cover = plt.figure(figsize=(LETTER_WIDTH_IN, LETTER_HEIGHT_IN))
        ax_cover = fig_cover.add_axes([0, 0, 1, 1])
        ax_cover.axis('off')
        report_title = "Reporte Estadístico - Estudiantes"
        report_description = (
            "Este es el reporte generado automáticamente que contiene gráficos y retroalimentación por cada estadística.\n"
            "Puede incluir hallazgos clave y recomendaciones."
        )
        ax_cover.text(0.5, 0.65, report_title, ha='center', va='center', fontsize=22, weight='bold')
        ax_cover.text(0.5, 0.45, report_description, ha='center', va='center', fontsize=11)
        pdf.savefig(fig_cover)
        plt.close(fig_cover)

        for tipo_est in TipoEstadistica:
            estadisticas = await obtener_estadisticas(tipo_est, db, current_user)

            for tipo_diag in TipoDiagrama:
                # Preparar labels/values
                if tipo_est == TipoEstadistica.PROMEDIO:
                    datos = estadisticas.datos.rango_promedios
                    labels = list(datos.keys())
                    values = list(datos.values())
                else:
                    items = estadisticas.datos.items
                    labels = [item.etiqueta for item in items]
                    values = [item.cantidad for item in items]

                # Crear figura tamaño carta
                fig = plt.figure(figsize=(LETTER_WIDTH_IN, LETTER_HEIGHT_IN))
                plt.clf()

                # Definir áreas aprovechables dentro de márgenes
                left = margin_in / LETTER_WIDTH_IN
                right = 1 - left
                bottom = margin_in / LETTER_HEIGHT_IN
                top = 1 - bottom

                # Calcular áreas dentro de los márgenes y aplicar padding entre elementos
                available_h = top - bottom
                title_h = 0.08 * available_h          # espacio para el encabezado grande
                gap_title_graph = 0.03 * available_h  # separación entre título y gráfico
                graph_h = 0.58 * available_h          # espacio para el gráfico
                gap_graph_text = 0.03 * available_h   # separación entre gráfico y texto
                text_h = available_h - (title_h + gap_title_graph + graph_h + gap_graph_text)

                # Axes para título (encabezado formato descriptivo)
                header_title = _describe_chart_header(tipo_est, tipo_diag)
                ax_title = fig.add_axes([left, top - title_h, right - left, title_h])
                ax_title.axis('off')
                ax_title.text(0.02, 0.5, header_title, ha='left', va='center', fontsize=14, weight='bold')

                # Axes para la imagen centrada (respetando padding)
                img_bottom = bottom + text_h + gap_graph_text
                img_height = graph_h
                ax_img = fig.add_axes([left, img_bottom, right - left, img_height])
                ax_img.axis('off')

                # Eje interno para plotting con márgenes interiores
                inner_left = left + 0.05 * (right - left)
                inner_width = 0.90 * (right - left)
                inner_bottom = img_bottom + 0.06 * img_height
                inner_height = 0.88 * img_height

                ax_plot = fig.add_axes([inner_left, inner_bottom, inner_width, inner_height])

                if sum(values) == 0:
                    ax_plot.text(0.5, 0.5, 'No hay datos disponibles', horizontalalignment='center', verticalalignment='center')
                    ax_plot.axis('off')
                else:
                    if tipo_diag == TipoDiagrama.BARRAS:
                        ax_plot.bar(labels, values)
                        for label in ax_plot.get_xticklabels():
                            label.set_rotation(45)
                        ax_plot.set_ylabel('Cantidad')
                    elif tipo_diag == TipoDiagrama.TORTA:
                        # Si demasiadas categorías, limitar etiquetas visibles
                        if len(values) > 12:
                            # agrupar pequeñas en 'Otros'
                            combined = sorted(zip(labels, values), key=lambda x: x[1], reverse=True)
                            top = combined[:11]
                            others = combined[11:]
                            labels_plot = [t[0] for t in top] + (['Otros'] if others else [])
                            values_plot = [t[1] for t in top] + ([sum([o[1] for o in others])] if others else [])
                            ax_plot.pie(values_plot, labels=labels_plot, autopct='%1.1f%%')
                        else:
                            ax_plot.pie(values, labels=labels, autopct='%1.1f%%')
                    elif tipo_diag == TipoDiagrama.LINEAS:
                        # Para lineas, usar índices como x si labels no numéricos
                        x = list(range(len(values)))
                        ax_plot.plot(x, values, marker='o')
                        ax_plot.set_xticks(x)
                        ax_plot.set_xticklabels(labels, rotation=45)
                        ax_plot.set_ylabel('Cantidad')

                # Axes para la retroalimentación (texto)
                ax_text = fig.add_axes([left, bottom, right - left, text_h])
                ax_text.axis('off')
                feedback_text, used_ai = await generar_feedback(labels, values, tipo_est, tipo_diag)
                if used_ai:
                    feedback_text = feedback_text + "\n\nGenerado por Google AI Studio"

                # Encabezado pequeño para la sección de retroalimentación
                header_text = "Retroalimentación (IA Gemini)"
                ax_text.text(0.02, 0.96, header_text, ha='left', va='top', fontsize=10, weight='bold')

                # Wrap text a un ancho razonable (más estrecho para evitar cortes en página)
                wrapped = textwrap.fill(feedback_text, width=90)
                # Escribir el cuerpo de la retroalimentación debajo del encabezado
                ax_text.text(0.02, 0.78, wrapped, ha='left', va='top', fontsize=9)

                # Guardar la página
                pdf.savefig(fig)
                plt.close(fig)

    pdf_buffer.seek(0)

    headers = {"Content-Disposition": "attachment; filename=report_estadisticas.pdf"}
    return StreamingResponse(pdf_buffer, media_type='application/pdf', headers=headers)

@router.get("/catalogos/colegios/", response_model=List[dict])
async def obtener_colegios_indexados(
    db: AsyncSession = Depends(get_db),
    current_user: UsuarioModel = Depends(get_current_user)
):
    query = select(
        ColegioEgresadoModel.id,
        ColegioEgresadoModel.nombre,
        func.count(EstudianteModel.id).label('total_estudiantes')
    ).join(
        EstudianteModel,
        ColegioEgresadoModel.id == EstudianteModel.colegio_egresado_id
    ).join(
        UsuarioEstudianteModel,
        EstudianteModel.id == UsuarioEstudianteModel.estudiante_id
    ).where(
        UsuarioEstudianteModel.usuario_id == current_user.id
    ).group_by(
        ColegioEgresadoModel.id,
        ColegioEgresadoModel.nombre
    ).order_by(
        ColegioEgresadoModel.nombre
    )

    result = await db.execute(query)
    colegios = result.all()
    
    return [
        {
            "id": colegio.id,
            "nombre": colegio.nombre,
            "total_estudiantes": colegio.total_estudiantes
        }
        for colegio in colegios
    ]

@router.get("/catalogos/municipios/", response_model=List[dict])
async def obtener_municipios_indexados(
    db: AsyncSession = Depends(get_db),
    current_user: UsuarioModel = Depends(get_current_user)
):
    query = select(
        MunicipioNacimientoModel.id,
        MunicipioNacimientoModel.nombre,
        func.count(EstudianteModel.id).label('total_estudiantes')
    ).join(
        EstudianteModel,
        MunicipioNacimientoModel.id == EstudianteModel.municipio_nacimiento_id
    ).join(
        UsuarioEstudianteModel,
        EstudianteModel.id == UsuarioEstudianteModel.estudiante_id
    ).where(
        UsuarioEstudianteModel.usuario_id == current_user.id
    ).group_by(
        MunicipioNacimientoModel.id,
        MunicipioNacimientoModel.nombre
    ).order_by(
        MunicipioNacimientoModel.nombre
    )

    result = await db.execute(query)
    municipios = result.all()
    
    return [
        {
            "id": municipio.id,
            "nombre": municipio.nombre,
            "total_estudiantes": municipio.total_estudiantes
        }
        for municipio in municipios
    ]

@router.get("/catalogos/semestres/", response_model=List[dict])
async def obtener_semestres_indexados(
    db: AsyncSession = Depends(get_db),
    current_user: UsuarioModel = Depends(get_current_user)
):
    query = select(
        EstudianteModel.semestre,
        func.count(EstudianteModel.id).label('total_estudiantes')
    ).join(
        UsuarioEstudianteModel,
        EstudianteModel.id == UsuarioEstudianteModel.estudiante_id
    ).where(
        UsuarioEstudianteModel.usuario_id == current_user.id
    ).group_by(
        EstudianteModel.semestre
    ).order_by(
        EstudianteModel.semestre
    )

    result = await db.execute(query)
    semestres = result.all()
    
    return [
        {
            "semestre": semestre.semestre,
            "total_estudiantes": semestre.total_estudiantes
        }
        for semestre in semestres
    ]

@router.get("/catalogos/ingresos/", response_model=List[dict])
async def obtener_ingresos_indexados(
    db: AsyncSession = Depends(get_db),
    current_user: UsuarioModel = Depends(get_current_user)
):
    query = select(
        EstudianteModel.ingreso,
        func.count(EstudianteModel.id).label('total_estudiantes')
    ).join(
        UsuarioEstudianteModel,
        EstudianteModel.id == UsuarioEstudianteModel.estudiante_id
    ).where(
        UsuarioEstudianteModel.usuario_id == current_user.id
    ).group_by(
        EstudianteModel.ingreso
    ).order_by(
        EstudianteModel.ingreso
    )

    result = await db.execute(query)
    ingresos = result.all()
    
    return [
        {
            "ingreso": ingreso.ingreso,
            "total_estudiantes": ingreso.total_estudiantes
        }
        for ingreso in ingresos
    ] 