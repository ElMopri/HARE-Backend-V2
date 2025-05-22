from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Query, Response
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List
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
from enum import Enum

router = APIRouter(prefix="/estudiantes", tags=["estudiantes"])

class TipoDiagrama(str, Enum):
    BARRAS = "barras"
    TORTA = "torta"
    LINEAS = "lineas"

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

def calcular_nivel_riesgo(promedio: float) -> NivelRiesgo:
    if 0.0 <= promedio <= 1.0:
        return NivelRiesgo.ALTO
    elif 1.1 <= promedio <= 2.9:
        return NivelRiesgo.MEDIO
    else:
        return NivelRiesgo.BAJO

@router.get("/mis-estudiantes/", response_model=ListaEstudiantesResponse)
async def listar_estudiantes_usuario(
    db: AsyncSession = Depends(get_db),
    current_user: UsuarioModel = Depends(get_current_user),
    skip: int = 0,
    limit: int = 100
):
    # Consulta para obtener los estudiantes del usuario con sus métricas
    query = select(
        EstudianteModel.codigo,
        EstudianteModel.nombre,
        EstudianteModel.semestre,
        EstudianteModel.email_institucional,
        MetricaEvaluacionModel.promedio
    ).join(
        UsuarioEstudianteModel,
        EstudianteModel.id == UsuarioEstudianteModel.estudiante_id
    ).join(
        MetricaEvaluacionModel,
        EstudianteModel.id == MetricaEvaluacionModel.estudiante_id
    ).where(
        UsuarioEstudianteModel.usuario_id == current_user.id
    ).offset(skip).limit(limit)

    result = await db.execute(query)
    estudiantes_raw = result.all()

    # Contar el total de estudiantes sin el límite
    query_count = select(
        EstudianteModel
    ).join(
        UsuarioEstudianteModel,
        EstudianteModel.id == UsuarioEstudianteModel.estudiante_id
    ).where(
        UsuarioEstudianteModel.usuario_id == current_user.id
    )
    result_count = await db.execute(query_count)
    total = len(result_count.scalars().all())

    # Convertir los resultados al formato requerido
    estudiantes = [
        EstudianteConRiesgo(
            codigo=estudiante.codigo,
            nombre=estudiante.nombre,
            semestre=estudiante.semestre,
            email_institucional=estudiante.email_institucional,
            nivel_riesgo=calcular_nivel_riesgo(estudiante.promedio),
            promedio=round(estudiante.promedio, 2)
        )
        for estudiante in estudiantes_raw
    ]

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