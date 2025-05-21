from pydantic import BaseModel
from typing import Optional

# Schemas base para catálogos
class CatalogBase(BaseModel):
    nombre: str

class CatalogCreate(CatalogBase):
    pass

class CatalogUpdate(BaseModel):
    nombre: Optional[str] = None

class Catalog(CatalogBase):
    id: int

    class Config:
        from_attributes = True

# Tipos de documento
class TipoDocumentoCreate(CatalogCreate):
    pass

class TipoDocumentoUpdate(CatalogUpdate):
    pass

class TipoDocumento(Catalog):
    pass

# Estados de matrícula
class EstadoMatriculaCreate(CatalogCreate):
    pass

class EstadoMatriculaUpdate(CatalogUpdate):
    pass

class EstadoMatricula(Catalog):
    pass

# Colegios egresados
class ColegioEgresadoCreate(CatalogCreate):
    pass

class ColegioEgresadoUpdate(CatalogUpdate):
    pass

class ColegioEgresado(Catalog):
    pass

# Municipios de nacimiento
class MunicipioNacimientoCreate(CatalogCreate):
    pass

class MunicipioNacimientoUpdate(CatalogUpdate):
    pass

class MunicipioNacimiento(Catalog):
    pass 