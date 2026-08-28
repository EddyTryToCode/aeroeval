"""
Model Management Routes for AeroEval API.
"""

from fastapi import APIRouter, Depends, HTTPException, status

from aeroeval.api.dependencies import get_model_registry
from aeroeval.api.schemas import ModelInfoResponse, ModelListResponse, ModelRegisterRequest
from aeroeval.models.registry import ModelRegistry

router = APIRouter(prefix="/models", tags=["Models"])


@router.post("/register", response_model=ModelInfoResponse, status_code=status.HTTP_201_CREATED)
def register_model(
    req: ModelRegisterRequest,
    registry: ModelRegistry = Depends(get_model_registry)
):
    """Registers a new model into the AeroEval Model Registry."""
    try:
        info = registry.register(
            name=req.name,
            path=req.path,
            format=req.format,
            imgsz=req.imgsz,
            description=req.description or "",
            metadata=req.metadata
        )
        return ModelInfoResponse(**info.to_dict())
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to register model: {str(e)}")


@router.get("", response_model=ModelListResponse)
def list_models(registry: ModelRegistry = Depends(get_model_registry)):
    """Lists all registered models in the registry."""
    all_models = registry.list()
    return ModelListResponse(
        total_models=len(all_models),
        models=[ModelInfoResponse(**m.to_dict()) for m in all_models]
    )


@router.get("/{name}", response_model=ModelInfoResponse)
def get_model_by_name(name: str, registry: ModelRegistry = Depends(get_model_registry)):
    """Retrieves metadata of a specific model by name."""
    model = registry.get(name)
    if not model:
        raise HTTPException(status_code=404, detail=f"Model '{name}' not found in registry.")
    return ModelInfoResponse(**model.to_dict())


@router.delete("/{name}", status_code=status.HTTP_200_OK)
def delete_model(name: str, registry: ModelRegistry = Depends(get_model_registry)):
    """Removes a model from the registry."""
    deleted = registry.remove(name)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Model '{name}' not found.")
    return {"message": f"Model '{name}' removed successfully."}
