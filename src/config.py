from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class TransformConfig:
    """ config for the transform pipeline"""
    name: str
    source_camera: str
    transforms: List[Dict[str, Any]]

    def __init__(self, name: str, source_camera: str, transforms: List[Dict[str, Any]]):
        self.name = name
        self.source_camera = source_camera
        if source_camera is None:
            raise ValueError("source_camera is required")
        if transforms is None:
            raise ValueError("transforms is required")
        self.transforms = transforms
        if not isinstance(transforms, list):
            raise ValueError("transforms must be a list of dicts")
    
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "TransformConfig":
        # val logic 
        return cls(**config)