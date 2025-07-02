from .utils.gpu_utils import get_device, viam_to_tensor, tensor_to_viam
import torch
from viam.media.video import ViamImage
from typing import List, Dict, Any, Callable
import torchvision.transforms as T

TRANSFORM_REGISTRY: Dict[str, Callable] = {
    "resize": {  
               "transform": T.Resize,
               "parser": lambda attrs: {
                   "size": (attrs["width_px"], attrs["height_px"])
               }
    },
    "to_tensor": {
        "transform": T.ToTensor,
        "parser": lambda attrs: {}
    },
     "normalize": {
         "transform": T.Normalize,  
         "parser": lambda attrs: {
             "mean": attrs["mean"],
             "std": attrs["std"]
         }
     },
     "grayscale": {
         "transform": T.Grayscale,  
         "parser": lambda attrs: {}
     },
     "rotate": {
         "transform": T.RandomRotation,
         "parser": lambda attrs: {
             "degrees": attrs["angle_degs"]
         }
     },
     "crop": {
         "transform": T.RandomCrop,
         "parser": lambda attrs: {
            "size": (attrs["width"], attrs["height"])
         }
     }
}

class GPUTransformPipeline:
    def __init__(self, transform_config: List[Dict[str, Any]]):
        self.device = get_device()
        # Build transform registry
        # Registry mapping transform names to their corresponding function
        # Parse config into GPU transforms
        self.transforms = self._build_pipeline(transform_config)

    def _build_pipeline(self, transform_config: List[Dict[str, Any]]) -> T.Compose:
        transform_list = []

        for transform_entry in transform_config:
            transform = transform_entry["type"]
            if transform not in TRANSFORM_REGISTRY:
                raise ValueError(f"Unknown transform: {transform}")
            
            info =  TRANSFORM_REGISTRY[transform]
            transform_class = info["transform"]
            params = info["parser"](transform_entry["attributes"])

            transform_fn = transform_class(**params)  # initialize transform
            transform_list.append(transform_fn)

        transform_pipeline = T.Compose(transform_list) #compose transforms into a single pipeline

        # Optionally JIT script the pipeline for optimization
       # if transform_config["jit_script"]: #just in time for optimized execution #not typically part of a viam config
       #     transform_pipeline = torch.jit.script(transform_pipeline)

        return transform_pipeline

    def transform(self, image: ViamImage) -> ViamImage:
        tensor = viam_to_tensor(image).to(self.device) #ViamImage to tensor
        transformed = self.transforms(tensor) #apply transforms 
        return tensor_to_viam(transformed) #return converted viam image