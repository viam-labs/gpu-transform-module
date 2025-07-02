#implement get image to read image from upstream source, push to GPU and apply transfomrs and return transfomred image in the right format (Encoded JPEG)

from viam.components.camera import Camera
from viam.media.video import ViamImage
from src.transform_pipeline import GPUTransformPipeline
from typing import Dict, Any, ClassVar
from viam.resource.types import Model, ModelFamily
from viam.components.camera import Camera

class GPUTransformCamera(Camera):
    MODEL: ClassVar[Model] = Model(ModelFamily("viam", "camera"), "transform")
    def __init__(self, name: str):
        super().__init__(name)
        self.source_camera = None
        self.pipeline = None
        
    @classmethod
    async def new(cls, config: Dict[str, Any], dependencies: Dict[str, Any]):
        camera = cls(config["name"])
        source_name = config["attributes"]["source"]
        if source_name not in dependencies:
            raise ValueError(f"Source camera {source_name} not found in dependencies")
        camera.source_camera = dependencies[source_name]
        camera.pipeline = GPUTransformPipeline(config["attributes"]["pipeline"])
        return camera

    async def get_image(self) -> ViamImage:
        source_image = await self.source_camera.get_image()
        return self.pipeline.transform(source_image)