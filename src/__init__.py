from viam.resource.registry import Registry, ResourceCreatorRegistration
from .camera_module import GPUTransformCamera
from .transform_pipeline import GPUTransformPipeline
from viam.components.camera import Camera

Registry.register_resource_creator(Camera.API, GPUTransformCamera.MODEL, ResourceCreatorRegistration(GPUTransformCamera.new))

__all__ = ['GPUTransformCamera', 'GPUTransformPipeline']