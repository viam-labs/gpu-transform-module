import pytest
from viam.media.video import ViamImage
import numpy as np
from PIL import Image
from src.transform_pipeline import GPUTransformPipeline
from src.utils.gpu_utils import pil_to_viam_image, viam_to_tensor
import io
from viam.media.video import CameraMimeType

@pytest.fixture
def sample_image():
    """Creates a sample image object for testing."""
    width, height = 128, 128
    pil_image = Image.new("RGB", (width, height), color="white")
    return pil_to_viam_image(pil_image, mime_type=CameraMimeType.JPEG)

@pytest.fixture
def transform_config():
    """Returns a sample transform configuration."""
    return [{
        "type": "resize",
        "attributes": {
            "width_px": 64,
            "height_px": 64
        }
    }]

def test_config_validation(transform_config):
    """Tests if invalid transform raise errors"""
    pipeline = GPUTransformPipeline(transform_config)
    assert pipeline is not None

    # Test invalid transform type
    invalid_config = [{
        "type": "invalid_transform",
        "attributes": {}
    }]
    with pytest.raises(ValueError):
        GPUTransformPipeline(invalid_config)

def test_missing_attributes(sample_image):
    """Tests if missing attributes raise errors"""
    invalid_attrs = [{
        "type": "resize",
        "attributes": {"width_px": 50}  
    }]
    with pytest.raises(KeyError):
        GPUTransformPipeline(invalid_attrs)

def test_resize_transform(sample_image, transform_config):
    pipeline = GPUTransformPipeline(transform_config)
    result = pipeline.transform(sample_image)
    assert isinstance (result, ViamImage)
    result_pil = Image.open(io.BytesIO(result.data))
    assert result_pil.size == (64, 64)

def test_device_handling(sample_image, transform_config):
    pipeline = GPUTransformPipeline(transform_config)
    tensor = viam_to_tensor(sample_image).to(pipeline.device)
    assert tensor.device == pipeline.device

def test_multiple_transforms(sample_image):
    """test multiple transforms on the same image"""
    config = [
        {
            "type": "resize",
            "attributes": {"width_px": 50, "height_px": 50}
        },
        {
            "type": "rotate",
            "attributes": {"angle_degs": 90}
        }
    ]
    pipeline = GPUTransformPipeline(config)
    result = pipeline.transform(sample_image)
    assert isinstance(result, ViamImage)

def test_normalize_transform(sample_image):
    """Test normalize transform"""
    config = [{
        "type": "normalize",
        "attributes": {
            "mean": [0.5],
            "std": [0.5]
        }
    }]
    pipeline = GPUTransformPipeline(config)
    result = pipeline.transform(sample_image)
    assert isinstance(result, ViamImage)

def test_crop_transform(sample_image):
    """Test crop transform"""
    config = [{
        "type": "crop",
        "attributes": {
            "top": 25,
            "left": 25,
            "height": 50,
            "width": 50,
        }
    }]
    pipeline = GPUTransformPipeline(config)
    result = pipeline.transform(sample_image)
    
    # assert type and dims of output
    assert isinstance(result, ViamImage)
    result_pil = Image.open(io.BytesIO(result.data))
    assert result_pil.size == (50, 50)