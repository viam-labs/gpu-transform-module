import torch
import torchvision.transforms as T
from viam.media.video import ViamImage, CameraMimeType
from viam.media.utils.pil import pil_to_viam_image
import io
from PIL import Image

def get_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def tensor_to_viam(tensor: torch.Tensor) -> ViamImage:
    # convert tensor to PIL first
    pil_image = T.ToPILImage()(tensor)
    return pil_to_viam_image(pil_image, mime_type=CameraMimeType.JPEG)

def viam_to_tensor(viam_image:ViamImage) -> torch.Tensor:
    pil_image = Image.open(io.BytesIO(viam_image.data))
    return T.ToTensor()(pil_image)