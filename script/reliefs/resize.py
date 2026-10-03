"""Antialiased image resizing with a differentiable MPS matrix product."""

from functools import lru_cache
import torch
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF


@lru_cache(maxsize=64)
def weights(source, target, mode, device):
    # Apply the reference filter to basis vectors once per size to obtain its linear map.
    basis = torch.eye(source, dtype=torch.float32).reshape(source, 1, 1, source)
    values = F.interpolate(
        basis, size=(1, target), mode=mode, align_corners=False, antialias=True
    )
    return values[:, 0, 0, :].T.contiguous().to(device)


def resize(image, size, mode):
    height, width = image.shape[-2:]
    if (height, width) == tuple(size):
        return image
    rows = weights(height, size[0], mode, str(image.device))
    columns = weights(width, size[1], mode, str(image.device))
    return rows @ image @ columns.T


class Resize(transforms.Resize):
    def forward(self, image):
        height, width = image.shape[-2:]
        short = self.size if isinstance(self.size, int) else self.size[0]
        size = (
            (short, int(short * width / height))
            if height <= width
            else (int(short * height / width), short)
        )
        return resize(image, size, self.interpolation.value)


class RandomResizedCrop(transforms.RandomResizedCrop):
    def forward(self, image):
        top, left, height, width = self.get_params(image, self.scale, self.ratio)
        crop = TF.crop(image, top, left, height, width)
        return resize(crop, self.size, self.interpolation.value)


def perspective(image, grid):
    batch, channels, height, width = image.shape
    x = ((grid[0, :, :, 0] + 1) * width - 1) / 2
    y = ((grid[0, :, :, 1] + 1) * height - 1) / 2
    left, top = x.floor(), y.floor()
    dx, dy = x - left, y - top
    output = torch.zeros_like(image)
    mask = torch.zeros_like(x)
    for ox, oy, weight in [
        (0, 0, (1 - dx) * (1 - dy)),
        (1, 0, dx * (1 - dy)),
        (0, 1, (1 - dx) * dy),
        (1, 1, dx * dy),
    ]:
        ix, iy = left + ox, top + oy
        valid = (ix >= 0) & (iy >= 0) & (ix < width) & (iy < height)
        indices = (
            (iy.clamp(0, height - 1) * width + ix.clamp(0, width - 1)).long().flatten()
        )
        samples = (
            image.flatten(2)
            .index_select(2, indices)
            .reshape(batch, channels, height, width)
        )
        weight = weight * valid
        output = output + samples * weight
        mask = mask + weight
    # Torchvision applies the sampled fill mask after bilinear interpolation.
    return output * mask


class RandomPerspective(transforms.RandomPerspective):
    def forward(self, image):
        if (
            self.fill != 0
            or self.interpolation != transforms.InterpolationMode.BILINEAR
        ):
            raise ValueError(
                "Appearance augmentation requires bilinear sampling and zero fill"
            )
        if torch.rand(1) >= self.p:
            return image
        from torchvision.transforms._functional_tensor import _perspective_grid

        height, width = image.shape[-2:]
        start, end = self.get_params(width, height, self.distortion_scale)
        coefficients = TF._get_perspective_coeffs(start, end)
        grid = _perspective_grid(coefficients, width, height, image.dtype, image.device)
        return perspective(image, grid)
