"""Acoustic Reliefs appearance objective using Mitsuba Metal and CLIP on MPS.

Derived from Acoustic Reliefs pyoptim/diffmesh.py and pyoptim/losses.py.
See ../../src/Bem/LICENSE.AcousticReliefs.
"""

from pathlib import Path
import numpy as np
import torch
from torchvision import transforms
import clip
import mitsuba as mi
import drjit as dr
from .resize import Resize, RandomResizedCrop, RandomPerspective

CAMERAS = [(np.pi / 2, 0)] + [(np.pi / 4, i * np.pi / 2) for i in range(4)]


class PatchEmbedding(torch.autograd.Function):
    @staticmethod
    def forward(ctx, image, weight):
        if weight.requires_grad or weight.shape[-2:] != (32, 32):
            raise ValueError("Expected frozen ViT-B/32 patch weights")
        if image.shape[-2] % 32 or image.shape[-1] % 32:
            raise ValueError("Patch dimensions must divide the image dimensions")
        ctx.save_for_backward(weight)
        ctx.shape = image.shape
        return torch.nn.functional.conv2d(image, weight, stride=32)

    @staticmethod
    def backward(ctx, gradient):
        (weight,) = ctx.saved_tensors
        batch, channels, height, width = ctx.shape
        # Nonoverlapping patches reduce the input derivative to one matrix product.
        patches = gradient.flatten(2).transpose(1, 2) @ weight.flatten(1)
        image = patches.reshape(batch, height // 32, width // 32, channels, 32, 32)
        return image.permute(0, 3, 1, 4, 2, 5).reshape(ctx.shape), None


class ImageLoss:
    def __init__(self, device="mps", models=Path("build/reliefs-models")):
        self.device = device
        self.model, prep = clip.load(
            "ViT-B/32", device, jit=False, download_root=str(models)
        )
        self.model.float().eval().requires_grad_(False)
        self.normalizers = transforms.Compose(
            [
                Resize(
                    prep.transforms[0].size,
                    interpolation=prep.transforms[0].interpolation,
                ),
                prep.transforms[1],
                prep.transforms[-1],
            ]
        )
        self.augs = transforms.Compose(
            [
                RandomPerspective(fill=0, p=1.0, distortion_scale=0.5),
                RandomResizedCrop(224, scale=(0.8, 0.8), ratio=(1.0, 1.0)),
                transforms.Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

    def features(self, image):
        visual = self.model.visual
        x = PatchEmbedding.apply(image.to(self.model.dtype), visual.conv1.weight)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        token = visual.class_embedding.to(x.dtype) + torch.zeros(
            x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
        )
        x = torch.cat([token, x], dim=1) + visual.positional_embedding.to(x.dtype)
        x = visual.ln_pre(x).permute(1, 0, 2)
        # The released objective uses only transformer layer 3 and zero semantic weight.
        for block in visual.transformer.resblocks[:4]:
            x = block(x)
        return x.permute(1, 0, 2)

    def __call__(self, image, target):
        x, y = image.float().to(self.device), target.float().to(self.device)
        xs, ys = [self.normalizers(x)], [self.normalizers(y)]
        for _ in range(4):
            pair = self.augs(torch.cat([x, y]))
            xs.append(pair[0].unsqueeze(0))
            ys.append(pair[1].unsqueeze(0))
        return torch.square(
            self.features(torch.cat(xs)) - self.features(torch.cat(ys))
        ).mean()

    def texture(self, pixels, target):
        x = torch.tensor(
            pixels, dtype=torch.float32, device=self.device, requires_grad=True
        )
        image = x.unsqueeze(0).unsqueeze(0).expand(1, 3, *x.shape)
        loss = self(image, target)
        gradient = torch.autograd.grad(loss, x)[0]
        return float(loss.detach().cpu()), gradient.detach().cpu().numpy()


class Appearance:
    def __init__(self, vertices, triangles, target, loss):
        mi.set_variant("metal_ad_rgb")
        transform = mi.ScalarTransform4f
        vertices, triangles = np.asarray(vertices), np.asarray(triangles)
        low, high = vertices.min(axis=0), vertices.max(axis=0)
        self.width, _, self.depth = high - low
        top_faces = triangles[
            np.all(np.isclose(vertices[triangles, 1], high[1]), axis=1)
        ]
        ids, faces = [], []
        index = {}
        for face in top_faces:
            row = []
            for i in face:
                if i not in index:
                    index[i] = len(ids)
                    ids.append(i)
                row.append(index[i])
            faces.append(row)
        points = vertices[ids].copy()
        points[:, 1] = 0
        uv = (points[:, [0, 2]] - low[[0, 2]]) / (
            high[[0, 2]] - low[[0, 2]]
        ) * 0.9998 + 0.0001
        mesh = mi.Mesh(
            "hfield",
            vertex_count=len(points),
            face_count=len(faces),
            has_vertex_normals=True,
            has_vertex_texcoords=True,
        )
        parameters = mi.traverse(mesh)
        parameters["vertex_positions"] = mi.Float(points.ravel())
        parameters["faces"] = mi.UInt(np.asarray(faces).ravel())
        parameters["vertex_texcoords"] = mi.Float(uv.ravel())
        parameters["bsdf.reflectance.value"] = 1
        parameters.update()
        base = dict(
            type="scene",
            integrator=dict(type="direct_projective", sppi=0, hide_emitters=True),
            emitter=dict(
                type="point", position=[0, 1, 0], intensity=dict(type="rgb", value=1.5)
            ),
        )
        self.scene = mi.load_dict(dict(base, hfield=mesh))
        self.scene_params = mi.traverse(self.scene)
        self.positions = dr.unravel(
            mi.Vector3f, self.scene_params["hfield.vertex_positions"]
        )
        self.normals = dr.unravel(
            mi.Vector3f, self.scene_params["hfield.vertex_normals"]
        )
        self.si = dr.zeros(mi.SurfaceInteraction3f, len(points))
        self.si.uv = dr.unravel(
            type(self.si.uv), self.scene_params["hfield.vertex_texcoords"]
        )
        self.texture = mi.load_dict(
            dict(
                type="bitmap",
                id="hfield_tex",
                bitmap=mi.Bitmap(dr.zeros(mi.TensorXf, (2, 2))),
                raw=True,
            )
        )
        self.params = mi.traverse(self.texture)
        self.params.keep(["data"])
        bitmap = mi.Bitmap(str(target)).convert(mi.Bitmap.PixelFormat.Y)
        self.reference = mi.load_dict(
            dict(
                base,
                ref_mesh=dict(
                    type="rectangle",
                    to_world=transform()
                    .scale([self.width / 2, 1, self.depth / 2])
                    .rotate([1, 0, 0], -90),
                    bsdf=dict(
                        type="diffuse",
                        reflectance=dict(
                            type="bitmap",
                            bitmap=bitmap,
                            to_uv=transform().scale([1, -1, 1]),
                        ),
                    ),
                ),
            )
        )
        self.loss = loss
        self.reference_images = {}

    def sensor(self, elevation, azimuth, radius=1, resolution=256):
        origin = [
            radius * np.cos(elevation) * np.cos(azimuth),
            radius * np.sin(elevation),
            -radius * np.cos(elevation) * np.sin(azimuth),
        ]
        return mi.load_dict(
            dict(
                type="perspective",
                fov=45,
                to_world=mi.ScalarTransform4f().look_at(
                    origin=origin, target=[0, 0, 0], up=[0, 0, -1]
                ),
                sampler=dict(type="independent", sample_count=512),
                film=dict(
                    type="hdrfilm",
                    width=resolution,
                    height=resolution,
                    sample_border=True,
                    pixel_format="rgb",
                ),
            )
        )

    def render(
        self, pixels, elevation, azimuth, radius=1, resolution=256, spp=512, seed=0
    ):
        self.params["data"] = mi.TensorXf(
            np.asarray(pixels, dtype=np.float32)[:, :, None]
        )
        dr.enable_grad(self.params["data"])
        self.params.update()
        heights = self.texture.eval_1(self.si)
        self.scene_params["hfield.vertex_positions"] = dr.ravel(
            heights * self.normals + self.positions
        )
        self.scene_params.update()
        sensor = self.sensor(elevation, azimuth, radius, resolution)
        return mi.render(
            self.scene, self.params, sensor=sensor, spp=spp, seed=seed
        ) ** (1 / 2.2)

    def gradient(
        self, pixels, elevation, azimuth, radius=1, resolution=256, spp=512, seed=0
    ):
        image = self.render(pixels, elevation, azimuth, radius, resolution, spp, seed)
        key = (elevation, azimuth, radius, resolution, spp, seed)
        if key not in self.reference_images:
            reference = mi.render(
                self.reference,
                sensor=self.sensor(elevation, azimuth, radius, resolution),
                spp=spp,
                seed=seed,
            ) ** (1 / 2.2)
            target = (
                torch.from_numpy(
                    np.asarray(mi.util.convert_to_bitmap(reference)) / 255.0
                )
                .permute(2, 0, 1)
                .unsqueeze(0)
            )
            self.reference_images[key] = target.float().to(self.loss.device)
        target = self.reference_images[key]
        x = torch.tensor(np.asarray(image), device=self.loss.device, requires_grad=True)
        value = self.loss(x.permute(2, 0, 1).unsqueeze(0), target)
        image_gradient = torch.autograd.grad(value, x)[0].detach().cpu().numpy()
        dr.set_grad(image, mi.TensorXf(image_gradient))
        dr.backward_from(image)
        gradient = np.asarray(dr.grad(self.params["data"])).squeeze().copy()
        return float(value.detach().cpu()), gradient
