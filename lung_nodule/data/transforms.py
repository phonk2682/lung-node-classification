
from pathlib import Path
import numpy as np
import numpy.linalg as npl
import scipy.ndimage as ndi
import torch

from lung_nodule.config import config


def _calculateAllPermutations(itemList):
    if len(itemList) == 1:
        return [[i] for i in itemList[0]]
    else:
        sub_permutations = _calculateAllPermutations(itemList[1:])
        return [[i] + p for i in itemList[0] for p in sub_permutations]


def worker_init_fn(worker_id):
    seed = int(torch.utils.data.get_worker_info().seed) % (2**32)
    np.random.seed(seed=seed)


def volumeTransform(
    image,
    voxel_spacing,
    transform_matrix,
    center=None,
    output_shape=None,
    output_voxel_spacing=None,
    **argv,
):
    if "offset" in argv:
        raise ValueError(
            "Cannot supply 'offset' to scipy.ndimage.affine_transform - already used by this function"
        )
    if "output_shape" in argv:
        raise ValueError(
            "Cannot supply 'output_shape' to scipy.ndimage.affine_transform - already used by this function"
        )

    if image.ndim != len(voxel_spacing):
        raise ValueError("Voxel spacing must have the same dimensions")

    if center is None:
        voxelCenter = (np.array(image.shape) - 1) / 2.0
    else:
        if len(center) != image.ndim:
            raise ValueError(
                "center point has not the same dimensionality as the image"
            )
        voxelCenter = np.asarray(center) / voxel_spacing

    transform_matrix = np.asarray(transform_matrix)
    if output_voxel_spacing is None:
        if output_shape is None:
            output_voxel_spacing = np.ones(transform_matrix.shape[0])
        else:
            output_voxel_spacing = np.ones(len(output_shape))
    else:
        output_voxel_spacing = np.array(output_voxel_spacing)

    if transform_matrix.shape[1] != image.ndim:
        raise ValueError(
            "transform_matrix does not have the correct number of columns (does not match image dimensionality)"
        )
    if transform_matrix.shape[0] != image.ndim:
        raise ValueError(
            "Only allowing square transform matrices here, even though this is unneccessary. However, one will need an algorithm here to create full rank-square matrices. 'QR decomposition with Column Pivoting' would probably be a solution, but the author currently does not know what exactly this is, nor how to do this..."
        )

    # Normalize the transform matrix
    transform_matrix = np.array(transform_matrix)
    transform_matrix = (
        transform_matrix.T
        / np.sqrt(np.sum(transform_matrix * transform_matrix, axis=1))
    ).T
    transform_matrix = np.linalg.inv(transform_matrix.T)

    # The forwardMatrix transforms coordinates from input image space into result image space
    forward_matrix = np.dot(
        np.dot(np.diag(1.0 / output_voxel_spacing), transform_matrix),
        np.diag(voxel_spacing),
    )

    if output_shape is None:
        # No output dimensions are specified
        # Calculate the region that will span the whole image
        image_axes = [[0 - o, x - 1 - o] for o, x in zip(voxelCenter, image.shape)]
        image_corners = _calculateAllPermutations(image_axes)

        transformed_image_corners = map(
            lambda x: np.dot(forward_matrix, x), image_corners
        )
        output_shape = [
            1 + int(np.ceil(2 * max(abs(x_max), abs(x_min))))
            for x_min, x_max in zip(
                np.amin(transformed_image_corners, axis=0),
                np.amax(transformed_image_corners, axis=0),
            )
        ]
    else:
        if len(output_shape) != transform_matrix.shape[1]:
            raise ValueError(
                "output dimensions must match dimensionality of the transform matrix"
            )
    output_shape = np.array(output_shape)

    # Calculate the backwards matrix which will be used for the slice extraction
    backwards_matrix = npl.inv(forward_matrix)
    target_image_offset = voxelCenter - backwards_matrix.dot((output_shape - 1) / 2.0)

    return ndi.affine_transform(
        image,
        backwards_matrix,
        offset=target_image_offset,
        output_shape=output_shape,
        **argv,
    )


def clip_and_scale(npzarray, maxHU=400.0, minHU=-1000.0):
    npzarray = (npzarray - minHU) / (maxHU - minHU)
    npzarray[npzarray > 1] = 1.0
    npzarray[npzarray < 0] = 0.0
    return npzarray


def rotateMatrixX(cosAngle, sinAngle):
    return np.asarray([[1, 0, 0], [0, cosAngle, -sinAngle], [0, sinAngle, cosAngle]])


def rotateMatrixY(cosAngle, sinAngle):
    return np.asarray([[cosAngle, 0, sinAngle], [0, 1, 0], [-sinAngle, 0, cosAngle]])


def rotateMatrixZ(cosAngle, sinAngle):
    return np.asarray([[cosAngle, -sinAngle, 0], [sinAngle, cosAngle, 0], [0, 0, 1]])


def elastic_deform_3d(image, alpha, sigma, random_state=None):
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape

    # Generate random displacement fields
    dx = ndi.gaussian_filter(
        (random_state.rand(*shape) * 2 - 1),
        sigma,
        mode="constant",
        cval=0
    ) * alpha
    dy = ndi.gaussian_filter(
        (random_state.rand(*shape) * 2 - 1),
        sigma,
        mode="constant",
        cval=0
    ) * alpha
    dz = ndi.gaussian_filter(
        (random_state.rand(*shape) * 2 - 1),
        sigma,
        mode="constant",
        cval=0
    ) * alpha

    # Create meshgrid
    z, y, x = np.meshgrid(
        np.arange(shape[0]),
        np.arange(shape[1]),
        np.arange(shape[2]),
        indexing='ij'
    )

    # Apply displacement
    indices = (
        np.reshape(z + dz, (-1, 1)),
        np.reshape(y + dy, (-1, 1)),
        np.reshape(x + dx, (-1, 1))
    )

    # Interpolate
    distorted_image = ndi.map_coordinates(
        image,
        indices,
        order=1,
        mode='reflect'
    ).reshape(shape)

    return distorted_image


def apply_intensity_augmentation(patch, config):
    # Intensity shift
    if config.INTENSITY_SHIFT and np.random.random() > 0.7:
        shift = np.random.uniform(*config.INTENSITY_SHIFT_RANGE)
        patch = patch + shift

    # Intensity scale
    if config.INTENSITY_SCALE and np.random.random() > 0.7:
        scale = np.random.uniform(*config.INTENSITY_SCALE_RANGE)
        patch = patch * scale

    # Gaussian noise
    if config.GAUSSIAN_NOISE and np.random.random() > 0.5:
        noise = np.random.normal(0, config.GAUSSIAN_NOISE_STD, patch.shape)
        patch = patch + noise

    # Gamma correction
    if config.GAMMA_CORRECTION and np.random.random() > 0.5:
        gamma = np.random.uniform(*config.GAMMA_RANGE)
        patch_min = patch.min()
        patch_max = patch.max()
        patch_norm = (patch - patch_min) / (patch_max - patch_min + 1e-8)
        patch_norm = np.power(patch_norm, gamma)
        patch = patch_norm * (patch_max - patch_min) + patch_min

    # Contrast adjustment
    if config.CONTRAST_ADJUSTMENT and np.random.random() > 0.5:
        factor = np.random.uniform(*config.CONTRAST_RANGE)
        mean = patch.mean()
        patch = (patch - mean) * factor + mean

    return patch


def sample_random_coordinate_on_sphere(radius):
    random_nums = np.random.normal(size=(3,))

    if np.all(random_nums == 0):
        return np.zeros((3,))

    return random_nums / np.sqrt(np.sum(random_nums * random_nums)) * radius


def extract_patch(
    CTData,
    coord,
    srcVoxelOrigin,
    srcWorldMatrix,
    srcVoxelSpacing,
    output_shape=(64, 64, 64),
    voxel_spacing=(50.0 / 64, 50.0 / 64, 50.0 / 64),
    rotations=None,
    translations=None,
    translation_radius=2.5,
    coord_space_world=False,
    mode="2D",
    config=None,
):
    from lung_nodule.config import config as default_config
    if config is None:
        config = default_config

    transform_matrix = np.eye(3)

    if rotations is not None:
        (zmin, zmax), (ymin, ymax), (xmin, xmax) = rotations

        angleX = np.multiply(np.pi / 180.0, np.random.randint(xmin, xmax, 1))[0]
        angleY = np.multiply(np.pi / 180.0, np.random.randint(ymin, ymax, 1))[0]
        angleZ = np.multiply(np.pi / 180.0, np.random.randint(zmin, zmax, 1))[0]

        transformMatrixAug = np.eye(3)
        transformMatrixAug = np.dot(
            transformMatrixAug, rotateMatrixX(np.cos(angleX), np.sin(angleX))
        )
        transformMatrixAug = np.dot(
            transformMatrixAug, rotateMatrixY(np.cos(angleY), np.sin(angleY))
        )
        transformMatrixAug = np.dot(
            transformMatrixAug, rotateMatrixZ(np.cos(angleZ), np.sin(angleZ))
        )

        transform_matrix = np.dot(transform_matrix, transformMatrixAug)

    if translations is not None:
        radius = np.random.random_sample() * translation_radius
        offset = sample_random_coordinate_on_sphere(radius=radius)
        offset = offset * (1.0 / srcVoxelSpacing)
        coord = np.array(coord) + offset

    thisTransformMatrix = transform_matrix
    thisTransformMatrix = (
        thisTransformMatrix.T
        / np.sqrt(np.sum(thisTransformMatrix * thisTransformMatrix, axis=1))
    ).T

    invSrcMatrix = np.linalg.inv(srcWorldMatrix)

    if coord_space_world:
        overrideCoord = invSrcMatrix.dot(coord - srcVoxelOrigin)
    else:
        overrideCoord = coord * srcVoxelSpacing

    overrideMatrix = (invSrcMatrix.dot(thisTransformMatrix.T) * srcVoxelSpacing).T

    patch = volumeTransform(
        CTData,
        srcVoxelSpacing,
        overrideMatrix,
        center=overrideCoord,
        output_shape=np.array(output_shape),
        output_voxel_spacing=np.array(voxel_spacing),
        order=1,
        prefilter=False,
    )

    # Apply elastic deformation
    if hasattr(config, 'ELASTIC_DEFORM') and config.ELASTIC_DEFORM:
        if np.random.random() > 0.5:
            patch = elastic_deform_3d(
                patch,
                alpha=config.ELASTIC_ALPHA,
                sigma=config.ELASTIC_SIGMA
            )

    # Apply intensity augmentation
    if hasattr(config, 'INTENSITY_SHIFT'):
        patch = apply_intensity_augmentation(patch, config)

    # Replicate channels for 2D or expand dims for 3D
    if mode == "2D":
        patch = np.repeat(patch, 3, axis=0)
    else:
        patch = np.expand_dims(patch, axis=0)

    orig_min, orig_max = patch.min(), patch.max()

    # if hasattr(config, "CLIP_HU_MIN") and hasattr(config, "CLIP_HU_MAX"):
    #     patch = np.clip(patch, config.CLIP_HU_MIN, config.CLIP_HU_MAX)
    # else:
    #     # if not provided, clip to original min/max ± some margin (optional)
    #     margin = getattr(config, "CLIP_MARGIN", 0.0)
    #     patch = np.clip(patch, orig_min - margin, orig_max + margin)

    return patch
