# Copyright 2024 AI-Assisted Healthcare Lab

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Stitching multiple sitk.Images to create one single large image
"""

import warnings
from typing import List, Tuple

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import gaussian_filter
from tqdm.autonotebook import tqdm


def no_negative_intensities(image: sitk.Image) -> bool:
    """
    Check if intensities in an image are equal or greather than 0.
    Motivation: In MRI sequences we can set the background to 0.
    (Requirement for the current implmentation of the gaussion filter)
    This is not possible for CT

    Args:
        image1 (sitk.Image): sitk image

    Returns:
        A boolean
    """

    image_array = sitk.GetArrayFromImage(image)
    min_value = np.min(image_array)

    return min_value >= 0


def left_upper(image1: sitk.Image, image2: sitk.Image) -> Tuple[int, int, int]:
    """
    Assuming that image1 and image2 are in the same coordinate system. This
    function returns the left upper point of the bunding box that encases
    the two images

    Args:
        image1 (sitk.Image): The first image
        image2 (sitk.Image): The secong image

    Returns:
        A tuple of the coordinates (x, y, z)
    """
    origin1 = image1.GetOrigin()
    origin2 = image2.GetOrigin()
    return tuple([min(o1, o2) for o1, o2 in zip(origin1, origin2)])


def get_spatial_size(image: sitk.Image) -> Tuple[int, int, int]:
    """
    While sitk.Image.GetSize returns the size of the pixel matrix,
    this function returns the spatial size of the image.

    Args:
        image (sitk.Image): The image to extract the spatial size

    Returns:
        tuple: A tuple with the spatial size in all dimensions
    """
    size = image.GetSize()
    spacing = image.GetSpacing()
    return tuple([sz * sp for sz, sp in zip(size, spacing)])


def right_lower(image1: sitk.Image, image2: sitk.Image) -> Tuple[int, int, int]:
    """
    Assuming that image1 and image2 are in the same coordinate system. This
    function returns the right lower point of the bunding box that encases
    the two images

    Args:
        image1 (sitk.Image): The first image
        image2 (sitk.Image): The secong image

    Returns:
        tuple: A tuple of the coordinates (x, y, z)
    """
    origin1 = image1.GetOrigin()
    origin2 = image2.GetOrigin()
    size1 = get_spatial_size(image1)
    size2 = get_spatial_size(image2)
    right_lower1 = [o + s for o, s in zip(origin1, size1)]
    right_lower2 = [o + s for o, s in zip(origin2, size2)]
    return tuple([max(r1, r2) for r1, r2 in zip(right_lower1, right_lower2)])


def resample_images_to_fit(
    fixed_image: sitk.Image, moving_image: sitk.Image, is_label: bool = False
) -> Tuple[sitk.Image, sitk.Image]:
    """
    Resamples both `moving_image` and `fixed_image`, so they occupy the same
    physical space. Useful as preprocessing before image registration

    o---o              o---------o    o---------o
    |im1| o---o        |         |    |         |
    |   | |im2|  = >   |   im1   |    |   im2   |
    o---o |   |        |         |    |         |
          o---o        o---------o    o---------o

    Args:
        fixed_image (sitk.Image): The first image
        moving_image (sitk.Image): The secong image

    Returns:
        tuple: A tuple of both images resampled.

    """
    origin_empty_image = left_upper(fixed_image, moving_image)
    right_lower_empty_image = right_lower(fixed_image, moving_image)
    spatial_size_emtpy_image = [abs(lu - rl) for lu, rl in zip(origin_empty_image, right_lower_empty_image)]
    px_size_emtpy_image = [round(sz / sp) for sz, sp in zip(spatial_size_emtpy_image, fixed_image.GetSpacing())]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(fixed_image.GetSpacing())
    resampler.SetSize(px_size_emtpy_image)
    resampler.SetOutputDirection(fixed_image.GetDirection())
    resampler.SetTransform(sitk.Transform())
    resampler.SetOutputOrigin(origin_empty_image)

    if no_negative_intensities(fixed_image) and no_negative_intensities(moving_image):
        resampler.SetDefaultPixelValue(0)
    else:
        resampler.SetDefaultPixelValue(fixed_image.GetPixelIDValue())

    if is_label:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resampler.SetInterpolator(sitk.sitkLinear)

    fixed_image = resampler.Execute(fixed_image)
    moving_image = resampler.Execute(moving_image)

    return fixed_image, moving_image


# deprecated
def translate_image(fixed_image: sitk.Image, moving_image: sitk.Image):
    """
    Translates a `moving_image` to align it with a `fixed_image` using the ElastixImageFilter.

    Args:
        fixed_image (sitk.Image): The fixed image that the `moving_image` will
            be aligned to.
        moving_image (sitk.Image): The image that will be translated to align
            with the `fixed_image`.

    Returns:
        tuple: A tuple containing the translated `moving_image` and the `ParameterMap`
            of the transformation applied.
    """
    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.LogToFileOff()
    elastixImageFilter.LogToConsoleOff()
    elastixImageFilter.SetFixedImage(fixed_image)
    elastixImageFilter.SetMovingImage(moving_image)
    elastixImageFilter.SetParameterMap(sitk.GetDefaultParameterMap("translation"))
    elastixImageFilter.Execute()
    # We need the translation parameter map, but rigid registration gives better results.
    parameter_map = elastixImageFilter.GetTransformParameterMap()[0].asdict()

    # Create a ParameterMap for rigid registration
    pm = sitk.GetDefaultParameterMap("rigid")
    elastixImageFilter.SetParameterMap(pm)
    elastixImageFilter.Execute()
    translated_image = elastixImageFilter.GetResultImage()

    return translated_image, parameter_map


def combine_images(
    array: np.ndarray,
    fixed_image: sitk.Image,
    moving_image: sitk.Image,
    z_direction: str,
    gaus: bool = True,
) -> np.ndarray:
    """
    Parse `fixed_image` and `moving_image` onto an empty array.

    Args:
        array (np.ndarray): An (empty) array providing the common canvas to parse
            both images
        fixed_image (sitk.Image): The first image to parse onto the array
        moving_image (sitk.Image): The second image to parse onto the array
        z_direction (str): If caudal, `first_image` is treated as top image
            (caudal to `moving_image`)
        gaus (bool): apply a gaussian filter (z-axis only) on the overlap of both images. Attention: Only works for MRI

    Returns:
        np.ndarray: The combined image as array
    """
    top = sitk.GetArrayFromImage(fixed_image if z_direction == "caudal" else moving_image)
    bottom = sitk.GetArrayFromImage(moving_image if z_direction == "caudal" else fixed_image)

    if top.shape == bottom.shape == array.shape:

        merged = np.stack([top, bottom], 0).max(0)

        # this assumes that the default value is zero
        # This holds for MRI but not for CT
        if gaus:

            if not no_negative_intensities(fixed_image) or not no_negative_intensities(moving_image):
                warnings.warn(
                    "The current implementation of the gaus filter cannot be applied on negative intensities. "
                    "Gaus Filter is skipped"
                )
                return merged

            cuttoffs_z = [
                np.max(np.argwhere(top > 0)[:, 0]),
                np.min(np.argwhere(top > 0)[:, 0]),
                np.max(np.argwhere(bottom > 0)[:, 0]),
                np.min(np.argwhere(bottom > 0)[:, 0]),
            ]
            cuttoffs_z.sort()

            z1 = cuttoffs_z[1] - 1  # start of overlap ( substract 1, so that the cut is at the next position)
            z2 = cuttoffs_z[2]  # end of overlap

            # Change depending on your images
            # Recommandation for UKBB:
            # sigma = (1.5, 0, 0)
            # margin = 3  # excluding start slice
            sigma = (1, 0, 0)
            margin = 2  # excluding start slice

            if z2 - z1 < 2 * (margin + 1):
                _slice = slice(z1 - margin, z2 + margin + 2)
                merged[_slice] = gaussian_filter(merged[_slice], sigma=sigma, mode="nearest")

            else:
                _slice1 = slice(z1 - margin, z1 + margin + 2)
                _slice2 = slice(z2 - margin, z2 + margin + 2)
                merged[_slice1] = gaussian_filter(merged[_slice1], sigma=sigma, mode="nearest")
                merged[_slice2] = gaussian_filter(merged[_slice2], sigma=sigma, mode="nearest")

        return merged

    if gaus:
        raise Exception("gaussian filter not yet implemented for this configuration")

    # TODO: enable smoother pooling of the images
    fx, fy, fz = top.shape
    mx, my, mz = bottom.shape
    top = np.stack([top, array[-fx:, -fy:, -fz:]], 0).max(0)
    array[-fx:, -fy:, -fz:] = top
    bottom = np.stack([bottom, array[:mx, :my, :mz]], 0).max(0)
    array[:mx, :my, :mz] = bottom

    return array


def stitch_two_images(
    fixed_image: sitk.Image, moving_image: sitk.Image, translate: bool = False, gaus: bool = True, is_label=False
) -> sitk.Image:
    """
    Stitches two SimpleITK images into a single image by aligning `moving_image`
    to `fixed_image` and combining them.

    Parameters:
        fixed_image (sitk.Image): The fixed image that the `moving_image` will be
            aligned to.
        moving_image (sitk.Image): The image that will be translated to align with
            the `fixed_image`.
        translate (bool): translate moving image with image registration
        gaus (bool): apply a gaussian filter (z-axis only) on the overlap of both images. Attention: Only works for MRI

    Returns:
        sitk.Image: A single image resulting from combining `fixed_image` and
            `moving_image` after aligning `moving_image` to `fixed_image`.
    """
    z_direction = (
        "caudal" if fixed_image.GetOrigin()[2] < moving_image.GetOrigin()[2] else "cranial"
    )  # caudal -> the moving image is placed caudal the fixed image

    # perform a translation registration and get parameters
    fixed_image, moving_image = resample_images_to_fit(fixed_image, moving_image, is_label)

    if translate:
        translated_image, parameter_map = translate_image(fixed_image, moving_image)
        transform_parameters = [float(x) for x in parameter_map["TransformParameters"]]

        # compute the final volume dimensions
        final_vol_dim = [
            x + abs(round(y / s))
            for x, y, s in zip(fixed_image.GetSize(), transform_parameters, fixed_image.GetSpacing())
        ]
        z_direction = (
            "caudal" if transform_parameters[2] < 0 else "cranial"
        )  # caudal -> the moving image is placed caudal the fixed image

        # create an empty volume with these dimensions
        empty_array = np.zeros(final_vol_dim[::-1])

    else:
        translated_image = moving_image
        assert (
            sitk.GetArrayFromImage(fixed_image).shape == sitk.GetArrayFromImage(moving_image).shape
        ), "Image not correctly resampled, size of fixed and moving image should be equal at this stage"

        empty_array = np.zeros_like(sitk.GetArrayFromImage(fixed_image))

    # compute volume affine data
    spacing = fixed_image.GetSpacing()
    origin = fixed_image.GetOrigin() if z_direction == "caudal" else moving_image.GetOrigin()
    direction = fixed_image.GetDirection()

    # fill the volume according to the transform paramters
    stitched_array = combine_images(empty_array, fixed_image, translated_image, z_direction, gaus=gaus)

    # convert array to sitk.Image and update affine
    stiched_image = sitk.GetImageFromArray(stitched_array)
    stiched_image.SetSpacing(spacing)
    stiched_image.SetOrigin(origin)
    stiched_image.SetDirection(direction)

    return stiched_image


def resample_to_isotropic(
    img: sitk.Image,
    new_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    is_label=False,
) -> sitk.Image:
    """
    This function resamples an image to isotropic pixel spacing.

    Args:
        img: input SimpleITK image.
        new_spacing: desired pixel spacing.
        interpolator: interpolation method (default is sitk.sitkLinear).

    Returns:
        Resampled SimpleITK image.
    """
    # Original image spacing and size
    original_spacing = img.GetSpacing()
    original_size = img.GetSize()

    # Compute new image size
    new_size = [int(round(osz * ospc / nspc)) for osz, ospc, nspc in zip(original_size, original_spacing, new_spacing)]

    # Resample image
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetTransform(sitk.Transform())

    if no_negative_intensities(img):
        resampler.SetDefaultPixelValue(0)
    else:
        resampler.SetDefaultPixelValue(img.GetPixelIDValue())

    if is_label:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resampler.SetInterpolator(sitk.sitkLinear)

    return resampler.Execute(img)


def stitch_images(
    images: List[sitk.Image],
    new_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    translate: bool = False,
    gaus: bool = True,
    is_label: bool = False,
) -> List[sitk.Image]:
    """
    Combine a list of images into a single large image with isotropnic spacing

    Args:
        images: list of SimpleITK images.
        new_spacing: desired pixel spacing.
        translate (bool): translate moving images with image registration
        gaus (bool): apply a gaussian filter (z-axis only) on the overlap of images

    Returns:
        Stitched SimpleITK image.
    """

    if is_label:
        gaus = False

    isotrophic_images = [
        resample_to_isotropic(img, new_spacing=new_spacing, is_label=is_label)
        for img in tqdm(images, postfix="Resampling Series")
    ]

    fixed_image, *moving_images = isotrophic_images
    for moving in tqdm(moving_images, postfix="Stitching Series"):
        fixed_image = stitch_two_images(fixed_image, moving, translate, gaus, is_label)
    return fixed_image
