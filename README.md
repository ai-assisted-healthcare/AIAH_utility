<h2 align="center"> AI-Assisted Healthcare Utility Scripts </h2>

***

<div align="center">
<a href="https://github.com/ai-assisted-healthcare/AIAH_utility/actions"><img alt="Continuous Integration" src="https://github.com/ai-assisted-healthcare/AIAH_utility/actions/workflows/ci.yml/badge.svg"></a>
<a href="https://github.com/ai-assisted-healthcare/AIAH_utility/blob/master/License.txt"><img alt="License: Apache" src="https://img.shields.io/badge/License-Apache_2.0-blue.svg"></a>  
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</div>

## Container for:
- Nifti Stitching
- 3D Image Viewing
- Dicom Reading

## Installation

```bash
git clone https://github.com/ai-assisted-healthcare/AIAH_utility.git
cd AIAH_utility
python -m pip install -e .
```
## Usage

### Viewing
```python
from AIAH_utility.viewer import BasicViewer, ListViewer
from monai.transforms import LoadImage()

img = LoadImage()("image.nii.gz")
seg = LoadImage()("segmentation.nii.gz")

BasicViewer(img,seg).show()
```

### Stitching
```python
from AIAH_utility.stitching import stitch_images
import SimpleITK as sitk

img1 = sitk.ReadImage("image1.nii.gz")
img2 = sitk.ReadImage("image2.nii.gz")
img3 = sitk.ReadImage("image3.nii.gz")

stitched_image = stitch_images([img1,img2,img3])
```

