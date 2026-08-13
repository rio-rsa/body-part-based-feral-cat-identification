# Code for "Body-Part-based Individual Feral Cat Identification from Camera Trap Images Using Deep Learning"

[![](https://img.shields.io/badge/Ecological_Informatics_Vol_90-Paper-blue)](https://doi.org/10.1016/j.ecoinf.2025.103258)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17586692.svg)](https://doi.org/10.5281/zenodo.17586692)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

**TL;DR**: This repository hosts the code and (links to) data for the paper:

> R. R. S. Akbar, M. W. Rees, P. A. Fleming, and F. Sohel, “Body-part-based individual feral cat identification from camera trap images using deep learning,” *Ecological Informatics*, vol. 90, p. 103258, Jun. 2025, doi: 10.1016/j.ecoinf.2025.103258.

---

### Step 1: Folder Setup
Crops of body part images should be arranged in the following folder setup to begin. 
Body Part / Individual Cat / Image.

```
├── body/
│   ├── cat_01/
│   ├── cat_02/
│   ├── ...
│   └── cat_10/
├── front_leg/
│   ├── cat_01/
│   ├── cat_02/
│   ├── ...
│   └── cat_10/
├── back_leg/
│   ├── cat_01/
│   ├── cat_02/
│   ├── ...
│   └── cat_10/
└── tail/
    ├── cat_01/
    ├── cat_02/
    ├── ...
    └── cat_10/
```

### Step 2: Data split with split.py

Place the split.py script in the root of your folder along with the other body part folders and run it. There is a minimum of 3 images that a cat must have for each body part for the script to run. The script should split the data (by default 70/15/15, but this can be changed in the script) into the following folder structure:

```
├── body/
|   ├── train/
|   |   ├── cat_01/
|   |   ...
|   |   ├── cat_10/
|   ├── val/
|   |   ├── cat_01/
|   |   ...
|   |   ├── cat_10/
|   └── test/
|       ├── cat_01/
|       ...
|       └── cat_10/
├── front_leg/
|   ├── train/
|   |   ├── cat_01/
|   |   ...
|   |   ├── cat_10/
|   ├── val/
|   |   ├── cat_01/
|   |   ...
|   |   ├── cat_10/
|   └──  test/
|       ├── cat_01/
|       ...
|       └──  cat_10/
├── back_leg/
|   ├── train/
|   |   ├── cat_01/
|   |   ...
|   |   ├── cat_10/
|   ├── val/
|   |   ├── cat_01/
|   |   ...
|   |   ├── cat_10/
|   └── test/
|       ├── cat_01/
|       ...
|       └── cat_10/
└── tail/
    ├── train/
    |   ├── cat_01/
    |   ...
    |   ├── cat_10/
    ├── val/
    |   ├── cat_01/
    |   ...
    |   ├── cat_10/
    └── test/
        ├── cat_01/
        ...
        └── cat_10/
```

### Step 3: Run featureconcat.py

Now that the images are in the correct folders, the featureconcat.py script can be run.
