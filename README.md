# Data Availability for Body-Part-based Individual Feral Cat Identification from Camera Trap Images Using Deep Learning

# Original Data
The data from the study that captured camera trap images of the feral cats used in this study is available here - https://datadryad.org/dataset/doi:10.5061/dryad.69p8cz95w

## Data Processing and Image Sorting Tutorial

### Step 1: Select the Cats

We begin by selecting the 10 individual cats used for this study. These were chosen based on having a sufficient number of usable images (300–580 per individual). The selected individuals are:

- **Annya Swirl Ben**
- **Otway Tabby Harriet**
- **Otway Tabby Maximus**
- **Otway Tabby Meg**
- **Otway Tabby Kingfluffy**
- **Otway Tabby Persy**
- **Otway Swirl Chowder**
- **Otway Tabby Squash**
- **Otway Swirl Bluey**
- **Otway Tabby Catsup**

Each cat will be represented by a unique folder (e.g., `cat_01`, `cat_02`, ..., `cat_10`).

### Step 2: Create Body Part Folders

Next, we create **four body part folders**, and copy the 10 cat folders into each one. This allows us to sort and manage images by both individual and body part.

The structure should look like this:

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
├── rear_leg/
│   ├── cat_01/
│   ├── cat_02/
│   ├── ...
│   └── cat_10/
└── tail/
    ├── cat_01/
    ├── cat_02/
    ├── ...
    └── cat_10/

### Step 3: Sort by Visibility and Orientation

For each image in each cat folder:

1. **Check if the target body part is clearly visible** in the image. If not, discard the image.
2. **Determine whether the image shows the left or right side** of the cat.
3. **Move the image into a `left/` or `right/` subfolder** under the corresponding individual’s folder.

Repeat this process for all four body parts and all 10 cats.

The updated structure will look like this:

├── body/
│   ├── cat_01/
│   │   ├── left/
│   │   └── right/
│   ├── cat_02/
│   │   ├── left/
│   │   └── right/
│   ├── ...
│   └── cat_10/
│       ├── left/
│       └── right/
├── front_leg/
│   ├── cat_01/
│   │   ├── left/
│   │   └── right/
│   ├── cat_02/
│   │   ├── left/
│   │   └── right/
│   ├── ...
│   └── cat_10/
│       ├── left/
│       └── right/
├── rear_leg/
│   ├── cat_01/
│   │   ├── left/
│   │   └── right/
│   ├── ...
│   └── cat_10/
│       ├── left/
│       └── right/
└── tail/
    ├── cat_01/
    │   ├── left/
    │   └── right/
    ├── ...
    └── cat_10/
        ├── left/
        └── right/

### Step 4: Choosing a side

To ensure consistency in the dataset, we standardise all images to show the **left side** of the cat.

- For each cat, compare the number of usable images between the left and right side views.
- **Choose the side with the most usable images**.
  - In this study, the **left side was more common**, so we chose it as the standard.
- For cats that had more right-side images, those images were **horizontally flipped** to simulate a left-side view.
- This ensures all images are aligned in orientation and feature placement, simplifying model training.

> Note: Since cat tails are roughly symmetrical, tail images from the right side were also flipped and used as left-side tail views.

After this step, all retained and flipped images represent the cat’s left side, and future cropping and training will use this orientation only.

Once this is complete, discard the side that was not selected, and **move all selected images into the main cat folder** for that body part. The folder structure should go back to looking like this (i.e., no `left/` or `right/` subfolders):

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
├── rear_leg/
│   ├── cat_01/
│   ├── cat_02/
│   ├── ...
│   └── cat_10/
└── tail/
    ├── cat_01/
    ├── cat_02/
    ├── ...
    └── cat_10/


# Models
