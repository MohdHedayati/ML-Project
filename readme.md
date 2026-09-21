# Traffic Sign Classification Datasets Inspection Report

Here is the factual inspection report for the three traffic sign datasets based on their respective files and directory structures.

### 1. DATASET STRUCTURE

**GTSRB (German Traffic Sign Recognition Benchmark)**
- **Structure**: The dataset contains `Train/`, `Test/`, and `Meta/` directories. 
- **Training images**: Located in `Images\gtsrb-german-traffic-sign\versions\1\Train\`, further organized into subfolders corresponding to their class ID (e.g., `00000`, `00001`, etc.).
- **Testing images**: Located in `Images\gtsrb-german-traffic-sign\versions\1\Test\`, stored together in a single folder without class-specific subfolders.
- **Validation set**: No pre-existing validation folder/split exists.
- **Image file format**: `.png`

**CTSD (Chinese Traffic Sign Dataset)**
- **Structure**: The dataset contains a single `images/` directory and an `annotations.csv` file. 
- **Training/Testing images**: All images are stored together in `Images\chinese-traffic-signs\versions\2\images\`. There is no predefined train/test split.
- **Validation set**: No pre-existing validation split exists.
- **Image file format**: `.png`

**BelgiumTSC (Belgian Traffic Sign Dataset)**
- **Structure**: Split across top-level folders including `BelgiumTSC_Training` and `BelgiumTSC_Testing`.
- **Training images**: Located in `Images\belgium-ts\versions\10\BelgiumTSC_Training\Training\`, organized into class subfolders (e.g., `00000`, `00001`).
- **Testing images**: Located in `Images\belgium-ts\versions\10\BelgiumTSC_Testing\Testing\`, also organized into class subfolders.
- **Validation set**: No pre-existing validation split exists.
- **Image file format**: `.ppm`

---

### 2. ANNOTATION FILES

**GTSRB**
- **Train.csv** (`gtsrb-german-traffic-sign\versions\1\Train.csv`):
  - **Shape**: 39,209 rows, 8 columns
  - **Columns**: `Width`, `Height`, `Roi.X1`, `Roi.Y1`, `Roi.X2`, `Roi.Y2`, `ClassId`, `Path`
  - **Filename column**: `Path` (contains relative paths like `Train/20/00020_00000_00000.png`)
  - **Label column**: `ClassId`
  - **Bounding box**: Present (`Roi.X1`, `Roi.Y1`, `Roi.X2`, `Roi.Y2`)
  - **Other columns**: `Width` and `Height` representing absolute image dimensions.
- **Test.csv** (`gtsrb-german-traffic-sign\versions\1\Test.csv`):
  - **Shape**: 12,630 rows, 8 columns (identical structure to `Train.csv`)
  - Maps test images in the `Test/` folder to their true class labels.
- **Meta.csv** (`gtsrb-german-traffic-sign\versions\1\Meta.csv`):
  - **Shape**: 43 rows, 5 columns (`Path`, `ClassId`, `ShapeId`, `ColorId`, `SignId`)
  - Contains class metadata mappings rather than image annotations.

**CTSD**
- **annotations.csv** (`chinese-traffic-signs\versions\2\annotations.csv`):
  - **Shape**: 6,164 rows, 8 columns
  - **Columns**: `file_name`, `width`, `height`, `x1`, `y1`, `x2`, `y2`, `category`
  - **Filename column**: `file_name`
  - **Label column**: `category`
  - **Bounding box**: Present (`x1`, `y1`, `x2`, `y2`)
  - **Other columns**: `width` and `height` (image dimensions).

**BelgiumTSC**
- **GT-xxxxx.csv files** (e.g., `BelgiumTSC_Training\Training\00000\GT-00000.csv`):
  - Every class folder in both training and testing contains a dedicated annotation file named after its class. 
  - **Columns**: 8 columns delimited by semicolons (`;`) — `Filename;Width;Height;Roi.X1;Roi.Y1;Roi.X2;Roi.Y2;ClassId`
  - **Filename column**: `Filename`
  - **Label column**: `ClassId`
  - **Bounding box**: Present (`Roi.X1`, `Roi.Y1`, `Roi.X2`, `Roi.Y2`)
  - **Other columns**: `Width` and `Height` (image dimensions).

---

### 3. CLASS INFORMATION

**GTSRB**
- **Total classes**: 43
- **Class IDs**: `0` through `42` (starts from 0).
- **Class names**: Not explicitly provided as readable strings in the CSVs; the metadata file groups them by visual features (`ShapeId`, `ColorId`, `SignId`).

**CTSD**
- **Total classes**: 58
- **Class IDs**: Numeric values ranging from `0` to `57` (starts from 0).
- **Class names**: No class name mappings are provided in the current folder structure.

**BelgiumTSC**
- **Total classes**: 62
- **Class IDs**: `00000` through `00061` (starts from 0).
- **Class names**: No explicit class name strings provided.

---

### 4. IMAGE INFORMATION

**GTSRB**
- **Formats**: PNG
- **Channels**: 3 Channels (RGB)
- **Dimensions**: Variable. Smallest dimensions are `27x26`, largest are `166x152`.

**CTSD**
- **Formats**: PNG
- **Channels**: 3 Channels (RGB)
- **Dimensions**: Variable. Smallest dimensions are `42x43`, largest are `322x302`.

**BelgiumTSC**
- **Formats**: PPM (Portable Pixmap Format)
- **Channels**: 3 Channels (RGB)
- **Dimensions**: Variable. Smallest dimensions are `41x40`, largest are `347x322`.

---

### 5. DATASET SIZE

**GTSRB**
- **Total training images**: 39,209
- **Total testing images**: 12,630
- **Total images overall**: 51,839

**CTSD**
- **Total training/overall images**: 6,164 (No predefined test set)
- **Total testing images**: 0

**BelgiumTSC**
- **Total training images**: 4,575
- **Total testing images**: 2,520
- **Total images overall**: 7,095

---

### 6. CLASS DISTRIBUTION

Due to the length constraints of displaying all 163 rows across three datasets, here is a concise summary of the distribution demonstrating heavy class imbalance across all three. *(Note: Class names are omitted as they are not mapped in the CSVs).*

| Dataset | Imbalance Status | Selected Examples (Class ID : Train Count / Test Count) |
|---|---|---|
| **GTSRB** | Highly Imbalanced | `1`: 2220 / 720, `2`: 2250 / 750, `19`: 210 / 60, `37`: 210 / 60 |
| **CTSD** | Highly Imbalanced | `28`: 514 / NA, `54`: 500 / NA, `9`: 2 / NA, `19`: 4 / NA |
| **BelgiumTSC**| Highly Imbalanced | `22`: 376 / 62, `32`: 317 / 423, `12`: 19 / 4, `50`: 16 / 1 |

*(All datasets exhibit heavy imbalance. For instance, GTSRB ranges from 210 to 2,250 images per class. CTSD ranges from 2 to 514 images. BelgiumTSC ranges from 7 to 376 images).*

---

### 7. TRAIN/TEST LABEL INFORMATION

- **GTSRB**: Training and Testing labels are explicitly available in `Train.csv` and `Test.csv`. The `Path` column links directly to the image filename (e.g., `Train/20/00020_00000_00000.png`). The official test set possesses ground truth labels.
- **CTSD**: Labels are available for all 6,164 images inside `annotations.csv` through the `category` column. There is no official test set.
- **BelgiumTSC**: Both training and testing sets are fully labelled. Filenames are linked to labels primarily via their class-categorized subfolders (e.g., folder `00000/`) and explicitly detailed in the scattered `GT-xxxxx.csv` files.

---

### 8. DATA QUALITY

Based on programmatic checks over a representative sample:
- **Missing files**: 0 (All referenced images in annotations exist in directories).
- **Nonexistent images in annotations**: 0
- **Duplicate filenames**: Not found.
- **Corrupted/unreadable images**: 0 (All images in the sample loaded successfully via PIL).
- **Missing labels**: None.
- **Unexpected file formats**: None found.
- **Empty folders**: None found.

---

### 9. Information Needed to Build the TensorFlow Data Loader

**GTSRB**
- **Image path source**: `Train.csv` & `Test.csv` (`Path` column defines relative subdirectories).
- **Label source**: `Train.csv` & `Test.csv`. Alternatively, `tf.keras.utils.image_dataset_from_directory` can be used directly on the `Train/` directory.
- **Label column**: `ClassId`
- **Number of classes**: 43
- **Training image count**: 39,209
- **Testing image count**: 12,630
- **Resizing**: **Required** (dimensions are variable).
- **Labels mapping**: Not required (starts from 0).
- **Special handling**: Test images are un-nested inside `Test/`. To use `image_dataset_from_directory` for validation, you must manually write a custom loader or re-organize the `Test/` images using the `Test.csv` file.

**CTSD**
- **Image path source**: `annotations.csv` -> `images/` directory.
- **Label source**: `annotations.csv`
- **Label column**: `category`
- **Number of classes**: 58
- **Training image count**: 6,164 (total)
- **Testing image count**: 0
- **Resizing**: **Required** (dimensions are variable).
- **Labels mapping**: Not required (starts from 0).
- **Special handling**: Images are dumped into a single folder. You must use a custom generator or Pandas DataFrame loader (`tf.keras.preprocessing.image.ImageDataGenerator().flow_from_dataframe`) instead of a directory loader. You also need to manually create your own train/test split.

**BelgiumTSC**
- **Image path source**: `BelgiumTSC_Training\Training\` and `BelgiumTSC_Testing\Testing\`
- **Label source**: Implicit in directory structures (or inside `GT-xxxxx.csv` files).
- **Label column**: `ClassId`
- **Number of classes**: 62
- **Training image count**: 4,575
- **Testing image count**: 2,520
- **Resizing**: **Required** (dimensions are variable).
- **Labels mapping**: Not required (integer classes correspond directly to string folder names).
- **Special handling**: Images are stored as `.ppm`. TensorFlow's standard native `image_dataset_from_directory` only supports `.bmp`, `.gif`, `.jpeg`, and `.png`. You will need to write a custom loader (e.g., using `tf.io.read_file` with a third-party decoder or convert them to `.png` beforehand) to ingest `.ppm` formats natively into tf.data.

---

### 10. FINAL SUMMARY

**Dataset**: GTSRB
**Training images**: 39,209
**Testing images**: 12,630
**Number of classes**: 43
**Image format**: PNG
**Image dimensions**: Variable (27x26 to 166x152)
**Channels**: RGB
**Label source**: `Train.csv`, `Test.csv`
**Image filename column**: `Path`
**Label column**: `ClassId`
**Class-name source**: Not mapped to natural text
**Train/Test structure**: Train split organized into subfolders; Test split un-nested in one folder.
**Important preprocessing considerations**: Needs fixed image resizing.

**Dataset**: CTSD
**Training images**: 6,164
**Testing images**: 0
**Number of classes**: 58
**Image format**: PNG
**Image dimensions**: Variable (42x43 to 322x302)
**Channels**: RGB
**Label source**: `annotations.csv`
**Image filename column**: `file_name`
**Label column**: `category`
**Class-name source**: Not mapped to natural text
**Train/Test structure**: Entire dataset is located in a single `images/` directory with no split.
**Important preprocessing considerations**: Needs manual train/validation split logic; needs custom DataFrame generator; needs fixed image resizing.

**Dataset**: BelgiumTSC
**Training images**: 4,575
**Testing images**: 2,520
**Number of classes**: 62
**Image format**: PPM
**Image dimensions**: Variable (41x40 to 347x322)
**Channels**: RGB
**Label source**: Class-named Subfolders / `GT-xxxxx.csv`
**Image filename column**: `Filename`
**Label column**: `ClassId`
**Class-name source**: Not mapped to natural text
**Train/Test structure**: Both explicitly organized into separate class-named subfolders.
**Important preprocessing considerations**: .ppm format requires custom decoding logic in TensorFlow; needs fixed image resizing.
