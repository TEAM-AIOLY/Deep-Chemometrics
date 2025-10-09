# Citrus Multispectral Dataset for HLB Detection  

## Overview  
This dataset contains a collection of **multispectral images with 14 Vis–NIR bands** acquired *in situ* in a commercial citrus orchard in São Paulo state, Brazil. The images capture **large canopy portions (~3 m²)** and were designed to study **Huanglongbing (HLB) disease** detection in orchards using computer vision, machine learning, and deep learning approaches.  

The dataset is formatted for **classification tasks** (HLB-symptomatic vs. non-HLB-symptomatic) and is divided into two experimental plots collected at different dates.  

---

## Dataset Structure  

### Plots  
- **Plot A**  
  - Location: 9 ha orchard of sweet orange *Pera Rio* (*Citrus sinensis* (L.) Osbeck)  
  - Age: 3 years old  
  - Acquisition period: **15–18 August 2023**  
  - Number of images: **1,297**  

- **Plot B**  
  - Acquisition period: **18–20 November 2024**  
  - Number of images: **1,681**  

### Classes  
- `HLB-symptomatic`  
- `Non-HLB-symptomatic`  

### Data Formats  
Two formats are provided:  

1. **Raw format**  
   - Each multispectral image is a folder containing **14 `.tiff` files** (one per band).  

2. **Datacube format**  
   - Each multispectral image is a single **`.h5` file** storing the spectral cube in the shape:  
     ```
     channels × width × height
     ```  
   - Example: `PlotA_datacubes_hlb.zip` contains all symptomatic images from Plot A in HDF5 format.  

> ⚠️ For convenience and to keep folder sizes under ~50 GB, some folders may be split into parts using `_part_` suffixes.  

### Example Folder Organization  

Dataset/
│
├── PlotA/
│ ├── Raw/
│ │ ├── HLB-symptomatic/
│ │ │ ├── IMG_001/
│ │ │ │ ├── band_405.tiff
│ │ │ │ ├── band_430.tiff
│ │ │ │ └── ... (14 bands)
│ │ │ └── IMG_002/ ...
│ │ └── Non-HLB-symptomatic/
│ │ └── IMG_101/ ...
│ │
│ └── Datacube/
│ ├── HLB-symptomatic/
│ │ ├── IMG_001.h5
│ │ ├── IMG_002.h5
│ │ └── ...
│ └── Non-HLB-symptomatic/
│ ├── IMG_101.h5
│ └── ...
│
└── PlotB/
├── Raw/
│ ├── HLB-symptomatic/ ...
│ └── Non-HLB-symptomatic/ ...
│
└── Datacube/
├── HLB-symptomatic/ ...
└── Non-HLB-symptomatic/ ...