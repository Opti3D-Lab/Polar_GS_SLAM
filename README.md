# P&#8203;olarimetric M&#8203;onocular G&#8203;aussian S&#8203;platting S&#8203;LAM for D&#8203;ense S&#8203;urface R&#8203;econstruction

This repository contains the official implementation for the paper: **P&#8203;olarimetric M&#8203;onocular G&#8203;aussian S&#8203;platting S&#8203;LAM**.

This system assumes a **Monocular** setup with polarization data input. The code is designed to be minimal and functional.

**Tested Platform:** Ubuntu 20.04 with NVIDIA GPU.

## **⚙️ Installation**

We provide a configured YAML file. Please ensure you have Anaconda or Miniconda installed.

\# Create the environment from the provided file  
conda env create \-f environment.yml

\# Activate the environment  
conda activate PolarGS

*If you encounter issues with specific build versions (e.g., hash mismatches), you may need to relax the version constraints in the yaml file.*

## **🔨 Preprocessing (Mandatory)**

Before running the SLAM system, you **must** generate plane segmentation results from the polarization images. This is a prerequisite for the pipeline.

Run the following script to generate the segmentation masks:

python Pol\_seg/Multi\_GPU\_HDBSCAN.py

This will produce the segmentation results required for the seg.txt association described below.

## **📂 Data Preparation**

The system uses a TUM-like file association format. Please organize your data folder as follows:

datasets/  
└── \<scene\_name\>/  
    ├── images/         \# RGB images  
    ├── polarization/   \# Polarization data (containing AoLP, DoLP files)  
    ├── segmentation/   \# Output from the preprocessing step  
    ├── rgb.txt         \# List of RGB images  
    ├── aolp.txt        \# List of AoLP images  
    ├── dolp.txt        \# List of DoLP images  
    └── seg.txt         \# List of Segmentation masks (from preprocessing)

### **File Format Requirements**

The .txt files should follow the standard format: timestamp relative/path/to/image.png.

1. **Synchronization**: rgb.txt, aolp.txt, and dolp.txt are assumed to be strictly synchronized (same number of lines, same timestamps).  
2. **Association**: The system will automatically associate seg.txt (and depth.txt if available) with the RGB frames based on timestamps.  
   * **Max delay:** The association allows a maximum time difference of **0.08s**.  
   * Ensure your seg.txt timestamps are close enough to your rgb.txt timestamps.
## **🚀 Run**

To run the system:

python slam.py \--config configs/polar/mono/my.yaml

If the environment is set up correctly, the GUI window should appear immediately.
