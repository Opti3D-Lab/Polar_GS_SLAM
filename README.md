# Polarimetric Monocular Gaussian Splatting SLAM for Dense Surface Reconstruction

<p align="center">
  Haitao Wang · [Sijia Wen](https://wensijia.work/) · Bo Guo
</p>

<h2 align="center">
  ACM MM 2025
</h2>

### [Paper](https://doi.org/10.1145/3746027.3754925) | Project Page (Coming Soon) | Dataset (Coming Soon)

This repository contains the official implementation of **Polarimetric Monocular Gaussian Splatting SLAM for Dense Surface Reconstruction**, accepted by **ACM MM 2025**.

We propose a polarimetric monocular 3D Gaussian Splatting SLAM framework for dense surface reconstruction. By introducing polarization cues into monocular Gaussian Splatting SLAM, our method improves localization and mapping quality in challenging indoor scenes with weak textures, dark colors, and reflective surfaces.

This system assumes a **monocular** setup with polarization data input. The code is designed to be minimal and functional.

**Tested Platform:** Ubuntu 20.04 with NVIDIA GPU.

## **⚙️ Installation**

We provide a configured YAML file. Please ensure you have Anaconda or Miniconda installed.

```bash
# Create the environment from the provided file
conda env create -f environment.yml

# Activate the environment
conda activate PolarGS
```

*If you encounter issues with specific build versions, such as hash mismatches, you may need to relax the version constraints in the YAML file.*

## **🔨 Preprocessing (Mandatory)**

Before running the SLAM system, you **must** generate plane segmentation results from the polarization images. This is a prerequisite for the pipeline.

Run the following script to generate the segmentation masks:

```bash
python Pol_seg/Multi_GPU_HDBSCAN.py
```

This will produce the segmentation results required for the `seg.txt` association described below.

## **📂 Data Preparation**

The system uses a TUM-like file association format. Please organize your data folder as follows:

```text
datasets/
└── <scene_name>/
    ├── images/         # RGB images
    ├── polarization/   # Polarization data containing AoLP and DoLP files
    ├── segmentation/   # Output from the preprocessing step
    ├── rgb.txt         # List of RGB images
    ├── aolp.txt        # List of AoLP images
    ├── dolp.txt        # List of DoLP images
    └── seg.txt         # List of segmentation masks from preprocessing
```

### **File Format Requirements**

The `.txt` files should follow the standard format:

```text
timestamp relative/path/to/image.png
```

1. **Synchronization:** `rgb.txt`, `aolp.txt`, and `dolp.txt` are assumed to be strictly synchronized, with the same number of lines and the same timestamps.
2. **Association:** The system will automatically associate `seg.txt`, and `depth.txt` if available, with the RGB frames based on timestamps.
   - **Max delay:** The association allows a maximum time difference of **0.08s**.
   - Please ensure that the timestamps in `seg.txt` are close enough to the timestamps in `rgb.txt`.

## **🚀 Run**

To run the system:

```bash
python slam.py --config configs/polar/mono/my.yaml
```

If the environment is set up correctly, the GUI window should appear immediately.

## Citation

If you find our work useful, please consider citing:

```bibtex
@inproceedings{wang2025polarimetric,
  title     = {Polarimetric Monocular Gaussian Splatting SLAM for Dense Surface Reconstruction},
  author    = {Wang, Haitao and Wen, Sijia and Guo, Bo},
  booktitle = {Proceedings of the 33rd ACM International Conference on Multimedia},
  pages     = {7519--7528},
  year      = {2025},
  publisher = {Association for Computing Machinery},
  doi       = {10.1145/3746027.3754925}
}
```
