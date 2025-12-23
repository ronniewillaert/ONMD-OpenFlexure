# ONMD-OpenFlexure
# Optical Nanomotion Detection (ONMD) with the OpenFlexure Microscope

This repository contains the acquisition software, analysis pipeline,
and documentation associated with the manuscript:

“Automated optical nanomotion detection enables low-cost rapid antimicrobial testing”
(Nature Methods, under consideration).

## Contents

- Onboard acquisition software for real-time ONMD measurements
- Offline analysis software (PyONMD) for post-processing and quantification
- User manuals and setup documentation
- Example datasets and demonstration videos

## Hardware
The system is based on the OpenFlexure microscope platform (openflexure.org) and a Raspberry Pi.

## Documentation

- **ONMD Onboard User Manual (PDF)**  
  `documentation/ONMD Onboard User Manual.pdf`

- **PyONMD User Manual (PDF)**  
  `documentation/PYONMD User Manual.pdf`

## Software

The ONMD software stack consists of two complementary components:
(1) real-time onboard acquisition on the OpenFlexure microscope, and
(2) offline analysis on standard desktop systems.

### Onboard acquisition (OpenFlexure / Raspberry Pi)

- `software/onboard/nanomotion_extension.py`

Python extension running directly on the OpenFlexure microscope
(Raspberry Pi) for real-time optical nanomotion detection during
image acquisition.

See the *ONMD Onboard User Manual* for installation and usage.

### Offline analysis (PyONMD)

- `software/analysis/PyONMD_Ana_02/`

Python-based ONMD analysis software for post-processing recorded videos
on macOS, Windows, and Linux systems.

Main components:
- `PyONMD_Ana_02.py`
- `PyONMD_Ana_02_support.py`
- `PyONMD_Ana_02.tcl`
- `PyONMD_Ana_02.spec`

Dependencies are listed in `requirements.txt`. Some workflows may require
`ffmpeg` for video conversion (see manuals in `documentation/`).

#### Precompiled Windows executable (PyONMD)

A standalone Windows executable of the PyONMD analysis software
(`PyONMD_Ana_02.exe`) is available via Zenodo:

https://zenodo.org/records/17940783

This binary is provided for user convenience. The full source code
remains available in this repository.

## Demonstration videos

Videos are hosted directly in the repository and can be viewed or
downloaded via GitHub.

- **Sample preparation (MP4)**  
  [Sample preparation video](documentation/videos/sample_preparation_movie.mp4)

- **Example analysis videos (MP4)**  
  [Example analysis videos](examples/videos/)


## Repository structure

- `software/` – onboard acquisition and offline analysis software
- `documentation/` – user manuals and setup guides
- `examples/` – example datasets and output files

## License
This project is released under the MIT License.

## Citation
If you use this software, please cite the associated publication listed in `CITATION.cff`.

