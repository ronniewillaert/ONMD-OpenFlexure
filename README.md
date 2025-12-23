# ONMD-OpenFlexure
# Optical Nanomotion Detection (ONMD) with the OpenFlexure Microscope

This repository contains the acquisition software, analysis pipeline,
and documentation associated with the manuscript:

“Automated optical nanomotion detection enables low-cost rapid antimicrobial testing”
(Nature Methods, under consideration).

## Contents
- Software for video acquisition and onboard/offline ONMD analysis
- User manual and setup documentation
- Example datasets for testing and validation

## Hardware
The system is based on the OpenFlexure microscope platform and a Raspberry Pi.

## Documentation
- ONMD Onboard User Manual (PDF): documentation/ONMD Onboard User Manual.pdf
- PyONMD User Manual (PDF): documentation/PYONMD User Manual.pdf

# Analysis software

This folder contains analysis scripts for ONMD-OpenFlexure.

## PyONMD_Ana_02
- `PyONMD_Ana_02.py`
- `PyONMD_Ana_02_support.py`
- `PyONMD_Ana_02.tcl`
- `PyONMD_Ana_02.spec`

Dependencies are listed in `requirements.txt`. Some workflows may require `ffmpeg`
for video conversion (see the manuals in `documentation/`).

## Software

The acquisition and analysis software for optical nanomotion detection (ONMD)
is written in Python and supports both onboard execution on a Raspberry Pi
and offline analysis on standard desktop systems (macOS, Windows, Linux).

### Precompiled Windows executable

A standalone Windows executable of the PyONMD analysis software
(`PyONMD_Ana_02.exe`) is provided for users who do not wish to install
Python dependencies.

The executable is publicly available via Zenodo:
https://zenodo.org/records/17940783

The full source code, documentation, and example datasets are available
in this repository.

## Demonstration videos

- Sample preparation (MP4):  
  [Sample preparation video](documentation/videos/sample_preparation_movie.mp4)

- Example analysis videos (MP4):  
  [Example analysis videos](examples/videos/)


## Repository structure
- `software/` – acquisition and analysis scripts  
- `documentation/` – user manual and setup guides  
- `examples/` – example datasets and output files  

## License
This project is released under the MIT License.

## Citation
If you use this software, please cite the associated publication listed in `CITATION.cff`.

