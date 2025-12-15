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
- User manual (PDF): documentation/PyONMD_User_Manual.pdf
- Setup guide: documentation/Installation.md

## Software
The acquisition and analysis software is written in Python and supports both onboard
execution on a Raspberry Pi and offline analysis on standard desktop systems
(macOS, Windows, Linux). Dependencies and usage instructions are provided in the
documentation directory.

## Repository structure
- `software/` – acquisition and analysis scripts  
- `documentation/` – user manual and setup guides  
- `examples/` – example datasets and output files  

## License
This project is released under the MIT License.

## Citation
If you use this software, please cite the associated publication listed in `CITATION.cff`.

