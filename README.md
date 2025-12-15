Façade Solar Analysis Workflow

Geometry-Aware, Cloud-Adjusted Solar Potential Evaluation

This repository provides a complete, reproducible Google Colab workflow for analyzing the solar energy potential of vertical building façades. The pipeline integrates solar geometry, urban shading, and cloud-adjusted irradiance to evaluate façade-level power and energy generation at a 30-minute temporal resolution.

The workflow accompanies a research study focused on methodology and design support, rather than absolute energy prediction, enabling architects and urban designers to evaluate orientation-dependent façade performance in dense urban contexts.

Execution Environment

This workflow is implemented and tested in Google Colab.

Recommended: Google Colab (Python 3 runtime)

No local setup required

All dependencies are installed directly inside the notebook cells

Each cell can be run sequentially to reproduce the full analysis.

Workflow Overview

The computational pipeline follows these steps:

Location and date selection

Solar position calculation (azimuth and altitude)

Geometry-based urban shadow evaluation

Cloud-adjusted irradiance retrieval (SoDa / NSRDB)

Façade-level instantaneous power computation

Daily energy aggregation and visualization

CELL 1 — Setup & Solar Position Utilities

Installs core libraries and defines solar position utilities using pvlib (NREL SPA).
