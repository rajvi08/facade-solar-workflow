# Vertical Façade Solar Analysis Workflow  
### *A Geometry-Aware and Cloud-Adjusted Solar Assessment Pipeline*

This repository presents a **reproducible computational workflow** for analyzing the **solar energy potential of vertical building façades**. The workflow integrates **deterministic solar geometry**, **urban shading analysis**, and **cloud-adjusted irradiance** to support **façade-level comparative evaluation** in dense urban environments.

The implementation is designed for **Google Colab** and is intended as a **design-support and early-stage feasibility tool**, rather than a prediction of absolute building energy sufficiency.

---

##  Execution Environment

This workflow is implemented and tested in **Google Colab**.

**If running locally, ensure:**
- *Python 3.9+*
- *Jupyter Notebook or JupyterLab*

Each cell explicitly installs or imports the required dependencies.

---

##  Workflow Overview

The complete pipeline follows a structured sequence:

1. Location and date selection  
2. Solar position calculation  
3. Urban shadow evaluation  
4. Cloud-adjusted irradiance retrieval (SoDa / NSRDB)  
5. Façade-level power computation  
6. Daily energy aggregation and visualization  

---

##  CELL 1 — Setup & Solar Position Utilities

This cell installs core Python libraries and defines **solar position utilities**.  
Solar azimuth and altitude are computed using **pvlib’s NREL Solar Position Algorithm**, providing a deterministic and physics-based description of Earth–Sun geometry.

```python
!pip -q install numpy matplotlib ipywidgets pvlib pandas

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ipywidgets as widgets

from pvlib.solarposition import get_solarposition

plt.rcParams["figure.figsize"] = (6, 6)
plt.rcParams["axes.grid"] = True

LOCATIONS = {
    "NYC":   {"lat": 40.7128, "lon": -74.0060, "tz": "America/New_York"},
    "Pune":  {"lat": 18.5204, "lon": 73.8567,  "tz": "Asia/Kolkata"},
    "Paris": {"lat": 48.8566, "lon": 2.3522,   "tz": "Europe/Paris"},
}

def get_sun_position(location, date, hour, minute):
    meta = LOCATIONS[location]
    t = pd.Timestamp(f"{date} {hour:02d}:{minute:02d}", tz=meta["tz"])
    sp = get_solarposition(t, meta["lat"], meta["lon"], method="nrel_numpy")
    return float(sp["azimuth"].iloc[0]), float(sp["apparent_elevation"].iloc[0])

print("Cell 1 ready.")

---

---


