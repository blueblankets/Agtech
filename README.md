Agricultural Soil Compaction Detection & Prescription Mapping

The Problem

Agricultural soil compaction is a "silent killer" of crop yields, increasing soil strength and reducing water infiltration. Traditional methods for addressing this rely on uniform deep tillage, which is fuel-intensive and often unnecessary for the entire field. Furthermore, current precision agriculture stacks depend on expensive, proprietary APIs—such as Google Earth Engine and John Deere Ops Center—that create high barriers to entry for smaller operations.

This project provides a physics-informed, open-source alternative. By combining satellite imagery, global soil databases, and tractor physics, we generate a precision "Targeted Deep Tillage" map that tells a grower exactly where to rip, to what depth, and whether the intervention is economically viable.

The Technical Approach

The system is implemented as a 6-stage geospatial ML pipeline divided across three engineering roles.

1\. Data Ingestion & Alignment 

 \* Asynchronous Streams: Uses asyncio to concurrently pull data from three sources: NDVI from the Copernicus Data Space (CDSE), soil mechanics from ISRIC SoilGrids, and synthetic telemetry based on ASABE D497.5.

 \* Spatial Join: Reprojects all data to EPSG:3857 for meter-accurate Cartesian joins, ensuring 10m x 10m pixels align perfectly with tractor passes.

2\. Physics & ML Inference 

 \* Söhne Physics Model: Calculates peak vertical stress (MPa) at varying depths based on equipment weight, tire width, and soil bulk density.

 \* Uncertainty Quantification: An XGBoost regressor wrapped in MAPIE predicts the required ripper depth with 90% confidence intervals.

 \* Economic Filter: Calculates ROI based on avoided yield loss vs. tillage cost. Tillage is only recommended if ROI > 1.2.

3\. Full-Stack / UX

 \* Interactive Frontend: A Streamlit dashboard allows users to draw field boundaries and triggers the backend via a RESTful API.

 \* Choropleth Visualization: Renders a 10\\text{m} resolution map where pixels are color-coded by action: Targeted Tillage (Red), Monitor (Amber), or None (Green).

Results & Demonstration Outputs

 \* Precision Prescription: The pipeline outputs a final\_payload.json containing specific ripper depths and confidence bounds for every pixel in the field.

 \* Summary Analytics: The dashboard provides an acreage breakdown. Each pixel represents 0.0247 acres (10m²), allowing the user to see exactly what percentage of the field requires intervention.

 \* GIS Export: Users can download a zipped ESRI Shapefile bundle containing the prescription data for use in tractor GPS displays.

Instructions for Running the Prototype

Prerequisites

 \* Python 3.9+

 \* A free CDSE (Copernicus) account for Sentinel-2 data.

Installation

 \* Clone the repository and install dependencies:

   pip install flask geopandas xgboost mapie openeo owslib

 \* Configure your config.yaml with your CDSE OIDC credentials and API flags.

Running the App

Launch the Flask dashboard:

flask run engineer\_c/app.py

"Judge Mode" (Fallback System)

To ensure the prototype remains functional for demonstrations even if external APIs are offline, we have implemented Judge Mode.

 \* If an API timeout occurs (>30s), the system automatically pulls physically plausible data from the mock\_data/ directory.

 \* The manifest.json will flag which layers are "live" vs. "mock" for full transparency.

 Demo Video:
 https://drive.google.com/file/d/1vyipW1A_UNMyp3WZwhoAAYQsu_BAMLC9/view?usp=sharing
