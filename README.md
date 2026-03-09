# Flight Delay Prediction Project

This repository implements a machine learning pipeline to predict U.S. flight delays based on weather conditions, following the methodology of the paper *"Using Scalable Data Mining for Predicting Flight Delays"*.

## Dataset Setup

The dataset (approx. 5GB) is not included in this repository due to size limits. You can download and set up the data automatically using the following commands:

```bash
# Create the Data directory if it doesn't exist
mkdir -p Data

# Download the dataset from Dropbox (Direct Link)
curl -L -o Data/flights_data.zip "https://www.dropbox.com/sh/iasq7frk6f58ptq/AAAzSmk6cusSNfqYNYsnLGIXa?dl=1"

# Unzip the archive into the Data folder
unzip Data/flights_data.zip -d Data/

# Clean up
rm Data/flights_data.zip
```

## Getting Started

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/nicotranfr/Using-Scalable-Data-Mining-for-Predicting-U.S.-Flight-Delays.git
    cd Using-Scalable-Data-Mining-for-Predicting-U.S.-Flight-Delays
    ```
2.  **Install dependencies**:
    ```bash
    pip install pandas numpy scikit-learn matplotlib
    ```
3.  **Run the analysis**:
    Open `main.ipynb` in Jupyter Notebook or VS Code to follow the research pipeline.

## PySpark cleaning before join (FT / OT)

A ready-to-run PySpark script is available in `pyspark_ft_ot_cleaning.py` to:
- build `FT` by removing cancelled/diverted flights and creating explicit columns (snake_case),
- build `OT` by keeping only weather rows mapped to an airport through `WBAN -> AirportID`,
- create Spark timestamps from `HHMM` fields with proper zero-padding.

Run:
```bash
python pyspark_ft_ot_cleaning.py
```
