Country Relocation Recommender

Overview
The Country Relocation Recommender is an interactive dashboard that evaluates and ranks countries for relocation based on different user goals:

* Tourism
* Startup / Business
* Corporate Employment
* Family
* Education

It combines demographic, economic, infrastructure, and governance indicators to compute purpose-specific suitability scores and visualize them in an interactive web application. This is a decision-support tool focused on transparency and interpretability rather than predictive modeling.

Setup

1. Install dependencies

pip install dash pandas numpy plotly scipy functools

2. Data

Create a folder named data in the project directory and place the following CSV files inside it:

* communications_data_cluster_imputed.csv
* demographics_data_cluster_imputed.csv
* economy_data_cluster_imputed.csv
* energy_data_cluster_imputed.csv
* geography_data_cluster_imputed.csv
* government_and_civics_data_cluster_imputed.csv
* transportation_data_cluster_imputed.csv

3. Run

python app.py

Open the local URL shown in the terminal.

Methodology

* Purpose-specific weighted scoring models
* Robust percentile-based normalization
* Per-capita and composite indicators
* Inversion of negative indicators such as mortality, unemployment, and emissions
* Data-quality adjustment based on completeness
* Market-size adjustment for selected purposes
* k-means clustering and PCA for similarity analysis

Authorship
* Scoring framework and weight design
* Feature engineering and normalization pipeline
* Data-quality and market-size adjustments
* Government-type encoding logic
* Clustering and PCA pipeline
* Dashboard design and interactivity

External libraries used:

* Dash and Plotly for visualization and UI
* Pandas and NumPy for data processing
* SciPy for statistical utilities
* functools (Python standard library) for caching with lru_cache

