import os
import pandas as pd
import numpy as np
from dash import Dash, html, dcc, Input, Output, State, no_update
import plotly.express as px
import plotly.graph_objects as go
from functools import lru_cache
from scipy import stats

# =========================
# Constants
# =========================

BASE_DIR = os.path.join(os.path.dirname(__file__), "data")
KEY_COL = "Country"
MAX_RADAR_COUNTRIES = 3
TOP_N_COUNTRIES = 10
PARALLEL_TOP_N = 60
PURPOSES = ["Tourism", "Startup", "Corporate", "Family", "Education"]
MAP_LAT_RANGE = [-60, 90]
MAP_LON_RANGE = [-180, 180]
SMALL_MARKET_PURPOSES = {"Startup", "Corporate", "Education"}
CLUSTER_COUNT = 3
CLUSTER_RANDOM_SEED = 13
CLUSTER_MAX_ITER = 75

# Updated filenames to match cluster imputation output
CSV_FILES = [
    "communications_data_cluster_imputed.csv",
    "demographics_data_cluster_imputed.csv",
    "economy_data_cluster_imputed.csv",
    "energy_data_cluster_imputed.csv",
    "geography_data_cluster_imputed.csv",
    "government_and_civics_data_cluster_imputed.csv",
    "transportation_data_cluster_imputed.csv",
]

# =========================
# 1. Load & merge CIA data
# =========================

def load_and_merge_data():
    """Load and merge all CSV files with error handling."""
    dfs = []
    for fname in CSV_FILES:
        filepath = os.path.join(BASE_DIR, fname)
        if not os.path.exists(filepath):
            print(f"Warning: {fname} not found, skipping...")
            continue
        try:
            df_i = pd.read_csv(filepath)
            dfs.append(df_i)
            print(f"Loaded: {fname} ({len(df_i)} rows)")
        except Exception as e:
            print(f"Error loading {fname}: {e}")
            continue
    
    if not dfs:
        raise ValueError("No CSV files could be loaded")
    
    df = dfs[0]
    for other in dfs[1:]:
        df = df.merge(other, on=KEY_COL, how="outer")
    
    df = df.dropna(subset=[KEY_COL])
    df[KEY_COL] = df[KEY_COL].astype(str).str.strip()
    
    # Filter out non-country entities
    df = df[
        ~df[KEY_COL].str.contains(
            "european union|^eu$|international|world",
            case=False,
            regex=True,
            na=False
        )
    ]
    
    # Check for duplicates
    duplicates = df[df.duplicated(subset=[KEY_COL], keep=False)]
    if not duplicates.empty:
        print(f"Warning: Found {len(duplicates)} duplicate countries")
        df = df.drop_duplicates(subset=[KEY_COL], keep='first')
    
    return df

df = load_and_merge_data()

# =========================
# 2. Pre-processing & features
# =========================

def to_num(series):
    """Convert series to numeric, handling common formatting issues."""
    return pd.to_numeric(
        series.astype(str).str.replace(",", "").str.replace(" ", ""),
        errors="coerce"
    )

numeric_cols = [
    "telephone_fixed_subscriptions_total",
    "mobile_cellular_subscriptions_total",
    "internet_users_total",
    "broadband_fixed_subscriptions_total",
    "Total_Population",
    "Population_Growth_Rate",
    "Birth_Rate",
    "Death_Rate",
    "Net_Migration_Rate",
    "Median_Age",
    "Sex_Ratio",
    "Infant_Mortality_Rate",
    "Total_Fertility_Rate",
    "Total_Literacy_Rate",
    "Male_Literacy_Rate",
    "Female_Literacy_Rate",
    "Youth_Unemployment_Rate_percent",
    "Unemployment_Rate_percent",
    "Real_GDP_PPP_billion_USD",
    "Real_GDP_per_Capita_USD",
    "Real_GDP_Growth_Rate_percent",
    "carbon_dioxide_emissions_Mt",
    "Coastline",
    "roadways_km",
    "railways_km",
    "waterways_km",
    "gas_pipelines_km",
    "oil_pipelines_km",
    "airports_paved_runways_count",
    "Suffrage_Age",
]

for col in numeric_cols:
    if col in df.columns:
        df[col + "_num"] = to_num(df[col])

# Additional safety checks
if "Total_Population_num" not in df.columns and "Total_Population" in df.columns:
    df["Total_Population_num"] = to_num(df["Total_Population"])

for col in ["internet_users_total", "mobile_cellular_subscriptions_total"]:
    if col in df.columns and col + "_num" not in df.columns:
        df[col + "_num"] = to_num(df[col])

# Create per-capita metrics for better comparability
if "Total_Population_num" in df.columns:
    if "internet_users_total_num" in df.columns:
        df["internet_per_capita"] = df["internet_users_total_num"] / df["Total_Population_num"]
    if "mobile_cellular_subscriptions_total_num" in df.columns:
        df["mobile_per_capita"] = df["mobile_cellular_subscriptions_total_num"] / df["Total_Population_num"]

# Create infrastructure composite score (handles multicollinearity)
infrastructure_features = []
for feat in ["roadways_km_num", "railways_km_num", "waterways_km_num"]:
    if feat in df.columns:
        infrastructure_features.append(feat)

if infrastructure_features:
    # Normalize each and average
    infra_normed = pd.DataFrame()
    for feat in infrastructure_features:
        vals = df[feat].fillna(df[feat].median())
        infra_normed[feat] = (vals - vals.min()) / (vals.max() - vals.min() + 1e-9)
    df["infrastructure_composite"] = infra_normed.mean(axis=1)

# Non-linear sex balance score with sigmoid
if "Sex_Ratio_num" in df.columns:
    sex = df["Sex_Ratio_num"].fillna(1.0)
    deviation = (sex - 1.0).abs()
    # Sigmoid: closer to 0 deviation = score closer to 1
    df["sex_balance_score"] = 1 / (1 + np.exp(5 * (deviation - 0.05)))
else:
    df["sex_balance_score"] = 0.5

# Inverted metrics: make "higher is better" for all features
if "Death_Rate_num" in df.columns:
    dr = df["Death_Rate_num"].fillna(df["Death_Rate_num"].median())
    df["health_score"] = 1 / (1 + dr / 10)  # normalized death rate inverse

if "Infant_Mortality_Rate_num" in df.columns:
    im = df["Infant_Mortality_Rate_num"].fillna(df["Infant_Mortality_Rate_num"].median())
    df["infant_health_score"] = 1 / (1 + im / 20)

if "Unemployment_Rate_percent_num" in df.columns:
    unemp = df["Unemployment_Rate_percent_num"].fillna(df["Unemployment_Rate_percent_num"].median())
    df["employment_score"] = 1 / (1 + unemp / 10)

if "Youth_Unemployment_Rate_percent_num" in df.columns:
    yunemp = df["Youth_Unemployment_Rate_percent_num"].fillna(df["Youth_Unemployment_Rate_percent_num"].median())
    df["youth_employment_score"] = 1 / (1 + yunemp / 10)

if "carbon_dioxide_emissions_Mt_num" in df.columns:
    co2 = df["carbon_dioxide_emissions_Mt_num"].fillna(df["carbon_dioxide_emissions_Mt_num"].median())
    df["environmental_score"] = 1 / (1 + co2 / 1000)

def encode_government_type(value: str) -> float:
    """Encode government type as numeric score - improved validation."""
    if not isinstance(value, str):
        return 0.5
    v = value.lower()
    
    # Negative keywords override positive ones (e.g., "Democratic People's Republic")
    if any(word in v for word in ["authoritarian", "dictatorship", "absolute monarchy"]):
        return 0.2
    
    # Check for positive governance
    if "democracy" in v or "democratic" in v:
        # Verify it's not a false positive
        if "people's" in v or "socialist" in v:
            return 0.3
        return 0.95
    if "constitutional monarchy" in v or "parliamentary monarchy" in v:
        return 0.85
    if "republic" in v and "islamic" not in v:
        return 0.75
    if "monarchy" in v:
        return 0.6
    
    return 0.5

if "Government_Type" in df.columns:
    df["government_score"] = df["Government_Type"].apply(encode_government_type)
else:
    df["government_score"] = 0.5

# Calculate data quality score per country
def calculate_data_quality(row, feature_cols):
    """Calculate percentage of non-missing values."""
    return row[feature_cols].notna().sum() / len(feature_cols)

# =========================
# 3. Scoring model
# =========================

NUMERIC_COLUMNS = {
    "telephone_fixed": "telephone_fixed_subscriptions_total_num",
    "mobile_cellular": "mobile_cellular_subscriptions_total_num",
    "internet_users": "internet_users_total_num",
    "internet_per_capita": "internet_per_capita",
    "mobile_per_capita": "mobile_per_capita",
    "broadband_fixed": "broadband_fixed_subscriptions_total_num",
    "total_population": "Total_Population_num",
    "population_growth": "Population_Growth_Rate_num",
    "birth_rate": "Birth_Rate_num",
    "death_rate": "Death_Rate_num",
    "health_score": "health_score",
    "net_migration": "Net_Migration_Rate_num",
    "median_age": "Median_Age_num",
    "sex_balance": "sex_balance_score",
    "infant_mortality": "Infant_Mortality_Rate_num",
    "infant_health": "infant_health_score",
    "fertility": "Total_Fertility_Rate_num",
    "total_literacy": "Total_Literacy_Rate_num",
    "male_literacy": "Male_Literacy_Rate_num",
    "female_literacy": "Female_Literacy_Rate_num",
    "unemployment": "Unemployment_Rate_percent_num",
    "employment_score": "employment_score",
    "youth_unemployment": "Youth_Unemployment_Rate_percent_num",
    "youth_employment": "youth_employment_score",
    "gdp_ppp": "Real_GDP_PPP_billion_USD_num",
    "gdp_per_capita": "Real_GDP_per_Capita_USD_num",
    "gdp_growth": "Real_GDP_Growth_Rate_percent_num",
    "co2_emissions": "carbon_dioxide_emissions_Mt_num",
    "environmental_score": "environmental_score",
    "coastline": "Coastline_num",
    "infrastructure": "infrastructure_composite",
    "roadways": "roadways_km_num",
    "railways": "railways_km_num",
    "waterways": "waterways_km_num",
    "gas_pipelines": "gas_pipelines_km_num",
    "oil_pipelines": "oil_pipelines_km_num",
    "airports_paved": "airports_paved_runways_count_num",
    "government": "government_score",
    "suffrage_age": "Suffrage_Age_num",
}

# IMPROVED: Weights now sum to 1.0, using transformed features
PURPOSE_CONFIG = {
    "Tourism": {
        "features": {
            "airports_paved": 0.25,
            "coastline": 0.20,
            "infrastructure": 0.15,  # composite instead of individual roads
            "internet_per_capita": 0.12,
            "mobile_per_capita": 0.10,
            "broadband_fixed": 0.08,
            "environmental_score": 0.10,  # inverted CO2
        },
    },
    "Startup": {
        "features": {
            "government": 0.20,
            "gdp_per_capita": 0.16,
            "gdp_ppp": 0.12,
            "total_population": 0.08,
            "gdp_growth": 0.12,
            "total_literacy": 0.10,
            "internet_per_capita": 0.08,
            "broadband_fixed": 0.06,
            "employment_score": 0.08,  # inverted unemployment
        },
    },
    "Corporate": {
        "features": {
            "government": 0.18,
            "employment_score": 0.18,  # inverted unemployment
            "gdp_ppp": 0.12,
            "gdp_per_capita": 0.10,
            "internet_per_capita": 0.10,
            "broadband_fixed": 0.08,
            "airports_paved": 0.10,
            "infrastructure": 0.08,
            "median_age": 0.06,
        },
    },
    "Family": {
        "features": {
            "government": 0.20,
            "infant_health": 0.20,  # inverted infant mortality
            "health_score": 0.15,   # inverted death rate
            "sex_balance": 0.15,
            "fertility": 0.10,
            "employment_score": 0.10,
            "environmental_score": 0.10,
        },
    },
    "Education": {
        "features": {
            "total_literacy": 0.22,
            "youth_employment": 0.16,  # inverted youth unemployment
            "gdp_per_capita": 0.14,
            "gdp_ppp": 0.10,
            "internet_per_capita": 0.10,
            "government": 0.12,
            "infrastructure": 0.08,
            "airports_paved": 0.04,
            "broadband_fixed": 0.04,
        },
    },
}

def normalize_column_robust(series, clip_percentile=95):
    """
    Robust normalization with outlier clipping.
    Uses percentile-based clipping to prevent extreme outliers from dominating.
    """
    series = pd.to_numeric(series, errors="coerce")
    series_clean = series.dropna()
    
    if len(series_clean) == 0:
        return pd.Series(0, index=series.index)
    
    # Clip at percentile to handle outliers
    upper_bound = series_clean.quantile(clip_percentile / 100)
    lower_bound = series_clean.quantile((100 - clip_percentile) / 100)
    
    series_clipped = series.clip(lower_bound, upper_bound)
    
    min_val = series_clipped.min()
    max_val = series_clipped.max()
    
    if pd.isna(min_val) or pd.isna(max_val) or max_val == min_val:
        return pd.Series(0, index=series.index)
    
    return (series_clipped - min_val) / (max_val - min_val)

def compute_market_size_factor(df_local):
    """Down-weight tiny markets using population and GDP PPP (log-scaled)."""
    pop = df_local.get("Total_Population_num")
    gdp = df_local.get("Real_GDP_PPP_billion_USD_num")

    if pop is None and gdp is None:
        return pd.Series(1.0, index=df_local.index)

    if pop is None:
        base = pd.to_numeric(gdp, errors="coerce")
        base = base.fillna(base.median())
        base = np.log1p(base)
    elif gdp is None:
        base = pd.to_numeric(pop, errors="coerce")
        base = base.fillna(base.median())
        base = np.log1p(base)
    else:
        pop_num = pd.to_numeric(pop, errors="coerce")
        gdp_num = pd.to_numeric(gdp, errors="coerce")
        pop_num = pop_num.fillna(pop_num.median())
        gdp_num = gdp_num.fillna(gdp_num.median())
        base = 0.6 * np.log1p(pop_num) + 0.4 * np.log1p(gdp_num)

    if base.dropna().empty:
        return pd.Series(1.0, index=df_local.index)

    normalized = normalize_column_robust(base)
    return 0.4 + 0.6 * normalized

def compute_scores(data, purpose):
    """
    Compute suitability scores with statistical improvements:
    - Robust normalization with outlier handling
    - Single normalization (no double normalization)
    - Data quality tracking
    - Use of pre-transformed features
    """
    cfg = PURPOSE_CONFIG[purpose]
    df_local = data.copy()
    
    feature_keys = list(cfg["features"].keys())
    feature_cols = [NUMERIC_COLUMNS[k] for k in feature_keys if k in NUMERIC_COLUMNS]
    
    # Calculate data quality
    df_local["data_quality"] = df_local[feature_cols].notna().sum(axis=1) / len(feature_cols)
    
    # Normalize features with robust method
    for key in feature_keys:
        if key not in NUMERIC_COLUMNS:
            continue
        col_name = NUMERIC_COLUMNS[key]
        if col_name not in df_local.columns:
            df_local[col_name] = 0.0
        df_local[f"norm_{col_name}"] = normalize_column_robust(df_local[col_name])
    
    # Compute weighted score (NO second normalization)
    score = pd.Series(0.0, index=df_local.index)
    for key, weight in cfg["features"].items():
        if key not in NUMERIC_COLUMNS:
            continue
        col_name = NUMERIC_COLUMNS[key]
        score = score + weight * df_local[f"norm_{col_name}"]
    
    df_local["score"] = score
    
    # Penalize low data quality
    market_size_factor = pd.Series(1.0, index=df_local.index)
    if purpose in SMALL_MARKET_PURPOSES:
        market_size_factor = compute_market_size_factor(df_local)
    df_local["market_size_factor"] = market_size_factor
    df_local["score_adjusted"] = df_local["score"] * df_local["data_quality"] * df_local["market_size_factor"]
    
    return df_local

@lru_cache(maxsize=32)
def compute_scores_cached(purpose):
    """Cached version for performance."""
    return compute_scores(df, purpose)

def run_kmeans(data, k, seed=42, max_iter=100):
    """Simple k-means clustering with deterministic seeding."""
    n_samples = data.shape[0]
    if n_samples == 0:
        return np.array([], dtype=int), np.empty((0, data.shape[1]))

    k = max(1, min(int(k), n_samples))
    rng = np.random.default_rng(seed)
    initial_idx = rng.choice(n_samples, size=k, replace=False)
    centroids = data[initial_idx].copy()
    labels = np.zeros(n_samples, dtype=int)

    for _ in range(max_iter):
        distances = np.linalg.norm(data[:, None, :] - centroids[None, :, :], axis=2)
        new_labels = distances.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels

        new_centroids = []
        for i in range(k):
            members = data[labels == i]
            if len(members) == 0:
                new_centroids.append(centroids[i])
            else:
                new_centroids.append(members.mean(axis=0))
        new_centroids = np.vstack(new_centroids)

        if np.allclose(new_centroids, centroids, atol=1e-5):
            centroids = new_centroids
            break
        centroids = new_centroids

    return labels, centroids

def project_pca_2d(data):
    """Project high-dimensional data to 2D using PCA (SVD)."""
    if data.size == 0:
        return np.empty((0, 2))
    centered = data - data.mean(axis=0, keepdims=True)
    if data.shape[1] == 1:
        return np.column_stack([centered[:, 0], np.zeros(data.shape[0])])
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:2].T
    return centered @ components

def build_cluster_plot_df(scored, purpose, k=CLUSTER_COUNT):
    """Prepare clustered data for plotting."""
    feature_keys = list(PURPOSE_CONFIG[purpose]["features"].keys())
    norm_cols = []
    for key in feature_keys:
        col_name = NUMERIC_COLUMNS.get(key)
        if col_name:
            norm_cols.append(f"norm_{col_name}")

    if not norm_cols:
        return pd.DataFrame(), 0

    features = scored[norm_cols].replace([np.inf, -np.inf], np.nan)
    features = features.fillna(features.median())
    features = features.fillna(0.0)
    data = features.to_numpy()
    if data.size == 0:
        return pd.DataFrame(), 0

    k = max(1, min(int(k), data.shape[0]))
    if np.allclose(data, data[0]):
        labels = np.zeros(data.shape[0], dtype=int)
        coords = np.zeros((data.shape[0], 2))
        k = 1
    else:
        labels, _ = run_kmeans(data, k, seed=CLUSTER_RANDOM_SEED, max_iter=CLUSTER_MAX_ITER)
        coords = project_pca_2d(data)

    cluster_means = (
        pd.DataFrame({"cluster": labels, "score_adjusted": scored["score_adjusted"].fillna(0).values})
        .groupby("cluster")["score_adjusted"]
        .mean()
        .fillna(0)
    )
    ordered_clusters = cluster_means.sort_values(ascending=False).index.tolist()
    rank_map = {cluster: idx + 1 for idx, cluster in enumerate(ordered_clusters)}

    plot_df = scored[[KEY_COL, "score_adjusted"]].copy()
    plot_df["cluster_id"] = labels
    plot_df["cluster_rank"] = plot_df["cluster_id"].map(rank_map)
    plot_df["cluster_avg_score"] = plot_df["cluster_id"].map(cluster_means)
    plot_df["cluster_label"] = plot_df["cluster_rank"].apply(lambda x: f"Cluster {x}")
    plot_df["pc1"] = coords[:, 0]
    plot_df["pc2"] = coords[:, 1]

    return plot_df, k

def build_parallel_coords_df(allowed_countries=None):
    """Prepare normalized purpose scores for the parallel coordinates plot."""
    score_cols = []
    scores_df = df[[KEY_COL]].copy()
    for purpose in PURPOSES:
        scored = compute_scores_cached(purpose)
        col_name = f"score_{purpose}"
        scores_df[col_name] = scored["score_adjusted"]
        score_cols.append(col_name)

    scores_clean = scores_df.dropna(subset=score_cols, how="all").copy()
    if allowed_countries:
        scores_clean = scores_clean[scores_clean[KEY_COL].isin(allowed_countries)].copy()
    if scores_clean.empty:
        return scores_clean, []

    scores_clean["score_mean"] = scores_clean[score_cols].mean(axis=1)
    scores_sorted = scores_clean.sort_values("score_mean", ascending=False)
    if allowed_countries:
        scores_top = scores_sorted.copy()
    else:
        scores_top = scores_sorted.head(PARALLEL_TOP_N).copy()

    for col in score_cols:
        series = scores_top[col]
        min_val = series.min()
        max_val = series.max()
        if pd.isna(min_val) or pd.isna(max_val) or max_val == min_val:
            scores_top[col + "_norm"] = series
        else:
            scores_top[col + "_norm"] = (series - min_val) / (max_val - min_val)

    norm_cols = [col + "_norm" for col in score_cols]
    return scores_top, norm_cols

def build_radar_country_options(top10, selected=None):
    """Build radar dropdown options with starred shortlist at the top."""
    selected = set(selected or [])
    all_countries = sorted(df[KEY_COL].dropna().astype(str).unique().tolist())
    rest = [c for c in all_countries if c not in top10]
    disable_others = len(selected) >= MAX_RADAR_COUNTRIES
    starred = [
        {
            "label": f"⭐ {c}",
            "value": c,
            "disabled": disable_others and c not in selected,
        }
        for c in top10
    ]
    remaining = [
        {
            "label": c,
            "value": c,
            "disabled": disable_others and c not in selected,
        }
        for c in rest
    ]
    return starred + remaining

def build_scored_shortlist(purpose, selected_countries):
    """Return scored countries sorted by suitability, optionally filtered to selection."""
    scored_sorted = compute_scores_cached(purpose).sort_values("score_adjusted", ascending=False)
    if selected_countries:
        scored_sorted = scored_sorted[scored_sorted[KEY_COL].isin(selected_countries)]
    return scored_sorted

def resolve_theme(light_mode):
    """Return theme tokens for the selected mode."""
    return LIGHT_THEME if light_mode else DARK_THEME

def validate_config():
    """Validate that all purpose configs are properly structured."""
    print("\n=== Configuration Validation ===")
    for purpose, cfg in PURPOSE_CONFIG.items():
        total_weight = sum(cfg["features"].values())
        print(f"{purpose}: weights sum to {total_weight:.3f}")
        if not np.isclose(total_weight, 1.0, atol=0.01):
            print(f"  ⚠️ WARNING: Should sum to 1.0")
        
        # Check top 10
        scored = compute_scores(df, purpose)
        top10 = scored.nlargest(10, "score_adjusted")[[KEY_COL, "score_adjusted", "data_quality"]]
        print(f"\n  Top 10 for {purpose}:")
        for idx, row in top10.iterrows():
            print(f"    {row[KEY_COL]:30s} Score: {row['score_adjusted']:.3f} (Quality: {row['data_quality']:.2%})")

# Run validation at startup
validate_config()

# =========================
# 4. Dash app 
# =========================

FONT_STACK = "'Sora', 'Space Grotesk', 'Manrope', 'Segoe UI', sans-serif"
LAYOUT_BG = "var(--bg)"
LAYOUT_BG_GRADIENT = "var(--bg-gradient)"
LAYOUT_SURFACE_1 = "var(--surface-1)"
LAYOUT_SURFACE_2 = "var(--surface-2)"
LAYOUT_SURFACE_SOLID = "var(--surface-solid)"
LAYOUT_BORDER_SUBTLE = "var(--border-subtle)"
LAYOUT_TEXT_PRIMARY = "var(--text-primary)"
LAYOUT_TEXT_MUTED = "var(--text-muted)"
LAYOUT_ACCENT = "var(--accent)"
LAYOUT_ACCENT_SOFT = "var(--accent-soft)"
LAYOUT_SHADOW_LG = "0 18px 45px rgba(6, 10, 16, 0.65)"
LAYOUT_QUALITY_HIGH = "var(--quality-high)"
LAYOUT_QUALITY_MED = "var(--quality-med)"
LAYOUT_QUALITY_LOW = "var(--quality-low)"
DARK_THEME = {
    "background": "#0b0f14",
    "surface": "#121821",
    "grid": "rgba(148, 163, 184, 0.22)",
    "text_primary": "#e6edf3",
    "text_muted": "#9aa6b2",
    "accent": "#b7f28a",
    "map_scale": [
        [0.0, "#0b0f0c"],
        [0.2, "#12351d"],
        [0.4, "#1f6b33"],
        [0.6, "#2f8a40"],
        [0.8, "#4bb45f"],
        [1.0, "#b7f28a"],
    ],
    "map_scale_cb": [
        [0.0, "#0b1020"],
        [0.2, "#17335f"],
        [0.4, "#2453a0"],
        [0.6, "#2f6db0"],
        [0.8, "#f0c96b"],
        [1.0, "#e6a735"],
    ],
    "cluster_colors": ["#00e5ff", "#ff3d8a", "#7cff4f", "#ffb000"],
    "cluster_colors_cb": ["#2a83ff", "#FFAF26", "#F8FFF1", "#ff5fb1"],
    "radar_colors": ["#00e5ff", "#ff3d8a", "#7cff4f", "#ffb000"],
    "radar_colors_cb": ["#2a83ff", "#FFAF26", "#F8FFF1", "#ff5fb1"],
    "parallel_scale": [
        [0.0, "#0b0f0c"],
        [0.5, "#2f8a40"],
        [1.0, "#b7f28a"],
    ],
    "parallel_scale_cb": [
        [0.0, "#0b1020"],
        [0.5, "#2f6db0"],
        [1.0, "#e6a735"],
    ],
}

LIGHT_THEME = {
    "background": "#f5f7fb",
    "surface": "#f0f4f8",
    "grid": "rgba(100, 116, 139, 0.22)",
    "text_primary": "#0f172a",
    "text_muted": "#5f6c7b",
    "accent": "#1f8bff",
    "map_scale": [
            [0.0, "#ebf2ff"],
        [0.2, "#cbe0d2"],
        [0.4, "#81b38f"],
        [0.6, "#559F71"],
        [0.8, "#15773B"],
        [1.0, "#02461A"],
    ],
    "map_scale_cb": [
        [0.0, "#f5f7fb"],
        [0.2, "#cfe0f7"],
        [0.4, "#7aa5e6"],
        [0.6, "#4c78c2"],
        [0.8, "#f0c96b"],
        [1.0, "#e6a735"],
    ],
    "cluster_colors": ["#1f8bff", "#ff3d8a", "#22c55e", "#f59e0b"],
    "cluster_colors_cb": ["#1f6feb", "#f97316", "#1C0132", "#facc15"],
    "radar_colors": ["#1f8bff", "#ff3d8a", "#22c55e", "#f59e0b"],
    "radar_colors_cb": ["#1f6feb", "#f97316", "#1C0132", "#facc15"],
    "parallel_scale": [
        [0.0, "#f5f7fb"],
        [0.5, "#b7f28a"],
        [1.0, "#2f8a40"],
        [0.8, "#3f8a5c"],
        [1.0, "#1f6f3a"],
    ],
    "parallel_scale_cb": [
        [0.0, "#f5f7fb"],
        [0.5, "#4c78c2"],
        [1.0, "#e6a735"],
    ],
}

app = Dash(__name__)

app.layout = html.Div(
    id="theme-root",
    **{"data-theme": "dark", "data-colorblind": "false"},
    style={
        "fontFamily": FONT_STACK,
        "backgroundColor": LAYOUT_BG,
        "backgroundImage": LAYOUT_BG_GRADIENT,
        "minHeight": "100vh",
        "padding": "24px 0",
        "color": LAYOUT_TEXT_PRIMARY,
    },
    children=[
        html.Div(
            style={"maxWidth": "1200px", "margin": "0 auto"},
            children=[
                html.Div(
                    className="mode-toggles",
                    children=[
                        dcc.Checklist(
                            id="colorblind-toggle",
                            options=[{"label": " Colorblind-friendly", "value": "cb"}],
                            value=[],
                            className="toggle-switch toggle-contrast",
                        ),
                        dcc.Checklist(
                            id="theme-toggle",
                            options=[{"label": " Light mode", "value": "light"}],
                            value=[],
                            className="toggle-switch toggle-theme",
                        ),
                    ],
                ),
                # Header
                html.Div(
                    style={"textAlign": "center", "marginBottom": "24px"},
                    children=[
                        html.H1(
                            "Country Relocation Recommender",
                            style={"fontSize": "2.3rem", "marginBottom": "8px"},
                        ),
                        html.P(
                            "Statistically-validated recommendations based on cluster-imputed CIA data",
                            style={"color": LAYOUT_TEXT_MUTED, "fontSize": "0.95rem"},
                        ),
                    ],
                ),

                # Controls card
                html.Div(
                    style={
                        "backgroundColor": LAYOUT_SURFACE_1,
                        "borderRadius": "16px",
                        "padding": "20px 24px",
                        "boxShadow": LAYOUT_SHADOW_LG,
                        "marginBottom": "20px",
                        "border": f"1px solid {LAYOUT_BORDER_SUBTLE}",
                        "backdropFilter": "blur(18px)",
                        "overflow": "visible",
                    },
                    children=[
                        html.H3(
                            "1. Which aspect of relocation would you like to explore?",
                            style={"marginTop": 0, "fontSize": "1.1rem"},
                        ),
                        dcc.Dropdown(
                            id="purpose-dropdown",
                            options=[
                                {"label": "Tourism", "value": "Tourism"},
                                {"label": "Business / Startup", "value": "Startup"},
                                {"label": "Corporate Employment", "value": "Corporate"},
                                {"label": "Starting a Family", "value": "Family"},
                                {"label": "Education", "value": "Education"},
                            ],
                            placeholder="Select your purpose...",
                            clearable=False,
                            value="Tourism",
                            style={"marginTop": "8px", "color": LAYOUT_TEXT_PRIMARY, "zIndex": 2000, "position": "relative"},
                        ),
                        html.Div(
                            id="purpose-description",
                            style={
                                "fontSize": "0.9rem",
                                "color": LAYOUT_TEXT_MUTED,
                                "marginTop": "10px",
                            },
                        ),
                    ],
                ),

                # Map card
                html.Div(
                    style={
                        "backgroundColor": LAYOUT_SURFACE_1,
                        "borderRadius": "16px",
                        "padding": "18px 20px 14px",
                        "boxShadow": LAYOUT_SHADOW_LG,
                        "border": f"1px solid {LAYOUT_BORDER_SUBTLE}",
                        "position": "relative",
                        "backdropFilter": "blur(18px)",
                    },
                    children=[
                        html.H3(
                            "2. Country suitability map",
                            style={"marginTop": 0, "fontSize": "1.1rem"},
                        ),
                        html.P(
                            "Scores adjusted for data quality and market size where relevant. Brighter colors = better match.",
                            style={"fontSize": "0.85rem", "color": LAYOUT_TEXT_MUTED},
                        ),
                        html.Div(
                            style={
                                "display": "flex",
                                "gap": "16px",
                                "alignItems": "stretch",
                            },
                            children=[
                                dcc.Loading(
                                    id="loading-map",
                                    type="default",
                                    children=[
                                        dcc.Graph(
                                            id="recommendation-map",
                                            style={"flex": "3", "height": "60vh"},
                                            config={
                                                "displayModeBar": True,
                                                "displaylogo": False,
                                                "scrollZoom": True
                                            },
                                        ),
                                    ],
                                ),
                                html.Div(
                                    id="leaderboard-panel",
                                    style={
                                        "flex": "1",
                                        "backgroundColor": LAYOUT_SURFACE_2,
                                        "borderRadius": "10px",
                                        "padding": "10px 12px",
                                        "border": f"1px solid {LAYOUT_BORDER_SUBTLE}",
                                        "overflowY": "auto",
                                        "maxHeight": "60vh",
                                        "fontSize": "0.85rem",
                                        "color": LAYOUT_TEXT_PRIMARY,
                                    },
                                ),
                            ],
                        ),
                        html.Div(
                            style={"marginTop": "12px", "display": "flex", "gap": "16px", "alignItems": "center", "flexWrap": "wrap"},
                            children=[
                                html.Button(
                                    "Fullscreen map",
                                    id="fullscreen-btn",
                                    n_clicks=0,
                                    style={
                                        "padding": "6px 14px",
                                        "borderRadius": "999px",
                                        "border": f"1px solid {LAYOUT_ACCENT_SOFT}",
                                        "backgroundColor": LAYOUT_SURFACE_SOLID,
                                        "color": LAYOUT_TEXT_PRIMARY,
                                        "cursor": "pointer",
                                        "fontSize": "0.85rem",
                                    },
                                ),
                                html.Button(
                                    "Download Rankings",
                                    id="download-btn",
                                    n_clicks=0,
                                    style={
                                        "padding": "6px 14px",
                                        "borderRadius": "999px",
                                        "border": f"1px solid {LAYOUT_ACCENT}",
                                        "backgroundColor": LAYOUT_SURFACE_SOLID,
                                        "color": LAYOUT_TEXT_PRIMARY,
                                        "cursor": "pointer",
                                        "fontSize": "0.85rem",
                                    },
                                ),
                                dcc.Download(id="download-rankings"),
                                html.Button(
                                    "Clear selection",
                                    id="clear-selection-btn",
                                    n_clicks=0,
                                    style={
                                        "padding": "6px 14px",
                                        "borderRadius": "999px",
                                        "border": "1px solid #ef4444",
                                        "backgroundColor": LAYOUT_SURFACE_SOLID   ,
                                        "color": LAYOUT_TEXT_PRIMARY,
                                        "cursor": "pointer",
                                        "fontSize": "0.85rem",
                                    },
                                ),

                            ],
                        ),
                        dcc.Interval(id="pulse-interval", interval=900, n_intervals=0),
                    ],
                ),

                # Clusters
                html.Div(
                    style={
                        "backgroundColor": LAYOUT_SURFACE_1,
                        "borderRadius": "16px",
                        "padding": "18px 20px 14px",
                        "boxShadow": LAYOUT_SHADOW_LG,
                        "border": f"1px solid {LAYOUT_BORDER_SUBTLE}",
                        "marginTop": "20px",
                        "backdropFilter": "blur(18px)",
                    },
                    children=[
                        html.Div(
                            style={"display": "flex", "alignItems": "center", "gap": "8px"},
                            children=[
                                html.H3(
                                    "3. Similarity clusters for the selected purpose",
                                    style={"marginTop": 0, "marginBottom": 0, "fontSize": "1.1rem"},
                                ),
                                html.Button(
                                    "i",
                                    id="cluster-info",
                                    n_clicks=0,
                                    title="PC1 and PC2 are the first two principal components (PCA) from the normalized indicators, reducing many dimensions to 2D for display.",
                                    style={
                                        "width": "20px",
                                        "height": "20px",
                                        "borderRadius": "999px",
                                        "border": f"1px solid {LAYOUT_BORDER_SUBTLE}",
                                        "backgroundColor": LAYOUT_SURFACE_2,
                                        "color": LAYOUT_TEXT_MUTED,
                                        "fontSize": "0.75rem",
                                        "lineHeight": "18px",
                                        "textAlign": "center",
                                        "padding": "0",
                                        "cursor": "help",
                                    },
                                    **{"aria-label": "PCA components info"},
                                ),
                            ],
                        ),
                        html.P(
                            "Groups countries by the same indicators used in the map. Clusters are ranked by average score.",
                            style={"fontSize": "0.85rem", "color": LAYOUT_TEXT_MUTED},
                        ),
                        dcc.Loading(
                            id="loading-clusters",
                            type="default",
                            children=[
                                dcc.Graph(
                                    id="cluster-chart",
                                    style={"height": "420px"},
                                ),
                            ],
                        ),
                    ],
                ),

                # Radar
                html.Div(
                    style={
                        "backgroundColor": LAYOUT_SURFACE_1,
                        "borderRadius": "16px",
                        "padding": "18px 20px 14px",
                        "boxShadow": LAYOUT_SHADOW_LG,
                        "border": f"1px solid {LAYOUT_BORDER_SUBTLE}",
                        "marginTop": "20px",
                        "backdropFilter": "blur(18px)",
                    },
                    children=[
                        html.H3(
                            "4. Compare countries across all purposes",
                            style={"marginTop": 0, "fontSize": "1.1rem"},
                        ),
                        html.P(
                            f"Select up to {MAX_RADAR_COUNTRIES} countries",
                            style={"fontSize": "0.85rem", "color": LAYOUT_TEXT_MUTED},
                        ),
                        dcc.Dropdown(
                            id="radar-countries",
                            options=[],
                            multi=True,
                            placeholder="Search and select countries...",
                            style={"marginBottom": "12px"},
                        ),
                        dcc.Loading(
                            id="loading-radar",
                            type="default",
                            children=[
                                dcc.Graph(
                                    id="radar-chart",
                                    style={"height": "420px"},
                                ),
                            ],
                        ),
                    ],
                ),

                # Parallel coordinates
                html.Div(
                    style={
                        "backgroundColor": LAYOUT_SURFACE_1,
                        "borderRadius": "16px",
                        "padding": "18px 20px 14px",
                        "boxShadow": LAYOUT_SHADOW_LG,
                        "border": f"1px solid {LAYOUT_BORDER_SUBTLE}",
                        "marginTop": "20px",
                        "backdropFilter": "blur(18px)",
                    },
                    children=[
                        html.H3(
                            "5. Purpose score relationships",
                            style={"marginTop": 0, "fontSize": "1.1rem"},
                        ),
                        html.P(
                            f"Parallel coordinates for the top {PARALLEL_TOP_N} countries by average suitability (or the current map selection).",
                            style={"fontSize": "0.85rem", "color": LAYOUT_TEXT_MUTED},
                        ),
                        dcc.Loading(
                            id="loading-parallel",
                            type="default",
                            children=[
                                dcc.Graph(
                                    id="parallel-coordinates-chart",
                                    style={"height": "520px"},
                                ),
                            ],
                        ),
                    ],
                ),

                # Fullscreen overlay
                html.Div(
                    id="fullscreen-overlay",
                    style={
                        "position": "fixed",
                        "inset": 0,
                        "backgroundColor": "rgba(8, 12, 18, 0.95)",
                        "zIndex": 9999,
                        "display": "none",
                        "padding": "16px",
                    },
                    children=[
                        html.Div(
                            style={"display": "flex", "justifyContent": "space-between", "alignItems": "center"},
                            children=[
                                html.H3(
                                    "Fullscreen map",
                                    style={"color": LAYOUT_TEXT_PRIMARY, "margin": "0 0 8px 8px"},
                                ),
                                html.Button(
                                    "Close",
                                    id="close-fullscreen-btn",
                                    n_clicks=0,
                                    style={
                                        "padding": "6px 14px",
                                        "borderRadius": "999px",
                                        "border": f"1px solid {LAYOUT_ACCENT_SOFT}",
                                        "backgroundColor": LAYOUT_SURFACE_SOLID,
                                        "color": LAYOUT_TEXT_PRIMARY,
                                        "cursor": "pointer",
                                        "fontSize": "0.85rem",
                                        "marginRight": "8px",
                                    },
                                ),
                            ],
                        ),
                        dcc.Graph(
                            id="fullscreen-map",
                            style={"height": "90vh"},
                            config={"displayModeBar": True, "displaylogo": False, "scrollZoom": True},
                        ),
                    ],
                ),
            ],
        )
    ],
)

# =========================
# 5. Callbacks
# =========================

def normalize_country_value(value):
    """Normalize Plotly customdata entries to country strings."""
    for _ in range(3):
        if value is None:
            return None
        if isinstance(value, np.ndarray):
            if value.shape == ():
                value = value.item()
            elif value.size > 0:
                value = value.flat[0]
            else:
                return None
            continue
        if isinstance(value, np.generic):
            value = value.item()
            continue
        if isinstance(value, (list, tuple)):
            if not value:
                return None
            value = value[0]
            continue
        break
    return value if isinstance(value, str) else None

def extract_countries_from_points(points):
    """Extract country names from Plotly selection points."""
    countries = []
    for point in points or []:
        for key in ("customdata", "location", "hovertext", "text"):
            value = normalize_country_value(point.get(key))
            if value:
                countries.append(value)
                break
    return countries

def get_selected_countries(selected_data, click_data):
    """Combine lasso/box selections with single-click selections."""
    countries = set()
    if selected_data and "points" in selected_data:
        countries.update(extract_countries_from_points(selected_data["points"]))
    if click_data and "points" in click_data:
        countries.update(extract_countries_from_points(click_data["points"]))
    return sorted(countries)

@app.callback(
    Output("purpose-description", "children"),
    Input("purpose-dropdown", "value"),
)
def update_purpose_description(purpose):
    """Update the purpose description text."""
    descriptions = {
        "Tourism": "Highlights connectivity, infrastructure, and environmental quality. Uses inverted CO2 emissions.",
        "Startup": "Focuses on governance, economic indicators, and digital infrastructure. Market size adjustment applied.",
        "Corporate": "Prioritizes job market health, connectivity, and infrastructure for professionals. Market size adjustment applied.",
        "Family": "Emphasizes health metrics (inverted mortality rates), demographic balance, and governance quality.",
        "Education": "Weights literacy heavily with youth employment prospects and digital access. Market size adjustment applied.",
    }
    if purpose is None:
        return "Please choose a purpose to start."
    return descriptions.get(purpose, "")

@app.callback(
    Output("theme-root", "data-theme"),
    Input("theme-toggle", "value"),
)
def update_theme_mode(theme_value):
    """Toggle light/dark theme for the layout."""
    return "light" if theme_value and "light" in theme_value else "dark"

@app.callback(
    Output("theme-root", "data-colorblind"),
    Input("colorblind-toggle", "value"),
)
def update_colorblind_mode(cb_value):
    """Expose color-blind mode to CSS."""
    return "true" if cb_value and "cb" in cb_value else "false"

@app.callback(
    Output("radar-countries", "options"),
    Input("purpose-dropdown", "value"),
    Input("recommendation-map", "selectedData"),
    Input("recommendation-map", "clickData"),
    Input("radar-countries", "value"),
)
def update_radar_country_options(purpose, map_selected, map_clicked, selected_values):
    """Update radar dropdown options with shortlist stars and map filtering."""
    all_countries = sorted(df[KEY_COL].dropna().astype(str).unique().tolist())
    if purpose is None:
        return build_radar_country_options([], selected_values)

    selected_countries = get_selected_countries(map_selected, map_clicked)
    scored_sorted = build_scored_shortlist(purpose, selected_countries)
    top10 = scored_sorted[KEY_COL].dropna().astype(str).head(TOP_N_COUNTRIES).tolist()
    return build_radar_country_options(top10, selected_values)

@app.callback(
    Output("recommendation-map", "figure"),
    Output("fullscreen-map", "figure"),
    Input("purpose-dropdown", "value"),
    Input("colorblind-toggle", "value"),
    Input("theme-toggle", "value"),
    Input("radar-countries", "value"),
)
def update_maps(purpose, cb_value, theme_value, radar_countries):
    """Update both map figures."""
    light_mode = theme_value is not None and "light" in theme_value
    theme = resolve_theme(light_mode)
    if purpose is None:
        empty_fig = px.choropleth(title="Select a purpose to see recommendations")
        empty_fig.update_layout(
            margin=dict(l=0, r=0, t=40, b=0),
            paper_bgcolor=theme["background"],
            plot_bgcolor=theme["background"],
            font=dict(color=theme["text_primary"]),
        )
        return empty_fig, empty_fig

    scored = compute_scores_cached(purpose)
    use_colorblind = cb_value is not None and "cb" in cb_value
    color_scale = theme["map_scale_cb"] if use_colorblind else theme["map_scale"]

    fig = px.choropleth(
        scored,
        locations=KEY_COL,
        locationmode="country names",
        color="score_adjusted",
        custom_data=[KEY_COL],
        hover_name=KEY_COL,
        hover_data={
            "score_adjusted": ":.3f",
        },
        color_continuous_scale=color_scale,
        title=f"Country suitability for: {purpose}",
    )

    fig.update_geos(
        projection_type="natural earth",
        showcountries=True,
        showcoastlines=True,
        coastlinecolor=theme["grid"],
        fitbounds="locations",
    )

    fig.update_layout(
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor=theme["background"],
        plot_bgcolor=theme["background"],
        font=dict(color=theme["text_primary"]),
        title_x=0.02,
        dragmode="zoom",
        geo=dict(
            showframe=False,
            landcolor=theme["surface"],
            bgcolor=theme["background"],
        ),
        uirevision=purpose,
        coloraxis_colorbar=dict(
            title=dict(
                text="Score",
                font=dict(color=theme["text_primary"]),
            ),
            ticks="outside",
            tickfont=dict(color=theme["text_muted"]),
        ),
    )

    highlight_countries = (radar_countries or [])[:MAX_RADAR_COUNTRIES]
    if highlight_countries:
        marker_color = (
            "#eba71f" if light_mode else "#eba71f"
        ) if not use_colorblind else (
            "#1C0132" if light_mode else "#1C0132"
        )
        fig.add_trace(
            go.Scattergeo(
                locations=highlight_countries,
                locationmode="country names",
                mode="markers",
                marker=dict(
                    size=14,
                    color=marker_color,
                    symbol="circle-open",
                    line=dict(width=2.5, color=marker_color),
                ),
                hovertext=highlight_countries,
                hoverinfo="text",
                name="Radar selection",
                showlegend=False,
            )
        )

    return fig, fig

@app.callback(
    Output("leaderboard-panel", "children"),
    Input("purpose-dropdown", "value"),
    Input("recommendation-map", "selectedData"),
)
def update_leaderboard(purpose, map_selected):
    """Update shortlist based on the selected map region."""
    if purpose is None:
        return html.Div(
            "Select a purpose to see the shortlist.",
            style={"textAlign": "center", "padding": "20px", "color": LAYOUT_TEXT_MUTED},
        )

    selected_countries = extract_countries_from_points(
    map_selected["points"] if map_selected and "points" in map_selected else None
)

    shortlist = build_scored_shortlist(purpose, selected_countries)

    top10 = shortlist[[KEY_COL, "score_adjusted"]].head(TOP_N_COUNTRIES)
    if top10.empty or top10["score_adjusted"].isna().all():
        message = "No data available for this selection." if selected_countries else "No data available for this purpose."
        return html.Div(
            message,
            style={"textAlign": "center", "padding": "20px", "color": LAYOUT_TEXT_MUTED},
        )

    rows = []
    rows.append(
        html.Tr([
            html.Th("#", style={"padding": "4px 8px", "textAlign": "left"}),
            html.Th("Country", style={"padding": "4px 8px", "textAlign": "left"}),
            html.Th("Score", style={"padding": "4px 8px", "textAlign": "right"}),
        ], style={"borderBottom": f"2px solid {LAYOUT_BORDER_SUBTLE}"})
    )

    for i, row in top10.reset_index(drop=True).iterrows():        
        rows.append(
            html.Tr([
                html.Td(i + 1, style={"padding": "4px 8px"}),
                html.Td(row[KEY_COL], style={"padding": "4px 8px"}),
                html.Td(f"{row['score_adjusted']:.3f}", style={"padding": "4px 8px", "textAlign": "right"}),
            ], style={"borderBottom": f"1px solid {LAYOUT_BORDER_SUBTLE}"})
        )

    return html.Div(
        children=[
            html.H4(
                "Shortlist of top countries",
                style={
                    "margin": "0 0 8px 0",
                    "fontSize": "0.95rem",
                    "color": LAYOUT_TEXT_PRIMARY,
                },
            ),
            html.Table(
                rows,
                style={
                    "width": "100%",
                    "borderCollapse": "collapse",
                },
            ),
        ],
    )

@app.callback(
    Output("cluster-chart", "figure"),
    Input("purpose-dropdown", "value"),
    Input("recommendation-map", "selectedData"),
    Input("recommendation-map", "clickData"),
    Input("radar-countries", "value"),
    Input("colorblind-toggle", "value"),
    Input("theme-toggle", "value"),
)
def update_cluster_chart(purpose, map_selected, map_clicked, radar_countries, cb_value, theme_value):
    """Update similarity cluster chart for the selected purpose."""
    light_mode = theme_value is not None and "light" in theme_value
    theme = resolve_theme(light_mode)
    if purpose is None:
        fig = px.scatter(title="Select a purpose to see similarity clusters")
        fig.update_layout(
            margin=dict(l=0, r=0, t=40, b=0),
            paper_bgcolor=theme["background"],
            plot_bgcolor=theme["background"],
            font=dict(color=theme["text_primary"]),
        )
        return fig

    scored = compute_scores_cached(purpose).copy()
    plot_df, cluster_count = build_cluster_plot_df(scored, purpose, CLUSTER_COUNT)
    if plot_df.empty:
        fig = px.scatter(title="Not enough data to build clusters")
        fig.update_layout(
            margin=dict(l=0, r=0, t=40, b=0),
            paper_bgcolor=theme["background"],
            plot_bgcolor=theme["background"],
            font=dict(color=theme["text_primary"]),
        )
        return fig

    use_colorblind = cb_value is not None and "cb" in cb_value
    cluster_palette = theme["cluster_colors_cb"] if use_colorblind else theme["cluster_colors"]

    fig = px.scatter(
        plot_df,
        x="pc1",
        y="pc2",
        color="cluster_label",
        hover_name=KEY_COL,
        hover_data={
            "score_adjusted": ":.3f",
            "cluster_avg_score": ":.3f",
            "cluster_label": False,
            "pc1": False,
            "pc2": False,
        },
        custom_data=[KEY_COL],
        labels={
            "score_adjusted": "Suitability score",
            "cluster_avg_score": "Cluster avg score",
            "pc1": "Component 1",
            "pc2": "Component 2",
        },
        color_discrete_sequence=cluster_palette,
        template="plotly_dark" if not light_mode else "plotly_white",
    )

    fig.update_traces(marker=dict(size=8, opacity=0.85, line=dict(width=0.3, color=theme["text_muted"])))
    fig.update_layout(
        title=f"Similarity clusters for {purpose} (k={cluster_count})",
        margin=dict(l=0, r=0, t=50, b=0),
        legend_title_text="Cluster (ranked by avg score)",
        paper_bgcolor=theme["background"],
        plot_bgcolor=theme["background"],
        font=dict(color=theme["text_primary"]),
    )
    fig.update_xaxes(showgrid=True, gridcolor=theme["grid"], zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=theme["grid"], zeroline=False)

    map_selected_countries = get_selected_countries(map_selected, map_clicked)
    radar_selected = (radar_countries or [])[:MAX_RADAR_COUNTRIES]
    highlight_countries = set(map_selected_countries) | set(radar_selected)
    if highlight_countries:
        should_dim = len(map_selected_countries) > 0
        unselected_marker = dict(opacity=0.15, size=6) if should_dim else dict(opacity=0.8, size=8)
        selected_marker = dict(opacity=0.95, size=11)
        for trace in fig.data:
            selected_idx = []
            customdata = trace.customdata if trace.customdata is not None else []
            for i, item in enumerate(customdata):
                country = normalize_country_value(item)
                if country in highlight_countries:
                    selected_idx.append(i)
            trace.selectedpoints = selected_idx
            trace.selected = dict(marker=selected_marker)
            trace.unselected = dict(marker=unselected_marker)

    return fig

@app.callback(
    Output("radar-chart", "figure"),
    Input("radar-countries", "value"),
    Input("colorblind-toggle", "value"),
    Input("theme-toggle", "value"),
)
def update_radar(selected_countries, cb_value, theme_value):
    """Update radar chart comparing countries across purposes."""
    light_mode = theme_value is not None and "light" in theme_value
    theme = resolve_theme(light_mode)
    selected = selected_countries or []

    selected = selected[:MAX_RADAR_COUNTRIES]

    purposes = PURPOSES

    if len(selected) == 0:
        fig = go.Figure()
        fig.add_trace(
            go.Scatterpolar(
                r=[0],
                theta=["Select countries"],
                mode="lines",
                line=dict(color=theme["text_muted"]),
                showlegend=False,
            )
        )
        fig.update_layout(
            template="plotly_dark" if not light_mode else "plotly_white",
            margin=dict(l=20, r=20, t=40, b=20),
            polar=dict(
                radialaxis=dict(range=[0, 1], tickformat=".2f", gridcolor=theme["grid"]),
                angularaxis=dict(gridcolor=theme["grid"]),
                bgcolor=theme["background"],
            ),
            paper_bgcolor=theme["background"],
            plot_bgcolor=theme["background"],
            font=dict(color=theme["text_primary"]),
        )
        return fig

    rows = []
    for country in selected:
        for p in purposes:
            scored = compute_scores_cached(p)
            try:
                country_data = scored[scored[KEY_COL] == country]
                if len(country_data) > 0:
                    score_val = float(country_data["score_adjusted"].values[0])
                    quality_val = float(country_data["data_quality"].values[0])
                else:
                    score_val = 0.0
                    quality_val = 0.0
            except Exception as e:
                print(f"Error getting score for {country}, {p}: {e}")
                score_val = 0.0
                quality_val = 0.0
            
            rows.append({
                "Country": country,
                "Purpose": p,
                "Score": score_val,
                "Quality": quality_val
            })

    df_radar = pd.DataFrame(rows)

    fig = go.Figure()
    use_colorblind = cb_value is not None and "cb" in cb_value
    colors = theme["radar_colors_cb"] if use_colorblind else theme["radar_colors"]
    for idx, country in enumerate(selected):
        country_data = df_radar[df_radar["Country"] == country]
        fig.add_trace(
            go.Scatterpolar(
                r=country_data["Score"],
                theta=country_data["Purpose"],
                mode="lines",
                fill="toself",
                name=country,
                opacity=0.7,
                line=dict(color=colors[idx % len(colors)], width=2.2),
            )
        )

    fig.update_layout(
        template="plotly_dark" if not light_mode else "plotly_white",
        title="Comparison across purposes (quality-adjusted scores)",
        polar=dict(
            radialaxis=dict(
                range=[0, 1],
                tickformat=".2f",
                gridcolor=theme["grid"],
            ),
            angularaxis=dict(gridcolor=theme["grid"]),
            bgcolor=theme["background"],
        ),
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor=theme["background"],
        plot_bgcolor=theme["background"],
        font=dict(color=theme["text_primary"]),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        )
    )

    return fig

@app.callback(
    Output("parallel-coordinates-chart", "figure"),
    Input("purpose-dropdown", "value"),
    Input("recommendation-map", "selectedData"),
    Input("recommendation-map", "clickData"),
    Input("colorblind-toggle", "value"),
    Input("theme-toggle", "value"),
)
def update_parallel_coordinates(_purpose, map_selected, map_clicked, cb_value, theme_value):
    """Update the parallel coordinates plot of purpose scores."""
    light_mode = theme_value is not None and "light" in theme_value
    theme = resolve_theme(light_mode)
    selected_countries = get_selected_countries(map_selected, map_clicked)
    scores_top, norm_cols = build_parallel_coords_df(selected_countries if selected_countries else None)
    if scores_top.empty or not norm_cols:
        fig = px.parallel_coordinates()
        fig.update_layout(
            margin=dict(l=40, r=40, t=40, b=40),
            paper_bgcolor=theme["background"],
            plot_bgcolor=theme["background"],
            font=dict(color=theme["text_primary"]),
        )
        return fig

    color_col = f"score_{PURPOSES[-1]}_norm"
    labels = {f"score_{purpose}_norm": purpose for purpose in PURPOSES}
    use_colorblind = cb_value is not None and "cb" in cb_value
    color_scale = theme["parallel_scale_cb"] if use_colorblind else theme["parallel_scale"]

    dims = []
    for col, purpose in zip(norm_cols, PURPOSES):
        dims.append(
            dict(
                label=purpose,
                values=scores_top[col],
                range=[0, 1],
                tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1],
               
            )
        )

    fig = go.Figure(
        data=go.Parcoords(
            line=dict(
                color= "#22c55e",
                showscale=True,
            ),
            dimensions=dims,
        )
    )


    fig.update_layout(
        paper_bgcolor=theme["background"],
        plot_bgcolor=theme["background"],
        font=dict(color=theme["text_primary"]),
        margin=dict(l=40, r=40, t=40, b=40),
        coloraxis_colorbar_title_text="Score",
    )
    return fig

    
@app.callback(
    Output("fullscreen-overlay", "style"),
    Input("fullscreen-btn", "n_clicks"),
    Input("close-fullscreen-btn", "n_clicks"),
    Input("theme-toggle", "value"),
)
def toggle_fullscreen(n_open, n_close, theme_value):
    """Toggle fullscreen overlay visibility."""
    n_open = n_open or 0
    n_close = n_close or 0
    visible = n_open > n_close
    light_mode = theme_value is not None and "light" in theme_value
    overlay_bg = "rgba(248, 250, 252, 0.95)" if light_mode else "rgba(8, 12, 18, 0.96)"
    base_style = {
        "position": "fixed",
        "inset": 0,
        "backgroundColor": overlay_bg,
        "zIndex": 9999,
        "padding": "16px",
    }
    if not visible:
        base_style["display"] = "none"
    return base_style

@app.callback(
    Output("download-rankings", "data"),
    Input("download-btn", "n_clicks"),
    State("purpose-dropdown", "value"),
    prevent_initial_call=True,
)
def download_rankings(n_clicks, purpose):
    """Export rankings to CSV file with data quality information."""
    if purpose is None:
        return no_update
    
    scored = compute_scores_cached(purpose)
    export_df = scored[[KEY_COL, "score_adjusted", "score", "data_quality"]].sort_values("score_adjusted", ascending=False)
    export_df.columns = ["Country", "Quality_Adjusted_Score", "Raw_Score", "Data_Quality_Percent"]
    export_df["Data_Quality_Percent"] = (export_df["Data_Quality_Percent"] * 100).round(1)
    
    return dcc.send_data_frame(
        export_df.to_csv,
        f"{purpose.lower()}_country_rankings_statistically_improved.csv",
        index=False
    )
@app.callback(
    Output("recommendation-map", "selectedData"),
    Output("recommendation-map", "clickData"),
    Input("clear-selection-btn", "n_clicks"),
    prevent_initial_call=True,
)
def clear_map_selection(n_clicks):
    # Returning None clears both lasso/box selection and clicked country
    return None, None

# =========================
# 6. Run
# =========================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("Starting Country Relocation Recommender (Statistically Improved)")
    print("="*60)
    print(f"\nLoaded {len(df)} countries")
    print(f"Using cluster-imputed data from: {BASE_DIR}")
    print("\nKey improvements:")
    print("  ✓ Robust outlier-resistant normalization")
    print("  ✓ Single-pass normalization (weights preserved)")
    print("  ✓ Data quality scoring and adjustment")
    print("  ✓ Pre-transformed features (inverted mortality, unemployment, etc.)")
    print("  ✓ Infrastructure composite score (handles multicollinearity)")
    print("  ✓ Per-capita metrics for fair comparison")
    print("  ✓ Improved government type encoding with validation")
    print("  ✓ Configuration validation at startup")
    print("\n" + "="*60 + "\n")
    
    app.run(debug=True)
