"""
Configuration file for Pest Advisory System
"""

# ============================================================
# OLLAMA SETTINGS
# ============================================================
OLLAMA_BASE_URL = "http://localhost:11434"
ENCODER_MODEL = "llama3:8b"  # Fast, good at structured reasoning
ENCODER_TEMPERATURE = 0.1      # Low for consistent structured output
ENCODER_TIMEOUT = 120          # seconds

# ============================================================
# ETL THRESHOLDS (Economic Threshold Levels)
# Source: Punjab Agriculture Department Guidelines
# ============================================================
ETL_THRESHOLDS = {
    'W. FLY(ABOVE ETL)': 10,
    'JASSID(ABOVE ETL)': 5,
    'THRIPS(ABOVE ETL)': 20,
    'M.BUG(ABOVE ETL)': 2,
    'MITES(ABOVE ETL)': 5,
    'APHIDS(ABOVE ETL)': 10,
    'DUSKY COTTON BUG(ABOVE ETL)': 2,
    'SBW(ABOVE ETL)': 2,        # Spotted Bollworm
    'PBW(ABOVE ETL)': 5,         # Pink Bollworm
    'ABW(ABOVE ETL)': 2,         # American Bollworm
    'ARMY WORM(ABOVE ETL)': 3
}

# ============================================================
# SEVERITY SCORING
# ============================================================
SEVERITY_THRESHOLDS = {
    'LOW': 5,      # Score < 5
    'MEDIUM': 15,  # Score 5-15
    'HIGH': 30     # Score > 15
}

# Critical multiplier (how many times above ETL is critical)
CRITICAL_MULTIPLIER = 5  # 5x ETL = critical threat

# ============================================================
# PEST METADATA
# ============================================================
PEST_INFO = {
    'W. FLY': {
        'full_name': 'Whitefly',
        'scientific_name': 'Bemisia tabaci',
        'damage': 'Sucks sap, transmits CLCuV, honeydew causes sooty mold',
        'critical_stage': 'Vegetative to early flowering'
    },
    'JASSID': {
        'full_name': 'Jassid (Leafhopper)',
        'scientific_name': 'Amrasca biguttula',
        'damage': 'Leaf curling, hopper burn, stunted growth',
        'critical_stage': 'Early vegetative stage'
    },
    'THRIPS': {
        'full_name': 'Thrips',
        'scientific_name': 'Thrips tabaci',
        'damage': 'Silver leaf, distorted growth',
        'critical_stage': 'Seedling to early vegetative'
    },
    'M.BUG': {
        'full_name': 'Mealy Bug',
        'scientific_name': 'Phenacoccus solenopsis',
        'damage': 'Honeydew, sooty mold, stunted growth',
        'critical_stage': 'All stages'
    },
    'PBW': {
        'full_name': 'Pink Bollworm',
        'scientific_name': 'Pectinophora gossypiella',
        'damage': 'Boll damage, reduces lint quality',
        'critical_stage': 'Flowering to boll formation'
    },
    'ABW': {
        'full_name': 'American Bollworm',
        'scientific_name': 'Helicoverpa armigera',
        'damage': 'Feeds on squares, flowers, and bolls',
        'critical_stage': 'Squaring to boll formation'
    },
    'SBW': {
        'full_name': 'Spotted Bollworm',
        'scientific_name': 'Earias vittella',
        'damage': 'Bores into bolls, causes shedding',
        'critical_stage': 'Flowering to boll formation'
    }
}

# ============================================================
# OUTPUT SETTINGS
# ============================================================
ENCODER_OUTPUT_DIR = "outputs/encoder_results"
SAVE_INDIVIDUAL_RESULTS = True
SAVE_BATCH_SUMMARY = True

# ============================================================
# LOGGING
# ============================================================
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR



# ============================================================
# VISUALIZATION SETTINGS
# ============================================================

# Chart styling
CHART_STYLE = "seaborn"  # matplotlib style
CHART_DPI = 300
CHART_COLORS = {
    'LOW': '#4CAF50',      # Green
    'MEDIUM': '#FFC107',   # Amber
    'HIGH': '#FF5722',     # Deep Orange
    'CRITICAL': '#B71C1C', # Dark Red
    'ABOVE_ETL': '#F44336',
    'BELOW_ETL': '#4CAF50',
    'PRIMARY': '#1976D2',
    'SECONDARY': '#757575'
}

# Figure sizes (width, height in inches)
FIGURE_SIZES = {
    'pest_bar': (12, 6),
    'severity_gauge': (10, 4),
    'threat_horizontal': (10, 5),
    'disease_bar': (10, 4),
    'summary_grid': (14, 10)
}

# Chart output directory
CHARTS_OUTPUT_DIR = "outputs/charts"

# PDF Report settings
REPORT_OUTPUT_DIR = "outputs/reports"
REPORT_FONT_SIZES = {
    'title': 24,
    'section': 16,
    'subsection': 14,
    'body': 11,
    'caption': 9
}