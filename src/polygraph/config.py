from pathlib import Path

GAMMA_API = "https://gamma-api.polymarket.com"
COMBO_API = "https://combos-rfq-api.polymarket.com"
DATA_API = "https://data-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"

DEFAULT_DATA_DIR = Path("data")
DEFAULT_DB_PATH = DEFAULT_DATA_DIR / "polygraph.db"
DEFAULT_GRAPH_PATH = DEFAULT_DATA_DIR / "market_graph.graphml"

# Local embedding model (sentence-transformers)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_BATCH_SIZE = 128
EMBEDDING_INDEX_DIR = DEFAULT_DATA_DIR / "vectors"

# Gamma API caps page size at 100
PAGE_SIZE = 100

# Price history (CLOB)
PRICE_HISTORY_INTERVAL = "1d"
PRICE_HISTORY_FIDELITY = 60  # minutes — ~daily points with interval=1d
PRICE_FETCH_MIN_VOLUME = 10_000
PRICE_FETCH_DEFAULT_LIMIT = 3_000
PRICE_FETCH_SLEEP_SEC = 0.06

# Semantic similarity: keep top-k neighbors per market above this cosine score
SEMANTIC_TOP_K = 8
SEMANTIC_MIN_SIMILARITY = 0.55

# Tag edges: skip ultra-common tags that connect everything
TAG_EDGE_MAX_MARKET_FRACTION = 0.15
