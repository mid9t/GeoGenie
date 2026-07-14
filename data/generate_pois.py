"""
Synthetic POI (Point of Interest) dataset generator.

Generates 120,000+ POIs with:
  - lat / lon (clustered around real-world city centers, like actual POI data)
  - category
  - accessibility info
  - noise level
  - opening hours

Outputs:
  - poi_dataset.json   (JSON array of all records)
  - poi_dataset.db     (SQLite database, indexed, for querying)

Usage:
  python3 generate_poi_dataset.py --n 120000 --seed 42
"""

import argparse
import json
import random
import sqlite3
import uuid
from pathlib import Path


# ---------------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------------

# (name, lon, lat, relative weight) - city / neighborhood centers along the
# San Francisco -> San Jose peninsula corridor. Points cluster around these so
# the dataset looks like real-world POI density rather than uniform noise.
# Weights roughly track population / commercial density.
CITY_CENTERS = [
    ("San Francisco",        -122.4194, 37.7749, 20),
    ("Daly City",            -122.4702, 37.6879, 4),
    ("South San Francisco",  -122.4077, 37.6547, 3),
    ("San Bruno",            -122.4111, 37.6305, 2),
    ("Millbrae",             -122.3872, 37.5985, 2),
    ("Burlingame",           -122.3661, 37.5841, 3),
    ("San Mateo",            -122.3255, 37.5630, 5),
    ("Foster City",          -122.2711, 37.5585, 2),
    ("Belmont",              -122.2758, 37.5202, 2),
    ("Redwood City",         -122.2364, 37.4852, 5),
    ("Menlo Park",           -122.1817, 37.4530, 3),
    ("Palo Alto",            -122.1430, 37.4419, 6),
    ("Mountain View",        -122.0839, 37.3861, 6),
    ("Los Altos",            -122.1141, 37.3852, 2),
    ("Sunnyvale",            -122.0363, 37.3688, 7),
    ("Cupertino",            -122.0322, 37.3230, 4),
    ("Santa Clara",          -121.9552, 37.3541, 6),
    ("Milpitas",             -121.8996, 37.4323, 3),
    ("San Jose",             -121.8863, 37.3382, 18),
]

# Scatter width (std dev in degrees) for the Gaussian spread around each center.
# ~0.02 deg latitude ~= 2.2 km, appropriate for tightly-spaced metro cities.
# The old global dataset used ~0.10-0.12, which would overlap badly and push
# points into the bay/ocean at this scale.
SCATTER_LON = 0.022
SCATTER_LAT = 0.020

CATEGORIES = [
    ("restaurant",     16),
    ("cafe",           12),
    ("retail",         14),
    ("grocery",         7),
    ("bar",             6),
    ("park",            5),
    ("school",          5),
    ("hospital",        3),
    ("pharmacy",        4),
    ("bank",            4),
    ("hotel",           5),
    ("museum",          2),
    ("gym",             5),
    ("library",         2),
    ("place_of_worship",3),
    ("transit_station", 4),
    ("nightclub",       3),
]

ACCESSIBILITY_FEATURES = [
    "ramp",
    "elevator",
    "accessible_restroom",
    "braille_signage",
    "hearing_loop",
    "wide_doorways",
    "accessible_parking",
]

# categories that tend to be loud/quiet, used to weight noise-level sampling
LOUD_CATEGORIES = {"bar", "nightclub", "gym", "transit_station"}
QUIET_CATEGORIES = {"library", "place_of_worship", "park", "museum"}

NOISE_LEVELS = ["quiet", "moderate", "loud", "very_loud"]

NAME_ADJECTIVES = [
    "Blue", "Golden", "Sunny", "Old Town", "Central", "Riverside", "Hillside",
    "Green", "Silver", "Downtown", "Corner", "Main Street", "North", "South",
    "East Side", "West End", "Harbor", "Garden", "Maple", "Oak",
]
NAME_NOUNS = {
    "restaurant": ["Kitchen", "Bistro", "Grill", "Table", "Diner"],
    "cafe": ["Cafe", "Coffee House", "Roasters", "Espresso Bar"],
    "retail": ["Boutique", "Store", "Shop", "Market"],
    "grocery": ["Grocery", "Foods", "Mart", "Supermarket"],
    "bar": ["Bar", "Pub", "Tavern", "Lounge"],
    "park": ["Park", "Gardens", "Green", "Commons"],
    "school": ["School", "Academy", "Institute"],
    "hospital": ["Hospital", "Medical Center", "Clinic"],
    "pharmacy": ["Pharmacy", "Drugstore"],
    "bank": ["Bank", "Credit Union", "Financial Center"],
    "hotel": ["Hotel", "Inn", "Suites", "Lodge"],
    "museum": ["Museum", "Gallery", "Exhibit Hall"],
    "gym": ["Gym", "Fitness Center", "Athletic Club"],
    "library": ["Library", "Reading Room"],
    "place_of_worship": ["Chapel", "Temple", "Mosque", "Church", "Synagogue"],
    "transit_station": ["Station", "Transit Hub", "Terminal"],
    "nightclub": ["Nightclub", "Club", "Lounge"],
}

WEEKDAYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def weighted_choice(rng: random.Random, items):
    """items: list of (value, weight) tuples."""
    total = sum(w for _, w in items)
    r = rng.uniform(0, total)
    upto = 0
    for value, weight in items:
        upto += weight
        if upto >= r:
            return value
    return items[-1][0]


def sample_location(rng: random.Random):
    weighted_cities = [((name, lon, lat), w) for name, lon, lat, w in CITY_CENTERS]
    name, lon0, lat0 = weighted_choice(rng, weighted_cities)
    # Gaussian scatter around the city center (~city-scale spread in degrees).
    lon = lon0 + rng.gauss(0, SCATTER_LON)
    lat = lat0 + rng.gauss(0, SCATTER_LAT)
    # clamp to valid ranges
    lon = max(-180.0, min(180.0, lon))
    lat = max(-90.0, min(90.0, lat))
    return name, round(lon, 6), round(lat, 6)


def sample_category(rng: random.Random):
    return weighted_choice(rng, CATEGORIES)


def sample_name(rng: random.Random, category: str):
    adj = rng.choice(NAME_ADJECTIVES)
    noun = rng.choice(NAME_NOUNS.get(category, ["Place"]))
    return f"{adj} {noun}"


def sample_accessibility(rng: random.Random):
    wheelchair_accessible = rng.random() < 0.55
    features = []
    if wheelchair_accessible:
        n_features = rng.randint(1, 4)
        features = rng.sample(ACCESSIBILITY_FEATURES, k=min(n_features, len(ACCESSIBILITY_FEATURES)))
    return {
        "wheelchair_accessible": wheelchair_accessible,
        "features": sorted(features),
    }


def sample_noise_level(rng: random.Random, category: str):
    if category in LOUD_CATEGORIES:
        weights = [1, 3, 8, 6]
    elif category in QUIET_CATEGORIES:
        weights = [8, 6, 1, 0.2]
    else:
        weights = [3, 6, 3, 1]
    return weighted_choice(rng, list(zip(NOISE_LEVELS, weights)))


def sample_hours(rng: random.Random, category: str):
    """Returns a dict of weekday -> {"open": "HH:MM", "close": "HH:MM"} or None if closed."""
    always_open = category in {"hospital", "transit_station", "pharmacy"} and rng.random() < 0.4
    if always_open:
        return {day: {"open": "00:00", "close": "23:59"} for day in WEEKDAYS}

    base_open_hour = rng.choice([6, 7, 8, 9, 10, 11])
    base_close_hour = rng.choice([17, 18, 19, 20, 21, 22, 23])
    if category in {"bar", "nightclub"}:
        base_open_hour = rng.choice([16, 17, 18])
        base_close_hour = rng.choice([1, 2, 3])  # past midnight

    closed_days = set()
    if category in {"bank", "school"}:
        closed_days = {"sat", "sun"}
    elif rng.random() < 0.15:
        closed_days = {rng.choice(WEEKDAYS)}

    hours = {}
    for day in WEEKDAYS:
        if day in closed_days:
            hours[day] = None
        else:
            open_h = base_open_hour + rng.choice([-1, 0, 0, 1])
            close_h = base_close_hour
            hours[day] = {
                "open": f"{open_h % 24:02d}:{rng.choice(['00', '15', '30']):}",
                "close": f"{close_h % 24:02d}:{rng.choice(['00', '30']):}",
            }
    return hours


def generate_poi(rng: random.Random):
    city_name, lon, lat = sample_location(rng)
    category = sample_category(rng)
    return {
        "id": str(uuid.UUID(int=rng.getrandbits(128))),
        "name": sample_name(rng, category),
        "category": category,
        "city": city_name,
        "lon": lon,
        "lat": lat,
        "accessibility": sample_accessibility(rng),
        "noise_level": sample_noise_level(rng, category),
        "hours": sample_hours(rng, category),
    }


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

def build_sqlite_db(records, db_path: Path):
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE poi (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            city TEXT,
            lon REAL NOT NULL,
            lat REAL NOT NULL,
            wheelchair_accessible INTEGER NOT NULL,
            accessibility_features TEXT,
            noise_level TEXT NOT NULL,
            hours_json TEXT NOT NULL
        )
        """
    )

    cur.executemany(
        """
        INSERT INTO poi (
            id, name, category, city, lon, lat,
            wheelchair_accessible, accessibility_features, noise_level, hours_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            (
                r["id"],
                r["name"],
                r["category"],
                r["city"],
                r["lon"],
                r["lat"],
                int(r["accessibility"]["wheelchair_accessible"]),
                json.dumps(r["accessibility"]["features"]),
                r["noise_level"],
                json.dumps(r["hours"]),
            )
            for r in records
        ),
    )

    # Indexes for the query patterns you'd actually run against this table:
    # category filters, noise-level filters, accessibility filters, and
    # bounding-box lookups on lon/lat (SQLite has no native geo index, so a
    # composite index on lon/lat lets range queries use it efficiently).
    cur.execute("CREATE INDEX idx_poi_category ON poi(category)")
    cur.execute("CREATE INDEX idx_poi_noise ON poi(noise_level)")
    cur.execute("CREATE INDEX idx_poi_wheelchair ON poi(wheelchair_accessible)")
    cur.execute("CREATE INDEX idx_poi_lonlat ON poi(lon, lat)")

    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic POI dataset.")
    parser.add_argument("--n", type=int, default=120_000, help="Number of POIs to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--out-dir", type=str, default=".", help="Output directory")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.n:,} POIs (seed={args.seed})...")
    records = [generate_poi(rng) for _ in range(args.n)]

    json_path = out_dir / "poi_dataset.json"
    with open(json_path, "w") as f:
        json.dump(records, f, indent=None, separators=(",", ":"))
    print(f"Wrote JSON: {json_path} ({json_path.stat().st_size / 1_000_000:.1f} MB)")

    db_path = out_dir / "poi_dataset.db"
    build_sqlite_db(records, db_path)
    print(f"Wrote SQLite DB: {db_path} ({db_path.stat().st_size / 1_000_000:.1f} MB)")

    # quick sanity summary
    from collections import Counter
    cat_counts = Counter(r["category"] for r in records)
    print("\nCategory distribution (top 5):")
    for cat, cnt in cat_counts.most_common(5):
        print(f"  {cat:20s} {cnt:,}")


if __name__ == "__main__":
    main()