# polars-country — fast offline country-code lookup for Polars

A PyO3-based native extension wrapping
[country-boundaries-rust](https://github.com/westnordost/country-boundaries-rust).
Resolves (lat, lng) → ISO 3166-1 alpha-2 country code entirely offline, at
~10 million lookups/second on a single thread.

## Layout

```
polars-country/
├── Cargo.toml
├── pyproject.toml
├── Taskfile.yml
├── .gitignore
├── src/
│   └── lib.rs                          # Rust/PyO3 extension
└── python/
    └── polars_country/
        ├── __init__.py                 # Polars expression helper
        └── _geo_country.pyi            # Type stubs for the native extension
```

## Development

```bash
task build   # uv sync + maturin develop --release
task test    # cargo test + pytest
task clean   # cargo clean + remove Python cache dirs
```

Build a wheel for distribution:

```bash
uv run maturin build --release
uv pip install target/wheels/polars_country-*.whl
```

## Usage

### Low-level (native extension directly)

```python
from polars_country import _geo_country as gc

# Single point
gc.country_code(47.37, 8.54)          # "CH"
gc.country_code(0.0, 0.0)             # None  (open ocean)

# Vectorised — accepts plain Python lists
codes = gc.country_codes(
    [47.37, 48.85, 33.0],
    [ 8.54,  2.35, -97.0],
)  # ["CH", "FR", "US"]
```

### Polars expression API

`pc.code` accepts either two separate columns **or** a single `[lat, lng]` list column:

```python
import polars as pl
import polars_country as pc

df = pl.DataFrame({
    "lat": [47.37, 48.85, 33.0, -33.87, 0.0,  51.50],
    "lng": [ 8.54,  2.35, -97.0, 151.21, 0.0, -0.12],
})

# 1. Column names (most common)
df.with_columns(pc.code("lat", "lng").alias("country"))
# ┌────────┬────────┬─────────┐
# │ lat    ┆ lng    ┆ country │
# │ f64    ┆ f64    ┆ str     │
# ╞════════╪════════╪═════════╡
# │  47.37 ┆   8.54 ┆ CH      │
# │  48.85 ┆   2.35 ┆ FR      │
# │  33.0  ┆ -97.0  ┆ US      │
# │ -33.87 ┆ 151.21 ┆ AU      │
# │   0.0  ┆   0.0  ┆ null    │  ← open ocean
# │  51.5  ┆  -0.12 ┆ GB      │
# └────────┴────────┴─────────┘

# 2. pl.Series
df.with_columns(pc.code(df["lat"], df["lng"]).alias("country"))

# 3. pl.Expr
df.with_columns(pc.code(pl.col("lat"), pl.col("lng")).alias("country"))

# 5. Single [lat, lng] list column
df = pl.DataFrame({"latlng": [[47.37, 8.54], [48.85, 2.35]]})
df.with_columns(pc.code("latlng").alias("country"))

# 4. Works inside lazy pipelines
(
    df.lazy()
    .with_columns(pc.code("lat", "lng").alias("country"))
    .filter(pl.col("country") == "CH")
    .collect()
)
```

## Notes

- The bundled boundary data (`BOUNDARIES_ODBL_360X180`) is © OpenStreetMap
  contributors, licensed **ODbL**. Attribution is required.
- Sea borders are not modelled; positions in open ocean return `null`.
- The `CountryBoundaries` instance is a process-wide singleton (loaded once
  via `OnceLock`). Subsequent calls have zero startup cost.
- For subdivision codes (e.g. `"US-TX"`) extend `lookup` in `src/lib.rs` to
  return the full `ids()` list rather than filtering to the 2-letter code.
