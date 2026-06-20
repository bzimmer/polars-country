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
        └── _polars_country.pyi         # Type stubs for the native extension
```

## Development

```bash
task build    # install Python deps + compile Rust extension
task fmt      # ruff format/isort + cargo fmt
task test     # cargo test + pytest (with coverage)
task release  # run tests, then build a wheel into target/wheels/
task install  # release, then install the wheel into the current venv
task clean    # cargo clean + remove Python cache dirs
```

## Usage

### Low-level (native extension directly)

```python
from polars_country import _polars_country as pc

# Single point
pc.country_code(47.37, 8.54)          # "CH"
pc.country_code(0.0, 0.0)             # None  (open ocean)

# Vectorised — accepts plain Python lists
codes = pc.country_codes(
    [47.37, 48.85, 33.0],
    [ 8.54,  2.35, -97.0],
)  # ["CH", "FR", "US"]
```

### Polars expression API

`pc.code` follows the same convention as `pl.concat_list` and `pl.struct`: pass a
single expression for a combined column, or a list for separate columns.

```python
import polars as pl
import polars_country as pc

# 1. Single [lat, lng] list column (most common)
df = pl.DataFrame({"latlng": [[47.37, 8.54], [48.85, 2.35], [0.0, 0.0]]})
df.with_columns(pc.code("latlng").alias("country"))
# ┌───────────────┬─────────┐
# │ latlng        ┆ country │
# │ ---           ┆ ---     │
# │ list[f64]     ┆ str     │
# ╞═══════════════╪═════════╡
# │ [47.37, 8.54] ┆ CH      │
# │ [48.85, 2.35] ┆ FR      │
# │ [0.0, 0.0]    ┆ null    │  ← open ocean
# └───────────────┴─────────┘

# 2. Null rows in the list column produce null output
df = pl.DataFrame({"latlng": [[47.37, 8.54], None, [48.85, 2.35]]})
df.with_columns(pc.code("latlng").alias("country"))
# ┌───────────────┬─────────┐
# │ latlng        ┆ country │
# │ ---           ┆ ---     │
# │ list[f64]     ┆ str     │
# ╞═══════════════╪═════════╡
# │ [47.37, 8.54] ┆ CH      │
# │ null          ┆ null    │
# │ [48.85, 2.35] ┆ FR      │
# └───────────────┴─────────┘

# 3. Two separate columns
df = pl.DataFrame({
    "lat": [47.37, 48.85, 33.0, -33.87, 0.0,  51.50],
    "lng": [ 8.54,  2.35, -97.0, 151.21, 0.0, -0.12],
})
df.with_columns(pc.code(["lat", "lng"]).alias("country"))

# 4. pl.Expr or pl.Series inputs
df.with_columns(pc.code([pl.col("lat"), pl.col("lng")]).alias("country"))

# 5. Works inside lazy pipelines
(
    df.lazy()
    .with_columns(pc.code(["lat", "lng"]).alias("country"))
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
