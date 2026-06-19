"""
Polars expression helper for offline country-code lookup.

Wraps the ``_geo_country`` native extension (backed by country-boundaries-rust)
and exposes a single ``code`` expression that plugs into any Polars query.

    import polars as pl
    import polars_country as pc

    # Two separate columns
    df = pl.DataFrame({"lat": [47.37, 48.85], "lng": [8.54, 2.35]})
    df.with_columns(pc.code("lat", "lng").alias("country"))

    # Single list column [lat, lng]
    df = pl.DataFrame({"latlng": [[47.37, 8.54], [48.85, 2.35]]})
    df.with_columns(pc.code("latlng").alias("country"))
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
from polars.plugins import register_plugin_function

_LIB = Path(__file__).parent


def code(
    lat: str | pl.Series | pl.Expr,
    lng: str | pl.Series | pl.Expr | None = None,
) -> pl.Expr:
    """Return a Polars expression resolving to the ISO 3166-1 alpha-2 country code.

    Accepts either two separate lat/lng inputs or a single list/array column
    whose first element is latitude and second is longitude.

    Implemented as a native Polars expression plugin: Arrow buffers are passed
    directly to Rust with no Python-level materialisation.  Null or non-finite
    (NaN / ±inf) values in either input produce null output.

    Parameters
    ----------
    lat:
        Column name, ``pl.Expr``, or ``pl.Series`` containing latitudes —
        **or** a list/array column of shape ``[lat, lng]`` when ``lng`` is omitted.
    lng:
        Column name, ``pl.Expr``, or ``pl.Series`` containing longitudes.
        Omit when ``lat`` refers to a ``[lat, lng]`` list/array column.

    Returns
    -------
    pl.Expr
        A lazy expression of dtype ``pl.String``.

    Examples
    --------
    Two separate columns:

    >>> import polars as pl
    >>> import polars_country as pc
    >>> df = pl.DataFrame({"lat": [47.37, 48.85], "lng": [8.54, 2.35]})
    >>> df.with_columns(pc.code("lat", "lng").alias("country"))
    shape: (2, 3)
    ┌───────┬──────┬─────────┐
    │ lat   ┆ lng  ┆ country │
    │ ---   ┆ ---  ┆ ---     │
    │ f64   ┆ f64  ┆ str     │
    ╞═══════╪══════╪═════════╡
    │ 47.37 ┆ 8.54 ┆ CH      │
    │ 48.85 ┆ 2.35 ┆ FR      │
    └───────┴──────┴─────────┘

    Single ``[lat, lng]`` list column:

    >>> df = pl.DataFrame({"latlng": [[47.37, 8.54], [48.85, 2.35]]})
    >>> df.with_columns(pc.code("latlng").alias("country"))
    shape: (2, 2)
    ┌──────────────┬─────────┐
    │ latlng       ┆ country │
    │ ---          ┆ ---     │
    │ list[f64]    ┆ str     │
    ╞══════════════╪═════════╡
    │ [47.37, 8.54]┆ CH      │
    │ [48.85, 2.35]┆ FR      │
    └──────────────┴─────────┘
    """
    if lng is None:
        latlng = _to_expr(lat)
        lat_expr = latlng.list.get(0)
        lng_expr = latlng.list.get(1)
    else:
        lat_expr = _to_expr(lat)
        lng_expr = _to_expr(lng)

    return register_plugin_function(
        plugin_path=_LIB,
        function_name="geo_country_code",
        args=[lat_expr, lng_expr],
        is_elementwise=True,
    )


def _to_expr(x: str | pl.Series | pl.Expr) -> pl.Expr:
    if isinstance(x, str):
        return pl.col(x)
    if isinstance(x, pl.Series):
        return pl.lit(x)
    if isinstance(x, pl.Expr):
        return x
    raise TypeError(f"Expected str, pl.Series or pl.Expr, got {type(x)}")
