"""
Polars expression helper for offline country-code lookup.

Wraps the ``_geo_country`` native extension (backed by country-boundaries-rust)
and exposes a single ``code`` expression that plugs into any Polars query.

    import polars as pl
    import polars_country as pc

    # Single [lat, lng] list column
    df = pl.DataFrame({"latlng": [[47.37, 8.54], [48.85, 2.35]]})
    df.with_columns(pc.code("latlng").alias("country"))

    # Two separate columns
    df = pl.DataFrame({"lat": [47.37, 48.85], "lng": [8.54, 2.35]})
    df.with_columns(pc.code(["lat", "lng"]).alias("country"))
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import polars as pl
from polars.plugins import register_plugin_function

_LIB = Path(__file__).parent

IntoExpr = str | pl.Expr | pl.Series


def code(coords: IntoExpr | Sequence[IntoExpr]) -> pl.Expr:
    """Return a Polars expression resolving to the ISO 3166-1 alpha-2 country code.

    Accepts either a single list/array column containing ``[lat, lng]`` per row,
    or a two-element sequence of separate lat and lng inputs.

    Implemented as a native Polars expression plugin: Arrow buffers are passed
    directly to Rust with no Python-level materialisation.  Null or non-finite
    (NaN / ±inf) values in either input produce null output.

    Parameters
    ----------
    coords:
        One of:

        - A column name, ``pl.Expr``, or ``pl.Series`` whose values are
          ``[lat, lng]`` lists or arrays.
        - A two-element sequence ``[lat, lng]`` where each element is a column
          name, ``pl.Expr``, or ``pl.Series``.

    Returns
    -------
    pl.Expr
        A lazy expression of dtype ``pl.String``.

    Examples
    --------
    Single ``[lat, lng]`` list column:

    >>> import polars as pl
    >>> import polars_country as pc
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

    Two separate columns:

    >>> df = pl.DataFrame({"lat": [47.37, 48.85], "lng": [8.54, 2.35]})
    >>> df.with_columns(pc.code(["lat", "lng"]).alias("country"))
    shape: (2, 3)
    ┌───────┬──────┬─────────┐
    │ lat   ┆ lng  ┆ country │
    │ ---   ┆ ---  ┆ ---     │
    │ f64   ┆ f64  ┆ str     │
    ╞═══════╪══════╪═════════╡
    │ 47.37 ┆ 8.54 ┆ CH      │
    │ 48.85 ┆ 2.35 ┆ FR      │
    └───────┴──────┴─────────┘
    """
    if isinstance(coords, (list, tuple)):
        if len(coords) != 2:
            raise ValueError(f"coords sequence must have exactly 2 elements, got {len(coords)}")
        lat_expr = _to_expr(coords[0])
        lng_expr = _to_expr(coords[1])
    else:
        latlng = _to_expr(coords)
        lat_expr = latlng.list.get(0)
        lng_expr = latlng.list.get(1)

    return register_plugin_function(
        plugin_path=_LIB,
        function_name="geo_country_code",
        args=[lat_expr, lng_expr],
        is_elementwise=True,
    )


def _to_expr(x: IntoExpr) -> pl.Expr:
    if isinstance(x, str):
        return pl.col(x)
    if isinstance(x, pl.Series):
        return pl.lit(x)
    if isinstance(x, pl.Expr):
        return x
    raise TypeError(f"Expected str, pl.Series or pl.Expr, got {type(x)}")
