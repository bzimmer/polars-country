import math

import polars as pl
import pytest

from polars_country import code, _polars_country as gc

# ---------------------------------------------------------------------------
# Native extension (_geo_country)
# ---------------------------------------------------------------------------


class TestCountryCode:
    def test_zurich(self):
        assert gc.country_code(47.37, 8.54) == "CH"

    def test_paris(self):
        assert gc.country_code(48.85, 2.35) == "FR"

    def test_texas(self):
        assert gc.country_code(33.0, -97.0) == "US"

    def test_sydney(self):
        assert gc.country_code(-33.87, 151.21) == "AU"

    def test_london(self):
        assert gc.country_code(51.50, -0.12) == "GB"

    def test_tokyo(self):
        assert gc.country_code(35.68, 139.69) == "JP"

    def test_open_ocean_returns_none(self):
        assert gc.country_code(0.0, 0.0) is None

    def test_invalid_lat_raises(self):
        with pytest.raises(ValueError):
            gc.country_code(91.0, 0.0)

    def test_pole_returns_none(self):
        # Poles are valid coordinates but are not in any country.
        assert gc.country_code(90.0, 0.0) is None
        assert gc.country_code(-90.0, 0.0) is None


class TestCountryCodes:
    def test_batch(self):
        result = gc.country_codes([47.37, 48.85, 33.0], [8.54, 2.35, -97.0])
        assert result == ["CH", "FR", "US"]

    def test_ocean_in_batch(self):
        assert gc.country_codes([0.0], [0.0]) == [None]

    def test_empty(self):
        assert gc.country_codes([], []) == []

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            gc.country_codes([1.0, 2.0], [1.0])


# ---------------------------------------------------------------------------
# Polars expression API
# ---------------------------------------------------------------------------

DF = pl.DataFrame(
    {
        "lat": [47.37, 48.85, 33.0, -33.87, 0.0, 51.50],
        "lng": [8.54, 2.35, -97.0, 151.21, 0.0, -0.12],
    }
)
EXPECTED = ["CH", "FR", "US", "AU", None, "GB"]


class TestSeparateColumns:
    def test_column_names(self):
        result = DF.with_columns(code(["lat", "lng"]).alias("country"))
        assert result["country"].to_list() == EXPECTED

    def test_series_inputs(self):
        result = DF.with_columns(code([DF["lat"], DF["lng"]]).alias("country"))
        assert result["country"].to_list() == EXPECTED

    def test_expr_inputs(self):
        result = DF.with_columns(code([pl.col("lat"), pl.col("lng")]).alias("country"))
        assert result["country"].to_list() == EXPECTED

    def test_lazy_pipeline(self):
        result = (
            DF.lazy()
            .with_columns(code(["lat", "lng"]).alias("country"))
            .filter(pl.col("country") == "CH")
            .collect()
        )
        assert result.shape == (1, 3)
        assert result["country"][0] == "CH"

    def test_output_dtype_is_string(self):
        result = DF.with_columns(code(["lat", "lng"]).alias("country"))
        assert result["country"].dtype == pl.String

    def test_ocean_produces_null(self):
        ocean = pl.DataFrame({"lat": [0.0], "lng": [0.0]})
        result = ocean.with_columns(code(["lat", "lng"]).alias("country"))
        assert result["country"][0] is None

    def test_null_inputs_propagate_as_null(self):
        df = pl.DataFrame({"lat": [47.37, None], "lng": [8.54, None]})
        result = df.with_columns(code(["lat", "lng"]).alias("country"))
        assert result["country"][0] == "CH"
        assert result["country"][1] is None

    def test_nan_inputs_produce_null(self):
        df = pl.DataFrame({"lat": [47.37, math.nan], "lng": [8.54, math.nan]})
        result = df.with_columns(code(["lat", "lng"]).alias("country"))
        assert result["country"][0] == "CH"
        assert result["country"][1] is None

    def test_float32_columns(self):
        df = pl.DataFrame(
            {
                "lat": pl.Series([47.37], dtype=pl.Float32),
                "lng": pl.Series([8.54], dtype=pl.Float32),
            }
        )
        result = df.with_columns(code(["lat", "lng"]).alias("country"))
        assert result["country"][0] == "CH"

    def test_wrong_length_raises(self):
        with pytest.raises(ValueError, match="exactly 2"):
            code(["lat", "lng", "extra"])


class TestListColumn:
    def test_column_name(self):
        df = pl.DataFrame({"latlng": [[47.37, 8.54], [48.85, 2.35], [0.0, 0.0]]})
        result = df.with_columns(code("latlng").alias("country"))
        assert result["country"].to_list() == ["CH", "FR", None]

    def test_expr(self):
        df = pl.DataFrame({"latlng": [[47.37, 8.54], [33.0, -97.0]]})
        result = df.with_columns(code(pl.col("latlng")).alias("country"))
        assert result["country"].to_list() == ["CH", "US"]

    def test_series(self):
        s = pl.Series("latlng", [[51.50, -0.12], [-33.87, 151.21]])
        df = pl.DataFrame({"latlng": s})
        result = df.with_columns(code(df["latlng"]).alias("country"))
        assert result["country"].to_list() == ["GB", "AU"]

    def test_lazy_pipeline(self):
        df = pl.DataFrame({"latlng": [[47.37, 8.54], [48.85, 2.35]]})
        result = (
            df.lazy()
            .with_columns(code("latlng").alias("country"))
            .filter(pl.col("country") == "CH")
            .collect()
        )
        assert result.shape == (1, 2)
        assert result["country"][0] == "CH"

    def test_null_element_produces_null(self):
        df = pl.DataFrame({"latlng": [[47.37, 8.54], None]})
        result = df.with_columns(code("latlng").alias("country"))
        assert result["country"][0] == "CH"
        assert result["country"][1] is None

    def test_short_lists_produce_null(self):
        df = pl.DataFrame({"latlng": [[47.37, 8.54], [], [47.37]]})
        result = df.with_columns(code("latlng").alias("country"))
        assert result["country"][0] == "CH"
        assert result["country"][1] is None  # empty list
        assert result["country"][2] is None  # single element

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError):
            code(42)  # type: ignore[arg-type]
