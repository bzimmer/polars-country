import math

import polars as pl
import pytest

from polars_country import code

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
    def test_column_names(self) -> None:
        result = DF.with_columns(code(["lat", "lng"]).alias("country"))
        assert result["country"].to_list() == EXPECTED

    def test_expr_inputs(self) -> None:
        result = DF.with_columns(code([pl.col("lat"), pl.col("lng")]).alias("country"))
        assert result["country"].to_list() == EXPECTED

    def test_lazy_pipeline(self) -> None:
        result = (
            DF.lazy()
            .with_columns(code(["lat", "lng"]).alias("country"))
            .filter(pl.col("country") == "CH")
            .collect()
        )
        assert result.shape == (1, 3)
        assert result["country"][0] == "CH"

    def test_output_dtype_is_string(self) -> None:
        result = DF.with_columns(code(["lat", "lng"]).alias("country"))
        assert result["country"].dtype == pl.String

    def test_ocean_produces_null(self) -> None:
        ocean = pl.DataFrame({"lat": [0.0], "lng": [0.0]})
        result = ocean.with_columns(code(["lat", "lng"]).alias("country"))
        assert result["country"][0] is None

    def test_null_inputs_propagate_as_null(self) -> None:
        df = pl.DataFrame({"lat": [47.37, None], "lng": [8.54, None]})
        result = df.with_columns(code(["lat", "lng"]).alias("country"))
        assert result["country"][0] == "CH"
        assert result["country"][1] is None

    def test_nan_inputs_produce_null(self) -> None:
        df = pl.DataFrame({"lat": [47.37, math.nan], "lng": [8.54, math.nan]})
        result = df.with_columns(code(["lat", "lng"]).alias("country"))
        assert result["country"][0] == "CH"
        assert result["country"][1] is None

    def test_float32_columns(self) -> None:
        df = pl.DataFrame(
            {
                "lat": pl.Series([47.37], dtype=pl.Float32),
                "lng": pl.Series([8.54], dtype=pl.Float32),
            }
        )
        result = df.with_columns(code(["lat", "lng"]).alias("country"))
        assert result["country"][0] == "CH"

    def test_integer_columns(self) -> None:
        df = pl.DataFrame(
            {
                "lat": pl.Series([47, 48], dtype=pl.Int32),
                "lng": pl.Series([8, 2], dtype=pl.Int32),
            }
        )
        result = df.with_columns(code(["lat", "lng"]).alias("country"))
        assert result["country"].to_list() == ["CH", "FR"]

    def test_wrong_length_raises(self) -> None:
        with pytest.raises(ValueError, match="exactly 2"):
            code(["lat", "lng", "extra"])


class TestListColumn:
    def test_column_name(self) -> None:
        df = pl.DataFrame({"latlng": [[47.37, 8.54], [48.85, 2.35], [0.0, 0.0]]})
        result = df.with_columns(code("latlng").alias("country"))
        assert result["country"].to_list() == ["CH", "FR", None]

    def test_expr(self) -> None:
        df = pl.DataFrame({"latlng": [[47.37, 8.54], [33.0, -97.0]]})
        result = df.with_columns(code(pl.col("latlng")).alias("country"))
        assert result["country"].to_list() == ["CH", "US"]

    def test_lazy_pipeline(self) -> None:
        df = pl.DataFrame({"latlng": [[47.37, 8.54], [48.85, 2.35]]})
        result = (
            df.lazy()
            .with_columns(code("latlng").alias("country"))
            .filter(pl.col("country") == "CH")
            .collect()
        )
        assert result.shape == (1, 2)
        assert result["country"][0] == "CH"

    def test_null_element_produces_null(self) -> None:
        df = pl.DataFrame({"latlng": [[47.37, 8.54], None]})
        result = df.with_columns(code("latlng").alias("country"))
        assert result["country"][0] == "CH"
        assert result["country"][1] is None

    def test_short_lists_produce_null(self) -> None:
        df = pl.DataFrame({"latlng": [[47.37, 8.54], [], [47.37]]})
        result = df.with_columns(code("latlng").alias("country"))
        assert result["country"][0] == "CH"
        assert result["country"][1] is None  # empty list
        assert result["country"][2] is None  # single element

    def test_array_column(self) -> None:
        df = pl.DataFrame(
            {
                "latlng": pl.Series([[47.37, 8.54], [48.85, 2.35]]).cast(
                    pl.Array(pl.Float64, 2)
                )
            }
        )
        result = df.with_columns(code("latlng").alias("country"))
        assert result["country"].to_list() == ["CH", "FR"]

    def test_invalid_type_raises(self) -> None:
        with pytest.raises(AssertionError):
            code(42)  # type: ignore[arg-type]
