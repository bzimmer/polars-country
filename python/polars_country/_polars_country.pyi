def country_code(lat: float, lng: float) -> str | None:
    """Return the ISO 3166-1 alpha-2 code for (lat, lng), or None for open ocean.

    Raises ValueError for out-of-range or non-finite coordinates (|lat| > 90,
    NaN, ±inf).
    """
    ...

def country_codes(lats: list[float], lngs: list[float]) -> list[str | None]:
    """Vectorised form of country_code over two equal-length lists.

    Raises ValueError if len(lats) != len(lngs), any coordinate is out of
    range, or any coordinate is non-finite (NaN, ±inf).
    """
    ...
