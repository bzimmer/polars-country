use country_boundaries::{BOUNDARIES_ODBL_360X180, CountryBoundaries, LatLon};
use polars_core::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::derive::polars_expr;
use rayon::prelude::*;
use std::sync::OnceLock;

static BOUNDARIES: OnceLock<CountryBoundaries> = OnceLock::new();

fn boundaries() -> &'static CountryBoundaries {
    BOUNDARIES.get_or_init(|| {
        CountryBoundaries::from_reader(BOUNDARIES_ODBL_360X180)
            .expect("bundled boundary data must be valid")
    })
}

/// Resolves a `(lat, lng)` pair in decimal degrees to an ISO 3166-1 alpha-2 country code.
///
/// Returns `Ok(None)` for valid coordinates that lie outside any country boundary (open ocean,
/// poles). Returns `Err` for coordinates outside the valid range (lat ∉ [−90, 90]).
///
/// The returned `&str` borrows from the static boundary data — no allocation on the hot path.
fn lookup(lat: f64, lng: f64) -> Result<Option<&'static str>, String> {
    let ll = LatLon::new(lat, lng).map_err(|e| format!("invalid lat/lon ({lat},{lng}): {e}"))?;
    Ok(boundaries()
        .ids(ll)
        .into_iter()
        // ids() returns e.g. ["US-TX", "US"]; pick the bare 2-letter country code.
        .find(|id| id.len() == 2 && !id.contains('-')))
}

/// Polars expression plugin — receives two Arrow Series, returns a null-safe String Series.
/// Nulls and non-finite coordinates (NaN, ±inf) in either input produce null output.
#[polars_expr(output_type = String)]
fn polars_country_code(inputs: &[Series]) -> PolarsResult<Series> {
    let lat_s = inputs[0].cast(&DataType::Float64)?;
    let lat = lat_s.f64()?;
    let lng_s = inputs[1].cast(&DataType::Float64)?;
    let lng = lng_s.f64()?;
    let len = lat.len();

    // Process rows in parallel; the geospatial lookup is CPU-bound and
    // country_boundaries is stateless after initialisation.
    let values: PolarsResult<Vec<Option<&'static str>>> = (0..len)
        .into_par_iter()
        .map(|i| -> PolarsResult<Option<&'static str>> {
            match (lat.get(i), lng.get(i)) {
                (Some(la), Some(lo)) if la.is_finite() && lo.is_finite() => {
                    lookup(la, lo).map_err(|e| polars_err!(InvalidOperation: "{e}"))
                }
                _ => Ok(None),
            }
        })
        .collect();

    // Pre-allocate the output buffer with the known row count.
    let mut builder = StringChunkedBuilder::new("country".into(), len);
    for v in values? {
        builder.append_option(v);
    }
    Ok(builder.finish().into_series())
}

#[pymodule]
fn _polars_country(_m: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_cities() {
        assert_eq!(lookup(47.37, 8.54).unwrap(), Some("CH")); // Zurich
        assert_eq!(lookup(48.85, 2.35).unwrap(), Some("FR")); // Paris
        assert_eq!(lookup(33.0, -97.0).unwrap(), Some("US")); // Texas
        assert_eq!(lookup(-33.87, 151.21).unwrap(), Some("AU")); // Sydney
        assert_eq!(lookup(51.50, -0.12).unwrap(), Some("GB")); // London
        assert_eq!(lookup(35.68, 139.69).unwrap(), Some("JP")); // Tokyo
        assert_eq!(lookup(55.75, 37.62).unwrap(), Some("RU")); // Moscow
    }

    #[test]
    fn open_ocean_returns_none() {
        assert_eq!(lookup(0.0, 0.0).unwrap(), None);
    }

    #[test]
    fn invalid_lat_is_err() {
        assert!(lookup(91.0, 0.0).is_err());
        assert!(lookup(-91.0, 0.0).is_err());
    }

    #[test]
    fn non_finite_is_err() {
        assert!(lookup(f64::NAN, 0.0).is_err());
        assert!(lookup(f64::INFINITY, 0.0).is_err());
        assert!(lookup(f64::NEG_INFINITY, 0.0).is_err());
    }

    #[test]
    fn boundary_lat_is_valid() {
        assert!(lookup(90.0, 0.0).is_ok());
        assert!(lookup(-90.0, 0.0).is_ok());
    }
}
