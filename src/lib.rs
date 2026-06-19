use country_boundaries::{CountryBoundaries, LatLon, BOUNDARIES_ODBL_360X180};
use polars_core::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_polars::derive::polars_expr;
use std::sync::OnceLock;

static BOUNDARIES: OnceLock<CountryBoundaries> = OnceLock::new();

fn boundaries() -> &'static CountryBoundaries {
    BOUNDARIES.get_or_init(|| {
        CountryBoundaries::from_reader(BOUNDARIES_ODBL_360X180)
            .expect("bundled boundary data must be valid")
    })
}

// Pure Rust core — testable without a Python interpreter.
fn lookup(lat: f64, lng: f64) -> Result<Option<String>, String> {
    let ll = LatLon::new(lat, lng).map_err(|e| format!("invalid lat/lon ({lat},{lng}): {e}"))?;
    let code = boundaries()
        .ids(ll)
        .into_iter()
        // ids() returns e.g. ["US-TX", "US"]; pick the bare 2-letter country code.
        .find(|id| id.len() == 2 && !id.contains('-'))
        .map(|s| s.to_string());
    Ok(code)
}

/// Polars expression plugin — receives two Arrow Series, returns a null-safe String Series.
/// Nulls and non-finite coordinates (NaN, ±inf) in either input produce null output.
#[polars_expr(output_type = String)]
fn geo_country_code(inputs: &[Series]) -> PolarsResult<Series> {
    let lat_s = inputs[0].cast(&DataType::Float64)?;
    let lat = lat_s.f64()?;
    let lng_s = inputs[1].cast(&DataType::Float64)?;
    let lng = lng_s.f64()?;
    let out: StringChunked = lat
        .iter()
        .zip(lng.iter())
        .map(|(lat, lng)| -> PolarsResult<Option<String>> {
            match (lat, lng) {
                (Some(lat), Some(lng)) if lat.is_finite() && lng.is_finite() => {
                    lookup(lat, lng).map_err(|e| polars_err!(InvalidOperation: "{e}"))
                }
                _ => Ok(None),
            }
        })
        .collect::<PolarsResult<StringChunked>>()?;
    Ok(out.into_series())
}

/// Low-level single-point lookup exposed to Python directly.
/// Raises ValueError for out-of-range coordinates.
#[pyfunction]
fn country_code(lat: f64, lng: f64) -> PyResult<Option<String>> {
    lookup(lat, lng).map_err(PyValueError::new_err)
}

/// Low-level vectorised lookup exposed to Python directly.
/// Accepts two equal-length lists of floats; raises ValueError on length mismatch
/// or out-of-range coordinates.
#[pyfunction]
fn country_codes(lats: Vec<f64>, lngs: Vec<f64>) -> PyResult<Vec<Option<String>>> {
    if lats.len() != lngs.len() {
        return Err(PyValueError::new_err(
            "lats and lngs must have the same length",
        ));
    }
    lats.iter()
        .zip(lngs.iter())
        .map(|(&lat, &lng)| lookup(lat, lng).map_err(PyValueError::new_err))
        .collect()
}

#[pymodule]
fn _polars_country(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(country_code, m)?)?;
    m.add_function(wrap_pyfunction!(country_codes, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_cities() {
        assert_eq!(lookup(47.37, 8.54).unwrap(), Some("CH".into())); // Zurich
        assert_eq!(lookup(48.85, 2.35).unwrap(), Some("FR".into())); // Paris
        assert_eq!(lookup(33.0, -97.0).unwrap(), Some("US".into())); // Texas
        assert_eq!(lookup(-33.87, 151.21).unwrap(), Some("AU".into())); // Sydney
        assert_eq!(lookup(51.50, -0.12).unwrap(), Some("GB".into())); // London
        assert_eq!(lookup(35.68, 139.69).unwrap(), Some("JP".into())); // Tokyo
        assert_eq!(lookup(55.75, 37.62).unwrap(), Some("RU".into())); // Moscow
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
    fn boundary_lat_is_valid() {
        assert!(lookup(90.0, 0.0).is_ok());
        assert!(lookup(-90.0, 0.0).is_ok());
    }
}
