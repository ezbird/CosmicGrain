#!/usr/bin/env python3
"""Shared CosmicGrain extinction-law definitions.

CosmicGrain optical tables are sampled monochromatically at 0.44 and
0.55 micron.  Their direct ratio is therefore R(55), not broadband Johnson
R(V).  Zhang--Green prescribe the common reporting conversion

    R(V) = 1.1 R(55) + 0.07.

All paper scripts should form R(55) only after summing the relevant optical
depths along a sightline, then apply this conversion to the resulting cell or
projected sightline.
"""
from __future__ import annotations

import numpy as np

B44_MICRON = 0.44
V55_MICRON = 0.55
R55_TO_RV_SLOPE = 1.1
R55_TO_RV_OFFSET = 0.07
CONVENTION = "Zhang--Green: R(V) = 1.1 R(55) + 0.07"


def r55_from_extinction(a44, a55, *, valid=None):
    """Return monochromatic R(55)=A55/(A44-A55).

    ``a44`` and ``a55`` may be extinction, opacity-weighted mass, or optical
    depth, provided both use the same units and spatial aggregation.
    """
    a44, a55 = np.broadcast_arrays(np.asarray(a44, float),
                                   np.asarray(a55, float))
    denominator = a44 - a55
    good = (np.isfinite(a44) & np.isfinite(a55) &
            np.isfinite(denominator) & (a55 > 0) & (denominator > 0))
    if valid is not None:
        good &= np.broadcast_to(np.asarray(valid, bool), good.shape)
    return np.divide(a55, denominator,
                     out=np.full(a55.shape, np.nan, dtype=float),
                     where=good)


def r55_to_rv(r55):
    """Convert native R(55) to the ZG-reported broadband R(V) convention."""
    values = np.asarray(r55, float)
    return np.where(np.isfinite(values),
                    R55_TO_RV_SLOPE * values + R55_TO_RV_OFFSET,
                    np.nan)


def rv_to_r55(rv):
    """Invert the Zhang--Green reporting conversion."""
    values = np.asarray(rv, float)
    return np.where(np.isfinite(values),
                    (values - R55_TO_RV_OFFSET) / R55_TO_RV_SLOPE,
                    np.nan)


def rv_from_extinction(a44, a55, *, valid=None):
    """Form native R(55) from aggregated extinction, then convert to R(V)."""
    return r55_to_rv(r55_from_extinction(a44, a55, valid=valid))


def project_r55(a55, r55, *, axis=0, valid=None):
    """Project differential A55 and R(55) consistently along ``axis``.

    Extinction and E(44-55)=A55/R(55) are additive.  Conversion to R(V)
    belongs after this projection, via :func:`r55_to_rv`.
    """
    a55, r55 = np.broadcast_arrays(np.asarray(a55, float),
                                   np.asarray(r55, float))
    good = (np.isfinite(a55) & np.isfinite(r55) &
            (a55 > 0) & (r55 > 0))
    if valid is not None:
        good &= np.broadcast_to(np.asarray(valid, bool), good.shape)
    differential_e = np.divide(a55, r55,
                               out=np.zeros_like(a55, dtype=float),
                               where=good)
    total_a55 = np.sum(np.where(good, a55, 0.0), axis=axis)
    total_e4455 = np.sum(differential_e, axis=axis)
    projected = np.divide(total_a55, total_e4455,
                          out=np.full_like(total_a55, np.nan, dtype=float),
                          where=total_e4455 > 0)
    return projected


def project_rv(a55, r55, *, axis=0, valid=None):
    """Project native differential quantities and report converted R(V)."""
    return r55_to_rv(project_r55(a55, r55, axis=axis, valid=valid))


def project_reported_rv(a55, rv, *, axis=0, valid=None):
    """Project arrays already expressed in the ZG-reported R(V) convention."""
    return project_rv(a55, rv_to_r55(rv), axis=axis, valid=valid)
