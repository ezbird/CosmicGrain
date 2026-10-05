#!/usr/bin/env python3
"""Load and validate the frozen CosmicGrain paper analysis contract.

Figure scripts can use::

    from cosmicgrain_paper_contract import load_contract, measurement

    contract, provenance = load_contract()
    cfg = measurement(contract, "suite_extinction")

Run this file directly to validate, inspect, or export the resolved contract::

    python3 scripts/cosmicgrain_paper_contract.py --validate
    python3 scripts/cosmicgrain_paper_contract.py --show suite_extinction
    python3 scripts/cosmicgrain_paper_contract.py --export-json contract.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover - environment-dependent
    raise SystemExit(
        "PyYAML is required to read cosmicgrain_paper_contract.yaml. "
        "Install the python3-yaml system package or PyYAML in your venv."
    ) from exc


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONTRACT = SCRIPT_DIR / "cosmicgrain_paper_contract.yaml"

MEASUREMENT_SECTIONS = {
    "halo_identity",
    "integrated_galaxy",
    "orientation",
    "grain_size_distribution",
    "metallicity",
    "dust_ratios",
    "suite_extinction",
    "cg10_grain_rv",
    "zhang_green_comparison",
    "statistics",
    "optical_constants",
}


class ContractError(ValueError):
    """Raised when the analysis contract is missing or internally invalid."""


def _positive(value: Any, key: str) -> None:
    if not isinstance(value, (int, float)) or value <= 0:
        raise ContractError(f"{key} must be a positive number; got {value!r}")


def _fraction(value: Any, key: str) -> None:
    if not isinstance(value, (int, float)) or not 0 <= value < 1:
        raise ContractError(f"{key} must lie in [0, 1); got {value!r}")


def _require(mapping: dict[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise ContractError(f"Missing required key {context}.{key}")
    return mapping[key]


def validate_contract(data: dict[str, Any]) -> None:
    if not isinstance(data, dict):
        raise ContractError("Top-level YAML value must be a mapping")
    header = _require(data, "contract", "contract")
    if header.get("status") != "frozen":
        raise ContractError("contract.status must be 'frozen' for paper products")
    for key in ("name", "version", "frozen_date", "policy"):
        _require(header, key, "contract")

    missing = sorted(MEASUREMENT_SECTIONS - set(data))
    if missing:
        raise ContractError("Missing measurement sections: " + ", ".join(missing))

    _positive(data["integrated_galaxy"]["aperture_pkpc"],
              "integrated_galaxy.aperture_pkpc")
    _positive(data["orientation"]["radius_pkpc"], "orientation.radius_pkpc")
    gsd = data["grain_size_distribution"]
    for key in ("aperture_pkpc", "minimum_radius_nm", "maximum_radius_nm",
                "logarithmic_bins", "small_grain_threshold_nm"):
        _positive(gsd[key], f"grain_size_distribution.{key}")
    if gsd["minimum_radius_nm"] >= gsd["maximum_radius_nm"]:
        raise ContractError("GSD minimum radius must be smaller than maximum")

    for section in ("suite_extinction", "cg10_grain_rv"):
        cfg = data[section]
        _positive(cfg["pixel_size_pkpc"], f"{section}.pixel_size_pkpc")
        _positive(cfg["minimum_dust_particles_per_pixel"],
                  f"{section}.minimum_dust_particles_per_pixel")
        _fraction(cfg["minimum_tauV_fraction_of_aperture_maximum"],
                  f"{section}.minimum_tauV_fraction_of_aperture_maximum")
        _fraction(cfg["minimum_EBV_fraction_of_aperture_maximum"],
                  f"{section}.minimum_EBV_fraction_of_aperture_maximum")
        rv_range = cfg["displayed_RV_range"]
        if not (isinstance(rv_range, list) and len(rv_range) == 2
                and rv_range[0] < rv_range[1]):
            raise ContractError(f"{section}.displayed_RV_range must be [low, high]")

    registry = _require(data, "halo_registry", "halo_registry")
    expected_labels = {f"CG{i:02d}" for i in range(1, 13)}
    if set(registry) != expected_labels:
        raise ContractError(
            "halo_registry must contain exactly CG01 through CG12; got "
            + ", ".join(sorted(registry))
        )
    if registry["CG07"] != 1653:
        raise ContractError("CG07 must use current parent halo ID 1653")
    if len(set(registry.values())) != 12:
        raise ContractError("Every halo-registry parent ID must be unique")

    figures = _require(data, "paper_figures", "paper_figures")
    for number in range(1, 7):
        name = f"figure_{number}"
        fig = _require(figures, name, "paper_figures")
        sections = _require(fig, "measurements", f"paper_figures.{name}")
        unknown = sorted(set(sections) - MEASUREMENT_SECTIONS)
        if unknown:
            raise ContractError(f"{name} references unknown sections: {unknown}")


def load_contract(path: str | Path = DEFAULT_CONTRACT
                  ) -> tuple[dict[str, Any], dict[str, str]]:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Paper analysis contract not found: {path}")
    raw = path.read_bytes()
    data = yaml.safe_load(raw)
    validate_contract(data)
    provenance = {
        "contract_name": str(data["contract"]["name"]),
        "contract_version": str(data["contract"]["version"]),
        "contract_sha256": hashlib.sha256(raw).hexdigest(),
        "contract_path": str(path),
    }
    return data, provenance


def measurement(contract: dict[str, Any], section: str) -> dict[str, Any]:
    if section not in MEASUREMENT_SECTIONS:
        raise ContractError(f"Unknown measurement section: {section}")
    value = contract.get(section)
    if not isinstance(value, dict):
        raise ContractError(f"Measurement section is not a mapping: {section}")
    return value


def figure_contract(contract: dict[str, Any], figure_number: int
                    ) -> dict[str, Any]:
    key = f"figure_{figure_number}"
    if key not in contract["paper_figures"]:
        raise ContractError(f"Unknown paper figure: {figure_number}")
    assignment = contract["paper_figures"][key]
    return {
        "figure": key,
        "title": assignment["title"],
        "measurements": {
            name: measurement(contract, name)
            for name in assignment["measurements"]
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--show", choices=sorted(MEASUREMENT_SECTIONS))
    parser.add_argument("--figure", type=int, choices=range(1, 7))
    parser.add_argument("--export-json", type=Path)
    args = parser.parse_args()

    contract, provenance = load_contract(args.contract)
    if args.show:
        print(yaml.safe_dump({args.show: measurement(contract, args.show)},
                             sort_keys=False).rstrip())
    elif args.figure:
        print(yaml.safe_dump(figure_contract(contract, args.figure),
                             sort_keys=False).rstrip())
    else:
        print(
            f"VALID: {provenance['contract_name']} "
            f"v{provenance['contract_version']} "
            f"sha256={provenance['contract_sha256']}"
        )

    if args.export_json:
        payload = {"provenance": provenance, "contract": contract}
        args.export_json.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"Wrote {args.export_json.resolve()}")


if __name__ == "__main__":
    main()
