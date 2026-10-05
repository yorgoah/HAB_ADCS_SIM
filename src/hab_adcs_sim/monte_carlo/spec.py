"""Campaign TOML parsing and validation."""

from __future__ import annotations

import itertools
import json
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sim_tools.disturbance import wind_params_with_defaults

from .params import get_path, is_numeric_leaf

REPO_ROOT = Path(__file__).resolve().parents[3]

VALID_DISTS = ("normal", "uniform")
_UNSAFE_NAME = re.compile(r"[^A-Za-z0-9._=+-]")


class SpecError(ValueError):
    pass


@dataclass(frozen=True)
class Dispersion:
    path: str
    dist: str
    sigma: float | None = None
    sigma_rel: float | None = None
    rel: float | None = None
    low: float | None = None
    high: float | None = None
    minimum: float | None = None
    maximum: float | None = None
    link: tuple[str, ...] = ()


@dataclass(frozen=True)
class Case:
    name: str
    overrides: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CampaignSpec:
    name: str
    base_parameters: Path
    base_params: dict
    samples_per_case: int
    include_nominal: bool
    sampling_method: str
    sampling_seed: int
    noise_seed: int
    vary_wind: bool
    overrides: dict[str, Any]
    cases: tuple[Case, ...]
    dispersions: tuple[Dispersion, ...]
    source_text: str

    @property
    def runs_per_case(self) -> int:
        return self.samples_per_case + (1 if self.include_nominal else 0)

    @property
    def total_runs(self) -> int:
        return self.runs_per_case * len(self.cases)


def safe_name(name: str) -> str:
    return _UNSAFE_NAME.sub("_", name)


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SpecError(message)


def _check_path_exists(base_params: dict, path: str, what: str) -> Any:
    try:
        return get_path(base_params, path)
    except KeyError:
        raise SpecError(f"{what} refers to {path!r}, which is not in the parameter file") from None


def _parse_dispersion(path: str, raw: dict, base_params: dict) -> Dispersion:
    value = _check_path_exists(base_params, path, "dispersion")
    _require(
        is_numeric_leaf(value),
        f"dispersion {path!r} is not a number ({value!r})",
    )

    dist = raw.get("dist", "normal")
    _require(dist in VALID_DISTS, f"dispersion {path!r} has unknown dist {dist!r}, expected one of {VALID_DISTS}")

    d = Dispersion(
        path=path,
        dist=dist,
        sigma=raw.get("sigma"),
        sigma_rel=raw.get("sigma_rel"),
        rel=raw.get("rel"),
        low=raw.get("low"),
        high=raw.get("high"),
        minimum=raw.get("min"),
        maximum=raw.get("max"),
        link=tuple(raw.get("link", ())),
    )

    if dist == "normal":
        _require(
            (d.sigma is None) != (d.sigma_rel is None),
            f"dispersion {path!r} needs exactly one of sigma or sigma_rel",
        )
    else:
        has_bounds = d.low is not None and d.high is not None
        _require(
            (d.rel is not None) != has_bounds,
            f"dispersion {path!r} needs either rel, or both low and high",
        )

    for linked in d.link:
        linked_value = _check_path_exists(base_params, linked, f"dispersion {path!r} link")
        _require(
            is_numeric_leaf(linked_value),
            f"dispersion {path!r} links to {linked!r}, which is not a number",
        )
    return d


def _expand_cases(raw_cases: dict, grid: dict, base_params: dict) -> tuple[Case, ...]:
    for name, overrides in raw_cases.items():
        for path in overrides:
            _check_path_exists(base_params, path, f"case {name!r} override")
    for path, values in grid.items():
        _check_path_exists(base_params, path, "grid axis")
        _require(isinstance(values, list) and values, f"grid axis {path!r} must be a non-empty list")

    base_cases = [Case(safe_name(name), dict(overrides)) for name, overrides in raw_cases.items()]
    if not base_cases:
        base_cases = [Case("nominal", {})]
    if not grid:
        return tuple(base_cases)

    axes = [[(path, value) for value in values] for path, values in grid.items()]
    expanded: list[Case] = []
    for case in base_cases:
        for combo in itertools.product(*axes):
            overrides = dict(case.overrides)
            overrides.update(dict(combo))
            suffix = "__".join(f"{path.split('.')[-1]}={_format_value(v)}" for path, v in combo)
            expanded.append(Case(safe_name(f"{case.name}__{suffix}"), overrides))
    return tuple(expanded)


def _wind_model(base_params: dict, overrides: dict, case: Case) -> str:
    for source in (case.overrides, overrides):
        if "wind_params.model" in source:
            return source["wind_params.model"]
    return wind_params_with_defaults(base_params)["model"]


def load_spec(path: str | Path) -> CampaignSpec:
    """Parse and validate a campaign TOML file."""
    path = Path(path)
    source_text = path.read_text(encoding="utf-8")
    try:
        raw = tomllib.loads(source_text)
    except tomllib.TOMLDecodeError as exc:
        raise SpecError(f"{path} is not valid TOML: {exc}") from None

    _require("name" in raw, f"{path} has no campaign name")
    _require("base_parameters" in raw, f"{path} has no base_parameters")

    base_parameters = Path(raw["base_parameters"])
    if not base_parameters.is_absolute():
        base_parameters = REPO_ROOT / base_parameters
    _require(base_parameters.is_file(), f"base_parameters {base_parameters} does not exist")
    with base_parameters.open() as f:
        base_params = json.load(f)

    overrides = raw.get("overrides", {})
    for override_path in overrides:
        _check_path_exists(base_params, override_path, "override")

    dispersions = tuple(
        _parse_dispersion(disp_path, disp_raw, base_params)
        for disp_path, disp_raw in raw.get("dispersions", {}).items()
    )

    seen: set[str] = set()
    for d in dispersions:
        for touched in (d.path, *d.link):
            _require(touched not in seen, f"{touched!r} is dispersed or linked more than once")
            seen.add(touched)

    sampling = raw.get("sampling", {})
    samples_per_case = int(raw.get("samples_per_case", 0))
    _require(samples_per_case >= 0, "samples_per_case cannot be negative")
    _require(
        not (samples_per_case > 0 and not dispersions),
        "samples_per_case > 0 but no [dispersions] are declared",
    )

    cases = _expand_cases(raw.get("cases", {}), raw.get("grid", {}), base_params)

    vary_wind = raw.get("vary_wind", False)
    _require(isinstance(vary_wind, bool), f"vary_wind must be true or false, not {vary_wind!r}")
    if vary_wind:
        replaying = [c.name for c in cases if _wind_model(base_params, overrides, c) == "replay"]
        _require(
            not replaying,
            f"vary_wind = true, but case(s) {', '.join(replaying)} replay the flight log",
        )

    spec = CampaignSpec(
        name=safe_name(raw["name"]),
        base_parameters=base_parameters,
        base_params=base_params,
        samples_per_case=samples_per_case,
        include_nominal=bool(raw.get("include_nominal", True)),
        sampling_method=sampling.get("method", "random"),
        sampling_seed=int(sampling.get("seed", 0)),
        noise_seed=int(raw.get("noise_seed", 0)),
        vary_wind=vary_wind,
        overrides=overrides,
        cases=cases,
        dispersions=dispersions,
        source_text=source_text,
    )
    _require(spec.total_runs > 0, "campaign produces no runs")
    return spec
