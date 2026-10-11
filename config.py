"""Typed CSV settings. Pixel units always refer to the original image."""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass, fields, replace
from pathlib import Path

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / 'config.csv'


@dataclass(frozen=True)
class Settings:
    min_length_um: float = 0.0
    pixels_per_um: float = 2.232
    analysis_scale: float = 1.0
    min_area: int = 900
    min_aspect_ratio: float = 3.0
    min_eccentricity: float = 0.88
    max_width: float = 180.0
    boundary_threshold: float = 0.35
    cytoplasm_threshold: float = 0.15
    color_normalization: bool = False
    boundary_continuity: bool = True
    evidence_sigma_px: float = 0.0
    strand_bridge_gap: float = 12.0
    auto_select_limit: int = 0

    def __post_init__(self):
        for f in fields(self):
            value = getattr(self, f.name)
            if f.type == 'bool':
                if type(value) is not bool:
                    raise ValueError(f'{f.name}: expected true or false')
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f'{f.name}: expected a finite number')
            if f.type == 'int' and (int(value) != value or value < 0):
                raise ValueError(f'{f.name}: expected a non-negative integer')
        if self.min_length_um < 0 or self.strand_bridge_gap < 0 or self.evidence_sigma_px < 0:
            raise ValueError('Minimum length and bridge gap must be >= 0; zero disables them.')
        if self.pixels_per_um <= 0 or self.max_width <= 0 or self.min_area < 1:
            raise ValueError('Resolution, ray search distance and minimum area must be > 0.')
        if self.analysis_scale != 1.0:
            raise ValueError('This revision requires analysis_scale=1.0 (native resolution).')
        if self.min_aspect_ratio < 1 or not 0 <= self.min_eccentricity <= 1:
            raise ValueError('Aspect ratio must be >= 1; eccentricity must be in [0, 1].')
        for key in ('boundary_threshold', 'cytoplasm_threshold'):
            if not 0 < getattr(self, key) < 1:
                raise ValueError(f'{key}: expected a value strictly between 0 and 1')


# unit, description. These are heuristics, not ground-truth calibrated parameters.
INFO = {
    'min_length_um': ('um', 'Minimum centerline arc length; 0 disables length filtering'),
    'pixels_per_um': ('px/um', 'Original image resolution; 558 px / 250 um = 2.232'),
    'analysis_scale': ('ratio', 'Native-resolution revision: must remain 1.0'),
    'min_area': ('px^2', 'Below this area: Review; candidates >=20 pixels are retained'),
    'min_aspect_ratio': ('ratio', 'Below this shape ratio: Review, not automatic rejection'),
    'min_eccentricity': ('ratio', 'Below this value: Review; curved myotubes may score low'),
    'max_width': ('px', 'Maximum distance for each normal ray; missing edge becomes Review'),
    'boundary_threshold': ('0..1', 'White/yellow background evidence threshold; experimental'),
    'cytoplasm_threshold': ('0..1', 'Minimum purple/blue evidence outside boundary barriers'),
    'color_normalization': ('bool', 'Optional channel-gain correction; default false pending real images'),
    'boundary_continuity': ('bool', 'Extend short boundary gaps only where weaker color evidence exists'),
    'evidence_sigma_px': ('px', 'Gaussian color-evidence denoising at native resolution; 0 disables'),
    'strand_bridge_gap': ('px', 'Maximum endpoint gap; 0 disables experimental bridging'),
    'auto_select_limit': ('count', '0 selects all measurable candidates; 7 restores top-seven selection'),
}


def settings_from_values(values: dict[str, object], base: Settings | None = None) -> Settings:
    defaults = base or Settings()
    types = {f.name: f.type for f in fields(defaults)}
    unknown = set(values) - set(types)
    if unknown:
        raise ValueError('Unknown settings: ' + ', '.join(sorted(unknown)))
    parsed = {}
    for key, raw in values.items():
        text = str(raw).strip()
        if types[key] == 'bool':
            if text.lower() not in {'true', 'false'}:
                raise ValueError(f'{key}: use true or false')
            parsed[key] = text.lower() == 'true'
        else:
            try:
                number = float(text)
            except (TypeError, ValueError) as exc:
                raise ValueError(f'{key}: invalid number {text!r}') from exc
            if not math.isfinite(number):
                raise ValueError(f'{key}: expected a finite number')
            if types[key] == 'int':
                if not number.is_integer():
                    raise ValueError(f'{key}: expected an integer')
                parsed[key] = int(number)
            else:
                parsed[key] = number
    return replace(defaults, **parsed)


def load_config(path: str | Path = DEFAULT_CONFIG) -> Settings:
    values = {}
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        if not {'parameter', 'value'}.issubset(reader.fieldnames or []):
            raise ValueError('config.csv requires parameter,value columns')
        for row in reader:
            key = (row.get('parameter') or '').strip()
            if not key:
                raise ValueError('Empty parameter name in config.csv')
            if key in values:
                raise ValueError(f'Duplicate setting: {key}')
            if key in INFO and row.get('unit') and row['unit'].strip().replace('µ', 'u').replace('μ', 'u') != INFO[key][0]:
                raise ValueError(f'{key}: unit must be {INFO[key][0]}')
            values[key] = row.get('value')
    missing = set(INFO) - set(values)
    if missing:
        raise ValueError('Missing settings: ' + ', '.join(sorted(missing)))
    return settings_from_values(values)


def save_config(settings: Settings, path: str | Path = DEFAULT_CONFIG) -> None:
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.writer(handle)
        writer.writerow(['parameter', 'value', 'unit', 'description'])
        for f in fields(settings):
            value = getattr(settings, f.name)
            writer.writerow([f.name, str(value).lower() if isinstance(value, bool) else value, *INFO[f.name]])
    temporary.replace(path)
