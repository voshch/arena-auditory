from __future__ import annotations

import functools
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import attrs
import yaml
from ament_index_python.packages import get_package_share_path

DEFAULT_OCTAVE_BANDS_HZ = (
    125,
    250,
    500,
    1000,
    2000,
    4000,
)


@attrs.frozen
class AcousticMaterial:
    """Per-octave-band energy absorption and scattering in [0, 1] and transmission loss in dB."""

    material_id: str
    canonical_name: str
    center_frequencies_hz: tuple[int, ...]
    absorption: tuple[float, ...]
    transmission_loss_db: tuple[float, ...]
    scattering: tuple[float, ...]
    used_default: bool = False

    @property
    def mean_absorption(self) -> float:
        if not self.absorption:
            return 0.0
        return sum(self.absorption) / len(self.absorption)

    @property
    def mean_transmission_loss_db(self) -> float:
        if not self.transmission_loss_db:
            return 0.0
        return sum(self.transmission_loss_db) / len(self.transmission_loss_db)


def catalog_path() -> Path:
    return get_package_share_path("arena_auditory") / "config" / "acoustic_materials.yaml"


class AcousticMaterialCatalog:
    """Acoustic materials from a YAML catalog, unknown ids resolve to the marked default."""

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)

        raw = self._load_yaml(self._path)

        self._center_frequencies_hz = self._parse_center_frequencies(
            raw.get(
                "octave_bands_hz",
                DEFAULT_OCTAVE_BANDS_HZ,
            )
        )

        default_entry = raw.get("defaults")
        if not isinstance(default_entry, Mapping):
            raise ValueError(f"{self._path}: expected 'defaults' to be a mapping")

        self._default_material = self._parse_material(
            material_id="default",
            entry=default_entry,
            used_default=True,
        )

        raw_materials = raw.get("materials", {})
        if raw_materials is None:
            raw_materials = {}

        if not isinstance(raw_materials, Mapping):
            raise ValueError(f"{self._path}: expected 'materials' to be a mapping")

        materials: dict[str, AcousticMaterial] = {}

        for raw_material_id, raw_entry in raw_materials.items():
            material_id = str(raw_material_id).strip()

            if not material_id:
                raise ValueError(f"{self._path}: material ID cannot be empty")

            if material_id in materials:
                raise ValueError(f"{self._path}: duplicate material {material_id!r}")

            if not isinstance(raw_entry, Mapping):
                raise ValueError(f"{self._path}: material {material_id!r} must be a mapping")

            materials[material_id] = self._parse_material(
                material_id=material_id,
                entry=raw_entry,
                used_default=False,
            )

        self._materials = materials

    @property
    def path(self) -> Path:
        return self._path

    @property
    def center_frequencies_hz(self) -> tuple[int, ...]:
        return self._center_frequencies_hz

    def contains(self, material_id: str) -> bool:
        return str(material_id).strip() in self._materials

    def get(self, material_id: str) -> AcousticMaterial:
        """The material, or the default under the requested id with used_default set."""

        requested_id = str(material_id).strip()

        if not requested_id:
            requested_id = "default"

        material = self._materials.get(requested_id)
        if material is not None:
            return material

        return attrs.evolve(
            self._default_material,
            material_id=requested_id,
            canonical_name=requested_id,
            used_default=True,
        )

    def require(self, material_id: str) -> AcousticMaterial:
        """The material, KeyError for unknown ids."""

        requested_id = str(material_id).strip()

        if not requested_id:
            raise KeyError("acoustic material ID cannot be empty")

        try:
            return self._materials[requested_id]
        except KeyError:
            known = ", ".join(sorted(self._materials))
            raise KeyError(f"unknown acoustic material {requested_id!r}, known materials: {known or '<none>'}") from None

    def surface_damping_db(
        self,
        material_id: str,
        surface_role: str,
    ) -> float:
        """Wall: mean transmission loss. Floor and ceiling: 10 dB per unit mean absorption, capped at 6 dB."""
        material = self.get(str(material_id).strip() or "default")
        role = str(surface_role).strip().lower()

        if role == "wall":
            return float(material.mean_transmission_loss_db)

        if role in {"floor", "ceiling"}:
            absorption = material.mean_absorption
            if absorption <= 0.0:
                return 0.0

            floor_gain_db = 10.0 * absorption
            floor_gain_db = min(floor_gain_db, 6.0)
            return float(max(0.0, floor_gain_db))

        return 0.0

    @staticmethod
    def _load_yaml(path: Path) -> Mapping[str, Any]:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise OSError(f"failed to read acoustic material catalog {path}") from exc

        try:
            raw = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ValueError(f"failed to parse acoustic material catalog {path}") from exc

        if raw is None:
            raise ValueError(f"acoustic material catalog {path} is empty")

        if not isinstance(raw, Mapping):
            raise ValueError(f"{path}: catalog root must be a mapping")

        return raw

    def _parse_center_frequencies(
        self,
        raw_frequencies: object,
    ) -> tuple[int, ...]:
        if not self._is_sequence(raw_frequencies):
            raise ValueError(f"{self._path}: 'octave_bands_hz' must be a sequence")

        frequencies: list[int] = []

        for index, raw_value in enumerate(raw_frequencies):
            if isinstance(raw_value, bool):
                raise ValueError(f"{self._path}: octave-band frequency at index {index} cannot be boolean")

            try:
                frequency = int(raw_value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{self._path}: invalid octave-band frequency at index {index}: {raw_value!r}") from exc

            if frequency <= 0:
                raise ValueError(f"{self._path}: octave-band frequencies must be positive")

            frequencies.append(frequency)

        if not frequencies:
            raise ValueError(f"{self._path}: at least one octave-band frequency is required")

        if len(set(frequencies)) != len(frequencies):
            raise ValueError(f"{self._path}: octave-band frequencies must be unique")

        if frequencies != sorted(frequencies):
            raise ValueError(f"{self._path}: octave-band frequencies must be strictly increasing")

        return tuple(frequencies)

    def _parse_material(
        self,
        *,
        material_id: str,
        entry: Mapping[str, Any],
        used_default: bool,
    ) -> AcousticMaterial:
        canonical_name = str(entry.get("canonical_name", material_id)).strip()

        if not canonical_name:
            canonical_name = material_id

        absorption = self._parse_band_values(
            entry.get("absorption"),
            field_name="absorption",
            material_id=material_id,
            minimum=0.0,
            maximum=1.0,
        )

        transmission_loss_db = self._parse_band_values(
            entry.get("transmission_loss_db"),
            field_name="transmission_loss_db",
            material_id=material_id,
            minimum=0.0,
            maximum=None,
        )

        scattering = self._parse_band_values(
            entry.get("scattering", 0.1),
            field_name="scattering",
            material_id=material_id,
            minimum=0.0,
            maximum=1.0,
        )

        material = AcousticMaterial(
            material_id=material_id,
            canonical_name=canonical_name,
            center_frequencies_hz=self._center_frequencies_hz,
            absorption=absorption,
            transmission_loss_db=transmission_loss_db,
            scattering=scattering,
            used_default=used_default,
        )

        self._validate_material(material)
        return material

    def _parse_band_values(
        self,
        raw_values: object,
        *,
        field_name: str,
        material_id: str,
        minimum: float | None,
        maximum: float | None,
    ) -> tuple[float, ...]:
        if raw_values is None:
            raise ValueError(f"{self._path}: material {material_id!r} is missing {field_name!r}")

        if self._is_number(raw_values):
            values = [float(raw_values) for _ in self._center_frequencies_hz]
        elif self._is_sequence(raw_values):
            values = []

            for index, raw_value in enumerate(raw_values):
                if not self._is_number(raw_value):
                    raise ValueError(f"{self._path}: material {material_id!r} has a non-numeric {field_name} value at index {index}: {raw_value!r}")

                values.append(float(raw_value))
        else:
            raise ValueError(f"{self._path}: material {material_id!r} field {field_name!r} must be a number or a sequence of numbers")

        expected_count = len(self._center_frequencies_hz)

        if len(values) != expected_count:
            raise ValueError(f"{self._path}: material {material_id!r} field {field_name!r} has {len(values)} values, expected {expected_count}")

        for index, value in enumerate(values):
            if not self._is_finite(value):
                raise ValueError(f"{self._path}: material {material_id!r} field {field_name!r} contains a non-finite value at index {index}")

            if minimum is not None and value < minimum:
                raise ValueError(f"{self._path}: material {material_id!r} field {field_name!r} contains {value} below the minimum {minimum}")

            if maximum is not None and value > maximum:
                raise ValueError(f"{self._path}: material {material_id!r} field {field_name!r} contains {value} above the maximum {maximum}")

        return tuple(values)

    def _validate_material(
        self,
        material: AcousticMaterial,
    ) -> None:
        expected_count = len(self._center_frequencies_hz)

        fields = {
            "absorption": material.absorption,
            "transmission_loss_db": (material.transmission_loss_db),
            "scattering": material.scattering,
        }

        for field_name, values in fields.items():
            if len(values) != expected_count:
                raise ValueError(f"{self._path}: material {material.material_id!r} field {field_name!r} has {len(values)} values, expected {expected_count}")

    @staticmethod
    def _is_sequence(value: object) -> bool:
        return isinstance(value, Sequence) and not isinstance(
            value,
            (str, bytes, bytearray),
        )

    @staticmethod
    def _is_number(value: object) -> bool:
        return isinstance(value, (int, float)) and not isinstance(
            value,
            bool,
        )

    @staticmethod
    def _is_finite(value: float) -> bool:
        return value == value and value not in (
            float("inf"),
            float("-inf"),
        )


@functools.cache
def default_catalog() -> AcousticMaterialCatalog:
    """The package catalog, loaded once per process."""
    return AcousticMaterialCatalog(catalog_path())
