"""Pure-data property questions for a physical-property button family.

This module deliberately has no MuJoCo, BDDL, registry, or environment
dependency.  A task adapter can use :func:`build_question` to create three
public answer options and keep the returned private answer record in its
evaluator-only state.

The public view contains only the question and the rendered option cues.  The
candidate properties and the correct option are available only through
``private_dict``.  In particular, ``public_dict`` never contains a target
index, target candidate id, or candidate property values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence


SEEDS: tuple[int, ...] = (100, 101, 102, 103, 104)
POSITIONS: tuple[str, ...] = ("left", "center", "right")
FAMILY_IDS: tuple[str, ...] = (
    "weight",
    "magnetism",
    "density",
    "friction",
    "thermal_expansion",
    "speed_of_sound",
    "reflection",
)

# The permutation is intentionally independent from the question wording.
# Together with _RANK_BY_SEED it places the correct option at all three
# screen positions over the five prescribed seeds for every family.
_POSITION_PERMUTATIONS: Mapping[int, tuple[int, int, int]] = {
    100: (0, 1, 2),
    101: (2, 1, 0),
    102: (0, 1, 2),
    103: (1, 0, 2),
    104: (2, 0, 1),
}
_RANK_BY_SEED: Mapping[int, int] = {
    100: 0,
    101: 1,
    102: 2,
    103: 0,
    104: 1,
}


@dataclass(frozen=True)
class Candidate:
    """A canonical option.

    ``properties`` are evaluator-side semantic facts.  They are intentionally
    represented as an immutable tuple so that a caller cannot accidentally
    mutate the answer metadata while constructing a public observation.
    """

    candidate_id: str
    appearance: str
    properties: tuple[tuple[str, object], ...]

    def public_dict(self) -> dict[str, str]:
        return {
            "candidate_id": self.candidate_id,
            "appearance": self.appearance,
        }

    def private_dict(self) -> dict[str, object]:
        payload = self.public_dict()
        payload["properties"] = dict(self.properties)
        return payload


@dataclass(frozen=True)
class PublicOption:
    position: str
    candidate_id: str
    appearance: str

    def to_dict(self) -> dict[str, str]:
        return {
            "position": self.position,
            "candidate_id": self.candidate_id,
            "appearance": self.appearance,
        }


@dataclass(frozen=True)
class PublicQuestion:
    family_id: str
    seed: int
    instruction: str
    options: tuple[PublicOption, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "family_id": self.family_id,
            "seed": self.seed,
            "instruction": self.instruction,
            "options": [option.to_dict() for option in self.options],
        }


@dataclass(frozen=True)
class PropertyQuestion:
    """A question with a public observation and a private evaluator answer."""

    public: PublicQuestion
    canonical_candidates: tuple[Candidate, ...]
    _target_candidate_id: str
    _target_index: int

    @property
    def target_candidate_id(self) -> str:
        """Evaluator-only answer identity."""

        return self._target_candidate_id

    @property
    def target_index(self) -> int:
        """Evaluator-only left/center/right index."""

        return self._target_index

    def public_dict(self) -> dict[str, object]:
        return self.public.to_dict()

    def private_dict(self) -> dict[str, object]:
        """Serialize all evaluator metadata separately from the public view."""

        return {
            "public": self.public_dict(),
            "canonical_candidates": [
                candidate.private_dict() for candidate in self.canonical_candidates
            ],
            "position_order": [
                option.candidate_id for option in self.public.options
            ],
            "answer": {
                "target_candidate_id": self._target_candidate_id,
                "target_index": self._target_index,
                "target_position": POSITIONS[self._target_index],
            },
        }


@dataclass(frozen=True)
class _FamilySpec:
    family_id: str
    build_candidates: Callable[[int], tuple[Candidate, ...]]
    build_instruction: Callable[[int], tuple[str, int]]


def _candidate(
    candidate_id: str,
    appearance: str,
    **properties: object,
) -> Candidate:
    return Candidate(candidate_id, appearance, tuple(properties.items()))


def _rank_instruction(
    prompts: Sequence[str], seed: int
) -> tuple[str, int]:
    rank = _RANK_BY_SEED[seed]
    return prompts[rank], rank


def _weight_candidates(seed: int) -> tuple[Candidate, ...]:
    del seed
    # The candidates are solid objects made from the same wood and carry the
    # same red paint.  Their visible volume is therefore the intended mass
    # cue; shape names are deliberately omitted from the public contract.
    return (
        _candidate(
            "candidate_a",
            "a solid red-painted wooden object",
            property="mass",
            shape="sphere",
            relative_mass=1.0,
            density_relation="equal",
        ),
        _candidate(
            "candidate_b",
            "a solid red-painted wooden object",
            property="mass",
            shape="cylinder",
            relative_mass=1.93,
            density_relation="equal",
        ),
        _candidate(
            "candidate_c",
            "a solid red-painted wooden object",
            property="mass",
            shape="cube",
            relative_mass=2.93,
            density_relation="equal",
        ),
    )


def _density_candidates(seed: int) -> tuple[Candidate, ...]:
    pools = (
        (("wood", 0.60), ("rubber", 0.92), ("glass", 2.50)),
        (("wood", 0.60), ("glass", 2.50), ("stone", 2.70)),
        (("rubber", 0.92), ("glass", 2.50), ("steel", 7.80)),
        (("wood", 0.60), ("stone", 2.70), ("steel", 7.80)),
        (("rubber", 0.92), ("stone", 2.70), ("steel", 7.80)),
    )
    values = pools[SEEDS.index(seed)]
    appearances = {
        "wood": "a brown wooden block",
        "rubber": "a black rubber block",
        "glass": "a blue-tinted glass block",
        "stone": "a gray stone block",
        "steel": "a silver-white steel block",
    }
    return tuple(
        _candidate(
            f"candidate_{chr(ord('a') + index)}",
            appearances[material],
            property="density",
            material=material,
            density_g_cm3=density,
        )
        for index, (material, density) in enumerate(values)
    )


def _friction_candidates(seed: int) -> tuple[Candidate, ...]:
    # The same material triplet is kept across seeds; only option placement and
    # rank query vary.  A physical adapter can replace the cues with matching
    # meshes while retaining the private coefficients.
    del seed
    values = (
        ("PTFE", "a white PTFE puck", 0.05),
        ("unfinished wood", "a brown unfinished-wood puck", 0.40),
        ("rubber", "a black rubber puck", 0.80),
    )
    return tuple(
        _candidate(
            f"candidate_{chr(ord('a') + index)}",
            appearance,
            property="kinetic_friction",
            surface_material=material,
            coefficient_of_kinetic_friction=coefficient,
        )
        for index, (material, appearance, coefficient) in enumerate(values)
    )


def _thermal_candidates(seed: int) -> tuple[Candidate, ...]:
    del seed
    values = (
        ("fused silica", "a colorless translucent fused-silica bar", 0.5e-6),
        ("steel", "a dark-gray steel bar", 12.0e-6),
        ("aluminum", "a light-silver aluminum bar", 23.0e-6),
    )
    return tuple(
        _candidate(
            f"candidate_{chr(ord('a') + index)}",
            appearance,
            property="linear_thermal_expansion",
            material=material,
            coefficient_per_kelvin=coefficient,
        )
        for index, (material, appearance, coefficient) in enumerate(values)
    )


def _sound_candidates(seed: int) -> tuple[Candidate, ...]:
    del seed
    values = (
        ("rubber", "a black rubber rod", 1600.0),
        ("glass", "a blue-tinted glass rod", 5000.0),
        ("steel", "a silver steel rod", 5960.0),
    )
    return tuple(
        _candidate(
            f"candidate_{chr(ord('a') + index)}",
            appearance,
            property="longitudinal_sound_speed",
            material=material,
            speed_m_per_s=speed,
        )
        for index, (material, appearance, speed) in enumerate(values)
    )


def _reflection_candidates(seed: int) -> tuple[Candidate, ...]:
    del seed
    values = (
        ("black rubber", "a black rubber panel", 0.05),
        ("unfinished wood", "a brown unfinished-wood panel", 0.35),
        ("mirror-polished steel", "a mirror-polished silver-steel panel", 0.70),
    )
    return tuple(
        _candidate(
            f"candidate_{chr(ord('a') + index)}",
            appearance,
            property="visible_specular_reflectance",
            surface=surface,
            reflectance=reflectance,
        )
        for index, (surface, appearance, reflectance) in enumerate(values)
    )


def _magnetism_candidates(seed: int) -> tuple[Candidate, ...]:
    # Alternate the question direction while keeping exactly one valid answer
    # in either direction.  The target is candidate_b in both variants, but
    # that fact is private and not represented in the public candidate ids.
    if seed % 2 == 0:
        values = (
            ("wood", "a brown wooden cylinder", False),
            ("low-carbon steel", "a silver low-carbon-steel cylinder", True),
            ("glass", "a blue-tinted glass cylinder", False),
        )
    else:
        values = (
            ("low-carbon steel", "a silver low-carbon-steel cylinder", True),
            ("wood", "a brown wooden cylinder", False),
            ("iron", "a dark-gray iron cylinder", True),
        )
    return tuple(
        _candidate(
            f"candidate_{chr(ord('a') + index)}",
            appearance,
            property="magnetism",
            material=material,
            attracted_by_permanent_magnet=attracted,
        )
        for index, (material, appearance, attracted) in enumerate(values)
    )


def _magnetism_instruction(seed: int) -> tuple[str, int]:
    if seed % 2 == 0:
        return "Which candidate would be attracted to a permanent magnet?", 1
    return "Which candidate would not be attracted to a permanent magnet?", 1


_RANK_SPECS: tuple[tuple[str, _FamilySpec], ...] = (
    (
        "weight",
        _FamilySpec(
            "weight",
            _weight_candidates,
            lambda seed: _rank_instruction(
                (
                    "Which object has the smallest mass?",
                    "Which object has an intermediate mass?",
                    "Which object has the largest mass?",
                ),
                seed,
            ),
        ),
    ),
    (
        "density",
        _FamilySpec(
            "density",
            _density_candidates,
            lambda seed: _rank_instruction(
                (
                    "Which object has the lowest density?",
                    "Which object has an intermediate density?",
                    "Which object has the highest density?",
                ),
                seed,
            ),
        ),
    ),
    (
        "friction",
        _FamilySpec(
            "friction",
            _friction_candidates,
            lambda seed: _rank_instruction(
                (
                    "Against the same clean, dry steel surface under equal normal load and sliding speed, which object has the lowest coefficient of kinetic friction?",
                    "Against the same clean, dry steel surface under equal normal load and sliding speed, which object has an intermediate coefficient of kinetic friction?",
                    "Against the same clean, dry steel surface under equal normal load and sliding speed, which object has the highest coefficient of kinetic friction?",
                ),
                seed,
            ),
        ),
    ),
    (
        "thermal_expansion",
        _FamilySpec(
            "thermal_expansion",
            _thermal_candidates,
            lambda seed: _rank_instruction(
                (
                    "If these equal-length bars experience the same change in temperature, which object would expand the least?",
                    "If these equal-length bars experience the same change in temperature, which object would expand by an intermediate amount?",
                    "If these equal-length bars experience the same change in temperature, which object would expand the most?",
                ),
                seed,
            ),
        ),
    ),
    (
        "speed_of_sound",
        _FamilySpec(
            "speed_of_sound",
            _sound_candidates,
            lambda seed: _rank_instruction(
                (
                    "At room temperature, in which solid would a sound pulse travel most slowly?",
                    "At room temperature, in which solid would a sound pulse travel at an intermediate speed?",
                    "At room temperature, in which solid would a sound pulse travel fastest?",
                ),
                seed,
            ),
        ),
    ),
    (
        "reflection",
        _FamilySpec(
            "reflection",
            _reflection_candidates,
            lambda seed: _rank_instruction(
                (
                    "Under the same illumination and viewing angle, which separate panel has the lowest visible specular reflectance?",
                    "Under the same illumination and viewing angle, which separate panel has an intermediate visible specular reflectance?",
                    "Under the same illumination and viewing angle, which separate panel has the highest visible specular reflectance?",
                ),
                seed,
            ),
        ),
    ),
)

FAMILY_SPECS: Mapping[str, _FamilySpec] = dict(_RANK_SPECS)
FAMILY_SPECS["magnetism"] = _FamilySpec(
    "magnetism", _magnetism_candidates, _magnetism_instruction
)


def build_question(family_id: str, seed: int) -> PropertyQuestion:
    """Build one deterministic question for a supported family and seed."""

    if family_id not in FAMILY_SPECS:
        raise KeyError(f"unknown physical-property family: {family_id}")
    if seed not in _POSITION_PERMUTATIONS:
        raise ValueError(f"seed must be one of {SEEDS}, got {seed}")

    spec = FAMILY_SPECS[family_id]
    canonical_candidates = spec.build_candidates(seed)
    if len(canonical_candidates) != 3:
        raise ValueError(f"{family_id} must provide exactly three candidates")
    candidate_ids = [candidate.candidate_id for candidate in canonical_candidates]
    if len(set(candidate_ids)) != 3:
        raise ValueError(f"{family_id} candidates must have unique ids")

    instruction, target_canonical_index = spec.build_instruction(seed)
    if not 0 <= target_canonical_index < 3:
        raise ValueError("target canonical index must be in [0, 2]")
    permutation = _POSITION_PERMUTATIONS[seed]
    ordered_candidates = tuple(canonical_candidates[index] for index in permutation)
    target_id = canonical_candidates[target_canonical_index].candidate_id
    target_index = next(
        index
        for index, candidate in enumerate(ordered_candidates)
        if candidate.candidate_id == target_id
    )
    public = PublicQuestion(
        family_id=family_id,
        seed=seed,
        instruction=instruction,
        options=tuple(
            PublicOption(
                position=POSITIONS[index],
                candidate_id=candidate.candidate_id,
                appearance=candidate.appearance,
            )
            for index, candidate in enumerate(ordered_candidates)
        ),
    )
    return PropertyQuestion(
        public=public,
        canonical_candidates=canonical_candidates,
        _target_candidate_id=target_id,
        _target_index=target_index,
    )


def build_all_questions() -> tuple[PropertyQuestion, ...]:
    """Return the complete seven-family by five-seed question bank."""

    return tuple(
        build_question(family_id, seed)
        for family_id in FAMILY_IDS
        for seed in SEEDS
    )


__all__ = [
    "Candidate",
    "FAMILY_IDS",
    "FAMILY_SPECS",
    "POSITIONS",
    "PropertyQuestion",
    "PublicQuestion",
    "SEEDS",
    "build_all_questions",
    "build_question",
]
