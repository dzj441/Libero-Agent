"""Shared task descriptors for the physical-property button benchmark.

This small data module is imported by both the benchmark registry and the
private AgentEnv evaluator.  Keeping the mapping here avoids an import cycle
through ``libero.libero.agent_env`` and gives the two sides one authoritative
task order without exposing the answer mapping to the Agent.
"""

from __future__ import annotations

from dataclasses import dataclass

from .physical_property_questions import build_question


@dataclass(frozen=True)
class PhysicalPropertyTask:
    name: str
    family_id: str
    seed: int
    instruction: str


_TASK_FAMILY_SEEDS = (
    # Seed 102 asks for the highest rank, matching the legacy density task's
    # instruction and its right-hand diagnostic goal without confusing a
    # canonical rank with the screen index.
    ("select_highest_density", "density", 102),
    ("select_smallest_mass", "weight", 100),
    ("select_magnetic_material", "magnetism", 100),
    ("select_lowest_friction", "friction", 100),
    ("select_least_thermal_expansion", "thermal_expansion", 100),
    ("select_slowest_sound", "speed_of_sound", 100),
    ("select_highest_reflectance", "reflection", 102),
)


def _public_instruction(family_id: str, seed: int) -> str:
    public = build_question(family_id, seed).public
    question = public.instruction
    # Commas are intentional: BDDL treats semicolons as comments, so using
    # the same punctuation here and in the task files keeps one exact public
    # instruction contract without making the BDDL unparsable.
    if family_id == "weight":
        pairing_contract = (
            "The fixed candidate-button pairs, listed from left to right in the "
            "head-camera image, show "
            "three solid red-painted wooden objects made from the same wood and "
            "having visibly different volumes. Each object is paired with the red "
            "button directly beside it."
        )
    else:
        option_contract = ", ".join(option.appearance for option in public.options)
        pairing_contract = (
            "The fixed candidate-button pairs, listed from left to right in the "
            "head-camera image, are: "
            f"{option_contract}. Each object is paired with the red button directly beside it."
        )
    if family_id == "density":
        # The original task used "greatest density"; retain that stable phrase
        # while making the required physical action unambiguous.
        return (
            f"{pairing_contract} Which object has the greatest density? "
            "Press the button paired with that object."
        )
    answer_noun = {
        "magnetism": "candidate",
        "speed_of_sound": "solid",
        "reflection": "panel",
    }.get(family_id, "object")
    return (
        f"{pairing_contract} {question} "
        f"Press the button paired with that {answer_noun}."
    )


PHYSICAL_PROPERTY_TASKS: tuple[PhysicalPropertyTask, ...] = tuple(
    PhysicalPropertyTask(
        name=name,
        family_id=family_id,
        seed=seed,
        instruction=_public_instruction(family_id, seed),
    )
    for name, family_id, seed in _TASK_FAMILY_SEEDS
)


__all__ = ["PHYSICAL_PROPERTY_TASKS", "PhysicalPropertyTask"]
