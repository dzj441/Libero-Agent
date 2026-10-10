import abc
import os
import glob
import random
import torch

from typing import List, NamedTuple, Type
from libero.libero import get_libero_path
from libero.libero.benchmark.libero_suite_task_map import libero_task_map

BENCHMARK_MAPPING = {}
RELEASE_RESOURCE_DIRECTORY = "libero_agent"


def register_benchmark(target_class):
    """We design the mapping to be case-INsensitive."""
    BENCHMARK_MAPPING[target_class.__name__.lower()] = target_class


def get_benchmark_dict(help=False):
    if help:
        print("Available benchmarks:")
        for benchmark_name in BENCHMARK_MAPPING.keys():
            print(f"\t{benchmark_name}")
    return BENCHMARK_MAPPING


def get_benchmark(benchmark_name):
    return BENCHMARK_MAPPING[benchmark_name.lower()]


def print_benchmark():
    print(BENCHMARK_MAPPING)


class Task(NamedTuple):
    name: str
    language: str
    problem: str
    problem_folder: str
    bddl_file: str
    init_states_file: str


def _task_resource_path(root, task, filename):
    """Prefer the consolidated release resource, then the source-suite path."""

    release_path = os.path.join(root, RELEASE_RESOURCE_DIRECTORY, filename)
    if os.path.isfile(release_path):
        return release_path
    return os.path.join(root, task.problem_folder, filename)


def grab_language_from_filename(x):
    if x[0].isupper():  # LIBERO-100
        if "SCENE10" in x:
            language = " ".join(x[x.find("SCENE") + 8 :].split("_"))
        else:
            language = " ".join(x[x.find("SCENE") + 7 :].split("_"))
    else:
        language = " ".join(x.split("_"))
    en = language.find(".bddl")
    return language[:en]


TASK_LANGUAGE_OVERRIDES = {
    (
        "libero_goal",
        "push_the_plate_to_the_front_of_the_stove",
    ): "Push the plate into the green target area in front of the stove.",
}


libero_suites = [
    "vlabench_physical",
    "vlabench_world_knowledge",
    "vlabench_perception",
    "vlabench_tube_precision",
    "vlabench_additional",
    "vlabench_button",
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_90",
    "libero_10",
]
task_maps = {}
max_len = 0
for libero_suite in libero_suites:
    task_maps[libero_suite] = {}

    for task in libero_task_map[libero_suite]:
        language = TASK_LANGUAGE_OVERRIDES.get(
            (libero_suite, task), grab_language_from_filename(task + ".bddl")
        )
        task_maps[libero_suite][task] = Task(
            name=task,
            language=language,
            problem="Libero",
            problem_folder=libero_suite,
            bddl_file=f"{task}.bddl",
            init_states_file=f"{task}.pruned_init",
        )

        # print(language, "\n", f"{task}.bddl", "\n")
        # print("")


task_orders = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [4, 6, 8, 7, 3, 1, 2, 0, 9, 5],
    [6, 3, 5, 0, 4, 2, 9, 1, 8, 7],
    [7, 4, 3, 0, 8, 1, 2, 5, 9, 6],
    [4, 5, 6, 3, 8, 0, 2, 7, 1, 9],
    [1, 2, 3, 0, 6, 9, 5, 7, 4, 8],
    [3, 7, 8, 1, 6, 2, 9, 4, 0, 5],
    [4, 2, 9, 7, 6, 8, 5, 1, 3, 0],
    [1, 8, 5, 4, 0, 9, 6, 7, 2, 3],
    [8, 3, 6, 4, 9, 5, 1, 2, 0, 7],
    [6, 9, 0, 5, 7, 1, 2, 8, 3, 4],
    [6, 8, 3, 1, 0, 2, 5, 9, 7, 4],
    [8, 0, 6, 9, 4, 1, 7, 3, 2, 5],
    [3, 8, 6, 4, 2, 5, 0, 7, 1, 9],
    [7, 1, 5, 6, 3, 2, 8, 9, 4, 0],
    [2, 0, 9, 5, 3, 6, 8, 7, 1, 4],
    [3, 5, 9, 6, 2, 4, 8, 7, 1, 0],
    [7, 6, 5, 9, 0, 3, 4, 2, 8, 1],
    [2, 5, 0, 9, 3, 1, 6, 4, 8, 7],
    [3, 5, 1, 2, 7, 8, 6, 0, 4, 9],
    [3, 4, 1, 9, 7, 6, 8, 2, 0, 5],
]


class Benchmark(abc.ABC):
    """A Benchmark."""

    def __init__(self, task_order_index=0):
        self.task_embs = None
        self.task_order_index = task_order_index

    def _make_benchmark(self):
        tasks = list(task_maps[self.name].values())
        if self.name == "libero_90":
            self.tasks = tasks
        else:
            print(f"[info] using task orders {task_orders[self.task_order_index]}")
            self.tasks = [tasks[i] for i in task_orders[self.task_order_index]]
        self.n_tasks = len(self.tasks)

    def get_num_tasks(self):
        return self.n_tasks

    def get_task_names(self):
        return [task.name for task in self.tasks]

    def get_task_problems(self):
        return [task.problem for task in self.tasks]

    def get_task_bddl_files(self):
        return [task.bddl_file for task in self.tasks]

    def get_task_bddl_file_path(self, i):
        return _task_resource_path(
            get_libero_path("bddl_files"),
            self.tasks[i],
            self.tasks[i].bddl_file,
        )

    def get_task_demonstration(self, i):
        assert (
            0 <= i and i < self.n_tasks
        ), f"[error] task number {i} is outer of range {self.n_tasks}"
        # this path is relative to the datasets folder
        demo_path = f"{self.tasks[i].problem_folder}/{self.tasks[i].name}_demo.hdf5"
        return demo_path

    def get_task(self, i):
        return self.tasks[i]

    def get_task_emb(self, i):
        return self.task_embs[i]

    def get_task_init_states(self, i):
        init_states_path = _task_resource_path(
            get_libero_path("init_states"),
            self.tasks[i],
            self.tasks[i].init_states_file,
        )
        init_states = torch.load(init_states_path, weights_only=False)
        return init_states

    def set_task_embs(self, task_embs):
        self.task_embs = task_embs


@register_benchmark
class LIBERO_SPATIAL(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_spatial"
        self._make_benchmark()


@register_benchmark
class LIBERO_OBJECT(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_object"
        self._make_benchmark()


@register_benchmark
class LIBERO_GOAL(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_goal"
        self._make_benchmark()


@register_benchmark
class LIBERO_90(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        assert (
            task_order_index == 0
        ), "[error] currently only support task order for 10-task suites"
        self.name = "libero_90"
        self._make_benchmark()


@register_benchmark
class LIBERO_10(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_10"
        self._make_benchmark()


@register_benchmark
class MUJOCO_ADAPTED_V2(Benchmark):
    """First frozen V2 ports from compatible MuJoCo task sources."""

    _PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _INSTRUCTIONS = {
        "unlatch_and_open_door": (
            "Open the door. It is secured by a rotating latch."
        ),
        "drive_nail_with_hammer": (
            "Drive the nail into the wooden block."
        ),
        "pour_wine_into_mug": "Pour wine from the bottle into the mug.",
        "wipe_spill_from_table": (
            "Pick up the sponge, wipe the entire brown spill, then release "
            "the sponge on the blue rest area."
        ),
    }

    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        if task_order_index != 0:
            raise ValueError(
                "mujoco_adapted_v2 only supports task_order_index=0"
            )
        self.name = "mujoco_adapted_v2"
        self.tasks = tuple(
            Task(
                name=name,
                language=self._INSTRUCTIONS[name],
                problem="Libero",
                problem_folder=self.name,
                bddl_file=f"{name}.bddl",
                init_states_file=f"{name}.pruned_init",
            )
            for name in libero_task_map[self.name]
        )
        self.n_tasks = len(self.tasks)

    def get_task_bddl_file_path(self, i):
        task = self.get_task(i)
        return _task_resource_path(
            os.path.join(self._PACKAGE_ROOT, "bddl_files"),
            task,
            task.bddl_file,
        )

    def get_task_init_states(self, i):
        task = self.get_task(i)
        path = _task_resource_path(
            os.path.join(self._PACKAGE_ROOT, "init_files"),
            task,
            task.init_states_file,
        )
        return torch.load(path, weights_only=False)


@register_benchmark
class LONG_HORIZON_V2(Benchmark):
    """Diverse long-horizon candidates adapted from three audited sources."""

    _PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _INSTRUCTIONS = {
        "rotate_three_bowls_using_empty_plate": (
            "Cyclically rearrange the three black bowls from left to right "
            "in the head camera. No bowl may touch the tabletop at any time."
        ),
        "weigh_and_place_heaviest_mug": (
            "Determine which of the three visually identical mugs is "
            "heaviest by physically weighing them, then place the heaviest "
            "mug in the green bin."
        ),
        "place_mug_and_brew_coffee": (
            "Place the mug under the coffee machine dispenser, press the "
            "front button to brew coffee, then leave the mug under the "
            "dispenser."
        ),
        "prepare_cooking_with_two_stoves_and_microwave": (
            "Place the moka pot on the stove next to it and turn that stove "
            "on. Then place the frying pan on the other stove and turn it "
            "on. Finally, open the microwave."
        ),
        "prepare_cooking_text_goal": (
            "Prepare the cooking area. For a prepared cooking area, the "
            "frying pan should be on the stove. The stoves should be turned "
            "on. The moka pot should be on the nearest stove. The microwave "
            "should be open."
        ),
        "prepare_cooking_goal_image": (
            "Prepare the cooking area to match the provided goal image."
        ),
    }
    _SHARED_ASSET_STEMS = {
        "prepare_cooking_text_goal": (
            "prepare_cooking_with_two_stoves_and_microwave"
        ),
        "prepare_cooking_goal_image": (
            "prepare_cooking_with_two_stoves_and_microwave"
        ),
    }

    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        if task_order_index != 0:
            raise ValueError("long_horizon_v2 only supports task_order_index=0")
        self.name = "long_horizon_v2"
        tasks = []
        for name in libero_task_map[self.name]:
            asset_stem = self._SHARED_ASSET_STEMS.get(name, name)
            tasks.append(
                Task(
                    name=name,
                    language=self._INSTRUCTIONS[name],
                    problem="Libero",
                    problem_folder=self.name,
                    bddl_file=f"{asset_stem}.bddl",
                    init_states_file=f"{asset_stem}.pruned_init",
                )
            )
        self.tasks = tuple(tasks)
        self.n_tasks = len(self.tasks)

    def get_task_bddl_file_path(self, i):
        task = self.get_task(i)
        return _task_resource_path(
            os.path.join(self._PACKAGE_ROOT, "bddl_files"),
            task,
            task.bddl_file,
        )

    def get_task_init_states(self, i):
        task = self.get_task(i)
        path = _task_resource_path(
            os.path.join(self._PACKAGE_ROOT, "init_files"),
            task,
            task.init_states_file,
        )
        return torch.load(path, weights_only=False)


@register_benchmark
class PERCEPTION_V2(Benchmark):
    """Ten V2 perception tasks assembled from validated local scenes."""

    _TASKS = (
        (
            "select_upright_marker",
            "Pick up the only upright marker and place it in the collection bin.",
            "vlabench_perception",
            "select_upright_marker",
        ),
        (
            "select_between_condiments",
            "Pick up the object positioned between the other two objects and place it in the basket.",
            "perception_v2",
            "select_between_condiments",
        ),
        (
            "select_red_object",
            "Pick up the red object and place it in the collection bin.",
            "vlabench_perception",
            "select_red_object",
        ),
        (
            "select_striped_object",
            "Pick up the object with the striped texture and place it in the collection bin.",
            "vlabench_perception",
            "select_striped_object",
        ),
        (
            "select_marked_instance",
            "Pick up the object with exactly one square mark and place it in the collection bin.",
            "vlabench_perception",
            "select_marked_instance",
        ),
        (
            "select_ball_shape",
            "The three candidate-button pairs show a small ball, a cylinder, and a cube. Press the button paired with the small ball.",
            "perception_v2",
            "select_ball_shape",
        ),
        (
            "select_cola_drink",
            "Pick up the cola and place it in the collection bin.",
            "vlabench_world_knowledge",
            "select_cola_drink",
        ),
        (
            "weigh_and_place_heaviest_mug",
            "Determine which of the three visually identical mugs is heaviest by physically weighing them, then place the heaviest mug in the green bin.",
            "long_horizon_v2",
            "weigh_and_place_heaviest_mug",
        ),
        (
            "select_highest_reflectance",
            "The three candidate-button pairs have visibly different surface reflectance. Press the button paired with the most reflective object.",
            "perception_v2",
            "select_highest_reflectance",
        ),
        (
            "select_largest_cylinder",
            "The three candidate-button pairs show cylinders of different sizes. Press the button paired with the largest cylinder.",
            "perception_v2",
            "select_largest_cylinder",
        ),
    )

    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        if task_order_index != 0:
            raise ValueError("perception_v2 only supports task_order_index=0")
        self.name = "perception_v2"
        self.tasks = tuple(
            Task(
                name=name,
                language=language,
                problem="Libero",
                problem_folder=problem_folder,
                bddl_file=f"{source_name}.bddl",
                init_states_file=f"{source_name}.pruned_init",
            )
            for name, language, problem_folder, source_name in self._TASKS
        )
        self.n_tasks = len(self.tasks)


@register_benchmark
class VLABENCH_WORLD_KNOWLEDGE(Benchmark):
    """Matched VLABench world-knowledge task families.

    The original condiment quartet remains intact for backwards compatibility.
    The suite now contains sixteen tasks: four condiment controls, four
    previously added semantic tasks, and eight explicit/spatial variants that
    adapt the pinned VLABench fruit, drink, flower, and chemistry families.
    All terminal predicates are ordinary ``In`` relations into a receptacle,
    so the generic checker can distinguish the correct target from the
    candidates.
    """

    _PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        if task_order_index != 0:
            raise ValueError(
                "vlabench_world_knowledge only supports task_order_index=0"
            )
        self.name = "vlabench_world_knowledge"
        self.tasks = (
            Task(
                name="select_sweet_condiment",
                language=(
                    "Pick up the condiment that would make a dish sweeter and "
                    "place it in the basket."
                ),
                problem="Libero",
                problem_folder="vlabench_world_knowledge",
                bddl_file="select_sweet_condiment.bddl",
                init_states_file="select_sweet_condiment.pruned_init",
            ),
            Task(
                name="select_salty_condiment",
                language=(
                    "Pick up the condiment that would make a dish saltier and "
                    "place it in the basket."
                ),
                problem="Libero",
                problem_folder="vlabench_world_knowledge",
                bddl_file="select_salty_condiment.bddl",
                init_states_file="select_salty_condiment.pruned_init",
            ),
            Task(
                name="select_spicy_condiment",
                language=(
                    "Pick up the condiment that would make a dish spicier and "
                    "place it in the basket."
                ),
                problem="Libero",
                problem_folder="vlabench_world_knowledge",
                bddl_file="select_spicy_condiment.bddl",
                init_states_file="select_spicy_condiment.pruned_init",
            ),
            Task(
                name="select_sweet_condiment_explicit_name",
                language="Pick up the sugar and place it in the basket.",
                problem="Libero",
                problem_folder="vlabench_world_knowledge",
                bddl_file="select_sweet_condiment_explicit_name.bddl",
                init_states_file="select_sweet_condiment_explicit_name.pruned_init",
            ),
            Task(
                name="select_vitamin_c_fruit",
                language=(
                    "Pick up the fruit that provides the most vitamin C among "
                    "the choices and place it in the collection bin."
                ),
                problem="Libero",
                problem_folder="vlabench_world_knowledge",
                bddl_file="select_vitamin_c_fruit.bddl",
                init_states_file="select_vitamin_c_fruit.pruned_init",
            ),
            Task(
                name="select_calcium_drink",
                language=(
                    "Pick up the drink that is a common dietary source of "
                    "calcium and place it in the collection bin."
                ),
                problem="Libero",
                problem_folder="vlabench_world_knowledge",
                bddl_file="select_calcium_drink.bddl",
                init_states_file="select_calcium_drink.pruned_init",
            ),
            Task(
                name="select_rose_for_love",
                language=(
                    "Pick up the flower traditionally associated with romantic "
                    "love and place it in the collection bin."
                ),
                problem="Libero",
                problem_folder="vlabench_world_knowledge",
                bddl_file="select_rose_for_love.bddl",
                init_states_file="select_rose_for_love.pruned_init",
            ),
            Task(
                name="select_copper_sulfate_solution",
                language=(
                    "Pick up the solution associated with copper sulfate in a "
                    "school chemistry demonstration and place it in the "
                    "collection bin."
                ),
                problem="Libero",
                problem_folder="vlabench_world_knowledge",
                bddl_file="select_copper_sulfate_solution.bddl",
                init_states_file="select_copper_sulfate_solution.pruned_init",
            ),
            Task(
                name="select_banana_fruit",
                language="Pick up the banana and place it in the collection bin.",
                problem="Libero",
                problem_folder="vlabench_world_knowledge",
                bddl_file="select_banana_fruit.bddl",
                init_states_file="select_banana_fruit.pruned_init",
            ),
            Task(
                name="select_between_fruit",
                language=(
                    "Pick up the fruit positioned between the other two fruits "
                    "and place it in the collection bin."
                ),
                problem="Libero",
                problem_folder="vlabench_world_knowledge",
                bddl_file="select_between_fruit.bddl",
                init_states_file="select_between_fruit.pruned_init",
            ),
            Task(
                name="select_cola_drink",
                language="Pick up the cola and place it in the collection bin.",
                problem="Libero",
                problem_folder="vlabench_world_knowledge",
                bddl_file="select_cola_drink.bddl",
                init_states_file="select_cola_drink.pruned_init",
            ),
            Task(
                name="select_nearest_drink",
                language=(
                    "Pick up the drink closest to the collection bin and place "
                    "it in the collection bin."
                ),
                problem="Libero",
                problem_folder="vlabench_world_knowledge",
                bddl_file="select_nearest_drink.bddl",
                init_states_file="select_nearest_drink.pruned_init",
            ),
            Task(
                name="select_sunflower_flower",
                language="Pick up the sunflower and place it in the collection bin.",
                problem="Libero",
                problem_folder="vlabench_world_knowledge",
                bddl_file="select_sunflower_flower.bddl",
                init_states_file="select_sunflower_flower.pruned_init",
            ),
            Task(
                name="select_nearest_flower",
                language=(
                    "Pick up the flower closest to the collection bin and place "
                    "it in the collection bin."
                ),
                problem="Libero",
                problem_folder="vlabench_world_knowledge",
                bddl_file="select_nearest_flower.bddl",
                init_states_file="select_nearest_flower.pruned_init",
            ),
            Task(
                name="select_green_solution",
                language="Pick up the green solution and place it in the collection bin.",
                problem="Libero",
                problem_folder="vlabench_world_knowledge",
                bddl_file="select_green_solution.bddl",
                init_states_file="select_green_solution.pruned_init",
            ),
            Task(
                name="select_nearest_solution",
                language=(
                    "Pick up the solution closest to the collection bin and "
                    "place it in the collection bin."
                ),
                problem="Libero",
                problem_folder="vlabench_world_knowledge",
                bddl_file="select_nearest_solution.bddl",
                init_states_file="select_nearest_solution.pruned_init",
            ),
        )
        self.n_tasks = len(self.tasks)

    def get_task_bddl_file_path(self, i):
        task = self.get_task(i)
        return _task_resource_path(
            os.path.join(self._PACKAGE_ROOT, "bddl_files"),
            task,
            task.bddl_file,
        )


@register_benchmark
class VLABENCH_PERCEPTION(Benchmark):
    """Matched visual-state discrimination tasks from VLABench families.

    Public annotations remain anonymous, while the scene presents a unique
    visual cue: same-mesh pose (upright/fallen), bloom state, colour, texture,
    or an instance mark.  Every task ends with the correct object entering a
    neutral collection bin, which leaves terminal authority in the generic
    BDDL checker.
    """

    _PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        if task_order_index != 0:
            raise ValueError("vlabench_perception only supports task_order_index=0")
        self.name = "vlabench_perception"
        self.tasks = (
            Task(
                name="select_upright_marker",
                language=(
                    "Pick up the upright marker, rather than a fallen marker, "
                    "and place it in the collection bin."
                ),
                problem="Libero",
                problem_folder="vlabench_perception",
                bddl_file="select_upright_marker.bddl",
                init_states_file="select_upright_marker.pruned_init",
            ),
            Task(
                name="select_bloomed_flower",
                language=(
                    "Pick up the flower that is fully bloomed, not the wilted "
                    "flower, and place it in the collection bin."
                ),
                problem="Libero",
                problem_folder="vlabench_perception",
                bddl_file="select_bloomed_flower.bddl",
                init_states_file="select_bloomed_flower.pruned_init",
            ),
            Task(
                name="select_red_object",
                language="Pick up the red object and place it in the collection bin.",
                problem="Libero",
                problem_folder="vlabench_perception",
                bddl_file="select_red_object.bddl",
                init_states_file="select_red_object.pruned_init",
            ),
            Task(
                name="select_striped_object",
                language=(
                    "Pick up the object with the striped texture and place it "
                    "in the collection bin."
                ),
                problem="Libero",
                problem_folder="vlabench_perception",
                bddl_file="select_striped_object.bddl",
                init_states_file="select_striped_object.pruned_init",
            ),
            Task(
                name="select_marked_instance",
                language=(
                    "Pick up the instance with one square mark and place it in "
                    "the collection bin."
                ),
                problem="Libero",
                problem_folder="vlabench_perception",
                bddl_file="select_marked_instance.bddl",
                init_states_file="select_marked_instance.pruned_init",
            ),
            Task(
                name="select_spotted_object",
                language=(
                    "Pick up the object with the spotted texture and place it "
                    "in the collection bin."
                ),
                problem="Libero",
                problem_folder="vlabench_perception",
                bddl_file="select_spotted_object.bddl",
                init_states_file="select_spotted_object.pruned_init",
            ),
        )
        self.n_tasks = len(self.tasks)

    def get_task_bddl_file_path(self, i):
        task = self.get_task(i)
        return _task_resource_path(
            os.path.join(self._PACKAGE_ROOT, "bddl_files"),
            task,
            task.bddl_file,
        )


@register_benchmark
class VLABENCH_TUBE_PRECISION(Benchmark):
    """Chemistry-tube insertion and marked-rack rearrangement tasks."""

    _PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        if task_order_index != 0:
            raise ValueError("vlabench_tube_precision only supports task_order_index=0")
        self.name = "vlabench_tube_precision"
        self.tasks = (
            Task(
                name="insert_marked_chemistry_tube",
                language=(
                    "Pick up the blue chemistry tube and insert it vertically "
                    "into the rear-center empty slot of the tube rack."
                ),
                problem="Libero",
                problem_folder="vlabench_tube_precision",
                bddl_file="insert_marked_chemistry_tube.bddl",
                init_states_file="insert_marked_chemistry_tube.pruned_init",
            ),
            Task(
                name="rearrange_marked_chemistry_tubes",
                language=(
                    "Arrange the chemistry tubes according to the visible rack "
                    "NameTags."
                ),
                problem="Libero",
                problem_folder="vlabench_tube_precision",
                bddl_file="rearrange_marked_chemistry_tubes.bddl",
                init_states_file="rearrange_marked_chemistry_tubes.pruned_init",
            ),
        )
        self.n_tasks = len(self.tasks)

    def get_task_bddl_file_path(self, i):
        task = self.get_task(i)
        return _task_resource_path(
            os.path.join(self._PACKAGE_ROOT, "bddl_files"),
            task,
            task.bddl_file,
        )


@register_benchmark
class VLABENCH_ADDITIONAL(Benchmark):
    """Orthogonal VLABench search, decomposition, and clustering tasks."""

    _PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        if task_order_index != 0:
            raise ValueError("vlabench_additional only supports task_order_index=0")
        self.name = "vlabench_additional"
        self.tasks = (
            Task(
                name="find_unseen_object",
                language=(
                    "Find the hidden apple behind the cabinet drawers, open the "
                    "obstructing drawer, and place the apple in the basket."
                ),
                problem="Libero",
                problem_folder="vlabench_additional",
                bddl_file="find_unseen_object.bddl",
                init_states_file="find_unseen_object.pruned_init",
            ),
            Task(
                name="cluster_series",
                language=(
                    "Cluster the fruit by type, putting all apples in one basket "
                    "and all bananas in the other basket."
                ),
                problem="Libero",
                problem_folder="vlabench_additional",
                bddl_file="cluster_series.bddl",
                init_states_file="cluster_series.pruned_init",
            ),
        )
        self.n_tasks = len(self.tasks)

    def get_task_bddl_file_path(self, i):
        task = self.get_task(i)
        return _task_resource_path(
            os.path.join(self._PACKAGE_ROOT, "bddl_files"),
            task,
            task.bddl_file,
        )


@register_benchmark
class VLABENCH_BUTTON(Benchmark):
    """Seven VLABench-inspired physical-property button questions.

    The language field contains the world-knowledge question and the required
    physical response.  Candidate descriptions, answer identities, and all
    source/geometry information stay in the task-local scene and private
    evaluator rather than the Agent observation.
    """

    _PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        if task_order_index != 0:
            raise ValueError("vlabench_button only supports task_order_index=0")
        self.name = "vlabench_button"
        # Keep this import local so the base benchmark module remains usable
        # for installations that only need the original LIBERO suites.
        from .benchmark_physical_property import PHYSICAL_PROPERTY_TASKS

        property_tasks = tuple(
            Task(
                name=task.name,
                language=task.instruction,
                problem="Libero",
                problem_folder="vlabench_button",
                bddl_file=f"{task.name}.bddl",
                init_states_file=f"{task.name}.pruned_init",
            )
            for task in PHYSICAL_PROPERTY_TASKS
        )
        self.tasks = property_tasks + (
            Task(
                name="press_button",
                language="Press the red button.",
                problem="Libero",
                problem_folder="vlabench_button",
                bddl_file="press_button.bddl",
                init_states_file="press_button.pruned_init",
            ),
        )
        self.n_tasks = len(self.tasks)

    def get_task_bddl_file_path(self, i):
        task = self.get_task(i)
        return _task_resource_path(
            os.path.join(self._PACKAGE_ROOT, "bddl_files"),
            task,
            task.bddl_file,
        )


@register_benchmark
class VLABENCH_PHYSICAL(Benchmark):
    """VLABench-derived physical-interaction tasks.

    The suite keeps the original Basic Seesaw Usage task and adds two
    dynamics-grounded variants.  It is explicit because each task's private
    ordered checker is stronger than its flat BDDL diagnostic goal.
    """

    _PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        if task_order_index != 0:
            raise ValueError("vlabench_physical only supports task_order_index=0")
        self.name = "vlabench_physical"
        self.tasks = (
            Task(
                name="basic_seesaw_usage",
                language="Lift the hidden object and place it in the basket.",
                problem="Libero",
                problem_folder="vlabench_physical",
                bddl_file="basic_seesaw_usage.bddl",
                init_states_file="basic_seesaw_usage.pruned_init",
            ),
            Task(
                name="adaptive_seesaw_usage",
                language=(
                    "Probe the seesaw with a subset of neutral weights until the "
                    "board tilts, then place the payload in the basket."
                ),
                problem="Libero",
                problem_folder="vlabench_physical",
                bddl_file="adaptive_seesaw_usage.bddl",
                init_states_file="adaptive_seesaw_usage.pruned_init",
            ),
            Task(
                name="active_weight_comparison",
                language=(
                    "Compare two neutral weights by placing each on the marked "
                    "far end in turn, then place the heavier one in the basket."
                ),
                problem="Libero",
                problem_folder="vlabench_physical",
                bddl_file="active_weight_comparison.bddl",
                init_states_file="active_weight_comparison.pruned_init",
            ),
        )
        self.n_tasks = len(self.tasks)

    def get_task_bddl_file_path(self, i):
        task = self.get_task(i)
        return _task_resource_path(
            os.path.join(self._PACKAGE_ROOT, "bddl_files"),
            task,
            task.bddl_file,
        )


@register_benchmark
class LIBERO_100(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_100"
        self._make_benchmark()
