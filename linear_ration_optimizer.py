"""Sequential piecewise-linear (LP) ration formulation for the NASEM standard.

This module provides the ration optimizer used by the simulation engine: a
sequential piecewise-linear approximation (LP) of the full NASEM constraint
set (energy, metabolizable protein, NDF, forage NDF, fat, calcium,
phosphorus, and the dry matter intake bounds).  Each LP is valid in a trust
region around the current ration; successive LPs update that region until the
solution stabilizes.  Only the NASEM nutrient standard is supported; the
simulation engine raises for the NRC standard and for user-defined rations.
"""

from dataclasses import dataclass, fields, replace
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import OptimizeResult, linprog

from RUFAS.biophysical.animal.data_types.animal_combination import AnimalCombination
from RUFAS.biophysical.animal.data_types.nutrition_data_structures import NutritionRequirements
from RUFAS.biophysical.animal.ration.ration_optimizer import RationConfig, RationOptimizer
from RUFAS.data_structures.feed_storage_to_animal_connection import Feed, NutrientStandard


@dataclass(frozen=True)
class LinearRationResult:
    """Result of the sequential LP approximation."""

    success: bool
    message: str
    amounts: dict[int | str, float]
    objective: float
    iterations: int
    raw_result: OptimizeResult


@dataclass(frozen=True)
class SensitivityRecord:
    """One re-optimized sensitivity scenario."""

    parameter: str
    multiplier: float
    success: bool
    objective: float
    amounts: dict[int | str, float]


class LinearRationOptimizer:
    """Solve a linearized approximation of the RuFaS NASEM ration problem."""

    constraint_names = ("energy", "protein", "ndf", "forage_ndf", "fat", "calcium", "phosphorus", "dmi")

    def __init__(
        self,
        feeds: Sequence[Feed],
        requirements: NutritionRequirements,
        body_weight: float,
        enteric_methane: float = 0.0,
        urinary_nitrogen: float = 0.0,
        initial_dry_matter_requirement: float | None = None,
        initial_protein_requirement: float | None = None,
        enabled_constraints: Sequence[str] = constraint_names,
        segment_width: float = 0.25,
        max_iterations: int = 25,
        convergence_tolerance: float = 1e-5,
        animal_combination: AnimalCombination = AnimalCombination.LAC_COW,
    ) -> None:
        if not feeds:
            raise ValueError("At least one feed is required.")
        if body_weight <= 0:
            raise ValueError("body_weight must be positive.")
        if segment_width <= 0 or segment_width > 1:
            raise ValueError("segment_width must be in the interval (0, 1].")
        if max_iterations < 1:
            raise ValueError("max_iterations must be at least 1.")
        if convergence_tolerance <= 0:
            raise ValueError("convergence_tolerance must be positive.")
        unknown = set(enabled_constraints) - set(self.constraint_names)
        if unknown:
            raise ValueError(f"Unknown linearized constraints: {sorted(unknown)}")

        self.feeds = sorted(feeds, key=lambda feed: feed.rufas_id)
        self.requirements = requirements
        self.body_weight = body_weight
        self.enteric_methane = enteric_methane
        self.urinary_nitrogen = urinary_nitrogen
        self.initial_dry_matter_requirement = (
            requirements.dry_matter
            if initial_dry_matter_requirement is None
            else initial_dry_matter_requirement
        )
        self.initial_protein_requirement = (
            requirements.metabolizable_protein
            if initial_protein_requirement is None
            else initial_protein_requirement
        )
        self.enabled_constraints = tuple(enabled_constraints)
        self.segment_width = segment_width
        self.max_iterations = max_iterations
        self.convergence_tolerance = convergence_tolerance
        self.animal_combination = animal_combination

    def build_config(self, requirements: NutritionRequirements | None = None) -> RationConfig:
        return RationConfig(
            nutrient_standard=NutrientStandard.NASEM,
            animal_requirements=requirements or self.requirements,
            pen_available_feeds=list(self.feeds),
            initial_dry_matter_requirement=self.initial_dry_matter_requirement,
            initial_protein_requirement=self.initial_protein_requirement,
            pen_average_body_weight=self.body_weight,
            pen_average_enteric_methane=self.enteric_methane,
            pen_average_urine_nitrogen=self.urinary_nitrogen,
            animal_combination=self.animal_combination,
        )

    def _constraint_values(
        self, amounts: np.ndarray, requirements: NutritionRequirements | None = None
    ) -> np.ndarray:
        """Return nonnegative-form NASEM constraints at ``amounts``."""
        config = self.build_config(requirements)
        values: list[float] = []
        if "energy" in self.enabled_constraints:
            values.append(RationOptimizer.NASEM_net_energy_constraint(amounts, config))
        if "protein" in self.enabled_constraints:
            values.append(RationOptimizer.protein_constraint_lower(amounts, config))
            values.append(RationOptimizer.protein_constraint_upper(amounts, config))
        if "ndf" in self.enabled_constraints:
            values.append(RationOptimizer.NDF_constraint_lower(amounts, config))
            values.append(RationOptimizer.NDF_constraint_upper(amounts, config))
        if "forage_ndf" in self.enabled_constraints:
            values.append(RationOptimizer.forage_NDF_constraint(amounts, config))
        if "fat" in self.enabled_constraints:
            values.append(RationOptimizer.fat_constraint(amounts, config))
        if "calcium" in self.enabled_constraints:
            values.append(RationOptimizer.calcium_constraint(amounts, config))
        if "phosphorus" in self.enabled_constraints:
            values.append(RationOptimizer.phosphorus_constraint(amounts, config))
        if "dmi" in self.enabled_constraints:
            values.append(RationOptimizer.DMI_constraint_lower(amounts, config))
            values.append(RationOptimizer.DMI_constraint_upper(amounts, config))
        return np.asarray(values, dtype=float)

    def _linearized_constraints(
        self, reference: np.ndarray, requirements: NutritionRequirements | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build ``A_ub @ x <= b_ub`` from first-order piecewise segments."""
        values = self._constraint_values(reference, requirements)
        jacobian = np.empty((len(values), len(reference)))
        for index in range(len(reference)):
            step = max(abs(reference[index]) * 1e-5, 1e-6)
            forward = reference.copy()
            backward = reference.copy()
            forward[index] += step
            backward[index] = max(0.0, backward[index] - step)
            denominator = forward[index] - backward[index]
            jacobian[:, index] = (
                self._constraint_values(forward, requirements)
                - self._constraint_values(backward, requirements)
            ) / denominator

        # f(x) ~= f(x0) + grad f(x0)(x-x0) >= 0
        # scipy.linprog uses A_ub x <= b_ub, hence the sign reversal.
        intercept = values - jacobian @ reference
        return -jacobian, intercept

    def solve(
        self,
        initial_amounts: Sequence[float] | None = None,
        costs: Mapping[int | str, float] | None = None,
    ) -> LinearRationResult:
        """Solve successive local LPs and return the final ration."""
        lower = np.asarray([feed.lower_limit for feed in self.feeds], dtype=float)
        upper = np.asarray([feed.limit for feed in self.feeds], dtype=float)
        if initial_amounts is not None and len(initial_amounts) != len(self.feeds):
            raise ValueError("initial_amounts must have one value per feed.")
        reference = (
            np.clip(np.asarray(initial_amounts, dtype=float), lower, upper)
            if initial_amounts is not None
            else np.clip(np.full(len(self.feeds), self.initial_dry_matter_requirement / len(self.feeds)), lower, upper)
        )
        objective = np.asarray(
            [
                (costs or {}).get(feed.rufas_id, feed.purchase_cost)
                for feed in self.feeds
            ],
            dtype=float,
        )
        last_result: OptimizeResult | None = None
        for iteration in range(1, self.max_iterations + 1):
            width = np.maximum(upper * self.segment_width, 1e-6)
            local_bounds = list(
                zip(np.maximum(lower, reference - width), np.minimum(upper, reference + width))
            )
            a_ub, b_ub = self._linearized_constraints(reference)
            result = linprog(objective, A_ub=a_ub, b_ub=b_ub, bounds=local_bounds, method="highs")
            last_result = result
            if not result.success:
                amounts = dict[int | str, float](
                    zip([feed.rufas_id for feed in self.feeds], reference)
                )
                return LinearRationResult(False, result.message, amounts, float("inf"), iteration, result)
            if np.max(np.abs(result.x - reference)) <= self.convergence_tolerance:
                reference = result.x
                break
            reference = result.x

        assert last_result is not None
        amounts = dict[int | str, float](zip([feed.rufas_id for feed in self.feeds], reference))
        return LinearRationResult(
            bool(last_result.success),
            last_result.message,
            amounts,
            float(np.dot(objective, reference)),
            iteration,
            last_result,
        )

    def _rhs_scenario_records(
        self, rhs_parameters: Mapping[str, Sequence[float]]
    ) -> list[SensitivityRecord]:
        """Re-solve for scenarios that scale a nutrition requirement."""
        records: list[SensitivityRecord] = []
        valid_parameters = {field.name for field in fields(self.requirements)}
        for parameter, multipliers in rhs_parameters.items():
            if parameter not in valid_parameters:
                raise ValueError(f"Unknown nutrition requirement: {parameter}")
            baseline_value = getattr(self.requirements, parameter)
            for multiplier in multipliers:
                if multiplier <= 0:
                    raise ValueError("Sensitivity multipliers must be positive.")
                scenario_requirements = replace(
                    self.requirements, **{parameter: baseline_value * multiplier}
                )
                scenario_dmi = self.initial_dry_matter_requirement
                scenario_protein = self.initial_protein_requirement
                if parameter == "dry_matter":
                    scenario_dmi = baseline_value * multiplier
                if parameter == "metabolizable_protein":
                    scenario_protein = baseline_value * multiplier
                scenario = LinearRationOptimizer(
                    self.feeds,
                    scenario_requirements,
                    self.body_weight,
                    self.enteric_methane,
                    self.urinary_nitrogen,
                    scenario_dmi,
                    scenario_protein,
                    self.enabled_constraints,
                    self.segment_width,
                    self.max_iterations,
                    self.convergence_tolerance,
                    self.animal_combination,
                ).solve()
                records.append(
                    SensitivityRecord(parameter, multiplier, scenario.success, scenario.objective, scenario.amounts)
                )
        return records

    def _cost_scenario_records(
        self, cost_multipliers: Mapping[int | str, Sequence[float]]
    ) -> list[SensitivityRecord]:
        """Re-solve for scenarios that scale one feed purchase cost."""
        records: list[SensitivityRecord] = []
        for feed_id, multipliers in cost_multipliers.items():
            for multiplier in multipliers:
                if multiplier <= 0:
                    raise ValueError("Sensitivity multipliers must be positive.")
                costs: dict[int | str, float] = {
                    feed.rufas_id: feed.purchase_cost for feed in self.feeds
                }
                if feed_id not in costs:
                    raise ValueError(f"Unknown feed ID: {feed_id}")
                costs[feed_id] *= multiplier
                scenario = self.solve(costs=costs)
                records.append(
                    SensitivityRecord(
                        f"cost:{feed_id}", multiplier, scenario.success, scenario.objective, scenario.amounts
                    )
                )
        return records

    def sensitivity_analysis(
        self,
        rhs_parameters: Mapping[str, Sequence[float]] | None = None,
        cost_multipliers: Mapping[int | str, Sequence[float]] | None = None,
    ) -> list[SensitivityRecord]:
        """Re-solve for RHS and cost scenarios and return their impacts."""
        records: list[SensitivityRecord] = []
        baseline = self.solve()
        records.extend(self._rhs_scenario_records(rhs_parameters or {}))
        records.extend(self._cost_scenario_records(cost_multipliers or {}))
        if not records:
            return [
                SensitivityRecord("baseline", 1.0, baseline.success, baseline.objective, baseline.amounts)
            ]
        return records
