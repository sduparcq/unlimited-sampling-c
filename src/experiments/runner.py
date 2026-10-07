from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from ..methods.fista import Fista, FistaResult
from ..metrics.fista import FistaMetrics, compute_fista_metrics
from ..signals.factory import SignalFactory
from .experiments import FistaExperiment

from ..results import *


@dataclass
class ExperimentRun:
    experiment: FistaExperiment
    result: FistaResult


@dataclass
class BatchResult:
    experiment: FistaExperiment
    metrics: FistaMetrics


def run_fista(
    experiment: FistaExperiment,
    fista: Fista | None = None,
):
    if fista is None:
        fista = Fista()

    factory = SignalFactory()

    prepared = fista.prepare(
        N=experiment.signal.N,
        params=experiment.preparation,
    )

    signal = factory.generate_signal(
        experiment.signal,
    )

    solve_result = prepared.solve(
        y_mod=signal.mod_samples,
        params=experiment.solver,
    )

    result = fista.post_process(
        y_mod=signal.mod_samples,
        solve_result=solve_result,
        params=experiment.post_process,
    )

    return ExperimentRun(
        experiment=experiment,
        result=result,
    )


def run_fista_batch(
    experiments: list[FistaExperiment],
    output_path: str | Path,
):
    fista = Fista()
    factory = SignalFactory()

    prepared_cache = {}

    fieldnames = [
        "M",
        "Omega",
        "L",
        "sigma",
        "s_seed",
        "n_seed",
        "delta",
        "omega",
        "Te",
        "tau",
        "max_iter",
        "threshold",
    ]

    with CSVWriter(
        path=output_path,
        fieldnames=fieldnames,
    ) as writer:
        for experiment in tqdm(
            experiments,
            desc="FISTA experiments",
        ):
            preparation_key = (
                experiment.signal.N,
                experiment.preparation.delta,
                experiment.preparation.omega,
                experiment.preparation.Te,
            )

            if preparation_key not in prepared_cache:
                prepared_cache[preparation_key] = fista.prepare(
                    N=experiment.signal.N,
                    params=experiment.preparation,
                )

            prepared = prepared_cache[preparation_key]

            signal = factory.generate_signal(
                experiment.signal,
            )

            solve_result = prepared.solve(
                y_mod=signal.mod_samples,
                params=experiment.solver,
            )

            result = fista.post_process(
                y_mod=signal.mod_samples,
                solve_result=solve_result,
                params=experiment.post_process,
            )

            metrics = compute_fista_metrics(
                signal,
                result,
            )

            writer.write(
                fista_result_to_row(
                    experiment,
                    metrics,
                )
            )
