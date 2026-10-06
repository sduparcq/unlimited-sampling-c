from dataclasses import dataclass

from ..methods.fista import Fista, FistaResult
from ..signals.factory import SignalFactory
from .experiments import FistaExperiment


@dataclass
class ExperimentRun:
    experiment: FistaExperiment
    result: FistaResult


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
):
    fista = Fista()
    factory = SignalFactory()

    prepared_cache = {}
    results = []

    for experiment in experiments:
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

        results.append(
            ExperimentRun(
                experiment=experiment,
                result=result,
            )
        )

    return results
