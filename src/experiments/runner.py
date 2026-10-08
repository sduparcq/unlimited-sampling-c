from dataclasses import asdict
from pathlib import Path

from tqdm import tqdm

from ..results import *
from ..signals.factory import SignalFactory


def run_batch(
    experiments,
    method,
    compute_metrics,
    result_to_row,
    output_path: str | Path,
):
    factory = SignalFactory()

    prepared_cache = {}
    signal_cache = {}
    solve_cache = {}

    with CSVWriter(
        path=output_path,
    ) as writer:
        for experiment in tqdm(
            experiments,
            desc="Experiments",
        ):
            preparation_key = (
                experiment.signal.N,
                tuple(
                    asdict(
                        experiment.preparation,
                    ).items()
                ),
            )

            if preparation_key not in prepared_cache:
                prepared_cache[preparation_key] = method.prepare(
                    N=experiment.signal.N,
                    params=experiment.preparation,
                )

            prepared = prepared_cache[preparation_key]

            signal_key = tuple(
                asdict(
                    experiment.signal,
                ).items()
            )

            if signal_key not in signal_cache:
                signal_cache[signal_key] = factory.generate_signal(
                    experiment.signal,
                )

            signal = signal_cache[signal_key]

            solver_key = (
                preparation_key,
                signal_key,
                tuple(
                    asdict(
                        experiment.solver,
                    ).items()
                ),
            )

            if solver_key not in solve_cache:
                solve_cache[solver_key] = prepared.solve(
                    nms=signal.nms,
                    params=experiment.solver,
                )

            solve_result = solve_cache[solver_key]

            result = method.post_process(
                nms=signal.nms,
                solve_result=solve_result,
                params=experiment.post_process,
            )

            metrics = compute_metrics(
                signal,
                result,
            )

            writer.write(
                result_to_row(
                    experiment,
                    metrics,
                )
            )
