from pathlib import Path
import torchx
from torchx import specs
from torchx.components import utils

import tempfile
from ax.runners.torchx import TorchXRunner

from ax.core import ChoiceParameter, ParameterType, RangeParameter, SearchSpace

from ax.metrics.tensorboard import TensorboardCurveMetric

from ax.core import MultiObjective, Objective, ObjectiveThreshold
from ax.core.optimization_config import MultiObjectiveOptimizationConfig

from ax.core import Experiment

from ax.modelbridge.dispatch_utils import choose_generation_strategy

from ax.service.scheduler import Scheduler, SchedulerOptions


def create_experiment_scheduler(scriptname="NeurcompTraining.py", total_trials=48):

    def trainer(
        log_path: str,
        lambda_betas: float,
        lambda_weights: float,
        trial_idx: int = -1,
    ) -> specs.AppDef:

        # define the log path so we can pass it to the TorchX AppDef
        if trial_idx >= 0:
            log_path = Path(log_path).joinpath(str(trial_idx)).absolute().as_posix()

        config = 'experiment-config-files/mhd_p_HyperParamSearch.txt'  # TODO M: parse this in

        return utils.python(
            # command line args to the training script
            "--config",
            config,
            "--Tensorboard_log_dir",
            log_path,

            # M: hyperparam
            "--lambda_betas",
            str(lambda_betas),
            "--lambda_weights",
            str(lambda_weights),

            # M: TODO search for best rmse for compression rate: Diff NW Size + Num Layers + Pruning

            # other config options
            name="trainer",
            script=scriptname,
            image=torchx.version.TORCHX_IMAGE,
        )

    # Make a temporary dir to log our results into
    log_dir = tempfile.mkdtemp()
    print("LOG_DIR: ", log_dir)

    ax_runner = TorchXRunner(
        tracker_base="/tmp/",
        component=trainer,
        # NOTE: To launch this job on a cluster instead of locally you can
        # specify a different scheduler and adjust args appropriately.
        scheduler="local_cwd",
        component_const_params={"log_path": log_dir},
        cfg={},
    )

    parameters = [
        # NOTE: In a real-world setting, hidden_size_1 and hidden_size_2
        # should probably be powers of 2, but in our simple example this
        # would mean that num_params can't take on that many values, which
        # in turn makes the Pareto frontier look pretty weird.
        RangeParameter(
            name="lambda_betas",
            lower=5e-08,
            upper=5e-05,
            parameter_type=ParameterType.FLOAT,
            log_scale=True,
        ),
        RangeParameter(
            name="lambda_weights",
            lower=5e-08,
            upper=5e-05,
            parameter_type=ParameterType.FLOAT,
            log_scale=True,
        ),
    ]

    search_space = SearchSpace(
        parameters=parameters,
        # NOTE: In practice, it may make sense to add a constraint
        # hidden_size_2 <= hidden_size_1
        parameter_constraints=[],
    )

    class MyTensorboardMetric(TensorboardCurveMetric):

        # NOTE: We need to tell the new Tensorboard metric how to get the id /
        # file handle for the tensorboard logs from a trial. In this case
        # our convention is to just save a separate file per trial in
        # the pre-specified log dir.
        @classmethod
        def get_ids_from_trials(cls, trials):
            return {
                trial.index: Path(log_dir).joinpath(str(trial.index)).as_posix()
                for trial in trials
            }

        # This indicates whether the metric is queryable while the trial is
        # still running. We don't use this in the current tutorial, but Ax
        # utilizes this to implement trial-level early-stopping functionality.
        @classmethod
        def is_available_while_running(cls):
            return False

    compression_ratio = MyTensorboardMetric(
        name="compression_ratio",
        curve_name="compression_ratio",
        lower_is_better=False,
    )
    rmse = MyTensorboardMetric(
        name="rmse",
        curve_name="rmse",
        lower_is_better=True,
    )

    opt_config = MultiObjectiveOptimizationConfig(
        objective=MultiObjective(
            objectives=[
                Objective(metric=compression_ratio, minimize=False),
                Objective(metric=rmse, minimize=True),
            ],
        ),
        objective_thresholds=[
            ObjectiveThreshold(metric=compression_ratio, bound=105.0, relative=False),
            ObjectiveThreshold(metric=rmse, bound=0.011, relative=False),
        ],
    )

    experiment = Experiment(
        name="torchx_mhd_p_Smallify",
        search_space=search_space,
        optimization_config=opt_config,
        runner=ax_runner,
    )

    gs = choose_generation_strategy(
        search_space=experiment.search_space,
        optimization_config=experiment.optimization_config,
        num_trials=total_trials,
      )

    scheduler = Scheduler(
        experiment=experiment,
        generation_strategy=gs,
        options=SchedulerOptions(
            total_trials=total_trials, max_pending_trials=4
        ),
    )

    return experiment, scheduler

#scheduler.run_all_trials()
