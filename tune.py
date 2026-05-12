"""
tune.py — General-purpose Bayesian hyperparameter tuner for Orbit Wars agents.

Architecture overview
---------------------
``TunableStrategy``
    Holds the param-space definition, a factory that builds an agent from a
    sampled config, and helper methods for sampling, building, and rendering
    code snippets.  This is the unit of registration.

``StrategyRegistry``
    Maps string names -> TunableStrategy instances.  Use ``REGISTRY.register()``
    to add entries.  Exposes ``get()``, ``names()``, and iteration so the CLI
    can use it as a ``choices`` source without touching internals.

``MatchEvaluator``
    Resolves opponents once, then exposes ``evaluate(candidate, trial_number)``
    which plays all required games and returns raw per-game scores.  An optional
    ``report_fn`` callback enables Optuna intermediate-value pruning.

``TuningObjective``
    Optuna-compatible callable (``__call__(trial) -> float``) wrapping a
    ``TunableStrategy`` and ``MatchEvaluator``.  Sampling, building, evaluating,
    and pruning are all handled here.

``StudyConfig``
    Dataclass grouping all settings for one tuning run (n_trials, storage, …).
    Has a ``resolved_study_name`` property that fills in the default name.

``StudyRunner``
    Orchestrates one full Optuna lifecycle: creates the study, runs optimization,
    prints results, and emits a copy-paste snippet.  Can be called from the CLI
    or programmatically.

Registering a new tunable strategy
-----------------------------------
1. Write an agent function with signature  ``agent(obs, config: dict) -> list``.
2. Define a factory (or use a lambda) that closes over ``config`` and returns
   a 1-arg ``obs -> moves`` callable.
3. Register it::

    REGISTRY.register(
        "my_agent",
        TunableStrategy(
            factory=lambda config: (lambda obs: my_agent(obs, config)),
            params=[
                FloatParam("alpha",  0.0, 1.0),
                IntParam(  "depth",  1,   5),
                CategoricalParam("mode", ["greedy", "random"]),
            ],
            agent_fn_name="my_agent",
            description="my_agent: alpha, depth, mode",
        ),
    )

Usage
-----
    # List available tunable strategies
    python tune.py list

    # Tune never_miss against random + sniper, 50 trials, resume-safe
    python tune.py tune --strategy never_miss --n-trials 50

    # Tune with more matches per trial, parallel workers, custom storage
    python tune.py tune --strategy never_miss \\
        --n-trials 100 --matches-per-trial 20 \\
        --opponents random sniper roi \\
        --n-jobs 4 \\
        --storage sqlite:///custom.db

    # In-memory (no resume)
    python tune.py tune --strategy never_miss --storage none
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
import optuna

# Suppress TF/kaggle import noise before heavy imports.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from evaluation import AGENTS, run_match_raw
from agents.rule_based import AgentConfig, run_agent, StatefulAgentConfig, make_stateful_agent

RESULTS_DIR = Path(__file__).parent / "results"
RATINGS_FILE = RESULTS_DIR / "opponent_ratings.json"
DEFAULT_RATING = 1500


def load_ratings() -> dict[str, float]:
    """Load opponent ratings from ``results/opponent_ratings.json``.

    Returns a dict mapping opponent name to Elo-style rating.  Any opponent
    absent from the file gets ``DEFAULT_RATING`` (1500) at lookup time.
    """
    if RATINGS_FILE.exists():
        return json.loads(RATINGS_FILE.read_text())
    return {}


def _default_optuna_storage_uri() -> str:
    """SQLite file under ``results/`` (absolute path for stable Optuna URI)."""
    return "sqlite:///" + (RESULTS_DIR / "tune.db").resolve().as_posix()


DEFAULT_OPTUNA_STORAGE = _default_optuna_storage_uri()


# ---------------------------------------------------------------------------
# Param-space primitives
# ---------------------------------------------------------------------------

@dataclass
class FloatParam:
    """Continuous parameter sampled uniformly (or log-uniformly) in [low, high]."""

    name: str
    low: float
    high: float
    log: bool = False  # True → log-uniform (useful for learning rates, etc.)


@dataclass
class IntParam:
    """Integer parameter sampled uniformly in [low, high] inclusive."""

    name: str
    low: int
    high: int


@dataclass
class CategoricalParam:
    """Categorical parameter chosen from an explicit list of values."""

    name: str
    choices: list[Any]


Param = FloatParam | IntParam | CategoricalParam


def _suggest(trial: optuna.Trial, param: Param) -> Any:
    """Dispatch ``trial.suggest_*`` based on param type.

    Args:
        trial: Current Optuna trial.
        param: Parameter specification to sample from.

    Returns:
        Sampled value of the appropriate Python type.

    Raises:
        TypeError: If ``param`` is not a recognized ``Param`` subtype.
    """
    if isinstance(param, FloatParam):
        return trial.suggest_float(param.name, param.low, param.high, log=param.log)
    if isinstance(param, IntParam):
        return trial.suggest_int(param.name, param.low, param.high)
    if isinstance(param, CategoricalParam):
        return trial.suggest_categorical(param.name, param.choices)
    raise TypeError(f"Unknown param type: {type(param)}")


# ---------------------------------------------------------------------------
# TunableStrategy — the unit of registration
# ---------------------------------------------------------------------------

@dataclass
class TunableStrategy:
    """Encapsulates everything needed to tune one agent variant.

    Attributes:
        factory: Given a sampled config dict, returns a 1-arg callable ``obs -> moves``.
        params: Defines the hyperparameter search space.
        agent_fn_name: Python identifier of the underlying agent function, used
            by ``render_snippet`` to produce correct call syntax.
        description: Human-readable summary shown in ``tune.py list``.
    """

    factory: Callable[[dict], Callable]
    params: list[Param]
    agent_fn_name: str = ""
    description: str = ""

    def sample_config(self, trial: optuna.Trial) -> dict[str, Any]:
        """Sample one hyperparameter config from this strategy's param space.

        Args:
            trial: Current Optuna trial used for sampling.

        Returns:
            Dict mapping param names to sampled values.
        """
        return {param.name: _suggest(trial, param) for param in self.params}

    def build_agent(self, config: dict[str, Any]) -> Callable:
        """Instantiate an agent callable from a hyperparameter config.

        Args:
            config: Param-name -> value mapping (as returned by ``sample_config``).

        Returns:
            A 1-arg callable ``obs -> moves`` ready for the environment.
        """
        return self.factory(config)

    def render_snippet(self, strategy_name: str, best_params: dict[str, Any]) -> str:
        """Return a copy-pasteable Python function wrapping the best config.

        Args:
            strategy_name: Registry key (used to derive the generated function name).
            best_params: Best hyperparameter values from the completed study.

        Returns:
            Multi-line string containing a ready-to-paste Python function definition.
        """
        base_fn = self.agent_fn_name or f"{strategy_name.replace('-', '_')}_agent"
        fn_name = f"_{strategy_name.replace('-', '_')}_tuned"
        cfg_str = repr(best_params)
        return (
            "# ── Paste into rule_based.py or arena.py ────────────────────\n"
            f"def {fn_name}(obs):\n"
            f"    return {base_fn}(obs, {cfg_str})"
        )


# ---------------------------------------------------------------------------
# Strategy registry
# ---------------------------------------------------------------------------

class StrategyRegistry:
    """Central registry mapping strategy names to ``TunableStrategy`` instances.

    ``REGISTRY`` is the module-level singleton.  Add entries via ``register()``;
    look them up via ``get()``.  Supports ``in`` and ``iter`` so it can be used
    as an argparse ``choices`` source directly.
    """

    def __init__(self) -> None:
        self._strategies: dict[str, TunableStrategy] = {}

    def register(self, name: str, strategy: TunableStrategy) -> None:
        """Add a strategy to the registry.

        Args:
            name: Unique registry key (used on the CLI and in study names).
            strategy: The ``TunableStrategy`` to register.

        Raises:
            ValueError: If ``name`` is already registered.
        """
        if name in self._strategies:
            raise ValueError(f"Strategy {name!r} is already registered.")
        self._strategies[name] = strategy

    def get(self, name: str) -> TunableStrategy:
        """Retrieve a registered strategy by name.

        Args:
            name: Registry key.

        Returns:
            The corresponding ``TunableStrategy``.

        Raises:
            KeyError: If no strategy with ``name`` exists.
        """
        if name not in self._strategies:
            raise KeyError(
                f"Unknown strategy {name!r}. Known: {sorted(self._strategies)}"
            )
        return self._strategies[name]

    def names(self) -> list[str]:
        """Return all registered strategy names in sorted order."""
        return sorted(self._strategies)

    def items(self):
        """Iterate over ``(name, strategy)`` pairs in alphabetical order."""
        return sorted(self._strategies.items())

    def __contains__(self, name: str) -> bool:
        return name in self._strategies

    def __iter__(self):
        return iter(self.names())


# ---------------------------------------------------------------------------
# Opponent pool — keys are valid --opponents choices
# ---------------------------------------------------------------------------

OPPONENT_POOL: dict[str, object] = {k: k for k in AGENTS}


# ---------------------------------------------------------------------------
# Match evaluator
# ---------------------------------------------------------------------------

class MatchEvaluator:
    """Evaluates a candidate agent against a fixed opponent pool.

    Resolves opponent entries once at construction time.  Each call to
    ``evaluate`` plays all required game pairs and returns the raw per-game
    scores.

    Attributes:
        opponents: Ordered list of opponent names from ``OPPONENT_POOL``.
        matches_per_trial: Number of seed pairs played against each opponent.
        base_seed: Seed offset; per-trial seeds are derived automatically.
        episode_steps: Environment step budget per episode.
    """

    def __init__(
        self,
        opponents: list[str],
        matches_per_trial: int,
        base_seed: int,
        episode_steps: int = 400,
    ) -> None:
        """
        Args:
            opponents: Opponent keys from ``OPPONENT_POOL``.
            matches_per_trial: Game pairs to play against each opponent per trial.
            base_seed: Starting seed; individual seeds are derived from this.
            episode_steps: Maximum environment steps per episode.

        Raises:
            KeyError: If any opponent name is absent from ``OPPONENT_POOL``.
        """
        self.opponents = opponents
        self.matches_per_trial = matches_per_trial
        self.base_seed = base_seed
        self.episode_steps = episode_steps
        self._opp_fns = [OPPONENT_POOL[o] for o in opponents]  # validated once

        ratings = load_ratings()
        raw = [ratings.get(opp, DEFAULT_RATING) for opp in opponents]
        total = sum(raw)
        self._weights: list[float] = [r / total for r in raw]
        self._ratings: list[float] = raw

    def _play_pair(self, candidate: Callable, opponent: object, seed: int) -> float:
        """Play both slot orderings on one seed and return the candidate's average outcome.

        Running both orderings cancels starting-position bias at zero extra cost.

        Args:
            candidate: The agent being evaluated (``obs -> moves`` callable).
            opponent: Opponent agent — string name or callable.
            seed: Environment seed for this game pair.

        Returns:
            Candidate's average outcome in ``{0.0, 0.5, 1.0}``.
        """
        total = 0.0
        for my_slot in range(2):
            agents = [candidate, opponent] if my_slot == 0 else [opponent, candidate]
            raw = run_match_raw(agents, seed, episode_steps=self.episode_steps)
            total += raw["outcomes"][my_slot]
        return total / 2

    def evaluate(
        self,
        candidate: Callable,
        trial_number: int,
        *,
        report_fn: Callable[[float, int], None] | None = None,
    ) -> list[float]:
        """Evaluate ``candidate`` across all opponents and seeds.

        Args:
            candidate: Agent callable to evaluate.
            trial_number: Optuna trial number; used to derive per-trial seeds so
                different trials never share a game.
            report_fn: Optional ``(running_mean, step) -> None`` callback.
                Called after every game; intended for Optuna intermediate reporting
                and pruning.  If the callback raises ``optuna.TrialPruned``, this
                method lets it propagate immediately.

        Returns:
            Flat list of per-game scores in [0.0, 1.0],
            ordered opponent × seed pair.
        """
        scores: list[float] = []
        step = 0
        for opp_fn in self._opp_fns:
            for i in range(self.matches_per_trial):
                seed = self.base_seed + trial_number * 1000 + step
                score = self._play_pair(candidate, opp_fn, seed)
                scores.append(score)
                if report_fn is not None:
                    report_fn(self.weighted_mean(scores), step)
                step += 1
        return scores

    def weighted_mean(self, scores: list[float]) -> float:
        """Return the rating-weighted mean score across opponents.

        Scores are grouped by opponent in blocks of ``matches_per_trial``.
        Incomplete trailing blocks (mid-trial) are included with whatever
        games have finished so far, so this is safe to call from the
        intermediate-value callback.

        Args:
            scores: Flat per-game scores as returned by (or during) ``evaluate``.

        Returns:
            Weighted mean in [0.0, 1.0].
        """
        m = self.matches_per_trial
        per_opp_means: list[float] = []
        active_weights: list[float] = []
        for i, w in enumerate(self._weights):
            chunk = scores[i * m : (i + 1) * m]
            if not chunk:
                break
            per_opp_means.append(statistics.mean(chunk))
            active_weights.append(w)
        total_w = sum(active_weights)
        return sum(wt * s for wt, s in zip(active_weights, per_opp_means)) / total_w


# ---------------------------------------------------------------------------
# Tuning objective (Optuna-compatible callable)
# ---------------------------------------------------------------------------

class TuningObjective:
    """Optuna objective that evaluates a ``TunableStrategy`` via a ``MatchEvaluator``.

    Implements ``__call__(trial) -> float`` so it can be passed directly to
    ``study.optimize``.  Intermediate scores are reported after each game, and
    ``optuna.TrialPruned`` is raised automatically when the pruner fires.

    Attributes:
        strategy: The strategy whose hyperparameters are being optimized.
        evaluator: Pre-configured evaluator that runs actual games.
    """

    def __init__(self, strategy: TunableStrategy, evaluator: MatchEvaluator) -> None:
        """
        Args:
            strategy: Strategy to sample configs from and build agents with.
            evaluator: Evaluator that runs games and returns raw scores.
        """
        self.strategy = strategy
        self.evaluator = evaluator

    def __call__(self, trial: optuna.Trial) -> float:
        """Sample a config, build an agent, evaluate it, and return mean win-share.

        Args:
            trial: Current Optuna trial.

        Returns:
            Mean win-share in [0.0, 1.0] across all games (higher = better).

        Raises:
            optuna.TrialPruned: If the pruner decides to stop this trial early.
        """
        import optuna  # lazy: avoid slow import on --help

        config = self.strategy.sample_config(trial)
        candidate = self.strategy.build_agent(config)

        def _report(mean_score: float, step: int) -> None:
            trial.report(mean_score, step=step)
            if trial.should_prune():
                raise optuna.TrialPruned()

        scores = self.evaluator.evaluate(candidate, trial.number, report_fn=_report)
        return self.evaluator.weighted_mean(scores)


# ---------------------------------------------------------------------------
# Study configuration & runner
# ---------------------------------------------------------------------------

@dataclass
class StudyConfig:
    """All settings for one Optuna tuning run.

    Attributes:
        strategy_name: Registry key of the strategy to tune.
        n_trials: Total number of Optuna trials to execute.
        opponents: Opponent keys from ``OPPONENT_POOL`` to evaluate against.
        matches_per_trial: Game pairs per opponent per trial.
        base_seed: Starting seed for match generation.
        storage: Optuna storage URI.  ``None`` → in-memory (not resumable).
            Default: ``results/tune.db`` under this package (absolute URI).
        study_name: Explicit study name.  ``None`` → ``<strategy_name>_tuning``.
        n_jobs: Parallel trial workers; requires persistent ``storage``.
        quiet: Suppress per-trial Optuna log output when ``True``.
    """

    strategy_name: str
    n_trials: int = 60
    opponents: list[str] = field(default_factory=lambda: ["random", "sniper"])
    matches_per_trial: int = 8
    base_seed: int = 5000
    storage: str | None = DEFAULT_OPTUNA_STORAGE
    study_name: str | None = None
    n_jobs: int = 1
    quiet: bool = False

    @property
    def resolved_study_name(self) -> str:
        """Study name, defaulting to ``<strategy_name>_tuning`` if not set."""
        return self.study_name or f"{self.strategy_name}_tuning"


class StudyRunner:
    """Orchestrates one full Optuna tuning lifecycle.

    Creates the study, configures sampler/pruner, runs optimization, prints
    a result table, and emits a copy-paste snippet.  Can be called from the
    CLI handler or directly from a notebook / script.
    """

    def __init__(self, registry: StrategyRegistry) -> None:
        """
        Args:
            registry: Registry from which strategies are resolved by name.
        """
        self.registry = registry

    def run(self, cfg: StudyConfig) -> optuna.Study:
        """Execute a full tuning run and print results to stdout.

        Args:
            cfg: Configuration bundle for this study.

        Returns:
            The completed ``optuna.Study`` object (useful for programmatic
            post-processing, e.g. plotting the optimization history).
        """
        import optuna  # lazy: avoid slow import on --help

        strategy = self.registry.get(cfg.strategy_name)

        verbosity = optuna.logging.WARNING if cfg.quiet else optuna.logging.INFO
        optuna.logging.set_verbosity(verbosity)

        if cfg.storage:
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        sampler = optuna.samplers.TPESampler(seed=42)
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=4)

        study = optuna.create_study(
            study_name=cfg.resolved_study_name,
            storage=cfg.storage,
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
            load_if_exists=True,
        )

        evaluator = MatchEvaluator(
            opponents=cfg.opponents,
            matches_per_trial=cfg.matches_per_trial,
            base_seed=cfg.base_seed,
        )
        objective = TuningObjective(strategy=strategy, evaluator=evaluator)

        opp_info = "  ".join(
            f"{opp}(r={r:.0f},w={w:.3f})"
            for opp, r, w in zip(cfg.opponents, evaluator._ratings, evaluator._weights)
        )
        print(
            f"Tuning '{cfg.strategy_name}'  |  "
            f"trials={cfg.n_trials}  matches/trial={cfg.matches_per_trial}  "
            f"jobs={cfg.n_jobs}\n"
            f"Opponents: {opp_info}"
        )
        if cfg.storage:
            print(f"Storage: {cfg.storage}  (study '{cfg.resolved_study_name}', resumable)")

        study.optimize(objective, n_trials=cfg.n_trials, n_jobs=cfg.n_jobs)

        best = study.best_trial
        print("\n=== Best trial ===")
        print(f"  Win-share : {best.value:.4f}")
        print("  Params    :")
        for k, v in best.params.items():
            print(f"    {k}: {v!r}")

        print()
        print(strategy.render_snippet(cfg.strategy_name, best.params))

        return study


# ---------------------------------------------------------------------------
# Strategy registry — add new entries here
# ---------------------------------------------------------------------------

def _never_miss_factory(config: dict) -> Callable:
    """Build a 1-arg agent callable from a sampled hyperparameter dict.

    Constructs an ``AgentConfig`` once (not per call) so the closure is cheap.

    Args:
        config: Dict of param name -> sampled value, as produced by Optuna.

    Returns:
        A 1-arg ``obs -> moves`` callable ready for the environment.
    """
    agent_config = AgentConfig(
        source_sort          = config.get("source_sort",          "none"),
        target_mode          = config.get("target_mode",          "random_biased"),
        limit_top_k_fraction = config.get("limit_top_k_fraction", 0.5),
        bias_draws           = config.get("bias_draws",           2),
        accept_comet_src     = config.get("accept_comet_src",     True),
        cost_mode            = config.get("cost_mode",            "turns_weighted"),
        cost_weight          = config.get("cost_weight",          2.0),
        max_moves            = config.get("max_moves",            6),
    )

    def agent(obs):
        return run_agent(obs, agent_config)

    return agent


REGISTRY = StrategyRegistry()

REGISTRY.register(
    "never_miss",
    TunableStrategy(
        factory=_never_miss_factory,
        params=[
            CategoricalParam("source_sort",          ["none", "richest_first"]),
            # CategoricalParam("target_mode",          ["nearest", "roi", "random_biased"]),
            # FloatParam(      "limit_top_k_fraction", 0.1, 0.5),
            # IntParam(        "bias_draws",           1,   4),
            # CategoricalParam("accept_comet_src",     [True, False]),
            # CategoricalParam("cost_mode",            ["none", "turns_weighted"]),
            # FloatParam(      "cost_weight",          1.0, 3.0),
            # IntParam(        "max_moves",            1,   12),
        ],
        agent_fn_name="never_miss_agent",
        description="generalized agent: source_sort, target_mode, bias, cost_mode, max_moves",
    ),
)

def _stateful_factory(config: dict) -> Callable:
    """Build a fresh StatefulAgent instance from a sampled hyperparameter dict."""
    agent_config = StatefulAgentConfig(
        score_dist_base     = config.get("score_dist_base",     100.0),
        score_prod_mult     = config.get("score_prod_mult",      15.0),
        score_enemy_bonus   = config.get("score_enemy_bonus",    10.0),
        score_ships_penalty = config.get("score_ships_penalty",   0.7),
        score_eta_penalty   = config.get("score_eta_penalty",     2.0),
        min_ships_attack    = config.get("min_ships_attack",       10),
        top_k_targets       = config.get("top_k_targets",          3),
    )
    return make_stateful_agent(agent_config)


REGISTRY.register(
    "stateful",
    TunableStrategy(
        factory=_stateful_factory,
        params=[
            FloatParam("score_dist_base",      70.0, 130.0),
            FloatParam("score_prod_mult",        8.0,  25.0),
            FloatParam("score_enemy_bonus",      4.0,  18.0),
            FloatParam("score_ships_penalty",    0.3,   1.2),
            FloatParam("score_eta_penalty",      0.5,   4.0),
            IntParam(  "min_ships_attack",         5,    18),
            IntParam(  "top_k_targets",            2,     6),
        ],
        agent_fn_name="stateful_agent",
        description="stateful: score weights, min_ships_attack, top_k_targets",
    ),
)

# ── Template for a future agent ──────────────────────────────────────────────
# REGISTRY.register(
#     "my_agent",
#     TunableStrategy(
#         factory=lambda config: (lambda obs: my_agent(obs, config)),
#         params=[
#             FloatParam("alpha",  0.0, 1.0),
#             IntParam(  "depth",  1,   5),
#             CategoricalParam("mode", ["greedy", "cautious"]),
#         ],
#         agent_fn_name="my_agent",
#         description="my_agent: alpha, depth, mode",
#     ),
# )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Bayesian hyperparameter tuner for Orbit Wars agents.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # ── list ─────────────────────────────────────────────────────────────────
    sub.add_parser("list", help="List registered tunable strategies.")

    # ── tune ─────────────────────────────────────────────────────────────────
    t = sub.add_parser("tune", help="Run a Bayesian search for a strategy.")
    t.add_argument(
        "--strategy", required=True, choices=REGISTRY.names(),
        help="Name of the strategy to tune.",
    )
    t.add_argument(
        "--n-trials", type=int, default=60,
        help="Total Optuna trials (default 60).",
    )
    t.add_argument(
        "--matches-per-trial", type=int, default=20,
        help="Match pairs per opponent per trial (default 20).",
    )
    t.add_argument(
        "--opponents", nargs="+", default=["sniper"],
        choices=sorted(OPPONENT_POOL),
        help="Opponents to evaluate against (default: random sniper).",
    )
    t.add_argument(
        "--base-seed", type=int, default=5000,
        help="Starting seed for match generation (default 5000).",
    )
    t.add_argument(
        "--storage",
        default=DEFAULT_OPTUNA_STORAGE,
        help=(
            "Optuna storage URI.  Defaults to project's results/tune.db "
            "(absolute path) so runs are resumable.  Pass 'none' for a "
            "throw-away in-memory study."
        ),
    )
    t.add_argument(
        "--study-name", default=None,
        help="Study name (default: <strategy>_tuning).",
    )
    t.add_argument(
        "--n-jobs", type=int, default=1,
        help="Parallel trial workers; requires a persistent --storage (default 1).",
    )
    t.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-trial Optuna log lines.",
    )
    return p


def _cmd_list() -> int:
    print("Tunable strategies:")
    for name, strat in REGISTRY.items():
        print(f"  {name:<16} {strat.description}")
        for p in strat.params:
            if isinstance(p, FloatParam):
                spec = f"float [{p.low}, {p.high}]" + (" log" if p.log else "")
            elif isinstance(p, IntParam):
                spec = f"int   [{p.low}, {p.high}]"
            else:
                spec = f"cat   {p.choices}"
            print(f"    {p.name:<22} {spec}")
    return 0


def _cmd_tune(args: argparse.Namespace) -> int:
    storage = None if args.storage.lower() == "none" else args.storage
    cfg = StudyConfig(
        strategy_name=args.strategy,
        n_trials=args.n_trials,
        opponents=args.opponents,
        matches_per_trial=args.matches_per_trial,
        base_seed=args.base_seed,
        storage=storage,
        study_name=args.study_name,
        n_jobs=args.n_jobs,
        quiet=args.quiet,
    )
    StudyRunner(REGISTRY).run(cfg)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.cmd == "list":
        return _cmd_list()
    if args.cmd == "tune":
        return _cmd_tune(args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
