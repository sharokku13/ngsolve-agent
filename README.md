# NGSolve Agent

An experimental benchmark for reliable agentic finite element workflows in NGSolve.

The first milestone is **B0**, a deterministic Poisson solver with an independently known
manufactured solution. Agent architectures (D1, D4, and later D3) will use the same task
definitions, numerical checks, and run records.

Development decisions, run results, and next steps are tracked in
[`DEVELOPMENT_LOG.md`](DEVELOPMENT_LOG.md).

## Current baseline

Solve on the unit square

\[-\Delta u = 2\pi^2\sin(\pi x)\sin(\pi y), \qquad u|_{\partial\Omega}=0,\]

whose exact solution is `u(x,y) = sin(pi*x) sin(pi*y)`. The runner records L2 and H1-seminorm
errors, the free-DOF residual, solver status, runtime, mesh size, and degrees of freedom. It writes
a timestamped JSON report and a PNG comparing the numerical solution, exact solution, and
pointwise error. Reports are kept under `results/reports/`; plots are kept under
`results/figures/`, so later runs do not overwrite earlier results.

## First nontrivial NGSolve benchmark

`examples/nonlinear_elasticity.py` reproduces the official NGSolve 2D Neo-Hookean
cantilever tutorial: one edge is clamped, gravity is ramped from 0.1 to 5.0 in 50
load steps, and each step is solved with Newton linearization. It writes solver
metrics to JSON and displacement plots to the selected output folder. This problem
has no closed-form reference solution, so its checks are nonlinear convergence,
free-DOF equilibrium residual, and mesh-refinement comparison; it is not yet an
LLM-agent result.

```powershell
uv run python examples/nonlinear_elasticity.py
uv run python examples/nonlinear_elasticity.py --maxh 0.025 --output-dir results/tutorials/neo_hookean_cantilever_refined
```

The first two mesh runs (`maxh=0.05` and `0.025`) converged at the full tutorial
load. The L2 displacement norm changed from 0.185138 to 0.185219 (about 0.044%),
and the final energy from 8.599974 to 8.599914. These are initial discretization
checks, not a proof of full mesh independence. The task is sourced from the
[NGSolve nonlinear minimization tutorial](https://docu.ngsolve.org/latest/i-tutorials/unit-3.4-nonlmin/nonlmin.html).

## Mixed-flow benchmark

`examples/stokes_cylinder.py` reproduces the official Taylor-Hood Stokes example:
steady flow through a 2D channel around a cylinder, with parabolic inlet velocity,
no-slip walls/cylinder, and an open outlet. It checks the algebraic residual,
inlet/outlet mass balance, and the L2 divergence, then stores velocity and pressure
plots plus a JSON record.

```powershell
uv run python examples/stokes_cylinder.py
uv run python examples/stokes_cylinder.py --maxh 0.025 --output-dir results/tutorials/stokes_cylinder_refined
```

The two runs conserve inlet/outlet flux to roundoff (`0.41` on both meshes), and
the free-DOF residual is below `2e-14`. The L2 divergence decreases from `0.3157`
to `0.2293` when `maxh` is halved. It is not zero because Taylor-Hood enforces
incompressibility weakly; the refinement trend needs another mesh level before we
claim a convergence rate. This is the [official Stokes tutorial](https://docu.ngsolve.org/latest/i-tutorials/unit-2.6-stokes/stokes.html).

## Maxwell eigenvalue benchmark

`examples/maxwell_eigenvalue_fichera.py` reproduces the 3D Fichera-corner
curl-curl eigenproblem from the NGSolve Maxwell tutorial. It uses an `H(curl)`
space, projects out the gradient kernel, and computes 12 eigenpairs with PINVIT.
SciPy is a runtime dependency of NGSolve's PINVIT routine and is included in the
locked environment. The JSON report records the spectrum and its difference from
the tutorial's converged first eigenvalue.

```powershell
uv run python examples/maxwell_eigenvalue_fichera.py
```

The run completed with 42,103 degrees of freedom and 747 elements. The first
eigenvalue was `3.2202713671`, about `3.9e-6` relative difference from the
tutorial's converged value `3.2202838702`. The result is recorded in
`results/tutorials/maxwell_fichera/maxwell_eigenvalue.json`. The problem and
kernel projection follow the [official Maxwell eigenvalue tutorial](https://docu.ngsolve.org/latest/i-tutorials/unit-2.4-Maxwell/Maxwellevp.html),
which builds on the high-order Maxwell FEM work listed in the project literature.

## Setup (Windows PowerShell)

```powershell
uv python install 3.12
uv sync
uv run python -m ngsolve_agent --output-dir results
uv run python examples/run_benchmark_suite.py
uv run pytest
```

The combined deterministic benchmark run writes a timestamped `suite.json` and per-case
reports/figures under `results/benchmark_runs/`.

`uv.lock` pins the tested dependency set so the experiment can be recreated. Keep the solver
baseline free of LLM and workflow dependencies beyond the packages needed for the numerical task.

## Architecture comparison

- `B0`: hand-written deterministic NGSolve baseline (implemented first).
- `D1`: LLM selects among the same narrow, typed FEM tools.
- `D4`: deterministic LangGraph workflow controls ordering, validation, and bounded recovery.
- `D3`: LLM-generated Python driver for one allowlisted task; AST-checked and rebuilt
  before a subprocess dispatches to the same NGSolve solver/evaluator.
- `D2`: optional, only if one declarative configuration can express tasks without becoming brittle.

The primary experiment compares B0, D1, D3, and D4 through the same independent evaluator. D1 vs
D4 isolates orchestration; D3 tests constrained driver generation. The current D3 is a narrow
prototype, not free-form solver-code generation. D2 is not part of the first implementation.

The first D4 workflow is provider-neutral: it accepts a task-interpreter callable, so model
credentials and vendor-specific integrations remain outside the deterministic FEM/evaluation core.
The D1 loop exposes the same task schema and solver/evaluator operations, while the model selects
the tool sequence. Both interfaces currently accept provider callbacks rather than choosing an LLM
vendor or storing credentials in the repository.

## Real LLM adapters (D1/D3/D4 Poisson and Stokes pilot)

`src/ngsolve_agent/llm_adapter.py` now connects these callbacks to either Ollama's local chat API
or an OpenAI-compatible JSON chat endpoint. `agent_cli.py` runs one architecture and saves the
request, model/provider name, tool/graph trace, numerical result, LLM call timings, and report path
under `results/agent_runs/`. A deterministic task registry dispatches only validated Poisson or
the fixed NGSolve Stokes-cylinder tutorial tasks to their corresponding solver/evaluator. D3
accepts only a two-statement driver calling the fixed API; it validates the Python AST, rebuilds
the source from typed parameters, then runs that canonical file in an isolated Python process.
Model-authored statements are never executed directly. Python `-I` is interpreter isolation, not
an OS security sandbox; this prototype relies on the AST allowlist, canonicalization, bounded task
parameters, and the trusted solver API, and does not protect against defects inside NGSolve or the
fixed solver code.

Configure Ollama in PowerShell after it is installed, running, and has a model downloaded:

```powershell
$env:NGSOLVE_LLM_PROVIDER = "ollama"
$env:NGSOLVE_LLM_BASE_URL = "http://127.0.0.1:11434"
$env:NGSOLVE_LLM_MODEL = "<installed-model-name>"
uv run python -m ngsolve_agent.agent_cli --architecture d4 --request "Solve the supported unit-square smooth Poisson problem with mesh size 0.1 and quadratic elements."
```

For an OpenAI-compatible service, set `NGSOLVE_LLM_PROVIDER=openai-compatible`, its base URL,
`NGSOLVE_LLM_MODEL`, and `NGSOLVE_LLM_API_KEY` in the process environment. Never put the key in
source control or paste it into the project log. Supported agent tasks are the manufactured
unit-square Poisson problem and the fixed steady Stokes channel/cylinder tutorial. General
natural-language setup for other Stokes geometries, elasticity, and Maxwell has not been
implemented.

`data/agent_task_cases.json` contains Poisson P1–P6 and Stokes S1–S2 prompts, including two
paraphrased robustness cases. The comparison runner supports D1, D3, and D4 on identical prompts and scores exact task-parameter extraction
separately from the numerical pass status:

```powershell
uv run python examples/evaluate_agent_architectures.py --case P1
uv run python examples/evaluate_agent_architectures.py
uv run python examples/evaluate_agent_architectures.py --case P4 --repeats 3
uv run python examples/evaluate_agent_architectures.py --case P4 --architectures d1 d3 d4
uv run python examples/evaluate_agent_architectures.py --case S1 --architectures d1 d3 d4
```

The first command is the inexpensive integration check before running all ten architecture/case
combinations. Use `--repeats 2..10` for repeated runs; the summary includes pass rate, exact
parameter-match rate, mean call count, and mean/median LLM and end-to-end agent time for each
architecture. Results are written under `results/agent_benchmarks/`. The runner has a conservative scope guard that
requires an explicit supported Poisson or Stokes-cylinder request and rejects common unsupported
PDE names before a model or solver call. This pilot evaluates task interpretation and orchestration
for two fixed PDE tasks; it does not claim general FEM problem formulation. Each run records both
LLM time and end-to-end agent time, including driver validation, subprocess startup, solver, and
evaluation.

Evaluator limits are fixed in code and are not task fields the model can choose: Poisson L2 error
must be at most `2e-2`; Stokes relative inlet/outlet flux imbalance must be at most `1e-8`, and
the free-DOF residual at most `1e-10`. Stokes divergence is reported as a diagnostic, not a
pass/fail threshold: this tutorial's Taylor-Hood result has no manufactured exact solution here,
so a single arbitrary divergence cutoff would not establish accuracy. These are pilot acceptance
checks, not a substitute for a refinement study or an independent Stokes reference solution.
