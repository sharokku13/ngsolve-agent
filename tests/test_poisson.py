from ngsolve_agent.fem_solver import solve_poisson
from ngsolve_agent.schemas import PoissonTask
from ngsolve_agent.validation import validate_poisson_result


def test_manufactured_poisson_baseline_passes(tmp_path) -> None:
    task = PoissonTask(maxh=0.2, order=2)

    metrics, plot_path = solve_poisson(task, tmp_path)

    assert metrics.l2_error < 2.0e-2
    assert metrics.h1_seminorm_error > 0.0
    assert metrics.free_dof_residual < 1e-8
    assert metrics.ndof > 0
    assert metrics.nelements > 0
    assert validate_poisson_result(task, metrics) == []
    assert plot_path.exists()
    assert plot_path.stat().st_size > 0
