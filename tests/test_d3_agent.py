import pytest

from ngsolve_agent.d3_agent import validate_and_canonicalize_driver


def test_d3_accepts_and_canonicalizes_supported_stokes_driver() -> None:
    source = (
        "from ngsolve_agent.generated_driver_api import run\n"
        "run(task_id='ngsolve_tutorial_stokes_flow_around_cylinder', maxh=0.05)\n"
    )

    canonical, task = validate_and_canonicalize_driver(
        source, "ngsolve_tutorial_stokes_flow_around_cylinder"
    )

    assert task.task_id == "ngsolve_tutorial_stokes_flow_around_cylinder"
    assert task.maxh == 0.05
    assert canonical.startswith("from ngsolve_agent.generated_driver_api import run\n")
    assert "maxh=0.05" in canonical


@pytest.mark.parametrize(
    "source",
    [
        "import os\nfrom ngsolve_agent.generated_driver_api import run\nrun(task_id='poisson_unit_square_smooth')",
        "from ngsolve_agent.generated_driver_api import run\nrun(task_id='poisson_unit_square_smooth'); open('x','w')",
        "from ngsolve_agent.generated_driver_api import run\nrun(task_id='poisson_unit_square_smooth', maxh=__import__('os').getcwd())",
        "from ngsolve_agent.generated_driver_api import run\nrun(**{'task_id': 'poisson_unit_square_smooth'})",
    ],
)
def test_d3_rejects_driver_code_outside_the_narrow_contract(source: str) -> None:
    with pytest.raises(ValueError):
        validate_and_canonicalize_driver(source, "poisson_unit_square_smooth")


def test_d3_rejects_task_id_mismatch() -> None:
    source = (
        "from ngsolve_agent.generated_driver_api import run\n"
        "run(task_id='poisson_unit_square_smooth')\n"
    )

    with pytest.raises(ValueError, match="does not match"):
        validate_and_canonicalize_driver(
            source, "ngsolve_tutorial_stokes_flow_around_cylinder"
        )


def test_d3_rejects_oversized_ast_before_parameter_validation() -> None:
    nested_literal = "[" * 70 + "0" + "]" * 70
    source = (
        "from ngsolve_agent.generated_driver_api import run\n"
        "run(task_id='poisson_unit_square_smooth', maxh="
        + nested_literal
        + ")\n"
    )

    with pytest.raises(ValueError, match="128-node AST limit"):
        validate_and_canonicalize_driver(source, "poisson_unit_square_smooth")


def test_d3_rejects_resource_exhaustion_mesh_size() -> None:
    source = (
        "from ngsolve_agent.generated_driver_api import run\n"
        "run(task_id='poisson_unit_square_smooth', maxh=1e-20)\n"
    )

    with pytest.raises(ValueError):
        validate_and_canonicalize_driver(source, "poisson_unit_square_smooth")
