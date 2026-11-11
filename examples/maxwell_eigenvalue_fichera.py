"""Maxwell curl-curl eigenproblem on the 3D Fichera corner from NGSolve docs."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import ngsolve
from netgen.occ import Box, OCCGeometry
from ngsolve import (
    BilinearForm,
    HCurl,
    IdentityMatrix,
    Mesh,
    Preconditioner,
    TaskManager,
    dx,
    solvers,
)


def run(
    output_dir: Path,
    maxh: float = 0.4,
    hp_levels: int = 2,
    hp_factor: float = 0.2,
    order: int = 3,
    num_eigenpairs: int = 12,
    max_iterations: int = 20,
) -> dict[str, object]:
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)

    # L-shaped Fichera domain and refinement strategy used in NGSolve's EVP tutorial.
    outer_cube = Box((-1, -1, -1), (1, 1, 1))
    removed_cube = Box((0, 0, 0), (2, 2, 2))
    removed_cube.edges.hpref = 1
    geometry = outer_cube - removed_cube
    mesh = Mesh(OCCGeometry(geometry).GenerateMesh(maxh=maxh))
    mesh.RefineHP(levels=hp_levels, factor=hp_factor)

    fes = HCurl(mesh, order=order)
    trial, test = fes.TnT()
    curl_form = BilinearForm(fes)
    curl_form += ngsolve.curl(trial) * ngsolve.curl(test) * dx
    mass_form = BilinearForm(fes)
    mass_form += trial * test * dx
    preconditioner_form = BilinearForm(fes)
    preconditioner_form += (
        ngsolve.curl(trial) * ngsolve.curl(test) + trial * test
    ) * dx
    preconditioner = Preconditioner(
        preconditioner_form, "direct", inverse="sparsecholesky"
    )

    with TaskManager():
        curl_form.Assemble()
        mass_form.Assemble()
        preconditioner_form.Assemble()
        preconditioner.Update()

        # Project out grad(H1), the large zero-eigenvalue kernel of curl-curl.
        gradient_matrix, _scalar_space = fes.CreateGradient()
        gradient_transpose = gradient_matrix.CreateTranspose()
        scalar_laplacian = gradient_transpose @ mass_form.mat @ gradient_matrix
        scalar_laplacian[0, 0] += 1  # fix constant-potential kernel
        scalar_inverse = scalar_laplacian.Inverse(inverse="sparsecholesky")
        gradient_projector = (
            IdentityMatrix()
            - gradient_matrix
            @ scalar_inverse
            @ gradient_transpose
            @ mass_form.mat
        )
        projected_preconditioner = gradient_projector @ preconditioner.mat
        eigenvalues, _eigenvectors = solvers.PINVIT(
            curl_form.mat,
            mass_form.mat,
            pre=projected_preconditioner,
            num=num_eigenpairs,
            maxit=max_iterations,
        )

    values = [float(value) for value in eigenvalues]
    # Reference from the converged iteration table in the official documentation.
    reference_first = 3.22028387019447
    relative_error = abs(values[0] - reference_first) / reference_first
    report: dict[str, object] = {
        "task_id": "ngsolve_tutorial_maxwell_fichera_eigenproblem_3d",
        "source": "https://docu.ngsolve.org/latest/i-tutorials/unit-2.4-Maxwell/Maxwellevp.html",
        "formulation": "curl-curl generalized eigenproblem in H(curl), projected off grad(H1) kernel",
        "parameters": {
            "geometry": "Fichera corner: [-1,1]^3 minus [0,2]^3",
            "initial_maxh": maxh,
            "hp_refinement_levels": hp_levels,
            "hp_refinement_factor": hp_factor,
            "hcurl_order": order,
            "eigenpairs_requested": num_eigenpairs,
            "pinvit_max_iterations": max_iterations,
        },
        "results": {
            "eigenvalues": values,
            "reference_first_eigenvalue": reference_first,
            "first_eigenvalue_relative_difference_from_tutorial": relative_error,
            "ndof": fes.ndof,
            "elements": mesh.ne,
            "runtime_seconds": time.perf_counter() - started,
        },
        "environment": {
            "python": platform.python_version(),
            "ngsolve": getattr(ngsolve, "__version__", "unknown"),
        },
    }
    (output_dir / "maxwell_eigenvalue.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maxh", type=float, default=0.4)
    parser.add_argument("--hp-levels", type=int, default=2)
    parser.add_argument("--hp-factor", type=float, default=0.2)
    parser.add_argument("--order", type=int, default=3)
    parser.add_argument("--eigenpairs", type=int, default=12)
    parser.add_argument("--max-iterations", type=int, default=20)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/tutorials/maxwell_fichera")
    )
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.output_dir,
                maxh=args.maxh,
                hp_levels=args.hp_levels,
                hp_factor=args.hp_factor,
                order=args.order,
                num_eigenpairs=args.eigenpairs,
                max_iterations=args.max_iterations,
            ),
            indent=2,
        )
    )
