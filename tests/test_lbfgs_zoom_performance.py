"""Test to demonstrate LBFGSZoom outperforms LBFGS with BacktrackingArmijo.

This test validates that the Zoom linesearch implementation works correctly and
provides superior performance compared to the default BacktrackingArmijo linesearch
when used with LBFGS on challenging non-convex optimization problems.
"""

import jax.numpy as jnp
import optimistix as optx
import pytest

from .helpers import _himmelblau, beale, LBFGSZoom, rosenbrock, tree_allclose


@pytest.mark.parametrize("fn_name", ["rosenbrock", "himmelblau", "beale"])
def test_lbfgs_zoom_vs_backtracking(fn_name):
    """Demonstrate LBFGSZoom outperforms standard LBFGS with BacktrackingArmijo.

    This test compares the performance of LBFGS with two different linesearches:
    1. BacktrackingArmijo (default) - only satisfies Armijo decrease condition
    2. Zoom - satisfies strong Wolfe conditions (decrease + curvature)

    The Zoom linesearch should take fewer accepted steps due to better stepsize
    selection and stronger convergence guarantees.
    """

    # Test function configurations
    test_configs = {
        "rosenbrock": {
            "fn": lambda tree, args: jnp.sum(
                rosenbrock(tree, args)[0] ** 2 + rosenbrock(tree, args)[1] ** 2
            ),
            "init": [jnp.array(1.5), jnp.array(1.5)],
            "args": jnp.array(1.0),
            "expected_min": jnp.array(0.0),
        },
        "himmelblau": {
            "fn": _himmelblau,
            "init": [jnp.array(2.0), jnp.array(2.5)],
            "args": (jnp.array(11.0), jnp.array(7.0)),
            "expected_min": jnp.array(0.0),
        },
        "beale": {
            "fn": beale,
            "init": [jnp.array(2.0), jnp.array(0.0)],
            "args": (jnp.array(1.5), jnp.array(2.25), jnp.array(2.625)),
            "expected_min": jnp.array(0.0),
        },
    }

    config = test_configs[fn_name]
    rtol = atol = 1e-8
    max_steps = 10_000

    # Create solvers
    solver_armijo = optx.LBFGS(rtol=rtol, atol=atol, use_inverse=True , search=optx.BacktrackingArmijo())
    solver_zoom = LBFGSZoom(rtol=rtol, atol=atol, use_inverse=True)

    # Run optimizations
    sol_armijo = optx.minimise(
        config["fn"],
        solver_armijo,
        config["init"],
        args=config["args"],
        max_steps=max_steps,
        throw=False,
    )

    sol_zoom = optx.minimise(
        config["fn"],
        solver_zoom,
        config["init"],
        args=config["args"],
        max_steps=max_steps,
        throw=False,
    )

    # Both should successfully converge
    assert sol_armijo.result == optx.RESULTS.successful, (
        f"Armijo failed: {sol_armijo.result}"
    )
    assert sol_zoom.result == optx.RESULTS.successful, f"Zoom failed: {sol_zoom.result}"

    # Both should reach the minimum
    final_armijo = config["fn"](sol_armijo.value, config["args"])
    final_zoom = config["fn"](sol_zoom.value, config["args"])
    assert tree_allclose(final_armijo, config["expected_min"], atol=1e-4, rtol=1e-4)
    assert tree_allclose(final_zoom, config["expected_min"], atol=1e-4, rtol=1e-4)

    # KEY ASSERTION: Zoom should take fewer steps
    steps_armijo = sol_armijo.state.num_accepted_steps
    steps_zoom = sol_zoom.state.num_accepted_steps

    print(f"\n{'=' * 60}")
    print(f"Test function: {fn_name}")
    print(f"BacktrackingArmijo: {steps_armijo} accepted steps")
    print(f"Zoom:               {steps_zoom} accepted steps")
    improvement_pct = 100 * (steps_armijo - steps_zoom) / steps_armijo
    print(f"Improvement:        {improvement_pct:.1f}%")
    print(f"{'=' * 60}")

    # Zoom should be more efficient
    assert steps_zoom < steps_armijo, (
        f"Zoom should take fewer steps than Armijo. "
        f"Zoom: {steps_zoom}, Armijo: {steps_armijo}"
    )


def test_lbfgs_zoom_convergence():
    """Verify LBFGSZoom converges correctly on a simple quadratic.

    This is a sanity check to ensure the Zoom linesearch implementation
    works correctly on a trivial optimization problem.
    """

    def quadratic(y, args):
        """Simple quadratic: f(x,y) = (x-2)^2 + (y-3)^2"""
        del args
        return (y[0] - 2.0) ** 2 + (y[1] - 3.0) ** 2

    solver = LBFGSZoom(rtol=1e-8, atol=1e-8)
    init = [jnp.array(0.0), jnp.array(0.0)]

    sol = optx.minimise(quadratic, solver, init, args=None, max_steps=100, throw=False)

    assert sol.result == optx.RESULTS.successful
    assert jnp.allclose(sol.value[0], 2.0, atol=1e-6)
    assert jnp.allclose(sol.value[1], 3.0, atol=1e-6)
