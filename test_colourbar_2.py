import numpy as np
import numpy.linalg as la
import pytest
import matplotlib


from colourbar_2 import (
    initial_state,
    make_delta,
    make_constant,
    H_t,
    ts_step,
    evolve_time_dependent,
    MHz, 
    hbar
)


def test_initial_state_normalises_vector():
    psi_0 = initial_state(1, 0, 0, 0)
    psi_random = initial_state(1+9j, 2+1j, 3+20j, 4)
    assert psi_0.shape == (4,)
    assert psi_random.shape == (4,)
    assert np.isclose(la.norm(psi_0), 1.0)
    assert np.isclose(la.norm(psi_random), 1.0)


def test_initial_state_raises_on_zero_vector():
    with pytest.raises(ValueError):
        initial_state(0, 0, 0, 0)


def test_make_constant_returns_same_value_for_any_t():
    f = make_constant(3.14)
    assert f(0.0) == pytest.approx(3.14)
    assert f(1.23e-9) == pytest.approx(3.14)


def test_make_delta_constant_is_zero_everywhere(): # must be on resonance at t = t_max / 2 (so at all times)
    delta_fn = make_delta("constant", delta_max=50 * MHz, t_max=1e-6, t_array=np.linspace(0, 1e-6, 100), omega_fn=make_constant(1.0 * MHz))
    for t in [0.0, 0.3e-6, 1.0e-6]:
        assert delta_fn(t) == pytest.approx(0.0)


def test_make_delta_linear_is_zero_at_midpoint():  # must be on resonance at t = t_max / 2
    delta_max = 50 * MHz
    t_max = 1e-6
    delta_fn = make_delta("linear", delta_max=delta_max, t_max=t_max, t_array=np.linspace(0, t_max, 100), omega_fn=make_constant(1.0 * MHz))
    assert delta_fn(t_max / 2) == pytest.approx(0.0)


def test_make_delta_cubic_is_zero_at_midpoint():  # must be on resonance at t = t_max / 2
    delta_max = 50 * MHz
    t_max = 1e-6
    delta_fn = make_delta("cubic", delta_max=delta_max, t_max=t_max, t_array=np.linspace(0, t_max, 100), omega_fn=make_constant(1.0 * MHz))
    assert delta_fn(t_max / 2) == pytest.approx(0.0)


def test_make_delta_adiabatic_raises_if_omega_not_constant():
    delta_max = 50 * MHz
    t_max = 1e-6
    t_array = np.linspace(0.0, t_max, 100)


    def omega_time_dependent(t):
        return (1.0 + 1.0 * (t / t_max)) * MHz

    delta_fn = make_delta(
        "adiabatic",
        delta_max=delta_max,
        t_max=t_max,
        t_array=t_array,
        omega_fn=omega_time_dependent,
    )

    with pytest.raises(ValueError):
        delta_fn(t_max / 2) # try evaluating the function to trigger the check


def test_H_t_returns_4x4_hermitian_matrix():
    # simple constant parameter functions
    delta_fn = make_constant(1.0 * MHz)
    omega_fn = make_constant(2.0 * MHz)
    phiL_fn = make_constant(0.0)
    V_fn = make_constant(3.0 * MHz)

    Ht = H_t(delta_fn, omega_fn, phiL_fn, V_fn)
    H = Ht(0.0)

    assert H.shape == (4, 4)
    assert H.dtype == complex
    assert np.allclose(H, H.conj().T)


def test_ts_step_is_unitary():
    """Reconstruct U from ts_step and check U†U = I for a constant H."""
    H_const =  (hbar / 2) * MHz * np.array([
            [ 2*50,    1,   1,       0      ],
            [ 1,  0,       0,           1  ],
            [ 1,  0,       0,           1 ],
            [ 0,      1,   1,  -2*50 + 2*40  ]
        ], dtype=complex)


    def H_fn(t):
        return H_const

    dt = 1e-7
    basis = np.eye(4, dtype=complex)
    U_cols = []
    for j in range(4):
        psi0 = basis[:, j]
        psi1 = ts_step(psi0, H_fn, t=0.0, dt=dt)
        U_cols.append(psi1)

    U = np.column_stack(U_cols)
    identity = np.eye(4, dtype=complex)
    assert np.allclose(U.conj().T @ U, identity, atol=1e-10)


def test_eigenvector_matrix_R_is_unitary():
    """Check that the eigenvector matrix R is unitary (columns orthonormal).""" 
    delta_fn = make_constant(0.1 * MHz)
    omega_fn = make_constant(2.0 * MHz)
    phiL_fn = make_constant(1.0)
    V_fn = make_constant(50.0 * MHz)

    Ht = H_t(delta_fn, omega_fn, phiL_fn, V_fn)
    H = Ht(0.0)
    evals, R = la.eig(H)
    id_maybe = R.conj().T @ R

    assert np.allclose(id_maybe, np.eye(4), atol=1e-10)
    assert not np.isclose(la.det(R), 0.0)


def test_evolve_time_dependent_returns_array_of_correct_shape_and_norm():
    psi0 = initial_state(1+1j, 4, 0.005j, 10+10j)

    def H_zero(t):
        return np.zeros((4, 4), dtype=complex)

    t_grid = np.linspace(0.0, 1e-6, 10)
    psi_series = evolve_time_dependent(psi0, H_zero, t_grid)

    assert psi_series.shape == (len(t_grid), 4)

    norms = [la.norm(psi_series[i]) for i in range(len(t_grid))]
    assert np.allclose(norms, np.ones_like(norms), atol=1e-10)

