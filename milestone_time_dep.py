import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

hbar = 6.63e-34 / (2 * np.pi)

# Initial state ------------------------------------------------
def initial_state(a00, a0r, ar0, arr):
    psi_0 = np.asarray([a00, a0r, ar0, arr], dtype=complex)
    nrm = la.norm(psi_0)
    if nrm == 0:
        raise ValueError("Initial state has zero norm.")
    return psi_0 / nrm

# time dep detuning = delta --------------------------------------
def make_delta(behaviour, sweep_range = 1.0):
    """Return a callable delta(t) in rad/s.
    We need to sweep delta through the resonance at delta=0."""
    MHz = 2 * np.pi * 1e6

    def delta_const(t):  return 0.0 * MHz
    def delta_linear(t): return sweep_range / (2 * np.pi) * ((t / 1e-6)-0.5) *  MHz   # sweep from - sweep_range/2 MHz to +sweep_range/2 MHz over 1 microsecond, linear 
    def delta_cubic(t):  return 2 * sweep_range / (2 * np.pi) * 8 * ((t / 1e-6)-0.5)**3 * MHz # sweep from -sweep_range/2 MHz to +sweep_range/2 MHz over 1 microsecond, cubic 

    behaviours = {
        'constant': delta_const, 
        'linear':   delta_linear,
        'cubic':    delta_cubic,
    }
    return behaviours[behaviour]

def make_constant(x):
    """Wrap a constant into a time-function. Ignore t and always return x."""
    return (lambda t: x)

# Hamiltonian ----------------------------------------------------
def H_t(delta_fn, omega_fn, phiL_fn, V_fn):
    """Return a callable H(t) using the supplied time-dependent parameter functions."""
    def H(t):
        d  = delta_fn(t) # detuning...... make all these variables of H functions of time
        w  = omega_fn(t) # Rabi frequency
        ph = phiL_fn(t) # laser (assume plane polarised EM wave) driving phase
        V  = V_fn(t) # interaction strength (Van der Waals)
        e_m = np.exp(-1j * ph)
        e_p = np.exp(+1j * ph)
        return (hbar / 2) * np.array([
            [ 2*d,    w*e_m,   w*e_m,       0      ],
            [ w*e_p,  0,       0,           w*e_m  ],
            [ w*e_p,  0,       0,           w*e_m  ],
            [ 0,      w*e_p,   w*e_p,  -2*d + 2*V  ]
        ], dtype=complex)
    return H

# single Trotter-Suzuki step (startpoint/ midpoint??) ------------------------
def ts_step(psi_t, Ht, t, dt, method='previous_point'):
    """
    psi(t+dt)> = R(t) D(t+dt) R^†(t) |psi(t)>
    previous_point: U = exp[-i H(t) dt / hbar].
    midpoint: U = exp[-i H(t+dt/2) dt / hbar].
    Implement via spectral decomposition of the 4x4 Hamiltonian......?????? do u understand this really
    """
    if method == 'midpoint':
        H_use = Ht(t+0.5*dt)
    
    elif method == 'previous_point':
        H_use = Ht(t)
    
    eigvals, eigvecs = la.eigh(H_use)
    phases = np.exp(-1j * eigvals * dt / hbar)
    U = eigvecs @ np.diag(phases) @ eigvecs.conj().T
    return U @ psi_t

def evolve_time_dependent(psi_0, Ht, t_grid):
    """iterate trotter step over time grid"""
    out = np.empty((len(t_grid), 4), dtype=complex) # initialise output array of psi's
    psi = psi_0.copy() # initialise initial state
    out[0] = psi # store initial state in output array
    for i in range(len(t_grid) - 1): 
        t = t_grid[i]
        dt = t_grid[1] - t_grid[0] #dt is fixed (uniform grid)
        psi = ts_step(psi, Ht, t, dt) # evolve state by one time step
        out[i+1] = psi
    return out

# plotting ---------------------------------------------------------
def plot_probability_amplitudes(psi_0, t_array, V_array, state, *, delta_fn, omega_fn, phiL_fn):
    
    idx = {'00': 0, '0r': 1, 'r0': 2, 'rr': 3}[state] # index the basis states for ease of access
    bell = (1/np.sqrt(2)) * np.array([0, 1, 1, 0], dtype=complex) # define the Bell state |Ψ+>

    fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True, sharey=False)

    for k, Vconst in enumerate(V_array):  # i feel like these Vconst are unneccessary as V counts builds in constant steps anyway (not a func of time)??
        V_fn = make_constant(Vconst)        # it is necessary bc what were doing is saying V is not a finc of time and then we plot we just plot for many ddifferent V bc were interested in the vdw effects
        Ht = H_t(delta_fn, omega_fn, phiL_fn, V_fn)
        psi_series = evolve_time_dependent(psi_0, Ht, t_array)          # shape (time period,4)?
        probs = np.abs(psi_series)**2 
        P_state = probs[:, idx]
        P_bell  = np.abs(psi_series @ bell.conj())**2 # probability of being in the bell state at each time (Born Rule)

        # do we really want this alpha scaling maybe its just building up not getting darker
        if np.isclose(Vconst, 0.0):
            color, alpha, lw = 'blue', 1.0, 2.0
        else:
            color = 'red'
            #alpha = 0.25 + 0.65 * (k / max(1, len(V_array)-1)) # prevents dividing by 0 if there is only one V value??
            alpha = 0.075
            lw = 1.0

        axs[0].plot(t_array, P_state, color=color, alpha=alpha, linewidth=lw)
        axs[1].plot(t_array, P_bell,  color=color, alpha=alpha, linewidth=lw)

    axs[0].set_ylabel('Probability') 
    axs[0].set_title(rf'Basis state |{state}⟩ probability, $\mathrm{{\Delta_{{sweep}}(t)=cubic}}$')
    axs[1].set_ylabel('Probability')
    axs[1].set_title(rf'Bell state |Ψ⁺⟩ probability, $\mathrm{{\Delta_{{sweep}}(t)=cubic}}$')
    axs[1].set_xlabel('Time (s)')
    plt.savefig('00_bell_cubic.png', dpi = 700)
    plt.show()

# if main ------------------------------------------
if __name__ == "__main__":
    # |00> initial state
    psi_0 = initial_state(1, 0, 0, 0)

    # units
    MHz = 2*np.pi*1e6

    # DELTA = CONSTANT JUST GIVES US THE TIME INDEPENDENT CASE?
    # choose parameter schedules (functions of t)
    delta_fn = make_delta('cubic', sweep_range=7)          # 'constant', 'linear', 'cubic'
    omega_fn = make_constant(1.0 * MHz)      # constant 1 MHz
    phiL_fn  = make_constant(0.0)            # constant phase

    # V family (red curves) + include 0 for the blue baseline
    V_array = np.linspace(0, 100, 1000) * MHz
    # time grid
    t_array = np.linspace(0, 1e-6, 500)

    plot_probability_amplitudes(
        psi_0, t_array, V_array, state='00',
        delta_fn=delta_fn, omega_fn=omega_fn, phiL_fn=phiL_fn
    )

# ISSUES
# not responding at akll strongly enough to V which should shift red curves to the left as V increases