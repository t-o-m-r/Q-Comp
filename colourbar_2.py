"""

"""

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

hbar = 6.63e-34 / (2 * np.pi)
MHz = 2*np.pi*1e6

# Initial state ------------------------------------------------
def initial_state(a00, a0r, ar0, arr):
    """Define the initial state of the two-qubit rydberg system as a normalized complex vector."""
    psi_0 = np.asarray([a00, a0r, ar0, arr], dtype=complex)
    nrm = la.norm(psi_0)
    if nrm == 0:
        raise ValueError("Initial state has zero norm.")
    return psi_0 / nrm

# time-dependant detuning = delta -------------------------------------
def make_delta(behaviour, delta_max, t_max, t_array, omega_fn):
    """Return a callable delta(t) in rad/s.
    We need to sweep delta through the resonance at delta=0 to generate entanglement."""

    def delta_const(t):  return 0.0 * MHz
    def delta_linear(t): return delta_max * ((t - t_max/2) / t_max)  # sweep from - delta_max to +delta_max over 1 microsecond, linear
    def delta_cubic(t): return 2**3* delta_max * ((t - t_max/2)/ t_max )**3 # sweep from -delta_max to +delta_max over 1 microsecond, cubic
    def delta_adiabatic(t): # see ref "Time-Optimal Two- and Three-Qubit Gates for Rydberg Atoms"
        N = t_array.size 
        Omega = omega_fn(t) # ASSUME CONSTANT 
        if not np.allclose(Omega, omega_fn(t_array[0])):
            raise ValueError("Omega assumed constant here!")
        times = np.linspace(0, t_max, N)
        delta = np.zeros_like(times)
        delta[0] = -delta_max
        for i in range(N - 1):
            gap = np.sqrt(Omega**2 + delta[i]**2)
            dDelta = (gap**2) * (t_max / N)
            delta[i + 1] = min(delta[i] + dDelta, delta_max)
        return np.interp(t, times, delta) # interpolate so that delta_adiabatic can be called continuously as a function of t


    behaviours = {
        'constant': delta_const, 
        'linear':   delta_linear,
        'cubic':    delta_cubic,
        'adiabatic': delta_adiabatic,
    }
    return behaviours[behaviour]

def make_constant(x):
    """Wrap a constant into a time-function. Ignore t and always return x."""
    return (lambda t: x)

# Hamiltonian ------------------------------------------------------
def H_t(delta_fn, omega_fn, phiL_fn, V_fn):
    """Return a callable H(t) using the supplied time-dependent parameter functions."""
    def H(t):
        d  = delta_fn(t) # detuning
        w  = omega_fn(t) # Rabi frequency
        ph = phiL_fn(t) # laser (assume plane polarised EM wave) driving phase
        V  = V_fn(t) # interaction strength (Van der Waals). Rydberg blockade creates large effective detuning for |rr>. 
        e_m = np.exp(-1j * ph)
        e_p = np.exp(+1j * ph)
        return (hbar / 2) * np.array([
            [ 2*d,    w*e_m,   w*e_m,       0      ],
            [ w*e_p,  0,       0,           w*e_m  ],
            [ w*e_p,  0,       0,           w*e_m  ],
            [ 0,      w*e_p,   w*e_p,  -2*d + 2*V  ]
        ], dtype=complex)
    return H

# Single Trotter-Suzuki step (startpoint/ midpoint) ------------------------
def ts_step(psi_t, Ht, t, dt, method='previous_point'):
    """
    psi(t+dt)> = R(t) D(t+dt) R^†(t) |psi(t)>
    previous_point: U = exp[-i H(t) dt / hbar].
    midpoint: U = exp[-i H(t+dt/2) dt / hbar].
    Implement via spectral decomposition of the 4x4 Hamiltonian.
    """
    if method == 'midpoint':
        H_use = Ht(t+0.5*dt)
    
    elif method == 'previous_point':
        H_use = Ht(t)
    
    eigvals, eigvecs = la.eigh(H_use)
    phases = np.exp(-1j * eigvals * dt / hbar)
    U = eigvecs @ np.diag(phases) @ eigvecs.conj().T
    return U @ psi_t

# Time evolution for a given initial state and Hamiltonian ------------------------
def evolve_time_dependent(psi_0, Ht, t_grid):
    """iterate trotter step over time grid"""
    out = np.empty((len(t_grid), 4), dtype=complex) # initialise output array of psi's
    psi = psi_0.copy() # initialise initial state
    out[0] = psi # store initial state in output array
    for i in range(len(t_grid) - 1): 
        t = t_grid[i]
        dt = t_grid[1] - t_grid[0] # dt is fixed (uniform grid)
        psi = ts_step(psi, Ht, t, dt) # evolve state by one time step
        out[i+1] = psi
    return out

# plotting ---------------------------------------------------------
def plot_probability_amplitudes(psi_0, t_array, V_array, state, *, delta_fn, omega_fn, phiL_fn, sweep, style='plasma', label_offset=None):
    """Plot probability amplitudes of a given basis state or Bell state over time for different VdW interaction strengths, V (MHz)."""
    
    idx = {'00': 0, '0r': 1, 'r0': 2, 'rr': 3}[state] # index the basis states for ease of access
    bell = (1/np.sqrt(2)) * np.array([0, 1, 1, 0], dtype=complex) # define the Bell state |Ψ⁺⟩

    fig, ax = plt.subplots(1, 1, figsize=(8,6 ))
    fig.patch.set_alpha(0.2)
    ax.set_facecolor((1, 1, 1, 0.2))
    
    N = len(V_array)
    cmap = cm.get_cmap('rainbow')  
    norm = mcolors.Normalize(vmin=0, vmax=100)  # normalize between 0 and 100 MHz

    for Vconst in V_array:  # loop over V values
        V_fn = make_constant(Vconst)        
        Ht = H_t(delta_fn, omega_fn, phiL_fn, V_fn)
        psi_series = evolve_time_dependent(psi_0, Ht, t_array)          
        probs = np.abs(psi_series)**2 
        P_state = probs[:, idx]
        P_bell  = np.abs(psi_series @ bell.conj())**2 # probability of being in the bell state at each time step (Born Rule)

        color = cmap(norm(Vconst / MHz))
        alpha = 0.9
        if Vconst == 100*MHz:
            lw = 2.5
        else:           
            lw = 1.0

        ax.plot(t_array/1e-6, P_bell,  color=color, alpha=alpha, linewidth=lw)
        #ax.plot(t_array/1e-6, P_state, color=color, alpha=alpha, linewidth=lw) #toggle between plotting bell state prob or basis state prob

    ax.set_title(rf'Bell state |Ψ⁺⟩ probability, $\mathrm{{\Delta_{{sweep}}(t)={sweep}}}$')
    ax.set_xlabel('Time (µs)')
    ax.set_ylabel('Probability')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation='vertical', fraction=0.05, pad=0.04)
    cbar.set_label('V (MHz)')
    label_offset = 1.5
    cbar.ax.yaxis.set_label_coords(label_offset, 0.5)

    cbar.set_ticks(np.linspace(0, 100, 6))
    cbar.set_ticklabels([f'{int(x)}' for x in np.linspace(0, 100, 6)])
    plt.tight_layout(rect=[0, 0, 0.95, 1])
    #plt.savefig('bell_state_colourbar.png', dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.show()

# if main ------------------------------------------
if __name__ == "__main__":
    psi_0 = initial_state(1, 0, 0, 0)  # |00> initial state is ground state for both qubits
    
    t_max = 1e-6
    V_array = np.linspace(0, 100, 50) * MHz  # V values from 0 to 100 MHz
    t_array = np.linspace(0, t_max, 500) # time grid from 0 to t_max
    delta_max = 50 * MHz # sweep from -delta_max to + delta_max
    sweep = 'cubic'
    omega_fn = make_constant(1.0 * MHz) # constant 1 MHz
    delta_fn = make_delta(sweep, delta_max=delta_max, t_max=t_max, t_array=t_array, omega_fn=omega_fn) # 'constant', 'linear', 'cubic'
    phiL_fn  = make_constant(0.0) # constant phase can be set to 0 for simplicity


    plot_probability_amplitudes(
        psi_0, t_array, V_array, state='00',
        delta_fn=delta_fn, omega_fn=omega_fn, phiL_fn=phiL_fn, sweep=sweep
    )
