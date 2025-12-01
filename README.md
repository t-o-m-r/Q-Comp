## ROUGH NOTES ON CODE (TBC): 
colourbar_2 plots the probability of the system being in a Bell state over time, as it is driven into the bell state by a cubic detuning sweep that passes through the resonance at 0.5 microseconds. 

Can toggle 'cubic' to 'constant' (boring), 'linear' (poor at achieving entanglement) or 'adiabatic' (better at producing entanglement) [4]

N.B. 'adiabatic' is called 'optimal' in bell_state_comparison.png plot, but this sweep is by no means optimal in reality - this is just a sloppy naming convention that I've used to 
indicate that, given the set of detuning sweeps 
{'constant', 'linear', 'cubic', 'adiabatic/optimal'}, 
the adiabatic/optimal one produces the maximally entangled Bell state (i) fastest and (ii) with the highest fidelity. 

Van der Waals interaction potential in range 0 to 100 MHz shows that (predictably) interaction between the qubits is vital for a global laser pulse to produce entanglement between the qubits. 

test_colourbar_2 contains unit tests (pytest) for colourbar_2, which will be useful when I extend the code to model more complicated systems. 
Some extension ideas include modelling a CZ gate under decoherence (almost complete as of 1st Dec) and working with Dr Franzen/ Dr Gallagher on Yb atoms/ cuprous oxide.

## Background

A classical computer uses bits, which can be in state 0 or 1.
Quantum computers use qubits, which are superpositions of two states of a physical system — here, we use the energy levels of atoms.

To make logic gates and correct errors in a quantum computer, we need qubits to interact so that they become entangled.
We can mediate this interaction via van der Waals (VdW) forces by exciting atoms to a Rydberg state |r> (where the electron is far from the nucleus) using a laser pulse [1].

## Aim

To simulate the time evolution of a two-qubit system into the entangled Bell state |\psi^+\rangle, under varying VdW interaction strengths and different laser pulse profiles.

## Hamiltonian [2]

The system evolves under the time-dependent Hamiltonian:

$$
\hat{H}(t) = \frac{\hbar}{2}
\begin{pmatrix}
2\Delta & \Omega e^{-i\phi_L} & \Omega e^{-i\phi_L} & 0 \\
\Omega e^{i\phi_L} & 0 & 0 & \Omega e^{-i\phi_L} \\
\Omega e^{i\phi_L} & 0 & 0 & \Omega e^{-i\phi_L} \\
0 & \Omega e^{i\phi_L} & \Omega e^{i\phi_L} & -2\Delta + 2V
\end{pmatrix}
$$

## Methodology uses [3]

## Referances
[1] Nielsen, M.A. & Chuang, I.L. (2010). Quantum Computation and Quantum Information. Cambridge University Press.
[2] Department of Physics, Durham University. (2025). PHYS3561 Computing Project Booklet 2025/26, pp. 21–29.
[3] Suzuki, M. (1990). Fractal decomposition of exponential operators with applications to many-body theories and Monte Carlo simulations. Physics Letters A, 146(6–7), 319–323.
[4] Saffman, M., Jones, L.A. & Adams, C.S. (2020). Symmetric Rydberg controlled-Z gates with adiabatic pulses. Physical Review A, 101, 062309.

