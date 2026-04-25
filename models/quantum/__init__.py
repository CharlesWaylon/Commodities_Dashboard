"""
Quantum models package.

Original (Tier 1 foundation):
  embedding        — RY/RZ angle encoding into quantum state
  kernel           — fidelity kernel K(x1,x2) = |⟨ψ(x1)|ψ(x2)⟩|²
  hybrid           — kernel ridge regression with quantum Gram matrix

Tier-4 expansion:
  kernel_benchmark — 4/6/8-qubit × RY-RZ/ZZFeatureMap/IQP IC comparison
  qaoa_portfolio   — QUBO commodity basket selection via QAOA
  qnn_hybrid       — parameterized quantum circuit as a PyTorch layer
"""
