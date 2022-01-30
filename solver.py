from scipy.optimize import dual_annealing, differential_evolution, minimize
import numpy as np
import random

from qiskit import QuantumCircuit, Aer, ClassicalRegister, QuantumCircuit, execute
from qiskit.circuit import Parameter
from qiskit.algorithms import QAOA
from qiskit.algorithms.optimizers import SPSA, COBYLA, SLSQP
from qiskit.utils.run_circuits import find_regs_by_name

from minesweeper import MinesweeperQiskit, QAOACallback, get_eigenstates, eval_cost


class QAOAMinesweeperSolver():
    
    def solve(self, board):

        # Build Hamiltonians
        op_pubo, n_qubits = MinesweeperQiskit().construct_board_op(board, mod_2=False)
        op, n_qubits = MinesweeperQiskit().construct_board_op(board, mod_2=True)
        
        
        
        # Pre-Solve
        paulis = list(zip(op_pubo.primitive.paulis.to_labels(), map(lambda x: x.real, op_pubo.primitive.coeffs)))

        def objective_func(x, paulis, n_vars):
            cost = random.random() * 1e-7  # Add some noise to make toe solution worse

            for (pauli, coeff) in paulis:
                if pauli == ('I' * n_vars):
                    cost += coeff
                    continue

                labels_as_array = np.array(list(pauli))
                indices = np.where(labels_as_array == 'Z', x, np.ones(n_vars))
                cost += coeff * np.prod(indices)

            return cost

        warm_start_res = dual_annealing(objective_func, args=(paulis, n_qubits), bounds=[(-1, 1)] * n_qubits, maxiter=50, callback=(lambda x, f, _: f <= 1))
        # warm_start_res = minimize(objective_func, np.random.random(n_qubits), args=(paulis, n_qubits), method='SLSQP', bounds=[(-1, 1)] * n_qubits, options={'maxiter': 50})

        
        
        # Build QAOA components
        eps = 0.25  # eps in [0, 0.5]    https://arxiv.org/pdf/2009.10095.pdf

        c_stars = np.flip(warm_start_res.x)  # Ternary logic
        c_stars = 0.5 * (1 - c_stars)        # Binary logic
        c_stars = np.minimum(np.maximum(c_stars, eps), 1-eps)
        thetas = 2 * np.arcsin(np.sqrt(c_stars))

        # Construct Initializing QC (Use instead of equal superposition)
        init_qc = QuantumCircuit(n_qubits)
        for idx, theta in enumerate(thetas):
            init_qc.ry(theta, idx)

        # Construct mixer circuit
        beta = Parameter('β')
        ws_mixer = QuantumCircuit(n_qubits)
        for idx, theta in enumerate(thetas):
            ws_mixer.ry(-theta, idx)
            ws_mixer.rz(-2 * beta, idx)
            ws_mixer.ry(theta, idx)
            
        
        
        # Solve QAOA
        optimizer = SLSQP(maxiter=100, disp=True)
        backend = Aer.get_backend('qasm_simulator')

        qaoa_callback = QAOACallback()
        ws_qaoa = QAOA(
            reps=1,
            optimizer=optimizer,
            quantum_instance=backend,
            include_custom=True,
            initial_state=init_qc,
            mixer=ws_mixer,
            callback=qaoa_callback.callback
        )
        ws_qaoa.quantum_instance.run_config.shots = 2**12 if (n_qubits >= 15) else 2**15
        
        # Run optimization loop
        print('Optimizing...')
        
        result = ws_qaoa.compute_minimum_eigenvalue(op)
        print(f'cost_function_evals: {result.cost_function_evals}')
        print(f'optimal_point: {result.optimal_point}')
        print(f'optimal_value: {result.optimal_value}')
        print(f'optimizer_time: {result.optimizer_time}')
        
        

        # Simulate Circuit and check for optimal result
        qc = ws_qaoa.get_optimal_circuit()

        c = ClassicalRegister(qc.width(), name='c')
        q = find_regs_by_name(qc, 'q')
        qc.add_register(c)
        qc.barrier(q)
        qc.measure(q, c)
    
        min_cost = n_qubits
        min_cost_sample = ''
        for (sample, counts) in execute(qc, backend, shots=2**12).result().get_counts().items():
            cost = (eval_cost(op, sample) + 1) * eval_cost(op_pubo, sample)
            
            if cost < min_cost:
                min_cost = cost
                min_cost_sample = sample
            
            if cost == 0:
                return (sample, cost)
        
        return (min_cost_sample, min_cost)