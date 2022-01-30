import itertools
import math
import copy

from qiskit.opflow import I, Z
import numpy as np

# GENERAL HELPER FUNCTIONS

def eval_cost(op, state):
    return np.absolute(op.eval(state).primitive[state]) 

def generate_binary_strings(bit_count):
    binary_strings = []
    def genbin(n, bs=''):
        if len(bs) == n:
            binary_strings.append(bs)
        else:
            genbin(n, bs + '0')
            genbin(n, bs + '1')

    genbin(bit_count)
    return binary_strings

def get_eigenstates(result, bit_strings=None):
    if type(result.eigenstate) is dict:
        return result.eigenstate
    else:
        
        if bit_strings is None:
            bit_strings = generate_binary_strings(round(math.log2(len(result.eigenstate))))
        
        amp = np.absolute(np.array(result.eigenstate))
        return dict(zip(bit_strings, amp))
    
def generate_random_board(w, h, n_bombs):
    board = [['-' for _ in range(w)] for _ in range(h)]
    
    rng = np.random.default_rng()
    for (x,y) in zip(rng.integers(0, w, n_bombs), rng.integers(0, h, n_bombs)):
        board[y][x] = 'x'
        
    # Construct solution
    solution = copy.deepcopy(board)
    for y in range(h):
        for x in range(w):
            if board[y][x] == 'x':
                continue
                
            bombs_count = 0
            for (_x, _y) in itertools.product([x-1,x,x+1], [y-1,y,y+1]):
                if not (0 <= _x < w and 0 <= _y < h): 
                    continue
                
                bombs_count += 1 if board[_y][_x] == 'x' else 0
            
            solution[y][x] = str(bombs_count)
    
    return '\n'.join(map(lambda x: ''.join(x), solution))


# CLASSES

class QAOACallback:

    def __init__(self):
        self.iteration = []
        self.params = []
        self.mean = []
        self.std_dev = []
        
    def callback(self, i, p, m, sd):
        self.iteration.append(i)
        self.params.append(p)
        self.mean.append(m)
        self.std_dev.append(sd)
        
        print(f'{i:04} {m:.2f}  -  {p}')
    
    
class MinesweeperQiskit:
    
    def construct_op(self, i_true, i_false, n_qubits):
        op = None

        for i in range(n_qubits):
            i_op = I
            if i in i_false:
                i_op = (0.5*I + 0.5*Z)
            elif i in i_true:
                i_op = (0.5*I - 0.5*Z)

            if i == 0:
                op = i_op
            else:
                op = op ^ i_op

        return op

    def construct_field_op(self, val, fields, n_qubits):
        op = None
        num_fields = len(fields)

        for perm in set(list(itertools.permutations([False] * (len(fields) - val) + [True] * val))):
            i_true  = [fields[i] for i in range(num_fields) if perm[i]]
            i_false = [fields[i] for i in range(num_fields) if not perm[i]]

            p_op = self.construct_op(i_true, i_false, n_qubits)

            if op is None:
                op = p_op
            else:
                op = op + p_op

        return ((I^n_qubits) - op)  # Invert, because we want to minimize

    def construct_field_op_mod_2(self, val, fields, n_qubits):
        op = None

        for i in range(n_qubits):
            i_op = Z if i in fields else I
            if i == 0:
                op = i_op
            else:
                op = op ^ i_op

        return 0.5 * ((I^n_qubits) - ((-1)**val) * op)

    def construct_board_op(self, board, mod_2=False):
        """
        mod_2:
            False -> Construct a PUBO hamiltonian
            True -> Construc a ISING hamiltonian
        """
        
        lines = board.split('\n')

        # Create coordinate -> qubit map
        n_qubits = 0
        qubits = {}
        for (x, line) in enumerate(lines):
            for (y, char) in enumerate(line):
                if char == 'x' or char == '.':
                    qubits[(x,y)] = n_qubits
                    n_qubits += 1

        # Construct hamiltonian
        op = 0 * (I ^ n_qubits)
        for (x, line) in enumerate(lines):
            for (y, char) in enumerate(line):
                if char.isdigit() and char != '0':
                    fields = []
                    for (dy, dx) in itertools.product([-1,0,1], [-1,0,1]):
                        coord = (x + dx, y + dy)
                        if coord in qubits:
                            fields.append(qubits[coord])

                    if len(fields) > 0:
                        if mod_2:
                            op += self.construct_field_op_mod_2(int(char), fields, n_qubits)
                        else:
                            op += self.construct_field_op(int(char), fields, n_qubits)

        return (op.reduce(), n_qubits)