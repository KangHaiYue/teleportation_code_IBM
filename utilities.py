# -*- coding: utf-8 -*-
"""
Last Modified on 19 August, 2023
@author: Haiyue Kang (Modified from John. F. Kam)
@Link to John. F. Kam's original codes: https://github.com/jkfids/qiskit-entangle/blob/main/code/utilities.py
"""

# Standard libraries
from numpy import array, kron
# Qiskit libraries
from qiskit import IBMQ
from qiskit_ibm_provider import IBMProvider
import mthree

# Pytket libraries
from pytket.extensions.quantinuum import QuantinuumBackend

# IBM Quantum account utils
from token_id import token_id
token = token_id


def startup(check=False, token=token, hub='ibm-q-melbourne', group='unimelb', project='hub'):
    """Start up session"""
    if IBMQ.active_account() is None:
        IBMQ.enable_account(token)
        print("Account enabled")
    else:
        print("Account already enabled")
    
    provider = IBMQ.get_provider(hub=hub, group=group, project=project)
    print('Provider:', f'{hub}-{group}-{project}')
    
    if check:
        check_provider(hub=hub, group=group, project=project)
            
    return provider

def IBM_startup(token = token):
    """Start up session"""
    
    try:
        provider = IBMProvider()
    except:
        IBMProvider.save_account(token=token)
        provider = IBMProvider()
    print("Account enabled")
            
    return provider

def Quantinuum_startup(device='H1-1SC'):
    backend=QuantinuumBackend(device_name=device)#,provider='Microsoft')
    backend.login()
    return backend

def check_provider(hub='ibm-q-melbourne', group='unimelb', project='hub'):
    """Check list of providers with queue size and qubit count for input hub"""
    provider = IBMQ.get_provider(hub=hub, group=group, project=project)
    
    for backend in provider.backends():
        try:
            qubit_count = len(backend.properties().qubits)
        except:
            qubit_count = 'simulated'
        print(f'{backend.name()} has {backend.status().pending_jobs} queud and {qubit_count} qubits')
      
      
# Math objects

pauli = {'I': array([[1, 0], [0, 1]], dtype=complex),
         'X': array([[0, 1], [1, 0]], dtype=complex),
         'Y': array([[0, -1j], [1j, 0]], dtype=complex),
         'Z': array([[1, 0], [0, -1]], dtype=complex)}

pauli_product = {'II':(1,'I'), 'IX':(1,'X'), 'IY':(1,'Y'), 'IZ':(1,'Z'),
                 'XI':(1,'X'), 'XX':(1,'I'), 'XY':(1j,'Z'), 'XZ':(-1j,'Y'),
                'YI':(1,'Y'), 'YX':(-1j,'Z'), 'YY':(1,'I'), 'YZ':(1j,'X'),
                'ZI':(1,'Z'), 'ZX':(1j,'Y'), 'ZY':(-1j,'X'), 'ZZ':(1,'I')}
# Math functions

def bit_str_list(n):
    """Create list of all n-bit binary strings"""
    return [format(i, 'b').zfill(n) for i in range(2**n)]
      
def pauli_n(basis_str):
    """Calculate kronecker tensor product sum of basis from basis string"""
    
    M = pauli[basis_str[0]]
    try:
        for basis in basis_str[1:]:
            M_new = kron(M, pauli[basis])
            M = M_new
    except: pass # Single basis case
    
    return M 

# Run and load mthree calibrations
def run_cal(backend, filename=None):
    mit = mthree.M3Mitigation(backend)
    mit.cals_from_system(list(range(len(backend.properties().qubits))), shots=8192)
    if filename is None:
        filename = f'calibrations/{backend.name}_cal.json'
    mit.cals_to_file(filename)
    
    return mit
    
def load_cal(backend=None, filename=None):
    mit = mthree.M3Mitigation(backend)
    if filename is None:
        filename = f'calibrations/{backend.name}_cal.json'
    mit.cals_from_file(filename)
    
    return mit
      

    