# -*- coding: utf-8 -*-
"""
Last Modified on 19 August, 2023
@author: Haiyue Kang (Modified from John. F. Kam)
@Link to John. F. Kam's original codes: https://github.com/jkfids/qiskit-entangle/blob/main/code/entanglebase.py
"""

# Standard libraries
import networkx as nx
import datetime

class Free_EntangleBase:
    """
    Parent class for device entangled state analysis. Different from entanglebase.py,
    this class takes one more argument 'qubits_to_connect' to freely choose which set
    of qubits to form an induced subgraph of the full graph
    """
    
    def __init__(self, backend, qubits_to_connect):
        """initialize the self objects including all topological informations

        Args:
            backend (IBMProvider().backend object): IBM quantum backend
            qubits_to_connect (list): list of qubits to be induced
        """
        self.backend = backend
        properties = backend.properties()
        
        self.device_name = properties.backend_name
        self.nqubits = len(qubits_to_connect)
        self.device_size = len(properties.qubits)
        self.graph, self.connections, self.edge_params = self.gen_graph(qubits_to_connect)
        self.edge_list = sorted(list(self.edge_params.keys()), key=lambda q: (q[0], q[1]))
        self.nedges = len(self.edge_params)
    
    def gen_graph(self, qubits_to_connect):
        """
        Obtain the Graph of the IBM QPU, including connections (neighbour of each qubit) and edges
        """
        graph = nx.Graph()
        #graph = nx.DiGraph()
        connections = {} #{0:[1], 1:[2], 2:[3,1],...} key is the index of the qubit, value is/are the qubit(s) connected to it
        edges = {} #{(0,1):0.2, (1,2):0.5,...} the values are the CNOT errors for this pair of qubits
        for i in qubits_to_connect:
            connections[i] = []
        # Iterate over possible CNOT or ECR connections, first check which type of basis coupling gate is for the backend
        #if (self.device_name == 'ibm_sherbrooke') or (self.device_name == 'ibm_brisbane') or (self.device_name == 'ibm_seattle'):
        #for gate in self.backend.properties(datetime=datetime.datetime(2024,2,10,20)).gates:
        for gate in self.backend.properties().gates:
            if gate.gate == 'ecr':
                q0 = sorted(gate.qubits)[0]
                q1 = sorted(gate.qubits)[1] #originally there is no sorted in Fidel's code, adding this to account for non-cx gates
                if (q0 in connections) and (q1 in connections):
                    connections[q0].append(q1)
                    connections[q1].append(q0)
                    if q0 < q1:
                        graph.add_edge(q0, q1, weight=gate.parameters[0].value)
                        #graph.add_edge(q0, q1, weight=1)
                        edges[q0, q1] = gate.parameters[0].value
        #else:
        #    for gate in self.backend.properties().gates:
            if gate.gate == 'cx' or gate.gate == 'cz':
                q0 = gate.qubits[0]
                q1 = gate.qubits[1]
                if (q0 in connections) and (q1 in connections):
                    connections[q0].append(q1)
                    if q0 < q1:
                        graph.add_edge(q0, q1, weight=gate.parameters[0].value)
                        edges[q0, q1] = gate.parameters[0].value
        # Sort adjacent qubit list in ascending order
        for q in connections:
            connections[q].sort()
            
        return graph, connections, edges