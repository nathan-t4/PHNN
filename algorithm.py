import pandas as pd
import numpy as np
"""
    This realizability algorithm is based on the paper:
    "Passive Guaranteed Simulation of Analog Audio Circuits: A Port-Hamiltonian Approach"
    by Antoine Falaize and Thomas Helie
"""

voltage_controlled_components = ["inductor", "conductance", "diode", "transistor_bc", "transistor_be", "current_source", "OUT"]
current_controlled_components = ["capacitor", "resistor", "voltage_source", "IN", "GRD"]

class PHGraph():
    def __init__(self, N0: int, Bx: pd.DataFrame, Bw: pd.DataFrame, By: pd.DataFrame, Bp: pd.DataFrame):
        """
            input
            - N0: index of ground node
            - Bx: interconnection of storage edges 
                (capacitor, inductor)
            - Bw: interconnection of dissipative edges 
                (resistor, conductance, PH diode, NPN transistor, potentiometer)
            - By: interconnection of source edges 
                (voltage source, current source)
            - Bp: interconnection of external components
                (IN, OUT, GRD)
        """
        assert(Bx.shape[0] == Bw.shape[0] == By.shape[0] == Bp.shape[0])

        self.N = Bx.shape[0] # number of nodes
        self.N0 = N0 # index of ground node
        self.Bx = Bx # interconnection of storage edges
        self.Bw = Bw # interconnection of dissipative edges
        self.By = By # interconnection of source edges
        self.Bp = Bp # interconnection of external components

        self.I = pd.concat((self.Bx, self.Bw, self.By, self.Bp), axis=1) # incidence matrix
        self.A = self.I.map(abs) # adjacency matrix

def portHamiltonianRealizability(g: PHGraph):
    """
        input
        - a PHGraph

        output
        - sets of voltage-controlled edges B1 and current-controlled edges B2
    """
    A = g.A.copy()

    B1 = []
    B2 = []
    Bw = g.Bw.copy()

    # sort storage and source edges to be voltage or current controlled (Table 2 in paper)

    for component_name in A:  
        if component_name in voltage_controlled_components:
            B1.append(component_name)
            if component_name in Bw:
                Bw = Bw.drop(columns=[component_name])

        elif component_name in current_controlled_components and component_name not in g.Bw:
            B2.append(component_name)
    
    Bi = Bw
    A.iloc[g.N0,:] = A.shape[1] * [0] # N0 is the ground node

    # for each voltage controlled edge, set the corresponding column to 0
    for b in B1:
        A[b] = A.shape[0] * [0]
    
    condition = True
    while (condition):
        A_star = A.copy()
        for b in B2:
            if sum(A[b]) == 0:
                print("not realzable 1")
                return False # not realizable
            elif sum(A[b]) == 1:
                n = list(A[b]).index(1)
                cols_wo_b  = A.columns.tolist()
                cols_wo_b.remove(b)
                A.loc[n,cols_wo_b] = 0
        
        for b in Bi:
            if sum(A[b]) == 0:
                B1.append(b)
                Bi = Bi.drop(columns=[b])
            elif sum(A[b]) == 1:
                n = list(A[b]).index(1)
                cols_wo_b  = A.columns.tolist()
                cols_wo_b.remove(b)
                A.loc[n,cols_wo_b] = 0

                B2.insert(0, b)
                Bi = Bi.drop(columns=[b])
            
        nodes_excluding_ground = list(range(g.N))
        nodes_excluding_ground.remove(g.N0)
        for n in nodes_excluding_ground:
            if sum(A.iloc[n,:]) == 0:
                print("not realizable 2")
                return False # not realizable
                
        condition = not A_star.equals(A)
            
    if len(Bi) > 0:
        B1.extend(Bi)
    
    return B1, B2

def get_dissipation_matrix(g: PHGraph):
    """
        input
        - a PHGraph

        output
        - port-Hamiltonian dissipation matrix J

        J satisfies (v1; i2) = J (i1; v2)

        Need to then reorder J to recover port-Hamiltonian equation (5) in paper
    """
    output = portHamiltonianRealizability(g)
    if output == False:
        raise ValueError("Not realizable")
    else:
        B1, B2 = output

    I = g.I.copy()
    I = I.drop([g.N0], axis=0) # remove ground node

    L1 = I[B1]
    L2 = I[B2]

    if np.linalg.det(L2) == 0:
        raise ValueError("L2 is singular")
    else:
        L = np.linalg.inv(L2) @ L1
    
    # Get J from A
    n1,n2 = L.shape
    J = np.block([[np.zeros((n2,n2)),L.T],
                  [-L, np.zeros((n1,n1))]])
    
    return J

if __name__ == "__main__":
    """
        - Bx: interconnection of storage edges 
            (capacitor, inductor)
        - Bw: interconnection of dissipative edges 
            (resistor, conductance, PH diode, NPN transistor, potentiometer)
        - By: interconnection of source edges 
            (voltage source, current source)
        - Bp: interconnection of external components
            (IN, OUT, GRD)
    """
    Bx = pd.DataFrame({
    })

    Bw = pd.DataFrame({
    })

    By = pd.DataFrame({
    })

    Bp = pd.DataFrame({
    })

    g = PHGraph(N0=0, Bx=Bx, Bw=Bw, By=By, Bp=Bp)
    J = get_dissipation_matrix(g)
    print("J", J)