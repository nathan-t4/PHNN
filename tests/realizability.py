import unittest
import pandas as pd
import numpy as np
from algorithm import PHGraph, get_dissipation_matrix

class TestRealizability(unittest.TestCase):

    def bjt(self):
        """ Figure 2 in the paper"""
        Bx = pd.DataFrame({
            'capacitor': [0, 1, -1, 0, 0, 0],
        })

        Bw = pd.DataFrame({
            'resistor': [0, 0, 0, 1, -1, 0],
            'transistor_bc': [0, 0, 1, -1, 0, 0],
            'transistor_be': [0, 0, 1, 0, 0, -1],
        })

        By = pd.DataFrame({
            'voltage_source': [1, 0, 0, 0, -1, 0],
        })

        Bp = pd.DataFrame({
            'IN': [1, -1, 0, 0, 0, 0],
            'OUT': [1, 0, 0, -1, 0, 0],
            'GRD': [1, 0, 0, 0, 0, -1],
        })

        g = PHGraph(N0=0, Bx=Bx, Bw=Bw, By=By, Bp=Bp)
        J = get_dissipation_matrix(g)
        
        gamma1 = np.array(
                 [[0, 0, 0],
                  [1, 1, 0],
                  [-1, 0, -1],
                  [0, 0, 0],
                  [0, -1, 0]])

        gamma2 = np.array(
                [[0, 1, 0, -1, 0],
                 [0, -1, 0, 0, 0],
                 [1, 0, 0, 0, 0],
                 [-1, 0, -1, 0, 0],
                 [0, 0, 0, 0, -1]])
        
        gamma = np.linalg.inv(gamma2) @ gamma1
        # Get J from A
        n1,n2 = gamma.shape
        J_true = np.block([[np.zeros((n2,n2)),gamma.T],
                           [-gamma, np.zeros((n1,n1))]])
        
        self.assertTrue(np.allclose(J, J_true), f"J {J} is not equal to {J_true}")
    
    @unittest.skip("not implemented yet")
    def diode(self):
        Bx = pd.DataFrame({

        })

        By = pd.DataFrame({

        })

        Bw = pd.DataFrame({

        })

        Bp = pd.DataFrame({

        })
        
        g = PHGraph(N0=0, Bx=Bx, Bw=Bw, By=By, Bp=Bp)
        J = get_dissipation_matrix(g)

        J_true = np.array(
            [[0, -1, 1, 0, 0],
             [1, 0, 0, 1, 0],
             [-1, 0, 0, -1, 0],
             [0, -1, 1, 0, 0],
             [1, 0, 0, 1, 0]]
        )
        
        self.assertTrue(np.allclose(J, J_true))

    @unittest.skip("not implemented yet")
    def inverter(self):
        Bx = pd.DataFrame({

        })

        By = pd.DataFrame({
            
        })

        Bw = pd.DataFrame({

        })

        Bp = pd.DataFrame({

        })
        
        g = PHGraph(N0=0, Bx=Bx, Bw=Bw, By=By, Bp=Bp)
        J = get_dissipation_matrix(g)

        