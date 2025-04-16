# Define time-varying parameters as functions of time t
import numpy as np

class TDProbabilities():
    def __init__(self, fp, psi1, psi2, psi3, psi4, psi5, psi6, psi7, psi8, t0):
        self.fp = fp
        self.t0 = t0
        self.psi1 = psi1
        self.psi2 = psi2
        self.psi3 = psi3
        self.psi4 = psi4
        self.psi5 = psi5
        self.psi6 = psi6
        self.psi7 = psi7
        self.psi8 = psi8

    def alpha(self, t):
        return self.fp * np.exp(self.psi1 * (t - self.t0))

    def beta(self, t):
        return (1 - self.fp) * np.exp(- self.psi2 * (t - self.t0))

    def gamma(self, t):
        return self.fp * np.exp(-self.psi3 * (t - self.t0))

    def delta(self, t):
        return self.fp * np.exp(-self.psi4 * (t - self.t0))

    def lmbda(self, t):
        return 1 - ( self.fp * np.exp(-self.psi5 * (t - self.t0)) )

    def epsilon(self, t):
        return self.fp * np.exp(self.psi6 * (t - self.t0))

    def rho(self, t):
        return 1 - ( self.fp * np.exp(-self.psi7 * (t - self.t0)) )

    def mu(self, t):
        return 1 - ( self.fp * np.exp(self.psi8 * (t - self.t0)) )
    

def define_tdp(fp, T, t0=0):
    pos_psi = np.log(1/fp)/(T)
    psi1, psi2, psi3, psi4, psi5, psi6, psi7, psi8 = pos_psi, pos_psi, pos_psi, pos_psi, pos_psi, pos_psi, pos_psi, pos_psi
    tdp = TDProbabilities(fp, psi1, psi2, psi3, psi4, psi5, psi6, psi7, psi8, t0)

    return tdp


def define_trans_prob(tdp):
    # transition probabilites
    trans_prob = {}
    trans_prob['S'] = {}
    trans_prob['P'] = {}
    trans_prob['N'] = {}
    trans_prob['C'] = {}
    trans_prob['S']['P'] = tdp.alpha
    trans_prob['S']['N'] = tdp.beta
    trans_prob['P']['N'] = tdp.gamma
    trans_prob['P']['C'] = tdp.delta
    trans_prob['N']['P'] = tdp.epsilon
    trans_prob['N']['C'] = tdp.rho
    trans_prob['C']['P'] = tdp.lmbda
    trans_prob['C']['N'] = tdp.mu

    return trans_prob


class Probabilities:
    def __init__(self, trans_prob):
        self.trans_prob = trans_prob
    
    def change(a, b):
        pass


