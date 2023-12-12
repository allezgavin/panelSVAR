import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from plotting import plot_ir
from shortAndLong import shortAndLong

# Encapsulation of input to provide default values and keep modularity
class VAR_input:
    __slots__ = ['df', 'size', 'variables', 'shocks', 'td_col', 'member_col', 'sr_constraint', 'lr_constraint', 'sr_sign', 'lr_sign',
                 'maxlags', 'nsteps', 'lagmethod', 'bootstrap', 'ndraws', 'signif', 'plot', 'savefig_path']
    
    def __init__(self, variables, shocks, td_col=[], member_col="", sr_constraint=[], lr_constraint=[], sr_sign=np.array([]), lr_sign=np.array([]),
                 maxlags=5, nsteps=12, lagmethod='aic', bootstrap=True, ndraws=2000, signif=0.05,
                 excel_path="", excel_sheet_name="", df=None, plot=True, savefig_path=""):
        # Build input dataframe
        if excel_path != "":
            if excel_sheet_name != "":
                self.df = pd.read_excel(excel_path, sheet_name=excel_sheet_name)
            else:
                raise ValueError("Please specify excel sheet name.")
        else:
            if len(df) > 0:
                self.df = df.copy()
            else:
                raise ValueError("Empty input data.")
        
        self.variables = variables
        self.shocks = shocks
        self.size = len(self.variables)
        if len(self.shocks) != self.size:
            raise ValueError("Variable and shock have different dimensions.")
        
        self.td_col = td_col
        if len(td_col) == 0:
            print("Td column not specified. Assuming data is sorted.")
        else:
            self.df.sort_values(by=td_col, inplace=True)
        self.member_col = member_col
        
        self.sr_constraint = sr_constraint
        self.lr_constraint = lr_constraint

        if len(sr_sign) > 0:
            self.sr_sign = sr_sign
        else:
            sr_sign = np.asarray([['.' for i in range(self.size)] for j in range(self.size)])
        if len(lr_sign) > 0:
            self.lr_sign = lr_sign
        else:
            self.lr_sign = np.asarray([['.' for i in range(self.size)] for j in range(self.size)])

        self.maxlags = maxlags
        self.nsteps = nsteps
        self.lagmethod = lagmethod
        self.bootstrap = bootstrap
        self.ndraws = ndraws
        self.signif = signif
        self.plot = plot
        self.savefig_path = savefig_path

class VAR_output:
    __slots__ = ['lag_order', 'shock', 'ir', 'ir_upper', 'ir_lower', 'fevd']
    def __init__(self, lag_order=0, shock=np.array([]), ir=np.array([]),
                 ir_upper=np.array([]), ir_lower=np.array([]), fevd=np.array([])):
        self.lag_order = lag_order
        self.shock = shock
        self.ir = ir
        self.ir_upper = ir_upper
        self.ir_lower = ir_lower
        self.fevd = fevd
        
def SVAR(input): # possible mutation of input.df
    output = VAR_output()
    variable_names = [var if input.variables[var] == 0 else 'd'+var for var in input.variables.keys()]

    if len(input.td_col)>0:
        input.df.set_index(input.td_col, inplace = True)

    # Convert to stationary form
    input.df = input.df[list(input.variables.keys())]
    for var in input.variables:
        if input.variables[var] == 1:
            input.df[var] = np.log(input.df[var]) - np.log(input.df[var].shift(1))
    input.df.dropna(inplace = True)

    model = VAR(input.df)
    results = model.fit(maxlags=input.maxlags, ic=input.lagmethod)
    # print(results.params) # VAR coefficients
    # print(results.sigma_u) # Covariance matrix Omega_mu
    output.lag_order = results.k_ar
    # print(input.df)
    prediction = model.predict(params=results.params, lags=output.lag_order)
    output.shock = input.df.iloc[output.lag_order:, :] - prediction # This step calculates mu. Will be tranformed into epsilon
    
    # Estimate impulse response, without any transformation of the shocks (FACTOR = %identity(m))
    irf = results.irf(input.nsteps)
    # print(irf.irfs)

    # Calculate decomposition matrix M
    F1 = np.zeros((input.size, input.size))
    for f in irf.irfs:
        F1 += f
    M = shortAndLong(results.sigma_u, input.sr_constraint, input.lr_constraint, F1)
    A1 = np.dot(F1, M)
    # Sign constraint
    signmat = np.identity(input.size)
    for i in range(input.size):
        for j in range(input.size):
            switch_sign = ((input.sr_sign[i, j] == '+' and M[i,j] < 0)
                           or (input.sr_sign[i, j] == '-' and M[i,j] > 0)
                           or (input.lr_sign[i, j] == '+' and A1[i,j] < 0)
                           or (input.lr_sign[i, j] == '-' and A1[i,j] > 0))
            if switch_sign:
                signmat[j, j] = -1
    M = np.dot(M, signmat)
    # print(M) # The M, or A0 matrix

    output.ir = irf.irfs
    for i in range(input.nsteps+1):
        output.ir[i] = np.dot(output.ir[i], M)
    
    for i in range(len(output.shock)):
        output.shock.iloc[i, :] = np.dot(np.linalg.inv(M), output.shock.iloc[i,:].T).T # epsilon = M^(-1) * mu

    # VARIANCE DECOMPOSITION NEEDS MORE WORK
    fevd = results.fevd(input.nsteps)
    # fevd.summary()

    ALALT = []
    for i in range(len(output.ir)):
        ALALT.append(np.dot(output.ir[i], output.ir[i].T))
    ALALT = np.array(ALALT)

    VD = ALALT.cumsum(axis=0)
    VD = np.abs(VD)
    for i in range(len(VD)):
        VD[i] /= VD[i].sum(axis=1, keepdims=True)
    # print(VD)




    if input.bootstrap:
        errband = np.asarray(irf.errband_mc(repl=input.ndraws, signif=input.signif))
        # errband[0] is lower band and errband[1] is upper band
        for j in [0, 1]:
            for i in range(input.nsteps+1):
                errband[j, i] = np.dot(errband[j, i], M)
        output.ir_lower = errband[0]
        output.ir_upper = errband[1]

    if input.plot:
        plot_ir(variable_names, input.shocks, output.ir,
                lower_errband=output.ir_lower, upper_errband=output.ir_upper,
                show_plot=True, save_plot=True, plot_path=input.savefig_path)
    
    return output

if __name__ == "__main__":
    pass