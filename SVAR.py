import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from plotting import plot_ir
from shortAndLong import shortAndLong
import copy
from scipy import stats
    
# Encapsulation of input to provide default values and keep modularity
class VAR_input:
    __slots__ = ['df', 'size', 'variables', 'shocks', 'td_col', 'member_col', 'sr_constraint', 'lr_constraint', 'sr_sign', 'lr_sign',
                 'maxlags', 'nsteps', 'lagmethod', 'bootstrap', 'ndraws', 'signif', 'plot', 'savefig_path']
    
    def __init__(self, variables, shocks, td_col=[], member_col="", sr_constraint=[], lr_constraint=[], sr_sign=np.array([]), lr_sign=np.array([]),
                 maxlags=5, nsteps=12, lagmethod='aic', bootstrap=True, ndraws=2000, signif=0.05,
                 excel_path="", excel_sheet_name="", df=pd.DataFrame(), plot=True, savefig_path=""):
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
        
def SVAR(input):
    output = VAR_output()

    if len(input.td_col) > 0:
        # Would raise error if there are duplicate times.
        input.df.set_index(input.td_col, inplace = True)

    variable_names = list(input.variables.keys())
    df = input.df[variable_names]

    # Convert to stationary form
    # This cannot be done in __init__ of VAR_input because, in a data panel, the steady-state data must be averaged first before finding the log diff.
    def log_diff(arr):
        log_arr = np.log(arr)
        return log_arr - log_arr.shift(1)
    for var in input.variables:
        if input.variables[var][0] == 1:
            df[var] = log_diff(df[var])
            input.variables[var][0] = 0
    df.dropna(inplace=True)
    
    model = VAR(df)
    results = model.fit(maxlags=input.maxlags, ic=input.lagmethod)
    # print(results.params) # VAR coefficients
    # print(results.sigma_u) # Covariance matrix Omega_mu

    # Cannot invert to VMA form if no lags selected
    output.lag_order = results.k_ar
    if output.lag_order == 0:
        return output
    
    prediction = model.predict(params=results.params, lags=output.lag_order)
    output.shock = df.iloc[output.lag_order:, :] - prediction # This step calculates mu. Will be tranformed into epsilon

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
    
    
    if input.bootstrap:
        print("Bootstrapping in progress...")
        normal_interval = False
        draw_from_normal = True
        
        # Initialize output storage
        if normal_interval:
            mean_accum = np.zeros_like(output.ir)
            sq_diff_accum = np.zeros_like(output.ir)
        if not normal_interval:
            # This uses too much memory when dataset is large
            IRs = np.zeros((input.ndraws, input.nsteps+1, input.size, input.size))

        for i in range(input.ndraws):
            # Initialize input object
            boot_input = copy.deepcopy(input)
            boot_input.bootstrap = False
            boot_input.plot = False
            boot_input.td_col = ""

            if draw_from_normal:
                shuffled_shock = np.random.multivariate_normal(np.array([0,0]), results.sigma_u, output.shock.shape[0])
            else:
                # Draw randomly with replacement
                shuffled_shock = np.empty_like(output.shock)
                for j in range(shuffled_shock.shape[0]):
                    shuffled_shock[j, :] = output.shock.iloc[np.random.randint(0, output.shock.shape[0]), :]
            
            # Find impulse response subject to structrual restrictions
            boot_input.df = pd.DataFrame(columns = variable_names, data = prediction+shuffled_shock)
            boot_output = SVAR(boot_input)
            if boot_output.lag_order == 0:
                # Unsuccessful VAR. No lags selected. Treat all VMA coefs as zero.
                continue

            if normal_interval:
                mean_accum += boot_output.ir
                sq_diff_accum += boot_output.ir ** 2
            else:
                IRs[i, :, :, :] = boot_output.ir
            
        if normal_interval:
            boot_mean = mean_accum / input.ndraws
            boot_std = np.sqrt(sq_diff_accum / input.ndraws - boot_mean ** 2)
            z_score = stats.norm.ppf(1 - input.signif / 2)
            output.ir_lower = boot_mean - z_score * boot_std
            output.ir_upper = boot_mean + z_score * boot_std
        else:
            # Sort boot_output.ir in IRs and find thresholds
            output.ir_lower = np.empty_like(output.ir)
            output.ir_upper = np.empty_like(output.ir)
            for lg in range(input.nsteps+1):
                for vr in range(input.size):
                    for sk in range(input.size):
                        output.ir_lower[lg, vr, sk], output.ir_upper[lg, vr, sk] = stats.mstats.mquantiles(
                            IRs[:, lg, vr, sk], [input.signif / 2, 1 - input.signif / 2])
                    

    for i in range(len(output.shock)):
        output.shock.iloc[i, :] = np.dot(np.linalg.inv(M), output.shock.iloc[i,:].T).T # epsilon = M^(-1) * mu

    # Convert to impulse reponse of steady state for unit root variables
    for i, var in enumerate(variable_names):
        if input.variables[var][1] == 1:
            output.ir[:, i, :] = output.ir[:, i, :].cumsum(axis=0)

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

    if input.plot:
        plot_ir(variable_names, input.shocks, output.ir,
                lower_errband=output.ir_lower, upper_errband=output.ir_upper,
                show_plot=True, save_plot=True, plot_path=input.savefig_path)
    
    return output

if __name__ == "__main__":
    pass