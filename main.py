"""
Calculates VAR, SVAR, and panel SVAR. Panel SVAR methods are from
Pedroni, P. (2013), Econometrics, 1(2), 180-206; https://doi.org/10.3390/econometrics1020180

Code still in development. Use for ECON 371 only (for now).
Do NOT try to crash the code! I have not implemented much error handling.
Incorrect input could lead to incorrect output.

Your contribution is much appreciated!
Please contact Gavin Xia at gx1@williams.edu for to make contributions or use the code for projects.
"""

from SVAR import *
from panelSVAR import *

def run_panel():
    
    # EXAMPLE INPUT BELOW
    """
    plot = False
    savefig_path = ""
    excel_path = "pedroni_ppp.xls"
    excel_sheet_name = "Sheet1"
    variables = {
        # 1 for unit root, 0 for stationary
        'rf' : 1,
        'ae' : 1,
    }
    shocks = ['real', 'nominal']
    td_col = ["Year", "Month"]
    member_col = "country"
    sr_constraint = []
    lr_constraint = [(1,2)]
    sr_sign = np.array([['+','.'],
                        ['.','+']])
    lr_sign = np.array([['.','.'],
                        ['.','.']])
    maxlags = 4 # maximum lags to be considered for common shock responses
    nsteps = 36   # desired number of steps for the impulse responses
    lagmethod = 'aic'

    bootstrap = False
    ndraws = 2000
    signif = 0.05 # significance level of bootstrap
    """
    
    plot = False
    savefig_path = ""
    excel_path = "test-run.xls"
    excel_sheet_name = "Panel6_comm_all"
    variables = {
        # 1 for unit root, 0 for stationary
        'CommodityIndex' : [1, 1],
        'Yreal' : [1, 1]
    }
    shocks = ['Real', 'Nominal']
    td_col = ["time"]
    member_col = "country"
    sr_constraint = []
    lr_constraint = [(1,2)]
    sr_sign = np.array([['+','+'],
                        ['.','.']])
    lr_sign = np.array([['.','.'],
                        ['.','.']])
    maxlags = 4 # maximum lags to be considered for common shock responses
    nsteps = 15   # desired number of steps for the impulse responses
    lagmethod = 'aic'

    bootstrap = False
    ndraws = 2000
    signif = 0.05 # significance level of bootstrap
    
    # Run VAR
    panel_input = VAR_input(variables, shocks, td_col, member_col, sr_constraint, lr_constraint,
                          sr_sign, lr_sign, maxlags, nsteps, lagmethod, bootstrap, ndraws, signif,
                          excel_path, excel_sheet_name, pd.DataFrame(), plot, savefig_path)
    panelSVAR(panel_input)

def run_var():
    """
    # EXAMPLE IMPUT BELOW
    plot = True
    savefig_path = ""
    excel_path = "AustraliaData.xlsx"
    excel_sheet_name = "Panel6_comm_all"
    variables = {
        # 1 for unit root, 0 for stationary
        # First element in list for data input, second for output
        'Yreal' : [1, 1], # input in unit root form, output in unit root (steady-state) form
        'CPI' : [1, 0] # input in unit root form, output in stationary form (inflation rate)
    }
    shocks = ['AS', 'AD']
    td_col = ""
    member_col = ""
    sr_constraint = []
    lr_constraint = [(1,2)]
    sr_sign = np.array([['.','.'],
                        ['.','+']])
    lr_sign = np.array([['+','.'],
                        ['.','.']])
    maxlags = 4 # maximum lags to be considered for common shock responses
    nsteps = 15 # desired number of steps for the impulse responses
    lagmethod = 'aic'

    bootstrap = True
    ndraws = 200
    signif = 0.05 # significance level of bootstrap
    """
    
    # INPUT SECTION
    plot = True
    savefig_path = ""
    excel_path = "bqdata.xlsx"
    excel_sheet_name = "econ471-bqdata"
    variables = {
        # 1 for unit root, 0 for stationary
        'dGDPADJUST' : [0, 1],
        'URADJUST' : [0, 0]
    }
    shocks = ['AS', 'AD']
    td_col = ""
    member_col = "" # Not a panel so no member column
    sr_constraint = []
    lr_constraint = [(1,2)]
    sr_sign = np.array([['.','+'],
                        ['.','.']])
    lr_sign = np.array([['+','.'],
                        ['.','.']])
    maxlags = 8 # maximum lags to be considered for common shock responses
    nsteps = 20 # desired number of steps for the impulse responses
    lagmethod = 'aic'

    bootstrap = True
    ndraws = 200
    signif = 0.32 # significance level of bootstrap
    
    """
    # INPUT SECTION
    plot = True
    savefig_path = ""
    excel_path = ""
    excel_sheet_name = ""
    df = pd.read_excel("pedroni_ppp.xls", sheet_name="Sheet1")
    df = df.loc[df['country']==112]
    variables = {
        # 1 for unit root, 0 for stationary
        'rf' : [1, 1]
        'ae' : [1, 1]
    }
    shocks = ['real', 'nom']
    td_col = ""
    member_col = "" # Not a panel so no member column
    sr_constraint = []
    lr_constraint = [(1,2)]
    sr_sign = np.array([['+','.'],
                        ['.','.']])
    lr_sign = np.array([['.','.'],
                        ['.','.']])
    maxlags = 4 # maximum lags to be considered for common shock responses
    nsteps = 36 # desired number of steps for the impulse responses
    lagmethod = 'aic'

    bootstrap = True
    ndraws = 2000
    signif = 0.05 # significance level of bootstrap
    """

    # Run VAR
    var_input = VAR_input(variables, shocks, td_col, member_col, sr_constraint, lr_constraint,
                          sr_sign, lr_sign, maxlags, nsteps, lagmethod, bootstrap, ndraws, signif,
                          excel_path, excel_sheet_name, pd.DataFrame(), plot, savefig_path)
    output = SVAR(var_input)
    # print(output.ir)

if __name__ == "__main__":
    # run_var()
    run_panel()