from SVAR import *
from panelSVAR import *

def run_panel():
    plot = True
    savefig_path = ""
    excel_path = "AustraliaData.xlsx"
    excel_sheet_name = "Panel6_comm_all"
    variables = {
        # 1 for unit root, 0 for stationary
        'Yreal' : 1,
        # 'Govnom' : 1,
        # 'CPI' : 1,
        'G': 1
    }
    shocks = ['other', 'AutFP']
    td_col = "" # Does not contain a time column
    sr_constraint = [(2,2)]
    lr_constraint = []
    sr_sign = np.array([['.','.'],
                        ['.','.']])
    lr_sign = np.array([['.','.'],
                        ['.','+']])
    maxlags = 4 # maximum lags to be considered for common shock responses
    nsteps = 15   # desired number of steps for the impulse responses
    lagmethod = 'aic'

    bootstrap = True
    ndraws = 2000
    signif = 0.05 # significance level of bootstrap

    # Run VAR
    panel_input = VAR_input(variables, shocks, td_col, sr_constraint, lr_constraint,
                          sr_sign, lr_sign, maxlags, nsteps, lagmethod, bootstrap, ndraws, signif,
                          excel_path, excel_sheet_name, None, plot, savefig_path)
    panelSVAR(panel_input)

def run_var():
    """
    # EXAMPLE IMPUT BELOW
    
    """
    # INPUT SECTION
    plot = True
    savefig_path = ""
    excel_path = "AustraliaData.xlsx"
    excel_sheet_name = "Panel6_comm_all"
    variables = {
        # 1 for unit root, 0 for stationary
        'Yreal' : 1,
        # 'Govnom' : 1,
        # 'CPI' : 1,
        'G': 1
    }
    shocks = ['other', 'AutFP']
    td_col = ""
    sr_constraint = [(2,2)]
    lr_constraint = []
    sr_sign = np.array([['.','.'],
                        ['.','.']])
    lr_sign = np.array([['.','.'],
                        ['.','+']])
    maxlags = 4 # maximum lags to be considered for common shock responses
    nsteps = 15 # desired number of steps for the impulse responses
    lagmethod = 'aic'

    bootstrap = True
    ndraws = 2000
    signif = 0.05 # significance level of bootstrap

    # Run VAR
    var_input = VAR_input(variables, shocks, td_col, sr_constraint, lr_constraint,
                          sr_sign, lr_sign, maxlags, nsteps, lagmethod, bootstrap, ndraws, signif,
                          excel_path, excel_sheet_name, None, plot, savefig_path)
    SVAR(var_input)

if __name__ == "__main__":
    run_var()