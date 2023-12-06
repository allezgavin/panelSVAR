"""
Based on Pedroni (2013)
"""

from SVAR import *

class Panel_output:
    __slots__ = []
    def __init__(self, ir_comm=None):
        self.ir_comm = ir_comm

def panelSVAR(input):
    """
    __slots__ = ['df', 'size', 'variables', 'shocks', 'td_col', 'sr_constraint', 'lr_constraint', 'sr_sign', 'lr_sign',
                 'maxlags', 'nsteps', 'lagmethod', 'bootstrap', 'ndraws', 'signif', 'plot', 'savefig_path']
    """
    variable_cols = list(input.variables.keys())
    #member_col
    # Composite shock
    for member, member_df in input.df.groupby(input.member_col):
        member_svar_input = VAR_input(input.variables, input.shocks, input.td_col,
                                      input.sr_constraint, input.lr_constraint, input.sr_sign, input.lr_sign,
                                      input.maxlags, input.nsteps, input.lagmethod, input.bootstrap,
                                      input.ndraws, input.signif, excel_path="", excel_sheet_name="",
                                      # This df changes
                                      df=member_df, plot=False)
        member_output = SVAR(member_svar_input)
    
    # Common shock
    if input.td_col == "":
        raise ValueError("Must include time column for panel data.")
    input.df[variable_cols] = input.df.groupby(input.td_col)[variable_cols].mean()
    common_output = SVAR(input) # Use date as indices of pd.Series?

    # Regress for Lambda

    
    output = Panel_output()
    return output