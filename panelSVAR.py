"""
Based on Pedroni (2013)
"""

from SVAR import *
from sklearn.linear_model import LinearRegression
from scipy.stats import linregress

class Panel_output:
    __slots__ = []
    def __init__(self, comp_df=None, comm_df=None, idio_df=None, lambda_df=None):
        self.comp_df = comp_df
        self.comm_df = comm_df
        self.idio_df = idio_df
        self.lambda_df = lambda_df

def panelSVAR(input):
    """
    __slots__ = ['df', 'size', 'variables', 'shocks', 'td_col', 'sr_constraint', 'lr_constraint', 'sr_sign', 'lr_sign',
                 'maxlags', 'nsteps', 'lagmethod', 'bootstrap', 'ndraws', 'signif', 'plot', 'savefig_path']
    """
    # lambda_dict = dict() # String member -> np.ndarray Lambda
    # comp_dict = dict()
    members = list(input.df[input.member_col].unique())
    elements = ["IR"+str(vr)+str(sk)+"_"+str(lg) for vr in range(1,input.size)
                for sk in range(1,input.size) for lg in range(input.nsteps+1)] # lg(Lag) is the innermost loop
    
    # Initialize output spreadsheets
    comp_df = pd.DataFrame(index=members, columns=elements)
    comm_df = comp_df.copy()
    idio_df = comp_df.copy()
    lambda_df = pd.DataFrame(index=members, columns=["Lambda"+str(i)+str(j) for i in range(input.size) for j in range(input.size)])

    variable_cols = list(input.variables.keys())
    # Common shock
    if input.td_col == "":
        raise ValueError("Must include time column for panel data.")
    input.df[variable_cols] = input.df.groupby(input.td_col)[variable_cols].mean()
    common_output = SVAR(input)
    common_shock = common_output.shock

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

        composite_shock = member_output.shock
        # Merge with the common shock on index (td)
        merged_df = pd.merge(composite_shock, common_shock, left_index=True, right_index=True, how='inner')
        # Regress for diagonal matrix Lambda. Only estimate diagonal elements to improve efficiency.
        Lambda = np.zeros((input.size, input.size))
        for i in range(input.size):
            y = merged_df.iloc[:, i]
            x = merged_df.iloc[:, input.size+i]
            linear = linregress(x, y)
            Lambda[i, i] = linear.intercept
            # 95% confidence band width = linear.stderr * 2

        # INSTEAD OF THIS
        # y = merged_df.iloc[:,:input.size]
        # X = merged_df.iloc[:, input.size:]
        # lin = LinearRegression()
        # lin.fit(X, y)
        # Lambda = lin.coef_.T # Must transpose

        # comp_dict[member] = member_output
        # lambda_dict[member] = Lambda
        
        # Write into dataframes
        comp_df.loc[member, :] = member_output.reshape(1,)
        # impulse response to common shock = A*Lambda
        comm_df.loc[member, :] = np.dot(member_output, Lambda).reshape(1,)
        # impulse response to idiosyncratic shock = A*(I-Lambda*Lambda')^(1/2)
        idio_df.loc[member, :] = np.dot(member_output, np.sqrt(np.identity(input.size)-Lambda)).reshape(1,)
        
        lambda_df.loc[member, :] = Lambda.reshape(1,)
        
    output = Panel_output(comp_df, comm_df, idio_df, lambda_df)

    # Write into the spreadsheets
    # Same nomenclature as Pedroni's RATS code
    comp_df.to_excel("ind-IRs-to-composite-shocks.xlsx", sheet_name="ind-IRs-to-composite-shocks")
    comm_df.to_excel("ind-IRs-to-common-shocks.xlsx", sheet_name="ind-IRs-to-common-shocks")
    idio_df.to_excel("ind-IRs-to-idiosyncratic-shocks.xlsx", sheet_name="ind-IRs-to-idiosyncratic-shocks")
    lambda_df.to_excel("lambda-matrices.xlsx", sheet_name = "lambda-matrices")

    return output