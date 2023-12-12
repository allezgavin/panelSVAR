"""
Based on Pedroni (2013)
"""

from SVAR import *
from sklearn.linear_model import LinearRegression
from scipy.stats import linregress

class Panel_output:
    __slots__ = ['comp_df', 'comm_df', 'idio_df', 'lambda_df']
    def __init__(self, comp_df=None, comm_df=None, idio_df=None, lambda_df=None):
        self.comp_df = comp_df
        self.comm_df = comm_df
        self.idio_df = idio_df
        self.lambda_df = lambda_df

def panelSVAR(input):
    # Process input dataframe
    if len(input.td_col) == 0:
        raise ValueError("Must include time column for panel data.")
    if input.member_col not in input.df:
        raise ValueError("Invaid panel member column.")
    input.df.sort_values(by=[input.member_col]+input.td_col, inplace=True)
    
    members = list(input.df[input.member_col].unique())
    elements = ["IR"+str(vr)+str(sk)+"_"+str(lg) for vr in range(1,input.size+1)
                for sk in range(1,input.size+1) for lg in range(input.nsteps+1)] # lg(Lag) is the innermost loop
    variable_cols = list(input.variables.keys())
    
    unit_root_var = []
    for var in input.variables:
        if input.variables[var] == 1:
            input.variables[var] = 0
            unit_root_var.append(var)
    input.df[unit_root_var] = input.df.groupby(input.member_col)[unit_root_var].transform(lambda x : np.log(x) - np.log(x).shift(1))

    # Initialize output spreadsheets
    comp_df = pd.DataFrame(index=members, columns=elements)
    comm_df = comp_df.copy()
    idio_df = comp_df.copy()
    lambda_df = pd.DataFrame(index=members, columns=["Lambda"+str(i)+str(j) for i in range(1,input.size+1) for j in range(1,input.size+1)])
    # lambda_dict = dict() # String member -> np.ndarray Lambda
    # comp_dict = dict()

    # Common shock
    average_df = input.df.groupby(input.td_col)[variable_cols].mean()
    comm_svar_input = VAR_input(input.variables, input.shocks, [], "",
                                input.sr_constraint, input.lr_constraint, input.sr_sign, input.lr_sign,
                                input.maxlags, input.nsteps, input.lagmethod, input.bootstrap,
                                input.ndraws, input.signif, excel_path="", excel_sheet_name="",
                                # This df changes
                                df=average_df, plot=False)
    common_output = SVAR(comm_svar_input)
    common_shock = common_output.shock

    # print(input.df)
    # Composite shock
    for member, member_df in input.df.groupby(input.member_col):
        member_svar_input = VAR_input(input.variables, input.shocks, [], "",
                                      input.sr_constraint, input.lr_constraint, input.sr_sign, input.lr_sign,
                                      input.maxlags, input.nsteps, input.lagmethod, input.bootstrap,
                                      input.ndraws, input.signif, excel_path="", excel_sheet_name="",
                                      # This df changes
                                      df=member_df.set_index(input.td_col)[variable_cols], plot=False)
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
        def multiply_by_matrix(arr, mat):
            result = np.empty_like(arr)
            for i, m in enumerate(arr):
                result[i] = np.dot(m, mat)
            return result
        comp_df.loc[member, :] = np.transpose(member_output.ir, (1,2,0)).flatten()
        # impulse response to common shock = A*Lambda
        comm_ir = multiply_by_matrix(member_output.ir, Lambda)
        comm_df.loc[member, :] = np.transpose(comm_ir, (1,2,0)).flatten()
        # impulse response to idiosyncratic shock = A*(I-Lambda*Lambda')^(1/2)
        idio_ir = multiply_by_matrix(member_output.ir, np.sqrt(np.identity(input.size)-Lambda))
        idio_df.loc[member, :] = np.transpose(idio_ir, (1,2,0)).flatten()
        
        lambda_df.loc[member, :] = Lambda.flatten()
        
    output = Panel_output(comp_df, comm_df, idio_df, lambda_df)

    # Write into the spreadsheets
    # Same nomenclature as Pedroni's RATS code
    comp_df.to_excel("ind-IRs-to-composite-shocks.xlsx", sheet_name="ind-IRs-to-composite-shocks")
    comm_df.to_excel("ind-IRs-to-common-shocks.xlsx", sheet_name="ind-IRs-to-common-shocks")
    idio_df.to_excel("ind-IRs-to-idiosyncratic-shocks.xlsx", sheet_name="ind-IRs-to-idiosyncratic-shocks")
    lambda_df.to_excel("lambda-matrices.xlsx", sheet_name = "lambda-matrices")

    return output