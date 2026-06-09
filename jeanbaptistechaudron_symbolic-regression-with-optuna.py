import pandas as pd 

path = "playground-series-s4e12/train.csv"
df = pd.read_csv(path, index_col =0)


df = df.dropna()
# We re-encode some variables and drop some other
df["Gender"] = df["Gender"].map(lambda a : 0 if a == "Female" else 1)

m_status = {"Single":0,"Married":1,"Divorced":2}
df["Marital Status"] = df["Marital Status"].map(lambda status : m_status[status])

ed_lev = {"High School": 0, "Bachelor's":1,
         "Master's":2, "PhD": 3}
df["Education Level"] = df["Education Level"].map(lambda ed : ed_lev[ed])

occ_dict = {"Unemployed": 0, "Employed":2,
         "Self-Employed":1}
df["Occupation"] = df["Occupation"].map(lambda occ : occ_dict[occ])

loc_dict = {"Rural": 0, "Suburban":1,
         "Urban":2}
df["Location"] = df["Location"].map(lambda loc : loc_dict[loc])

loc_dict = {"Basic": 0, "Comprehensive":1,
         "Premium":2}
df["Policy Type"] = df["Policy Type"].map(lambda loc : loc_dict[loc])

loc_dict = {"Poor": 0, "Average":1,
         "Good":2}
df["Customer Feedback"] = df["Customer Feedback"].map(lambda loc : loc_dict[loc])

loc_dict = {"No": 0, "Yes":1}
df["Smoking Status"] = df["Smoking Status"].map(lambda loc : loc_dict[loc])

loc_dict = {"Rarely": 0, "Monthly":1,"Weekly":2,"Daily":3}
df["Exercise Frequency"] = df["Exercise Frequency"].map(lambda loc : loc_dict[loc])

loc_dict = {"Condo": 0, "Apartment":1,"House":2}
df["Property Type"] = df["Property Type"].map(lambda loc : loc_dict[loc])

df["Policy Start Date"] = pd.to_datetime(df["Policy Start Date"]).dt.year


from sympy import symbols, Add, Mul, sin, cos, Pow
import random

# Define possible operations
operations = [Add, Mul, sin, cos, Pow]

# A Recursive function that'll generate as many equation as we need
def generate_random_expression(trial,variables, operations, node_name):
    #### ---- Here is generated the terminal node of the recursion ---- ####
    # This will allow us to break the recursion
    if trial.suggest_int(node_name+"_stop",0,1):
        # A terminal node will include a coefficient and a variable
        coef = trial.suggest_float(node_name+"_coef", -10, 10)
        var_choosen =  trial.suggest_categorical(node_name+"_variable",
                                          list(range(len(variables))))
        # Send a multiplication between the coefficient and the chosen variable
        return Mul(coef,variables[var_choosen])

    #### ---- Here are generated intermediate nodes of the recursion ---- ####
    
    # We select an operator Add, Mul, sin, cos OR Pow
    operation = trial.suggest_categorical(node_name+"_op",
                                        list(range(len(operations))))
    operation = operations[operation]
    
    if operation in [Add, Mul]:
        # The addition and Multiplication need us to get 2 variable
        # We continue going into the depth of the recursion to find them !
        return operation(generate_random_expression(trial, 
                                variables, operations, node_name+"_op1"),
                         generate_random_expression(trial, 
                                variables, operations, node_name+"_op2"))
    elif operation == Pow:
        # For the Power we break the equation too, but no need to do so
        coef = trial.suggest_int(node_name+"_pow_coef", 0, 4)
        var_choosen =  trial.suggest_categorical(node_name+"_pow_variable",
                                        list(range(len(variables))))
        return Pow(variables[var_choosen],coef)
    else:
        # Here we compute more depth for the sine or cosine
        return operation(generate_random_expression(trial,variables, operations, node_name+"_op"))


from sklearn.metrics import mean_squared_error
from sympy import lambdify, symbols
import numpy as np 
def objectif(trial, X_train, y_train):
    # Here I only have 8 features so I generate 8 variable using SymPy's symbols function
    all_sym = [x for x in symbols(" ".join([f'x{i}' for i in range(X_train.shape[1])]))]
    
    # Here we generate our expression using the previous function
    expr = generate_random_expression(trial,
                all_sym, operations,"node_0")
    # We add an intercept
    intercept = trial.suggest_float("intercept",-10,10)
    expr = Add(expr,intercept)
    func = lambdify(all_sym, expr, modules='numpy')
    y_pred = np.array([func(*x_train) for x_train in X_train])
    score = mean_squared_error(y_train, y_pred)
    return score


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=  train_test_split(df.iloc[:,:-1],df.iloc[:,-1], shuffle=True, train_size=0.7)


import optuna
from optuna.samplers import QMCSampler
study_qmc = optuna.create_study(direction="minimize", 
                sampler=QMCSampler())
study_qmc.optimize(lambda trial : objectif(trial, X_train.to_numpy(),y_train.to_numpy()), 
                n_trials=1_000, n_jobs=4)


study_qmc.best_params


def reconstruct_equation(operations, variables, parametres, node_name):
    #print(parametres, node_name)
    find_variable = lambda list_in, var_name : [v for (k,v) in list_in if k.startswith(var_name)][0]
    try:
        #print([(k.split(node_name),v) for (k,v) in parametres.items() if k.startswith(node_name)])
        parametres = [(k[len(node_name):],v) for (k,v) in parametres.items() if k.startswith(node_name)]
        #print(parametres, node_name)
    except:
        parametres = [(k[len(node_name):],v) for (k,v) in parametres.items() if k.startswith(node_name)]
        print("err_1",parametres, node_name)

    try:
        stop = find_variable(parametres,"_stop")
    except:
        stop = find_variable(parametres,"_stop")
        print("error",parametres)

    if stop:
        coef = find_variable(parametres,"_coef")
        var_choosen = find_variable(parametres, "_variable")
        return Mul(coef, variables[var_choosen])

    operation = find_variable(parametres, "_op")
    operation = operations[operation]
    if operation in [Add, Mul]:
        return operation(reconstruct_equation(operations, variables, {k : v for (k,v) in parametres if k.startswith("_op1")}, "_op1"),
                         reconstruct_equation(operations, variables, {k : v for (k,v) in parametres if k.startswith("_op2")}, "_op2"))
    elif operation == Pow:
        coef = find_variable(parametres,"_pow_coef")
        var_choosen =  find_variable(parametres,"_pow_variable")
        return Pow(variables[var_choosen],coef)
    else:
        return operation(reconstruct_equation(operations, variables, {k : v for (k,v) in parametres if ((k.startswith("_op")) and (not k.startswith("_op1")) and(not k.startswith("_op2")) and (not k == "_op"))}, "_op"))

reconstruct_equation(operations, [x for x in symbols(" ".join([f'x{i}' for i in range(X_train.shape[1])]))],study_qmc.best_params, "node_0")


df.columns[[12,3]]


6.4*np.sin(df["Marital Status"]).unique()


mean_squared_error(y_test, len(y_test)*[np.mean(y_train)])


from sklearn.linear_model import LinearRegression

LR = LinearRegression()

LR.fit(X_train,y_train)
mean_squared_error(y_test, LR.predict(X_test))

