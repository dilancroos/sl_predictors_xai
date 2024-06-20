import pandas as pd

# MICE imputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


def load_data():
    """
    Load the data from the CSV file
    """
    data = pd.read_csv("data/BST_V1toV10.csv", header=0, sep=";")
    colNames = pd.read_csv("data/colNames.csv", sep=";",
                           header=0, index_col=0, encoding='MacRoman')

    for i in data.columns:  # vague
        for j in range(len(colNames.columns)):  # eg 0 - 104
            if (i == colNames.columns[j]):
                data.rename(columns={i: colNames.iloc[0, j]}, inplace=True)
    return data


def clean_data(data):
    """
    Clean the data
    Return: DataFrame
    """
    # Change AGE to categorical
    q2 = "Q2- How old are you?"
    for i in range(len(data[q2])):
        if data[q2][i] > 17:
            if data[q2][i] > 16 and data[q2][i] < 29.9:
                data.loc[i, q2] = 1
            elif data[q2][i] < 39.9:
                data.loc[i, q2] = 2
            elif data[q2][i] < 44.9:
                data.loc[i, q2] = 3
            elif data[q2][i] < 49.9:
                data.loc[i, q2] = 4
            elif data[q2][i] < 55.9:
                data.loc[i, q2] = 5
            else:
                data.loc[i, q2] = 6

    # Remove the "YEAR MMS" coloumn as it is not needed
    data.drop("YEAR MMS", axis=1, inplace=True)

    # Results columns cleaning
    # Change the >2 and 0 values to NaN
    results_cols = [
        "MONS ARRESTS FOR 3 DAYS",
        "STOPS OF 3 TO 5 DAYS",
        "STOPS OF MORE THAN 1 WEEK",
        "ARRESTS OF MORE THAN 1 MONTH",
        "(V5 V9) Sick leave of more than 3 months"
    ]

    for col in results_cols:
        # if the value is greater than 3, change the value to NaN
        data.loc[data[col] > 2, col] = None
        # if the value is 0, change the value to NaN
        data.loc[data[col] == 0, col] = None

    # No response category values changed to NaN
    q58_cols = [
        'Q58- For each of these drinks, indicate whether you consume them:-Every day',
        'Q58- For each of these drinks, indicate whether you consume them:-At least once a week'
    ]

    for col in q58_cols:
        data.loc[data[col] == 4, col] = None

    # Q22 systematic error correction
    # increasing the value by 1
    sys_err_cols = [
        "Q22- Over the last 12 months have you personally experienced one or more of the following events:-An imposed change of position or profession",
        "Q22- Over the last 12 months have you personally experienced one or more of the following events:-A restructuring or reorganization of your service or business",
        "Q22- Over the last 12 months have you personally experienced one or more of the following events:-A social plan, layoffs in your company",
        "Q22- Over the last 12 months have you personally experienced one or more of the following events:-One or more periods of technical unemployment"
    ]

    for col in sys_err_cols:
        for i in range(len(data[col])):
            data.loc[i, col] = data.loc[i, col] + 1

    return data


def mice(data, col):
    """
    Impute the missing values using MICE
    """
    imp = IterativeImputer(max_iter=10, random_state=0)
    # train
    data_train = data.loc[:, col]
    # fit
    imp.fit(data_train)
    # transform
    imputed = pd.DataFrame(imp.transform(data_train), columns=col).round()

    return imputed


def categorise(data):
    vShort_column = [
        'MONS ARRESTS FOR 3 DAYS'
    ]

    short_columns = [
        'STOPS OF 3 TO 5 DAYS'
    ]

    long_columns = [
        'STOPS OF MORE THAN 1 WEEK',
        'ARRESTS OF MORE THAN 1 MONTH',
        '(V5 V9) Sick leave of more than 3 months'
    ]

    data['outcome'] = 0

    for i in range(len(data[long_columns[0]])):
        for j in range(len(long_columns)):
            if data.loc[i, long_columns[j]] == 2:
                data.loc[i, 'outcome'] = 2  # Long Sick Leave
                break

        if data.loc[i, 'outcome'] == 2:
            continue

        for k in range(len(short_columns)):
            if data.loc[i, short_columns[k]] == 2:
                data.loc[i, 'outcome'] = 1  # Short Sick Leave
                break

        if data.loc[i, 'outcome'] == 1:
            continue

        if data.loc[i, vShort_column[0]] == 2:
            data.loc[i, 'outcome'] = 0  # Very Short Sick Leave
            continue

        # if does not fall into any of the above categories, set the value to NaN
        else:
            data.loc[i, 'outcome'] = 0

    # drop the columns that were used to create the outcome column
    data.drop(vShort_column + short_columns +
              long_columns, axis=1, inplace=True)

    # all columns are categorical columns except for the outcome column

    # categorical_cols = data.columns.tolist()
    # categorical_cols_wo = categorical_cols.remove('outcome')
    # encode = pd.get_dummies(data, columns=categorical_cols_wo, prefix="cat")

    # # add the encoded to data and remove the original columns
    # data = pd.concat([data, encode], axis=1)
    # data.drop(categorical_cols_wo, axis=1, inplace=True)

    # data = data.astype(int)

    return data


def time_e(st, et, v="cell"):
    """
    Calculate the elapsed time
    st: start time
    et: end time
    v: print statement
    """
    minutes, seconds = divmod(et - st, 60)
    return f"Elapsed time to compute {v}: {minutes:.0f} minutes and {seconds:.0f} seconds"
