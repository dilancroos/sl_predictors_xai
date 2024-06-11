import pandas as pd
from time import time

# MICE imputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

data = pd.read_csv("data/BST_V1toV10.csv", header=0, sep=";")
colNames = pd.read_csv("data/colNames.csv", sep=";",
                       header=0, index_col=0, encoding='MacRoman')


def time_e(st, et, v="cell"):
    """
    Calculate the elapsed time
    st: start time
    et: end time
    v: print statement
    """
    minutes, seconds = divmod(et - st, 60)

    return f"Elapsed time to compute {v}: {minutes:.0f} minutes and {seconds:.0f} seconds"


def load_data(data=data, colNames=colNames):
    """
    Load the data from the CSV file
    """
    st = time()
    for i in data.columns:  # vague
        for j in range(len(colNames.columns)):  # eg 0 - 104
            if (i == colNames.columns[j]):
                data.rename(columns={i: colNames.iloc[0, j]}, inplace=True)
    et = time()
    print(time_e(st, et, v="data loading"))
    return data


def clean_data(data):
    """
    Clean the data
    Return: DataFrame
    """
    st = time()
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
    et = time()
    print(time_e(st, et, v="data cleaning"))

    return data


def mice(data, col):
    """
    Impute the missing values using MICE
    """
    st = time()
    imp = IterativeImputer(max_iter=10, random_state=0)
    # train
    data_train = data.loc[:, col]
    # fit
    imp.fit(data_train)
    # transform
    imputed = pd.DataFrame(imp.transform(data_train), columns=col).round()
    et = time()
    print(time_e(st, et, v="MICE imputation"))

    return imputed


def categorise(data):
    vShort_column = [
        'MONS ARRESTS FOR 3 DAYS'
    ]

    short_columns = [
        'STOPS OF 3 TO 5 DAYS',
        'STOPS OF MORE THAN 1 WEEK'
    ]

    long_columns = [
        'ARRESTS OF MORE THAN 1 MONTH',
        '(V5 V9) Sick leave of more than 3 months'
    ]

    data['outcome'] = 0

    t1 = time()
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
    t2 = time()
    print(time_e(t1, t2, v="categorisation of outcome column"))

    # non-catagorical columns
    not_to_cat = [
        "Q4- (3 to 6 years old) In each of the following age groups, how many children live totally or partially with you?",
        "Q4- (7 to 12 years old) In each of the following age groups, how many children live totally or partially with you?",
        "Q4- (13 to 17 years old) In each of the following age groups, how many children live totally or partially with you?",
        "Q4- (18 years and over) In each of the following age groups, how many children live totally or partially with you?",
        "outcome"
    ]

    t3 = time()
    for i in data.columns:  # vague
        if i in not_to_cat:
            continue
        for j in range(len(colNames.columns)):  # eg 0 - 104
            if (i == colNames.iloc[0, j]):  # j = column index
                for k in range(12):  # 0 - 11 (rows)
                    if pd.isnull(colNames.iloc[k, j]):
                        break
                    for l in range(len(data)):  # 0 - 45000+ (rows)
                        if data[i][l] == k:  # Q1...== 0, 1 == 1
                            data[i][l] = colNames.iloc[k, j]
    t4 = time()
    print(time_e(t3, t4, v="change values in catagorical columns"))

    # drop the columns that were used to create the outcome column
    data.drop(vShort_column + short_columns +
              long_columns, axis=1, inplace=True)

    # catagorical columns (everything other than the non_categorical columns)
    categorical_cols = [col for col in data.columns if col not in not_to_cat]

    from sklearn.preprocessing import OneHotEncoder

    t5 = time()
    # Initialize the OneHotEncoder
    # drop='first' to avoid multicollinearity
    encoder = OneHotEncoder(drop='if_binary', dtype='int')

    # Fit and transform the categorical columns
    encoded_array = encoder.fit_transform(data[categorical_cols])

    # convert encoder.get_feature_names_out() numpy.ndarray object to an array
    encoded_array = encoded_array.toarray()

    # Create a DataFrame from the encoded array
    encoded_df = pd.DataFrame(
        encoded_array, columns=[encoder.get_feature_names_out()])

    data1 = data.drop(categorical_cols, axis=1)

    # concatinate the encoded to not_to_cat columns
    data = pd.concat([data1, encoded_df], axis=1)

    t6 = time()
    print(time_e(t5, t6, v="OneHotEncoding"))

    return data


def main():
    """
    Main function
    """
    st = time()
    # Load the data
    data = load_data()

    # Clean the data
    data = clean_data(data)

    mst = time()
    # Using MICE to impute missing values
    for col in data.columns:
        impulated = mice(data, [col])
        data.loc[:, col] = impulated
    met = time()
    print(time_e(mst, met, v="complete MICE imputation"))

    # Categorise the data
    data = categorise(data)
    et = time()
    print(time_e(st, et, v="Full process"))

    return data
