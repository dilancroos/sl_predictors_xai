import pandas as pd
from time import time

# MICE imputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE

# Makes sure we see all columns
pd.set_option('display.max_columns', None)

data = pd.read_csv("data/BST_V1toV10.csv", header=0, sep=";")
colNames = pd.read_csv("data/colNames.csv", sep=";",
                       header=0, index_col=0, encoding='MacRoman')
retained_cols = pd.read_csv("data/retained.csv", header=None)

# non-catagorical columns
not_cat = [
    "Q4- (3 to 6 years old) In each of the following age groups, how many children live totally or partially with you?",
    "Q4- (7 to 12 years old) In each of the following age groups, how many children live totally or partially with you?",
    "Q4- (13 to 17 years old) In each of the following age groups, how many children live totally or partially with you?",
    "Q4- (18 years and over) In each of the following age groups, how many children live totally or partially with you?",
    "outcome"
]


def time_e(st, et, v="cell"):
    """
    Calculate the elapsed time
    st: start time
    et: end time
    v: print statement
    """
    minutes, seconds = divmod(et - st, 60)

    return f"Elapsed time to compute {v}: {minutes:.0f} minutes and {seconds:.0f} seconds"


def age_cat(data=data):
    """
    Categorise the age
    Return: DataFrame

    ---
    data: DataFrame

    """

    # check each value in the "Q2" column, if the value in that row is > 17, change the value to the category in the dataset
    # bins=[16, 29.9, 39.9, 44.9, 49.9, 55.9, float('inf')], labels=[1, 2, 3, 4, 5, 6]
    t1 = time()
    for i in range(len(data["Q2"])):
        if data["Q2"][i] > 17:
            if data["Q2"][i] > 16 and data["Q2"][i] < 29.9:
                data['Q2'][i] = 1
            elif data["Q2"][i] > 29.9 and data["Q2"][i] < 39.9:
                data['Q2'][i] = 2
            elif data["Q2"][i] > 39.9 and data["Q2"][i] < 44.9:
                data['Q2'][i] = 3
            elif data["Q2"][i] > 44.9 and data["Q2"][i] < 49.9:
                data['Q2'][i] = 4
            elif data["Q2"][i] > 49.9 and data["Q2"][i] < 55.9:
                data['Q2'][i] = 5
            elif data["Q2"][i] > 55.9 and data["Q2"][i] < float('inf'):
                data['Q2'][i] = 6
    t2 = time()
    print(time_e(t1, t2, v="age categorisation"))
    return data


def correct_sys_err(data):
    """
    Correct the systematic error
    Return: DataFrame

    ---
    data: DataFrame

    """

    # Correct the systematic in the sys_err_cols coloumns by going through each coloumn and then each value and increasing the value by 1
    t1 = time()
    sys_err_cols = [
        "Q22_2",
        "Q22_3",
        "Q22_4",
        "Q22_5"
    ]

    for col in sys_err_cols:
        for i in range(len(data)):
            data[col][i] = data[col][i] + 1
    t2 = time()
    print(time_e(t1, t2, v="correct systematic error"))
    return data


def load_col_names(data=data, colNames=colNames):
    """
    Load the data from the CSV file
    Return: DataFrame

    ---
    data: DataFrame = data/BST_V1toV10.csv
    colNames: DataFrame = data/colNames.csv
    """

    st = time()
    for i in data.columns:  # vague
        for j in range(len(colNames.columns)):  # eg 0 - 104
            if (i == colNames.columns[j]):
                data.rename(columns={i: colNames.iloc[0, j]}, inplace=True)
    et = time()
    print(time_e(st, et, v="load column names"))

    return data


def clean_data(data, retained=False):
    """
    Clean the data
    Return: DataFrame

    ---
    data: DataFrame
    retained: bool = False (Default) -> if True, only the retained columns are used

    """
    st = time()

    if not retained:

        # Remove the "YEAR MMS" coloumn as it is not needed
        data.drop("YEAR MMS", axis=1, inplace=True)

    et = time()
    print(time_e(st, et, v="clean data"))

    return data


def mice(data, columns=data.columns, clip=False):
    """
    Impute the missing values using MICE
    Return: DataFrame

    ---
    data: DataFrame
    col: list = column names

    """
    # Initialize the IterativeImputer
    imputer = IterativeImputer(random_state=100, max_iter=10,
                               n_nearest_features=1, sample_posterior=True, min_value=1)

    # train
    df_train = data.loc[:, columns]
    # fit
    imputer.fit(df_train)
    # transform
    df_imputed = imputer.transform(df_train)
    df_imputed = pd.DataFrame(df_imputed).round()

    if clip:
        df_imputed = df_imputed.clip(lower=1)

    # replace the original dataset with the imputed dataset
    data.loc[:, columns] = df_imputed

    return data


def categorise(data, string_issue=False):
    """
    Categorise the data
    Return: DataFrame

    ---
    data: DataFrame

    """
    # for i in data.columns:
    #     print(data.columns.get_loc(i), i)

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

    t1 = time()
    for i in range(len(data)):
        for j in range(len(long_columns)):
            if data[long_columns[j]][i] == 2:
                data['outcome'][i] = 3  # Long Sick Leave
                break

        if data.loc[i, 'outcome'] == 3:
            continue

        for k in range(len(short_columns)):
            if data.loc[i, short_columns[k]] == 2:
                data.loc[i, 'outcome'] = 2  # Short Sick Leave
                break

        if data.loc[i, 'outcome'] == 2:
            continue

        if data.loc[i, vShort_column[0]] == 2:
            data.loc[i, 'outcome'] = 1  # Very Short Sick Leave
            continue

        # if does not fall into any of the above categories, set the value to NaN
        else:
            data.loc[i, 'outcome'] = 0
    t2 = time()
    print(time_e(t1, t2, v="categorisation of outcome column"))

    if string_issue == False:
        t3 = time()
        for i in data.columns:  # vague
            if i in not_cat:
                continue
            for j in range(len(colNames.columns)):  # eg 0 - 104
                opt = int(colNames.iloc[2, j])  # opt row value
                if (i == colNames.iloc[0, j]):  # j = column index
                    for k in range(1, opt + 1):  # 1 to "opt" row value +1
                        if pd.isnull(colNames.iloc[k, j]):
                            break
                        for l in range(len(data)):  # 0 - 45000+ (rows)
                            if data[i][l] == k:  # Q1...== 0, 1 == 1
                                # k+1 because "cat" and "opt" rows is row 1 and 2
                                data[i][l] = colNames.iloc[k + 2, j]
                            if not isinstance(data[i][l], str) and not pd.isnull(data[i][l]):
                                if int(data[i][l]) > opt:
                                    data[i][l] = None
                                elif data[i][l] == 0:
                                    data[i][l] = None
        t4 = time()
        print(time_e(t3, t4, v="change values in catagorical columns"))

    else:  # if string_issue == True
        t3 = time()
        for i in data.columns:  # vague
            if i in not_cat:
                continue
            for j in range(len(colNames.columns)):  # eg 0 - 104
                if (i == colNames.iloc[0, j]):  # j = column index
                    opt = int(colNames.iloc[2, j])  # opt row value
                    for k in range(1, opt + 1):  # 1 to "opt" row value +1
                        if pd.isnull(colNames.iloc[k, j]):
                            break
                        for l in range(len(data)):  # 0 - 45000+ (rows)
                            if not isinstance(data[i][l], str) and not pd.isnull(data[i][l]):
                                if int(data[i][l]) > opt:
                                    data[i][l] = None
                                elif data[i][l] == 0:
                                    data[i][l] = None
        t4 = time()
        print(time_e(t3, t4, v="change values > opt"))

    # drop the columns that were used to create the outcome column
    data.drop(vShort_column + short_columns +
              long_columns, axis=1, inplace=True)

    return data


def one_hot_encode(data):
    """
    One hot encode the data
    Return: DataFrame

    ---
    data: DataFrame

    """

    # catagorical columns (everything other than the non_categorical columns)
    categorical_cols = [col for col in data.columns if col not in not_cat]

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


def main(dataV=data, retained=False, one_hot=False, string_issue=False):
    """
    Main function
    retained: True/False 
    one_hot: True/False
    string_issue: True/False
    Return: DataFrame

    ---
    data: DataFrame - (Default) data/BST_V1toV10.csv
    retained: bool  - (Default) False 
                    -> if True, only the retained columns are used
    one_hot: bool - (Default) False
                    -> if True, one hot encoding is used
    string_issue: bool - (Default) False
                    -> if True, the data will NOT be changed to string values.

    """
    st = time()

    # Age categorisation
    dataV = age_cat(dataV)

    # Correct the systematic error
    dataV = correct_sys_err(dataV)

    # Load the column names
    dataV = load_col_names(dataV)

    # Clean the data
    dataV = clean_data(dataV, retained=retained)

    if retained == False:
        mst = time()
        # Using MICE to impute missing values
        columns = dataV.columns
        data = mice(dataV, columns)
        met = time()
        print(time_e(mst, met, v="complete MICE imputation"))

    # Categorise the data
    dataV = categorise(dataV, string_issue)

    if one_hot:
        # One hot encode the data
        dataV = one_hot_encode(dataV)

    et = time()
    print(time_e(st, et, v="Full process"))

    return dataV


def train_random_forests(X_train, y_train, X_test, y_test, num_forests=1, num_trees=100):
    """
    Train Random Forests
    Return: List, List

    ---
    X_train: DataFrame - training data - features
    y_train: DataFrame - training labels - target
    X_test: DataFrame - testing data - features
    y_test: DataFrame - testing labels - target
    num_forests: int - number of forests to train (Default) 1
    num_trees: int - number of trees in each forest (Default) 100

    """

    models = []
    test_accuracies = []

    for i in range(num_forests):
        # Initialize the Random Forest model
        t1 = time()
        rf = RandomForestClassifier(
            n_estimators=num_trees, random_state=i, class_weight='balanced')

        # Train the model
        rf.fit(X_train, y_train)

        # Predict using the trained model
        y_pred = rf.predict(X_test)

        # Predict probabilities
        y_pred_proba = rf.predict_proba(X_test)

        # Calculate accuracy (or any other performance metric)

        test_accuracy = accuracy_score(y_test, y_pred)
        train_accuracy = accuracy_score(y_train, rf.predict(X_train))
        # confusion matrix: \n{confusion_matrix(y_test, y_pred)} \n

        # Store the trained model and its accuracy
        models.append(rf)
        test_accuracies.append(test_accuracy)

        print(
            f"Forest {i+1}/{num_forests} trained with:")
        print(f"F1 score: {f1_score(y_test, y_pred, average='macro')}")
        print(f"test accuracy: {test_accuracy:.4f}")
        print(f"train accuracy: {train_accuracy:.4f}")
        print(f"ROAUC: {roc_auc_score(
            y_test, y_pred_proba[:, 1], multi_class='ovo')}")
        print(f"Classification report:\n{
              classification_report(y_test, y_pred)}")

        t2 = time()
        print(time_e(t1, t2, v=f"Random Forest {i+1}/{num_forests}"))

    return models, test_accuracies


def smote(X_train, y_train):
    """
    Apply SMOTE to the training data to balance the classes
    Return: DataFrame, DataFrame 

    ---
    X_train: DataFrame - training data - features
    y_train: DataFrame - training labels - target

    """
    t1 = time()
    # Apply SMOTE to the training data
    smote = SMOTE(random_state=42, sampling_strategy='minority')
    # Resample the training data only to prevent data leakage
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    t2 = time()

    print(f"y_train: \n{y_train_res.value_counts()}")

    print(time_e(t1, t2, v="oversampling using SMOTE"))

    return X_train_res, y_train_res
