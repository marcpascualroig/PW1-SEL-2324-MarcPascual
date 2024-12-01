from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.decomposition import PCA
import os

def pre_processing(file_name):
    current_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    file = os.path.join(current_dir, "Data", file_name)
    if file_name == "ContactLens.csv":
        df = pd.read_csv(file)
        print(len(df))
        df_shuffled = df.sample(frac=1, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(df_shuffled.iloc[:, :-1], df_shuffled.iloc[:, -1],
                                                            test_size=0.3, random_state=42)
        return X_train, X_test, y_train, y_test

    elif file_name == "mushroom.csv":
        df = pd.read_csv(file)
        print(len(df))
        df_shuffled = df.sample(frac=1, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(df_shuffled.iloc[:, :-1], df_shuffled.iloc[:, -1],
                                                            test_size=0.3, random_state=42)
        return X_train, X_test, y_train, y_test


    elif file_name == "winequality-white.csv":
        df = pd.read_csv("winequality-white.csv", delimiter=";", header=0)
        df_shuffled = df.sample(frac=1, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(df_shuffled.iloc[:, :-1], df_shuffled.iloc[:, -1],
                                                            test_size=0.3, random_state=42)
        pca = PCA(n_components=2)  # You can change the number of components as needed
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca= pca.transform(X_test)
        total_variance_explained = sum(pca.explained_variance_ratio_)
        print("Total variance explained by PCA:", total_variance_explained)
        X_train_indices = X_train.index
        X_test_indices = X_test.index

        X_train = pd.DataFrame(X_train_pca, index=X_train_indices,
                               columns=[f'PC{i}' for i in range(1, pca.n_components_ + 1)])
        X_test = pd.DataFrame(X_test_pca, index=X_test_indices,
                              columns=[f'PC{i}' for i in range(1, pca.n_components_ + 1)])

        for col in X_train.select_dtypes(include=['float64']):
            new_col_name = col + '_buckets'
            X_train[new_col_name] = pd.cut(X_train[col], bins=50)
            X_train.drop(col, axis=1, inplace=True)

        for col in X_test.select_dtypes(include=['float64']):
            new_col_name = col + '_buckets'
            X_test[new_col_name] = pd.cut(X_test[col], bins=50)
            X_test.drop(col, axis=1, inplace=True)

        return X_train, X_test, y_train, y_test


    elif file_name == 'ENB2012_data.xlsx':
        df = pd.read_excel(file)
        print(len(df))
        new_columns = []
        for col in df.columns:
            if col == "Y1":
                new_col_name = col + '_buckets'
                df[new_col_name] = pd.cut(df[col], bins=15)
                new_columns.append(new_col_name)
                # Remove original float columns
                df.drop(col, axis=1, inplace=True)
            if col == "Y2":
                df.drop(col, axis=1, inplace=True)

        df = df[[col for col in df.columns if col not in new_columns] + new_columns]
        df_shuffled = df.sample(frac=1, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(df_shuffled.iloc[:, :-1], df_shuffled.iloc[:, -1],
                                                            test_size=0.3, random_state=42)
        return X_train, X_test, y_train, y_test

    elif file_name == 'Dry_Bean_Dataset.xlsx':
        df = pd.read_excel(file)
        df = df.sample(frac=1, random_state=42)
        df = df.head(4000)
        df.reset_index(drop=True, inplace=True)

        for col in df.select_dtypes(include=['float64', 'int64']):
            new_col_name = col + '_buckets'
            df[new_col_name] = pd.cut(df[col], bins=6)
            df.drop(col, axis=1, inplace=True)

        #df_shuffled = df.sample(frac=1, random_state=42)

        X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, 1:], df.iloc[:, 0],
                                                            test_size=0.3, random_state=42)
        return X_train, X_test, y_train, y_test


    elif file_name == '.....Dry_Bean_Dataset.xlsx':
        df = pd.read_excel(file_name)
        df = df.sample(frac=1, random_state=42)
        df = df.head(4000)
        df.reset_index(drop=True, inplace=True)

        X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-1], df.iloc[:, -1],
                                                            test_size=0.3, random_state=42)

        pca = PCA(n_components=3)  # You can change the number of components as needed
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)
        total_variance_explained = sum(pca.explained_variance_ratio_)
        print("Total variance explained by PCA:", total_variance_explained)
        X_train_indices = X_train.index
        X_test_indices = X_test.index

        X_train = pd.DataFrame(X_train_pca, index=X_train_indices,
                               columns=[f'PC{i}' for i in range(1, pca.n_components_ + 1)])
        X_test = pd.DataFrame(X_test_pca, index=X_test_indices,
                              columns=[f'PC{i}' for i in range(1, pca.n_components_ + 1)])

        for col in X_train.select_dtypes(include=['float64', 'int64']):
            new_col_name = col + '_buckets'
            bins = pd.cut(X_train[col], bins=30, retbins=True)[1]
            X_train[new_col_name] = pd.cut(X_train[col], bins=bins)
            X_test[new_col_name] = pd.cut(X_test[col], bins=bins)
            X_train.drop(col, axis=1, inplace=True)
            X_test.drop(col, axis=1, inplace=True)

        return X_train, X_test, y_train, y_test

