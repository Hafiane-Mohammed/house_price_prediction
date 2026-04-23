import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectFromModel

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.metrics import r2_score
import joblib


df = pd.read_csv("kc_house_data.csv")


df.columns = df.columns.str.strip().str.replace(" ", "_")



def clean_data(df):
    df = df.copy()

    df = df.drop_duplicates()


    df = df.loc[:, df.isnull().mean() < 0.6]


    constant_cols = [c for c in df.columns if df[c].nunique() == 1]
    df.drop(columns=constant_cols, inplace=True)

    return df

df = clean_data(df)



df = df[[
    "sqft_living",
    "bedrooms",
    "bathrooms",
    "grade",
    "sqft_above",
    "sqft_living15",
    "floors",
    "price"
]]



X = df.drop(columns=["price"])
y = df["price"]



X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)



def handle_outliers(X):
    X = np.array(X)
    X = X.copy()

    for i in range(X.shape[1]):
        col = X[:, i]

        Q1 = np.percentile(col, 25)
        Q3 = np.percentile(col, 75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        col = np.clip(col, lower, upper)


        if (col > 0).all():
            col = np.log1p(col)

        X[:, i] = col

    return X


numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),  # remplissage dyal null
    ("outliers", FunctionTransformer(handle_outliers)),  # clipping + log
    ("scaler", RobustScaler())  # scaling resistant l outliers
])



preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, X.columns)
])



selector = SelectFromModel(RandomForestRegressor(n_estimators=50))



model = XGBRegressor(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    tree_method="hist",
    n_jobs=-1
)



pipe = Pipeline([
    ("preprocess", preprocessor),
    ("feature_selection", selector),
    ("model", model)
])


pipe.fit(X_train, y_train)



preds = pipe.predict(X_test)

print("R2:", r2_score(y_test, preds))



cv = KFold(n_splits=3, shuffle=True, random_state=42)
cv_score = cross_val_score(pipe, X, y, cv=cv)

print("CV Mean:", cv_score.mean())


joblib.dump(pipe, "model.pkl")

print("Model saved successfully")