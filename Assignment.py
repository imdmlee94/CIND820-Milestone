import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, OrdinalEncoder



#----------------------------------
#1. Loading & Standardising Column Names
#----------------------------------

# "Consumer Behavior and Shopping Habits Dataset" by zeesolver
df_zeesolver = pd.read_csv('Consumer Behavior and Shopping Habits Dataset.csv')

# "Analyzing Customer Spending Habits" by thedevastator
df_thedevastator = pd.read_csv('Analyzing Customer Spending Habits.csv')

#Deleting useless columns
df_thedevastator.drop(columns=[c for c in ["index", "Column1"] if c in df_thedevastator.columns],
                 inplace=True)

#------------------------------------
#2. Dataset 1 (zeesolver)
#------------------------------------

# Numeric parsing
num_col_zeesolver = ["Age", "Purchase Amount (USD)", "Review Rating", "Frequency of Purchases"]
for col in num_zeesolver:
    if col in df_zeesolver.columns:
        df_zeesolver[col] = pd.to_numeric(df_zeesolver[col], errors="coerce")

# Target label - "MadePurchase" is 1  for any positive-value sale
df_zeesolver["MadePurchase"] = (df_zeesolver["Purchase Amount (USD)"] > 0).astype(int)

# Missing-value imputation (median for Age, mode for Rating)
imputer_age = SimpleImputer(strategy="median")
imputer_rating = SimpleImputer(strategy="most_frequent")
if "Age" in df_zeesolver.columns:
    df_zeesolver[["Age"]] = imputer_age.fit_transform(df_zeesolver[["Age"]])
if "Review Rating" in df_zeesolver.columns:
    df_zeesolver[["Review Rating"]] = imputer_rating.fit_transform(
        df_zeesolver[["Review Rating"]]
    )

#Winsorise purchase amount
q1, q99 = df_zeesolver["Purchase Amount (USD)"].quantile([0.01, 0.99])
df_zeesolver["Purchase Amount (USD)"] = df_zeesolver["Purchase Amount (USD)"].clip(q1, q99)

# One-hot encode all low-cardinality categoricals
cat_zeesolver = ["Gender", "Category", "Season", "Subscription Status", "Payment Method", "Shipping Type", "Discount Applied", "Promo Code Used"]

cat_zeesolver = [c for c in cat_zeesolver if c in df_zeesolver.columns]
ohe_zeesolver = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
matrix_zeesolver = ohe_zeesolver.fit_transform(df_zeesolver[cat_zeesolver])
df_ohe_zeesolver = pd.DataFrame(
    matrix_zeesolver, columns=ohe_zeesolver.get_feature_names_out(cat_zeesolver), index=df_zeesolver.index
)
df_zeesolver_clean = pd.concat([df_zeesolver.drop(columns=cat_zeesolver), df_ohe_zeesolver], axis=1)

# Scale numeric fields
scaler_zeesolver = MinMaxScaler()
df_zeesolver_clean[num_col_zeesolver] = scaler_zeesolver.fit_transform(df_zeesolver_clean[num_col_zeesolver])


#--------------------------------
# Dataset 2 (thedevastator)
#--------------------------------

# Dates & numeric parsing
df_thedevastator["Date"] = pd.to_datetime(df_thedevastator["Date"], format= "%Y-%m-%d", errors="coerce")
num_col_dev = ["Customer Age", "Quantity", "Unit Cost", "Unit Price", "Cost", "Revenue"]
for col in num_col_dev:
    df_thedevastator[col] = pd.to_numeric(df_thedevastator[col], errors="coerce")


# Determining weekend dates
ref_date = df_thedevastator["Date"].max()
df_thedevastator["Weekend"] = df_thedevastator["Date"].dt.dayofweek >= 5

# Target label - "MadePurchase" is 1  for any positive-value sale
df_thedevastator["MadePurchase"] = (df_thedevastator["Revenue"] > 0).astype(int)

# Winsorise skewed numeric columns
for col in ["Revenue", "Quantity"]:
    q1, q99 = df_thedevastator[col].quantile([0.01, 0.99])
    df_thedevastator[col] = df_thedevastator[col].clip(q1, q99)

# Ordinal-encode high-cardinality Country
enc_country = None
if "Country" in df_thedevastator.columns:
    enc_country = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    df_thedevastator["CountryEnc"] = enc_country.fit_transform(
        df_thedevastator[["Country"]]
    )
    df_thedevastator.drop(columns="Country", inplace=True)

# One-hot encode remaining categoricals
cat_dev = ["Customer Gender", "Product Category", "Sub Category", "State"]
cat_dev = [c for c in cat_dev if c in df_thedevastator.columns]
ohe_dev = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
matrix_dev = ohe_dev.fit_transform(df_thedevastator[cat_dev])
df_ohe_dev = pd.DataFrame(
    matrix_dev, columns=ohe_dev.get_feature_names_out(cat_dev),
    index=df_thedevastator.index
)
df_dev_clean = pd.concat([df_thedevastator.drop(columns=cat_dev), df_ohe_dev], axis=1)

#--------------------------
# Saving clean files 
#---------------------------

df_zeesolver_clean.to_parquet("clean_zeesolver.parquet", index=False)
df_dev_clean.to_parquet("clean_thedevastator.parquet", index=False)


