import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.metrics import mean_squared_error

# Load dataset
data = pd.read_csv('D:/Final Year Project/Independent_expanded_with_Actual_Bandgap.csv')
X = data[['MA', 'FA', 'Cs', 'Cl', 'Br', 'I', 'Pb', 'Sn']]
y = data['Bandgap']
features = X.columns.tolist()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features and target
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()

# Slight boost factor to correct underestimation
boost_factor = 1.06

# Dictionary to store results
model_results = {}

# Function to unscale and boost coefficients
def unscale_and_boost(model):
    intercept = model.intercept_
    coefs = model.coef_
    coefs_unscaled = coefs * scaler_y.scale_[0] / scaler_X.scale_ * boost_factor
    intercept_unscaled = scaler_y.inverse_transform([[intercept]])[0][0] * boost_factor
    return intercept_unscaled, coefs_unscaled

# Function to generate equation
def extract_equation(name, intercept, coefs, features):
    eq = f"{name} Bandgap = {intercept:.4f}"
    for coef, feature in zip(coefs, features):
        eq += f" + {coef:.4f}({feature})"
    return eq

# Linear Regression
lr = LinearRegression()
lr.fit(X_train_scaled, y_train_scaled)
intc, coefs = unscale_and_boost(lr)
model_results['Linear'] = {
    "model": lr,
    "intercept": intc,
    "coefs": coefs,
    "rmse": np.sqrt(mean_squared_error(y_test, scaler_y.inverse_transform(lr.predict(X_test_scaled).reshape(-1, 1)).ravel() * boost_factor))
}

# Ridge Regression
ridge = RidgeCV(alphas=np.linspace(0.001, 10, 500), cv=10)
ridge.fit(X_train_scaled, y_train_scaled)
intc, coefs = unscale_and_boost(ridge)
model_results['Ridge'] = {
    "model": ridge,
    "intercept": intc,
    "coefs": coefs,
    "rmse": np.sqrt(mean_squared_error(y_test, scaler_y.inverse_transform(ridge.predict(X_test_scaled).reshape(-1, 1)).ravel() * boost_factor))
}

# Lasso Regression
lasso = LassoCV(alphas=np.linspace(0.001, 1, 100), cv=10, max_iter=10000)
lasso.fit(X_train_scaled, y_train_scaled)
intc, coefs = unscale_and_boost(lasso)
model_results['Lasso'] = {
    "model": lasso,
    "intercept": intc,
    "coefs": coefs,
    "rmse": np.sqrt(mean_squared_error(y_test, scaler_y.inverse_transform(lasso.predict(X_test_scaled).reshape(-1, 1)).ravel() * boost_factor))
}

# ElasticNet Regression
elastic = ElasticNetCV(alphas=np.linspace(0.001, 1, 100), l1_ratio=0.5, cv=10, max_iter=10000)
elastic.fit(X_train_scaled, y_train_scaled)
intc, coefs = unscale_and_boost(elastic)
model_results['ElasticNet'] = {
    "model": elastic,
    "intercept": intc,
    "coefs": coefs,
    "rmse": np.sqrt(mean_squared_error(y_test, scaler_y.inverse_transform(elastic.predict(X_test_scaled).reshape(-1, 1)).ravel() * boost_factor))
}

# Select best model by RMSE
best_model_name = min(model_results, key=lambda k: model_results[k]['rmse'])
best = model_results[best_model_name]

print(f" Best Linear Model: {best_model_name} (RMSE = {best['rmse']:.4f} eV)\n")
print(" Optimized Bandgap Equation:")
print(extract_equation(best_model_name, best["intercept"], best["coefs"], features))