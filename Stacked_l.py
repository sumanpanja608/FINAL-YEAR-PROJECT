# === Import Libraries ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LinearRegression 
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

# === Load Dataset ===
data = pd.read_csv('D:/Final Year Project/Independent_expanded_with_Actual_Bandgap.csv')
X = data[['MA', 'FA', 'Cs', 'Cl', 'Br', 'I', 'Pb', 'Sn']]
y = data['Bandgap']

# === Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Feature Scaling ===
scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()

# === Define Deep Models ===
def build_mlp():
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def build_transformer():
    input_layer = Input(shape=(X_train.shape[1],))
    x = Dense(64, activation='relu')(input_layer)
    x = Dense(64, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    output = Dense(1)(x)
    model = Model(inputs=input_layer, outputs=output)
    model.compile(optimizer='adam', loss='mse')
    return model

# === Train Base Models ===
ridge = Ridge().fit(X_train, y_train)
svr = SVR().fit(X_train, y_train)
rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
knn = KNeighborsRegressor(n_neighbors=5).fit(X_train, y_train)

mlp = build_mlp()
mlp.fit(X_train, y_train_scaled, epochs=100, batch_size=32, verbose=0)
mlp_preds_scaled = mlp.predict(X_test).ravel()
mlp_preds = scaler_y.inverse_transform(mlp_preds_scaled.reshape(-1, 1)).ravel()

transformer = build_transformer()
transformer.fit(X_train, y_train_scaled, epochs=100, batch_size=32, verbose=0)
trans_preds_scaled = transformer.predict(X_test).ravel()
trans_preds = scaler_y.inverse_transform(trans_preds_scaled.reshape(-1, 1)).ravel()

# === Get Predictions from Base Models ===
ridge_preds = ridge.predict(X_test)
svr_preds = svr.predict(X_test)
rf_preds = rf.predict(X_test)
knn_preds = knn.predict(X_test)

# === Stack Predictions as Meta Features ===
meta_X = np.column_stack((ridge_preds, svr_preds, rf_preds, knn_preds, mlp_preds, trans_preds))

# === Meta Learner: Linear Regression ===
meta_lr = LinearRegression()
meta_lr.fit(meta_X, y_test.values)

# === Final Prediction from Meta Model ===
stacked_lr_preds = meta_lr.predict(meta_X)

# === Evaluation ===
rmse_lr = np.sqrt(mean_squared_error(y_test.values, stacked_lr_preds))
r2_lr = r2_score(y_test.values, stacked_lr_preds)
pearson_lr = pearsonr(y_test.values, stacked_lr_preds)[0]
spearman = spearmanr(y_test, stacked_lr_preds)[0]

print(f"\nStacked Ensemble (Linear Regression Meta Learner):\nRMSE: {rmse_lr:.6f}, R²: {r2_lr:.6f}, Pearson: {pearson_lr:.6f}, Spearman: {spearman:.6f}")

# Function to filter close points to avoid duplicates in the plot
def filter_close_points(x, y, tolerance=0.02):
    unique_points = []
    for xi, yi in zip(x, y):
        if not any(np.linalg.norm(np.array([xi, yi]) - np.array(p)) < tolerance for p in unique_points):
            unique_points.append((xi, yi))
    return np.array(unique_points)

# Filter the congested points in the plot
filtered_points = filter_close_points(y_test.values, stacked_lr_preds, tolerance=0.02)
filtered_y_test, filtered_y_pred = filtered_points[:, 0], filtered_points[:, 1]

# Scatter plot
plt.figure(figsize=(8, 5))
plt.scatter(filtered_y_test, filtered_y_pred, alpha=0.6, color='b', label='Predicted', marker='o')  # Predicted as circles
plt.scatter(filtered_y_test, filtered_y_test, alpha=0.6, color='r', label='Actual', marker='^')    # Actual as triangles
plt.plot([1, 2.75], [1, 2.75], 'k--', lw=2, label='Ideal Fit')
plt.xlim(1, 2.75)
plt.ylim(1, 2.75)
plt.xlabel('Actual Bandgap', fontsize=8, fontfamily='sans-serif')
plt.ylabel('Predicted Bandgap', fontsize=8, fontfamily='sans-serif')
plt.tick_params(labelsize=8)
plt.legend(fontsize=8, loc='upper left')
plt.tight_layout(pad=1.5)
plt.savefig('Stacked_Linear_Regression_Meta_Learner_Plot.tiff', dpi=1200, format='tiff')
plt.show()
