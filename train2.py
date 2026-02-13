import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# --- 1. CARGA ---
df = pd.read_csv('palmerpenguins_extended.csv')

# --- 2. LIMPIEZA ---
df = df.drop_duplicates()
df = df.dropna()

# --- 3. TRANSFORMACIÓN ---
# Las redes neuronales requieren entradas numéricas.
categorical_cols = ['island', 'sex', 'diet', 'life_stage', 'health_metrics']
encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

target_le = LabelEncoder()
df['species'] = target_le.fit_transform(df['species'])

# --- 4. VALIDACIÓN (DE DATOS) ---
# Aseguramos que todo sea numérico
print("Tipos de datos:", df.dtypes)

# --- 5. INGENIERÍA DE CARACTERÍSTICAS ---
X = df.drop('species', axis=1)
y = df['species']

# IMPORTANTE: Las redes neuronales son muy sensibles a la escala de los datos.
# Normalizar (StandardScaler) es fundamental aquí, a diferencia de los árboles de decisión donde es opcional.
scaler = StandardScaler()
numerical_cols = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'year']
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

# --- 6. DIVISIÓN ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score

# --- 7. CONSTRUCCIÓN ---
# Usamos MLPClassifier (Multi-Layer Perceptron)
# hidden_layer_sizes=(64, 32): Dos capas ocultas, la primera con 64 neuronas y la segunda con 32.
# activation='relu': Función de activación estándar para capas ocultas.
# max_iter=1000: Damos suficientes iteraciones para asegurar que la red aprenda (converja).
model = MLPClassifier(hidden_layer_sizes=(64, 32), 
                      activation='relu', 
                      solver='adam', 
                      max_iter=1000, 
                      random_state=42)

# --- 8. ENTRENAMIENTO ---
print("Entrenando red neuronal...")
model.fit(X_train, y_train)

# --- 9. VALIDACIÓN (DEL MODELO) ---
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nPrecisión del modelo (Red Neuronal): {accuracy:.2f} ({(accuracy*100):.2f}%)")

print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred, target_names=target_le.classes_))

