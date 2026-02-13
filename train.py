import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# --- 1. CARGA (Paso previo necesario) ---
# Cargamos el dataset para poder trabajar con él
df = pd.read_csv('palmerpenguins_extended.csv')

# --- 2. LIMPIEZA ---
# Eliminamos duplicados para asegurar que no hay datos repetidos que sesguen el modelo
df = df.drop_duplicates()
# Eliminamos filas con valores nulos (si los hubiera) para evitar errores
df = df.dropna()
print("Datos después de limpieza:", df.shape)

# --- 3. TRANSFORMACIÓN ---
# Convertimos las variables categóricas (texto) a numéricas para que la IA las entienda.
# Variables a codificar: island, sex, diet, life_stage, health_metrics y la variable objetivo 'species'
categorical_cols = ['island', 'sex', 'diet', 'life_stage', 'health_metrics']
encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le # Guardamos el encoder por si queremos invertir la transformación luego

# Transformamos la variable objetivo (la especie que queremos predecir)
target_le = LabelEncoder()
df['species'] = target_le.fit_transform(df['species'])

# --- 4. VALIDACIÓN (DE DATOS) ---
# Verificamos que todos los datos sean numéricos y no haya problemas antes de entrenar
print("\nTipos de datos verificados:")
print(df.dtypes)
# Verificación rápida de integridad
assert df.isnull().sum().sum() == 0, "Error: Aún quedan valores nulos."

# --- 5. INGENIERÍA DE CARACTERÍSTICAS ---
# Separamos las características (X) de la etiqueta a predecir (y)
X = df.drop('species', axis=1)
y = df['species']

# Escalamos las variables numéricas para que todas tengan el mismo peso (media 0, desviación 1)
# Esto ayuda a modelos como Regresión Logística o Redes Neuronales (aunque Random Forest es robusto a esto)
scaler = StandardScaler()
numerical_cols = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'year']
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

# --- 6. DIVISIÓN ---
# Dividimos los datos en conjunto de entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nDatos de entrenamiento: {X_train.shape}")
print(f"Datos de prueba: {X_test.shape}")

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# --- 7. CONSTRUCCIÓN ---
# Inicializamos el modelo. Usamos RandomForest porque maneja bien datos tabulares complejos.
# n_estimators=100 significa que usará 100 árboles de decisión.
model = RandomForestClassifier(n_estimators=100, random_state=42)

# --- 8. ENTRENAMIENTO ---
# La IA "aprende" encontrando patrones entre X_train y y_train
model.fit(X_train, y_train)

# --- 9. VALIDACIÓN (DEL MODELO) ---
# Hacemos predicciones sobre datos que el modelo nunca ha visto (X_test)
y_pred = model.predict(X_test)

# Evaluamos qué tan bien funcionó
accuracy = accuracy_score(y_test, y_pred)
print(f"\nPrecisión del modelo: {accuracy:.2f} ({(accuracy*100):.2f}%)")

print("\nReporte de Clasificación:")
# Usamos target_names para mostrar los nombres reales de las especies en lugar de números
print(classification_report(y_test, y_pred, target_names=target_le.classes_))
