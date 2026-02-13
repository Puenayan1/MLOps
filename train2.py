import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# ============================
# 1. CARGA
# ============================
df = pd.read_csv('palmerpenguins_extended.csv')

# ============================
# 2. LIMPIEZA
# ============================
df = df.drop_duplicates()
df = df.dropna()

# ============================
# 3. TRANSFORMACIÓN
# ============================
categorical_cols = ['island', 'sex', 'diet', 'life_stage', 'health_metrics']
encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

target_le = LabelEncoder()
df['species'] = target_le.fit_transform(df['species'])

# ============================
# 4. VALIDACIÓN DE TIPOS
# ============================
print("Tipos de datos:", df.dtypes)

# ============================
# 5. INGENIERÍA DE CARACTERÍSTICAS
# ============================
X = df.drop('species', axis=1)
y = df['species']

scaler = StandardScaler()
numerical_cols = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'year']
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

# ============================
# 6. DIVISIÓN
# ============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ============================
# 7. MODELO
# ============================
model = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    solver='adam',
    max_iter=1000,
    random_state=42
)

# ============================
# 8. ENTRENAMIENTO
# ============================
print("Entrenando red neuronal...")
model.fit(X_train, y_train)

# ============================
# 9. VALIDACIÓN
# ============================
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nPrecisión del modelo (Red Neuronal): {accuracy:.2f} ({accuracy*100:.2f}%)")
print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred, target_names=target_le.classes_))

# ============================
# 10. GUARDAR MODELO + TRANSFORMADORES
# ============================
bundle = {
    "model": model,
    "scaler": scaler,
    "encoders": encoders,
    "target_encoder": target_le,
    "numerical_cols": numerical_cols,
    "categorical_cols": categorical_cols
}

joblib.dump(bundle, "modelo_penguins.pkl")
print("\nModelo y transformadores guardados en modelo_penguins.pkl")

# ============================
# 10. inferir
# ============================

def predecir_penguin(
    bill_length_mm,
    bill_depth_mm,
    flipper_length_mm,
    body_mass_g,
    year,
    island,
    sex,
    diet,
    life_stage,
    health_metrics,
    ruta_modelo="modelo_penguins.pkl"
):
    bundle = joblib.load(ruta_modelo)

    model = bundle["model"]
    scaler = bundle["scaler"]
    encoders = bundle["encoders"]
    target_le = bundle["target_encoder"]
    numerical_cols = bundle["numerical_cols"]
    categorical_cols = bundle["categorical_cols"]

    # --- 1. Crear DF con los datos de entrada ---
    df = pd.DataFrame([{
        "bill_length_mm": bill_length_mm,
        "bill_depth_mm": bill_depth_mm,
        "flipper_length_mm": flipper_length_mm,
        "body_mass_g": body_mass_g,
        "year": year,
        "island": island,
        "sex": sex,
        "diet": diet,
        "life_stage": life_stage,
        "health_metrics": health_metrics
    }])

    # --- 2. Manejo de categorías desconocidas ---
    for col in categorical_cols:
        le = encoders[col]
        valor = df[col].iloc[0]

        if valor not in le.classes_:
            print(f"Advertencia: valor desconocido '{valor}' en columna '{col}'. Se asigna -1.")
            df[col] = -1
        else:
            df[col] = le.transform(df[col])

    # --- 3. Escalar numéricos ---
    df[numerical_cols] = scaler.transform(df[numerical_cols])

    # --- 4. REORDENAR COLUMNAS EXACTAMENTE COMO EN ENTRENAMIENTO ---
    columnas_entrenamiento = list(model.feature_names_in_)
    df = df[columnas_entrenamiento]

    # --- 5. Inferencia ---
    pred = model.predict(df)[0]
    especie = target_le.inverse_transform([pred])[0]

    return especie


# ============================
# 10. probar función de inferencia
# ============================
resultado = predecir_penguin(
    bill_length_mm=45.1,
    bill_depth_mm=14.5,
    flipper_length_mm=210,
    body_mass_g=4800,
    year=2009,
    island="Biscoe",
    sex="male",
    diet="krill",
    life_stage="adult",
    health_metrics="heathy"
)

print("Especie predicha:", resultado)