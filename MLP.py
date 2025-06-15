import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib

# --- 1. LER DADOS ---
df = pd.read_csv("imc_prepared.csv", index_col="NSC", encoding="latin1", sep=";")

# --- 2. DEFINIR VARIÁVEIS ---

input_vars = ['Género', 'Idade_Cirurgia_anos', 'IMC_inicial', 'Var_Peso_max',
              'Soma_antecedentes', 'Idade_Comorb']

output_vars = ["Var_IMC_0_3", "Var_IMC_3_6", "Var_IMC_6_12",
               "Var_IMC_12_24", "Var_IMC_24_36", "Var_IMC_36_48", "Var_IMC_48_60"]

# --- 3. DIVISÃO TREINO/TESTE ---
df = df.dropna(subset=input_vars + output_vars)
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42, stratify=df["Género"])

# --- 4. SCALING DOS INPUTS ---
scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(df_train[input_vars])  # fit só no treino
X_test = scaler_X.transform(df_test[input_vars])

# --- 5. SCALING DOS OUTPUTS ---
scaler_y = StandardScaler()
Y_train = scaler_y.fit_transform(df_train[output_vars])  # fit só no treino
Y_test = scaler_y.transform(df_test[output_vars])

# --- 5B. Guardar os dados escalados para inspeção ---
df_train_scaled = pd.DataFrame(X_train, columns=input_vars, index=df_train.index)
for i, col in enumerate(output_vars):
    df_train_scaled[col] = Y_train[:, i]

# --- 6. TREINO EM CASCATA COM MLP ---

param_grid = [
    {"hidden_layer_sizes": (100,), "activation": "relu", "alpha": 0.0001},
    {"hidden_layer_sizes": (100,), "activation": "tanh", "alpha": 0.0001},
    {"hidden_layer_sizes": (100, 50), "activation": "relu", "alpha": 0.001},
    {"hidden_layer_sizes": (200,), "activation": "relu", "alpha": 0.01},
    {"hidden_layer_sizes": (50,), "activation": "tanh", "alpha": 0.001},
]

modelos = {}
maes_por_output = {}

for i, target in enumerate(output_vars):
    best_mae = float("inf")
    best_model = None
    best_params = None

    # preparar X/y para este output
    X_treino = pd.DataFrame(X_train, columns=input_vars, index=df_train.index)
    if i > 0:
        for j in range(i):
            X_treino[output_vars[j]] = Y_train[:, j]
    y_treino = Y_train[:, i]

    # preparar X_test
    X_teste = pd.DataFrame(X_test, columns=input_vars, index=df_test.index)
    if i > 0:
        for j in range(i):
            X_teste[output_vars[j]] = Y_test[:, j]
    y_teste = Y_test[:, i]

    for config in param_grid:
        mlp = MLPRegressor(
            hidden_layer_sizes=config["hidden_layer_sizes"],
            activation=config["activation"],
            alpha=config["alpha"],
            max_iter=1000,
            early_stopping=True,
            random_state=42
        )
        mlp.fit(X_treino, y_treino)
        y_pred = mlp.predict(X_teste)

        # desnormalizar só este target
        y_pred_real = scaler_y.inverse_transform(
            [[0]*i + [val] + [0]*(len(output_vars)-i-1) for val in y_pred])[:, i]
        y_real = df_test[target].values
        mae = mean_absolute_error(y_real, y_pred_real)
        if mae < best_mae:
            best_mae = mae
            best_model = mlp
            best_params = config

    modelos[target] = best_model
    maes_por_output[target] = (best_mae, best_params)
    print(f"✅ Melhor para {target}: MAE = {best_mae:.2f} com {best_params}")

joblib.dump(modelos, "modelos_imc_mlp.pkl")  # guarda os modelos


# --- 7. FUNÇÃO DE PREVISÃO EM CASCATA ---
def prever_mlp_cascata(modelos, paciente_serie, scaler_X, scaler_y, conhecidos=None):
    if conhecidos is None:
        conhecidos = {}

    resultado = {}

    input_norm = scaler_X.transform(paciente_serie[input_vars].to_frame().T)
    input_paciente = pd.DataFrame(input_norm, columns=input_vars)

    for i, target in enumerate(output_vars):
        if target in conhecidos:
            vetor_conhecidos = [conhecidos.get(col, 0) for col in output_vars]
            vetor_normalizado = scaler_y.transform([vetor_conhecidos])[0]
            resultado[target] = vetor_normalizado[i]
        else:
            for prev in output_vars[:i]:
                input_paciente[prev] = resultado.get(prev, 0)
            pred = modelos[target].predict(input_paciente)[0]
            resultado[target] = pred

    # Desnormalizar todos os outputs previstos
    valores_norm = [resultado[col] for col in output_vars]
    valores_desnorm = scaler_y.inverse_transform([valores_norm])[0]

    return {col: round(val, 2) for col, val in zip(output_vars, valores_desnorm)}

# --- 8. AVALIAR O MODELO NO TESTE ---
y_true_all = {col: [] for col in output_vars}
y_pred_all = {col: [] for col in output_vars}

# Prever em cascata para cada paciente no teste
for idx, paciente in df_test.iterrows():
    previsao = prever_mlp_cascata(modelos, paciente, scaler_X, scaler_y)
    for col in output_vars:
        y_true_all[col].append(df_test.loc[idx, col])  # valor real na escala original
        y_pred_all[col].append(previsao[col])

# --- 9. ERROS ---
print("\n--- ERROS DE PREVISÃO (MAE) POR VAR_IMC ---")
maes = []
for col in output_vars:
    mae = mean_absolute_error(y_true_all[col], y_pred_all[col])
    maes.append(mae)
    print(f"{col}: MAE = {mae:.2f}")

# --- 10. GRÁFICO ---
cores = ['blue' if m < 0.8 else 'green' if m < 1.5 else 'cornflowerblue' if m < 2.5 else 'orange' for m in maes]
plt.figure(figsize=(10, 5))
plt.bar(output_vars, maes, color=cores)
plt.title("Mean Absolute Error (MAE) of BMI variation per timepoint")
plt.xlabel("Timepoint (intervalo de meses)")
plt.ylabel("MAE (IMC)")
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# --- 11. OPCIONAL: Guardar scalers para uso no site ---
joblib.dump(scaler_X, "scaler_mlp_zscore_inputs.pkl")
joblib.dump(scaler_y, "scaler_mlp_zscore_outputs.pkl")

