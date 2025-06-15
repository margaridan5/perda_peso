# app.py
import pandas as pd
from flask import Flask, request, jsonify
from joblib import load
from flask_cors import CORS
import traceback

# Carregar os modelos
modelos_mlp = load("modelos_imc_mlp.pkl")  

# Carregar scalers
scaler_mlp_inputs = load("scaler_mlp_zscore_inputs.pkl")
scaler_mlp_outputs = load("scaler_mlp_zscore_outputs.pkl")

# Definir as variáveis
input_vars = ['Género', 'Idade_Cirurgia_anos', 'IMC_inicial', 'Var_Peso_max',
 'Soma_antecedentes', 'Idade_Comorb']

output_vars = ["Var_IMC_0_3", "Var_IMC_3_6", "Var_IMC_6_12",
               "Var_IMC_12_24", "Var_IMC_24_36", "Var_IMC_36_48", "Var_IMC_48_60"]

app = Flask(__name__)
CORS(app)  # permite chamadas de outros domínios

def prever_imc_hibrido_mlp(modelos, paciente_serie, scaler_X, scaler_y, conhecidos=None):
    
    if conhecidos is None:
        conhecidos = {}

    resultado = {}
    
    #input_norm = scaler_X.transform(paciente_serie[input_vars].to_frame().T)
    input_norm = scaler_X.transform(paciente_serie[input_vars])
    input_paciente = pd.DataFrame(input_norm, columns=input_vars)
    
    conhecidos = {k: v for k, v in conhecidos.items() if v is not None}

    vetor_conhecidos = [conhecidos.get(col, 0) for col in output_vars]
    vetor_normalizado = scaler_y.transform([vetor_conhecidos])[0]

    for i, target in enumerate(output_vars):
        if target in conhecidos:
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


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    paciente_dict = data.get("paciente")
    conhecidos = data.get("conhecidos")
    modelo_usado = None
    
    try:
        # Converte o dicionário num DataFrame com uma só linha
        paciente_df = pd.DataFrame([paciente_dict])
        
        resultado = prever_imc_hibrido_mlp(modelos_mlp, paciente_df, scaler_mlp_inputs, scaler_mlp_outputs, conhecidos)
        modelo_usado = "MLP"
        #print("Resultado antes de jsonify:", resultado)

        return jsonify({"success": True, "resultado": resultado, "modelo": modelo_usado})
    except Exception as e:
        print("ERRO DETETADO NO BACKEND:")
        traceback.print_exc()  # <<< imprime o erro completo no terminal
        return jsonify({"success": False, "erro": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)