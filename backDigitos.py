import numpy as np

# =========================
# Funções de ativação
# =========================
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    return x * (1 - x)

# =========================
# Dados dos dígitos 0-9 (7 segmentos)
# =========================
X = np.array([
    [1,1,1,1,1,1,0],  # 0
    [0,1,1,0,0,0,0],  # 1
    [1,1,0,1,1,0,1],  # 2
    [1,1,1,1,0,0,1],  # 3
    [0,1,1,0,0,1,1],  # 4
    [1,0,1,1,0,1,1],  # 5
    [1,0,1,1,1,1,1],  # 6
    [1,1,1,0,0,0,0],  # 7
    [1,1,1,1,1,1,1],  # 8
    [1,1,1,1,0,1,1]   # 9
])

# Saída em 4 bits (one-hot simplificado)
y = np.array([
    [0,0,0,0],  # 0
    [0,0,0,1],  # 1
    [0,0,1,0],  # 2
    [0,0,1,1],  # 3
    [0,1,0,0],  # 4
    [0,1,0,1],  # 5
    [0,1,1,0],  # 6
    [0,1,1,1],  # 7
    [1,0,0,0],  # 8
    [1,0,0,1]   # 9
])

# =========================
# Inicialização dos pesos
# =========================
np.random.seed(42)
W1 = np.random.uniform(-1,1,(5,7))
b1 = np.random.uniform(-1,1,(5,1))
W2 = np.random.uniform(-1,1,(4,5))
b2 = np.random.uniform(-1,1,(4,1))

eta = 0.5
epochs = 10000

# =========================
# Treinamento
# =========================
for epoch in range(epochs):
    # Forward
    z1 = np.dot(W1, X.T) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(W2, a1) + b2
    a2 = sigmoid(z2)
    
    # Backprop
    erro = y.T - a2
    delta2 = erro * sigmoid_deriv(a2)
    delta1 = np.dot(W2.T, delta2) * sigmoid_deriv(a1)
    
    # Atualização dos pesos
    W2 += eta * np.dot(delta2, a1.T) / X.shape[0]
    b2 += eta * np.sum(delta2, axis=1, keepdims=True) / X.shape[0]
    W1 += eta * np.dot(delta1, X) / X.shape[0]
    b1 += eta * np.sum(delta1, axis=1, keepdims=True) / X.shape[0]
    
    # Print do progresso
    if epoch % 2000 == 0:
        loss = np.mean(np.square(erro))
        print(f"Epoca {epoch}, erro medio: {loss:.6f}")

# =========================
# Teste final para todos os dígitos
# =========================
z1 = np.dot(W1, X.T) + b1
a1 = sigmoid(z1)
z2 = np.dot(W2, a1) + b2
a2 = sigmoid(z2)

# Converter para 0/1
saida_final = (a2 > 0.5).astype(int)
print("\nSaida final da rede (bits):")
print(saida_final)

# Converter bits para dígito decimal
digitos_reconhecidos = [int("".join(bits.astype(str)), 2) for bits in saida_final.T]
print("Digitos reconhecidos:", digitos_reconhecidos)

# =========================
# Função para teste manual
# =========================
def testar_digito_manual(X_manual):
    if len(X_manual) != 7:
        print("Entrada inválida! Deve ter 7 segmentos (0 ou 1).")
        return
    
    X_manual = np.array(X_manual).reshape(7,1)
    
    z1 = np.dot(W1, X_manual) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(W2, a1) + b2
    a2 = sigmoid(z2)
    
    a2_bin = (a2 > 0.5).astype(int).flatten()
    digito = int("".join(a2_bin.astype(str)), 2)
    
    print("Saida da rede (bits):", a2_bin)
    print("Digito reconhecido:", digito)

# =========================
# Exemplo de teste manual
# =========================
# Dígito 3: segmentos a,b,c,d,g ligados
X_teste = [0,1,1,0,0,0,0]
testar_digito_manual(X_teste)

# Dígito 0: segmentos a,b,c,d,e,f ligados
X_teste = [1,1,1,1,1,1,1]
testar_digito_manual(X_teste)
