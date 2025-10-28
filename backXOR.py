import numpy as np

# =========================
# Funções de ativação
# =========================
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    return x * (1 - x)

# =========================
# Dados do XOR
# =========================
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
y = np.array([[0], [1], [1], [0]])

# =========================
# Inicialização aleatória
# =========================
np.random.seed(50)
W1 = np.random.uniform(-1, 1, (2, 2))
b1 = np.random.uniform(-1, 1, (2, 1))
W2 = np.random.uniform(-1, 1, (1, 2))
b2 = np.random.uniform(-1, 1, (1, 1))

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
    b1 += np.sum(delta1, axis=1, keepdims=True) / X.shape[0]

    # Print do erro a cada 2000 épocas
    if epoch % 2000 == 0:
        loss = np.mean((y.T - a2)**2)
        print(f"Epoca {epoch}, erro medio: {loss:.6f}")

# =========================
# Teste final de todos os pares do XOR
# =========================
z1 = np.dot(W1, X.T) + b1
a1 = sigmoid(z1)
z2 = np.dot(W2, a1) + b2
a2 = sigmoid(z2)

print("\nSaida final da rede (aproximada):")
print(a2.round(3))

# =========================
# Teste com entradas definidas dentro do código
# =========================
entradas_teste = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]

for entrada in entradas_teste:
    X_manual = np.array(entrada).reshape(2,1)
    z1 = np.dot(W1, X_manual) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(W2, a1) + b2
    a2 = sigmoid(z2)
    
    print(f"\nEntrada: {entrada}")
    print("Saída da rede (aproximada):", a2.round(3))
    print("Saída XOR prevista (0 ou 1):", int(a2 > 0.5))
