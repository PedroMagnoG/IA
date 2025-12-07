from heapq import heappush, heappop
from collections import deque
import time

# ---------------------------------------------------------
# Representação do estado
# ---------------------------------------------------------

GOAL = "123456780"  # estado objetivo (0 = espaço vazio)

def movimentos(pos):
    moves = {
        0: [1, 3],
        1: [0, 2, 4],
        2: [1, 5],
        3: [0, 4, 6],
        4: [1, 3, 5, 7],
        5: [2, 4, 8],
        6: [3, 7],
        7: [4, 6, 8],
        8: [5, 7]
    }
    return moves[pos]

# ---------------------------------------------------------
# Geração de sucessores
# ---------------------------------------------------------

def sucessores(estado):
    s = list(estado)
    pos0 = s.index('0')
    result = []

    for mov in movimentos(pos0):
        new_s = s[:]
        new_s[pos0], new_s[mov] = new_s[mov], new_s[pos0]
        result.append("".join(new_s))

    return result

# ---------------------------------------------------------
# Heurísticas
# ---------------------------------------------------------

def h_misplaced(estado):
    return sum(1 for i in range(9) if estado[i] != GOAL[i] and estado[i] != '0')

def h_manhattan(estado):
    dist = 0
    for i, val in enumerate(estado):
        if val == '0':
            continue
        val = int(val) - 1
        dist += abs(i//3 - val//3) + abs(i%3 - val%3)
    return dist

# ---------------------------------------------------------
# BFS com contadores
# ---------------------------------------------------------

def bfs(start):
    inicio = time.time()

    fila = deque([start])
    visited = {start: None}

    gerados = 1
    expandidos = 0

    while fila:
        atual = fila.popleft()
        expandidos += 1

        if atual == GOAL:
            fim = time.time()
            return visited, expandidos, gerados, fim - inicio

        for suc in sucessores(atual):
            if suc not in visited:
                visited[suc] = atual
                fila.append(suc)
                gerados += 1

    fim = time.time()
    return None, expandidos, gerados, fim - inicio

# ---------------------------------------------------------
# Gulosa com contadores
# ---------------------------------------------------------

def greedy(start, heuristic):
    inicio = time.time()

    pq = []
    heappush(pq, (heuristic(start), start))
    visited = {start: None}

    gerados = 1
    expandidos = 0

    while pq:
        _, atual = heappop(pq)
        expandidos += 1

        if atual == GOAL:
            fim = time.time()
            return visited, expandidos, gerados, fim - inicio

        for suc in sucessores(atual):
            if suc not in visited:
                visited[suc] = atual
                heappush(pq, (heuristic(suc), suc))
                gerados += 1

    fim = time.time()
    return None, expandidos, gerados, fim - inicio

# ---------------------------------------------------------
# A* com contadores
# ---------------------------------------------------------

def astar(start, heuristic):
    inicio = time.time()

    pq = []
    heappush(pq, (heuristic(start), 0, start))
    parent = {start: None}
    cost = {start: 0}

    gerados = 1
    expandidos = 0

    while pq:
        f, g, atual = heappop(pq)
        expandidos += 1

        if atual == GOAL:
            fim = time.time()
            return parent, expandidos, gerados, fim - inicio

        for suc in sucessores(atual):
            new_cost = g + 1
            h = heuristic(suc)
            f2 = new_cost + h

            if suc not in cost or new_cost < cost[suc]:
                cost[suc] = new_cost
                parent[suc] = atual
                heappush(pq, (f2, new_cost, suc))
                gerados += 1

    fim = time.time()
    return None, expandidos, gerados, fim - inicio

# ---------------------------------------------------------
# Reconstrução do caminho
# ---------------------------------------------------------

def reconstruir(parent, estado):
    if parent is None:
        return None
    caminho = []
    while estado is not None:
        caminho.append(estado)
        estado = parent[estado]
    return caminho[::-1]

# ---------------------------------------------------------
# Execução dos algoritmos
# ---------------------------------------------------------

# Entrada
start = input("Digite o estado inicial: ").strip()

def imprimir_tabuleiro(estado):
    print(estado[0:3])
    print(estado[3:6])
    print(estado[6:9])
    print()

if len(start) != 9 or not all(c in "0123456789" for c in start):
    print("Estado inválido! Deve conter 9 dígitos.")
    exit()


print("\n=== A* (Manhattan) ===")
parent, expandidos, gerados, tempo = astar(start, h_manhattan)
if parent is None:
    print("A*: sem solução")
else:
    caminho = reconstruir(parent, GOAL)
    print("Movimentos:", len(caminho) - 1)
    print("Nós expandidos:", expandidos)
    print("Nós gerados:", gerados)
    print("Tempo:", tempo, "s")

    print("\n--- Caminho da Solução (A*) ---")
    for estado in caminho:
        imprimir_tabuleiro(estado)


print("\n=== BFS ===")
parent, expandidos, gerados, tempo = bfs(start)
if parent is None:
    print("BFS: sem solução")
else:
    caminho = reconstruir(parent, GOAL)
    print("Movimentos:", len(caminho) - 1)
    print("Nós expandidos:", expandidos)
    print("Nós gerados:", gerados)
    print("Tempo:", tempo, "s")

    print("\n--- Caminho da Solução (BFS) ---")
    for estado in caminho:
        imprimir_tabuleiro(estado)


print("\n=== Gulosa (Manhattan) ===")
parent, expandidos, gerados, tempo = greedy(start, h_manhattan)
if parent is None:
    print("Gulosa: sem solução")
else:
    caminho = reconstruir(parent, GOAL)
    print("Movimentos:", len(caminho) - 1)
    print("Nós expandidos:", expandidos)
    print("Nós gerados:", gerados)
    print("Tempo:", tempo, "s")

    print("\n--- Caminho da Solução (Gulosa) ---")
    for estado in caminho:
        imprimir_tabuleiro(estado)
