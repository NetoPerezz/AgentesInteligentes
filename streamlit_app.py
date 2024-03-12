### Integrantes: Oswaldo Arceo, Ernesto Perez, Brian Avalos

import numpy as np                  #instalar librería: pip install numpy
from tabulate import tabulate       #instalar librería: pip install tabulate
import networkx as nx               #instalar librería: pip install networkx
import matplotlib.pyplot as plt     #instalar librería: pip install matplotlib

print('\n\n\n■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ Inicio Programa Q-Learning "REFORZADO" | Definición de los estados ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■\n')

print('\n■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ Etapa Inicial | Definición de los estados ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■\n')

def_de_estados = {'A': 0,
                  'B': 1,
                  'C': 2,
                  'D': 3,
                  'E': 4,
                  'F': 5,
                  'G': 6,
                  'H': 7,
                  'I': 8,
                  'J': 9,
                  'K': 10,
                  'L': 11,
                  'M': 12,
                  'N': 13,
                  'Ñ': 14,
                  'O': 15,
                  'P': 16,
                  'Q': 17,
                  'R': 18,
                  'S': 19,
                  'T': 20,
                  'U': 21,
                  'V': 22,
                  'W': 23,
                  'X': 24,
                  'Y': 25,
                  'Z': 26,
                  }

print(def_de_estados)


print('\n\n\n■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ Siguiente Etapa | Definición de las acciones ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■\n')

acciones = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]
print(acciones)


print('\n\n\n■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ Siguiente Etapa | Definición MAtríz de las recompensas en cada estado ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■\n')

matriz_recompensas = np.array([
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 9, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 3, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 5, 0, 4, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 6, 0, 7, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 1, 8, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 7, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 4, 0, 0, 0, 0, 8, 0, 0, 0, 6],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 1, 3, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 2, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
])


print('\n\n\n■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ Siguiente Etapa | Punto Intermedio ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■\n')

# Solicitar al usuario un estado intermedio
estado_intermedio = input("Ingresa un estado intermedio (entre A y Z): ").upper()

# Validar que el estado intermedio sea válido
while estado_intermedio not in def_de_estados:
    print("Estado intermedio no válido. Debe ser una letra entre A y Z.")
    estado_intermedio = input("Ingresa un estado intermedio (entre A y Z): ").upper()

# Definir una nueva matriz de recompensas que incluya el estado intermedio
# Se asignan recompensas más altas para las conexiones que involucran el estado intermedio
matriz_recompensas_con_intermedio = matriz_recompensas.copy()

for i in range(26):
    matriz_recompensas_con_intermedio[i, def_de_estados[estado_intermedio]] += 10

print('\n\n■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ Siguiente Etapa | Matrpiz de Caminos accesibles de cada estado ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■\n')

grafo = {
        'A': ['B'],
        'B': ['A', 'C'],
        'C': ['B', 'F'],
        'D': ['G'],
        'E': ['H'],
        'F': ['C', 'G'],
        'G': ['D', 'F', 'H'],
        'H': ['G', 'I'],
        'I': ['H', 'K'],
        'J': ['K'],
        'K': ['I', 'J', 'M', 'L'],
        'L': ['K', 'X'],
        'M': ['K', 'S', 'R'],
        'N': ['Ñ', 'W'],
        'O': ['Y', 'Ñ','U', 'P'],
        'P': ['O', 'T','S'],
        'Q': ['R'],
        'R': ['M', 'Q'],
        'S': ['M', 'P', 'Z'],
        'T': ['U', 'P'],
        'U': ['O', 'T'],
        'V': ['Ñ', 'W', 'Z'],
        'W': ['N', 'V'],
        'X': ['L'],
        'Y': ['O'],
        'Z': ['V', 'S']
    }


print('\n\n\n■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ Función para encontrar la ruta óptima utilizando el grafo de Caminos ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■\n')

def encontrar_ruta_optima(inicial, objetivo, grafo, visitados=None, ruta_actual=None):
    if visitados is None:
        visitados = set()
    if ruta_actual is None:
        ruta_actual = [inicial]
    
    visitados.add(inicial)
    
    if inicial == objetivo:
        return ruta_actual
    
    for vecino in grafo.get(inicial, []):
        if vecino not in visitados:
            
            nueva_ruta = encontrar_ruta_optima(vecino, objetivo, grafo, visitados.copy(), ruta_actual + [vecino]) # Hacemos llamada recursiva con el vecino
            if nueva_ruta is not None:
                return nueva_ruta
    
    return None # Si no se encuentra una ruta desde el estado actuall, regresamos ninguno
    

print('\n\n■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ Siguiente Etapa | parámetros Gamma y Alfa ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■\n')

gamma = 0.25 #valor original 0.75
alpha = 1.9  #valor original 0.9
num_iteraciones = 100 #valor original 1000

Q = np.array(np.zeros([26, 26])) #Inicializamos los valores de toda la matrizQ que siempre debe ser cuadrada n ceros

print('Valor "GAMA" utilizado de {}'.format(gamma))
print('Valor "Alpha" utilizado de {}'.format(alpha))
print('Número de "Iteraciones" utilizado es de {}'.format(num_iteraciones))


print('\n\n■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ Siguiente Etapa | Implementación del proceso de Q-Learning ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■\n')

for i in range(num_iteraciones):
    estado_actual = np.random.randint(0, 26)
    accion_realizable = []
    for j in range(26):
        if matriz_recompensas[estado_actual, j] > 0:
            accion_realizable.append(j)
    if accion_realizable:  # Verificando que la lista no está vacía porque sino me falla
        estado_siguiente = np.random.choice(accion_realizable)
        TD = matriz_recompensas[estado_actual, estado_siguiente] + gamma * Q[estado_siguiente, np.argmax(Q[estado_siguiente,])] - Q[estado_actual, estado_siguiente]
        Q[estado_actual, estado_siguiente] = Q[estado_actual, estado_siguiente] + alpha * TD
    else:#En el caso en el que no haya más acciones realizables desde el estado actual
        continue

# print("Q-Values:")
# print(Q.astype(int))


print('\n\n■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ Siguiente Etapa | Imprimir la Matriz de Q-Learning ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■\n')

print("·································· Matriz Q:\n")
print(tabulate(Q.astype(int), tablefmt='plain')) #se ve mucho mejor


print('\n\n\n■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ Función para encontrar la ruta óptima sin usar lo aprendido ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■\n')

def encontrar_ruta_optima_sin_aprendizaje(inicial, objetivo, grafo):
    ruta_optima = encontrar_ruta_optima(inicial, objetivo, grafo)  # buscamos por los estados alcanzasos
    return ruta_optima

estado_inicial= input("Ingresa el estado inicial: ")
estado_inicial = estado_inicial.upper()

estado_objetivo= input("Ingresa el estado objetivo: ")
estado_objetivo = estado_objetivo.upper()


ruta_optima = encontrar_ruta_optima_sin_aprendizaje(estado_inicial, estado_objetivo, grafo)
if ruta_optima:
    print("Ruta óptima encontrada:", ruta_optima)


    #print('\n■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ plot de ruta ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■\n')

    dib_ruta_optima = ruta_optima

    # Creamos un objeto de grafo dirigido
    G = nx.DiGraph()

    # Agregamos nodos y bordes al grafo
    for nodo, vecinos in grafo.items():
        for vecino in vecinos:
            G.add_edge(nodo, vecino)

    # Creamos una lista de colores para los nodos
    colores = ['blue' if nodo in dib_ruta_optima else 'lightgray' for nodo in G.nodes()]

    # Dibujamos el grafo
    pos = nx.spring_layout(G)  # Posiciones de los nodos
    nx.draw(G, pos, with_labels=True, node_color=colores, arrowsize=20, node_size=2000)

    # Mostramos el plot
    plt.show()


    #print('\n■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ plot de ruta ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■\n')



else:
    print("No se encontró ninguna ruta óptima :(\n")






#print('\n■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ Menú ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■\n')

reiniciarQL = True
while reiniciarQL:
    reiniciar_str = input("\n¿Deseas reiniciar el programa? (si/no): ")
    if reiniciar_str.lower() == "si" or reiniciar_str.lower() == "s":
        print("\n")
        estado_inicial = input('Ingresa el estado inicial: ').upper()
        estado_objetivo = input('Ingresa el estado objetivo: ').upper()
        print("\n")
        ruta_optima = encontrar_ruta_optima(estado_inicial, estado_objetivo, grafo)
        print('\n*** Ruta óptima encontrada:', ruta_optima)
        #dib_ruta_optima = ruta_optima
        plt.show()
        print("\n\n")
    elif reiniciar_str.lower() == "no" or reiniciar_str.lower() == "n":
        print("\n\n\n           *** Fin de programa....\n\n\n\n\n")
        reiniciarQL = False
    else:
        print("\n\n     << Respuesta no válida. Por favor, responde 'si' o 'no' >>\n")
    
print('\n■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ Fin Programa Q-Learning "REFORZADO" ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■\n')




