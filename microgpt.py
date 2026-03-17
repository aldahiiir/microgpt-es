"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                         TUTORIAL COMPLETO: microgpt.py                           ║
║                    por Andrej Karpathy — Documentado en Español                  ║
╚══════════════════════════════════════════════════════════════════════════════════╝

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

¿QUÉ ES ESTO?
═════════════
Este archivo es un GPT completo (sí, como ChatGPT), implementado en Python puro,
sin ninguna dependencia externa. Ni NumPy, ni PyTorch, ni TensorFlow. Solo os, math
y random de la librería estándar de Python.

¿QUÉ ES UN GPT?
════════════════
GPT significa "Generative Pre-trained Transformer". Es la arquitectura de red neuronal
que impulsa modelos de lenguaje como ChatGPT, Claude, Gemini, etc. Un GPT aprende
patrones en texto y puede generar texto nuevo que sigue esos patrones.

¿QUIÉN ES ANDREJ KARPATHY?
═══════════════════════════
Andrej Karpathy es una de las figuras más importantes en la historia de la IA moderna:
  • Miembro fundador de OpenAI (la empresa que creó ChatGPT)
  • Ex-Director de IA y Autopilot en Tesla
  • Profesor de Stanford (CS231n, el curso de Deep Learning más conocido del mundo)
  • Fundador de Eureka Labs (AI-native education)
  • Creador de micrograd, makemore, nanoGPT, y ahora microgpt

Cuando alguien con ese currículum dice "esto es todo lo que necesitas para entender
un LLM", HAY QUE ESCUCHAR.

¿POR QUÉ SE LLAMA "microgpt"?
══════════════════════════════
Es parte de una serie de proyectos educativos "micro" de Karpathy:
  - micrograd  → Motor de autograd en ~100 líneas
  - makemore   → Modelos de lenguaje simples (bigramas, MLPs, RNNs)
  - nanoGPT    → GPT en PyTorch (más eficiente pero menos educativo)
  - microgpt   → Todo junto, sin dependencias, en ~200 líneas.

Karpathy lo describió como un "proyecto de arte" en X (Twitter) el 11 de febrero de
2026. En sus palabras: "No puedo simplificarlo más. Esto es el contenido algorítmico
completo de lo que se necesita. Todo lo demás es solo eficiencia."

¿POR QUÉ ES SORPRENDENTE?
══════════════════════════
GPT-2 (el GPT original de OpenAI, 2019) tenía 1.5 BILLONES de parámetros y miles
de líneas de código en múltiples archivos, con docenas de dependencias.

Este microgpt tiene:
  - 4,192 parámetros
  - ~200 líneas de código
  - CERO dependencias externas
  - La MISMA arquitectura matemática fundamental

Es como si alguien destilara el Manual de Vuelo del Boeing 747 en un plano de papel
de origami que de todas formas vuela. La misma física, el mismo principio, pero sin
ningún ruido que te distraiga de cómo funciona de verdad.

La comunidad lo compara con "The C Programming Language" de Kernighan & Ritchie:
la expresión mínima y canónica de un concepto fundamental.

¿QUÉ APRENDERÁS LEYENDO ESTE ARCHIVO?
═══════════════════════════════════════
Al terminar este tutorial entenderás:
  1. Cómo se tokeniza texto (convertir caracteres a números)
  2. Cómo funciona el autograd (diferenciación automática, el corazón de toda IA)
  3. Cómo funciona un Transformer con Atención Multi-Cabeza
  4. Cómo funciona el optimizador Adam (el que usa ChatGPT)
  5. Cómo es un loop de entrenamiento de LLM
  6. Cómo hace inferencia (generar texto nuevo) un LLM

ESTRUCTURA DEL ARCHIVO (de arriba a abajo):
═══════════════════════════════════════════
  [1] IMPORTS y DATASET       — Los datos con los que aprenderá el modelo
  [2] TOKENIZER               — Traduce texto ↔ números
  [3] AUTOGRAD (clase Value)  — Motor de diferenciación automática
  [4] PARÁMETROS DEL MODELO   — La "memoria" del GPT (sus pesos)
  [5] ARQUITECTURA GPT        — Las funciones matemáticas del Transformer
  [6] OPTIMIZADOR ADAM        — Cómo actualizamos los pesos para aprender
  [7] LOOP DE ENTRENAMIENTO   — El ciclo forward → loss → backward → update
  [8] INFERENCIA              — Generar texto nuevo con el modelo entrenado

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

# ══════════════════════════════════════════════════════════════════════════════════
# [1] IMPORTS
# ══════════════════════════════════════════════════════════════════════════════════
#
# Observa esto con atención: SOLO hay 3 imports. Eso es todo.
# Sin NumPy, sin PyTorch, sin TensorFlow, sin scikit-learn.
# Solo librerías de la biblioteca estándar de Python:
#
#   os      → Para verificar si ya existe el archivo de datos (os.path.exists)
#   math    → Para operaciones matemáticas: logaritmo y exponencial
#   random  → Para inicializar pesos aleatoriamente y para muestreo en inferencia
#
# En producción, frameworks como PyTorch reemplazan esto con operaciones de tensor
# masivamente paralelizadas en GPU. La matemática es idéntica, solo que aquí
# operamos en escalares (un número a la vez) en lugar de tensores (millones a la vez).

"""
The most atomic way to train and run inference for a GPT in pure, dependency-free Python.
This file is the complete algorithm.
Everything else is just efficiency.

@karpathy
"""

import os       # os.path.exists
import math     # math.log, math.exp
import random   # random.seed, random.choices, random.gauss, random.shuffle

# ══════════════════════════════════════════════════════════════════════════════════
# random.seed(42) — Semilla de aleatoriedad
# ══════════════════════════════════════════════════════════════════════════════════
#
# Fijar la semilla hace que el programa sea DETERMINISTA: si lo corres dos veces
# con la misma semilla, obtendrás exactamente los mismos resultados.
#
# Esto es fundamental en ciencia e investigación: permite reproducir experimentos.
#
# El número 42 es un guiño cultural a "The Hitchhiker's Guide to the Galaxy" de
# Douglas Adams, donde 42 es "la respuesta a la vida, el universo y todo lo demás".
# La comunidad de programación lo usa como número "aleatorio de broma" desde los 90s.

random.seed(42) # Let there be order among chaos


# ══════════════════════════════════════════════════════════════════════════════════
# [2] DATASET — Los datos de entrenamiento
# ══════════════════════════════════════════════════════════════════════════════════
#
# Todo modelo de lenguaje necesita datos para aprender. Sin datos, no hay
# aprendizaje. El dataset es la "experiencia de vida" del modelo.
#
# En este caso, usamos ~32,000 nombres de personas en inglés (emma, olivia, ava...).
# Es un dataset pequeño y manejable para demostración.
#
# En producción (ChatGPT, Claude, etc.), el dataset es el INTERNET ENTERO:
# Wikipedia, libros, código, artículos científicos, billones de tokens.
#
# ¿Cómo funciona la descarga?
# ───────────────────────────
# 1. Primero verifica si 'input.txt' ya existe (para no descargarlo dos veces)
# 2. Si no existe, lo descarga desde el repositorio de GitHub de Karpathy
# 3. Lee cada línea del archivo, eliminando espacios en blanco
# 4. Filtra las líneas vacías con `if line.strip()`
# 5. El resultado: una lista de strings donde cada elemento es un nombre
#
# ¿Por qué random.shuffle(docs)?
# ────────────────────────────────
# Mezclar los datos aleatoriamente evita que el modelo aprenda el ORDEN en que
# están los datos en lugar de los PATRONES en los datos. Si los nombres estuvieran
# ordenados (primero todos los que empiezan con 'a', luego 'b', etc.), el modelo
# podría sobre-ajustarse al orden alfabético.

# Let there be a Dataset `docs`: list[str] of documents (e.g. a list of names)
if not os.path.exists('input.txt'):
    import urllib.request
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt'
    urllib.request.urlretrieve(names_url, 'input.txt')
docs = [line.strip() for line in open('input.txt') if line.strip()]
random.shuffle(docs)
print(f"num docs: {len(docs)}")


# ══════════════════════════════════════════════════════════════════════════════════
# [3] TOKENIZER — El traductor entre texto y números
# ══════════════════════════════════════════════════════════════════════════════════
#
# Las redes neuronales solo trabajan con NÚMEROS. No entienden texto.
# El tokenizer es el diccionario que traduce caracteres → números y viceversa.
#
# En este caso se usa un tokenizer de NIVEL DE CARÁCTER (character-level):
# Cada carácter único del dataset se convierte en un entero único.
#
# Paso a paso:
# ─────────────
# 1. ''.join(docs)  → Concatena todos los documentos en un solo string gigante
# 2. set(...)        → Obtiene los caracteres únicos (elimina duplicados)
# 3. sorted(...)     → Los ordena alfabéticamente para consistencia
# 4. El resultado 'uchars': lista con los caracteres únicos, ej: ['a','b','c',...,'z']
#
# La codificación funciona así:
#   'a' → 0, 'b' → 1, 'c' → 2, ... 'z' → 25
#   uchars.index('a') → 0
#   uchars[0]         → 'a'
#
# ¿Qué es el token BOS?
# ──────────────────────
# BOS = Beginning of Sequence (Inicio de Secuencia)
# Es un token ESPECIAL que marca el comienzo y el fin de un documento.
# Se usa como señal para que el modelo sepa:
#   - "empieza a generar" (cuando lo recibe como entrada)
#   - "para de generar" (cuando lo produce como salida)
#
# En GPT-2 y modelos modernos, el token equivalente se llama <|endoftext|>.
# Aquí es simplemente el entero siguiente al último carácter: len(uchars)
#
# Por ejemplo, si hay 26 letras (a-z), entonces:
#   a=0, b=1, ..., z=25, BOS=26
#   vocab_size = 27 (26 letras + 1 token BOS)
#
# Comparación con producción:
# ─────────────────────────────
# GPT-2 usa BPE (Byte Pair Encoding): vocabulario de 50,257 tokens
# GPT-4 usa cl100k_base: vocabulario de 100,277 tokens
# Aquí: ~27 tokens (26 letras + BOS). Simple, puro, perfecto para aprender.

# Let there be a Tokenizer to translate strings to sequences of integers ("tokens") and back
uchars = sorted(set(''.join(docs))) # unique characters in the dataset become token ids 0..n-1
BOS = len(uchars) # token id for a special Beginning of Sequence (BOS) token
vocab_size = len(uchars) + 1 # total number of unique tokens, +1 is for BOS
print(f"vocab size: {vocab_size}")


# ══════════════════════════════════════════════════════════════════════════════════
# [4] AUTOGRAD — El corazón del aprendizaje automático
# ══════════════════════════════════════════════════════════════════════════════════
#
# Esta es la parte más importante del archivo, sin discusión.
# El autograd (diferenciación automática) es lo que permite que una red neuronal
# aprenda. Sin esto, tendríamos un modelo que hace predicciones pero no puede
# mejorarlas.
#
# ¿Qué problema resuelve el autograd?
# ─────────────────────────────────────
# Para entrenar una red neuronal necesitamos calcular cómo cambia el error (loss)
# cuando cambiamos cada parámetro. Esto es el GRADIENTE: ∂loss/∂parámetro
#
# Para una red con 4,192 parámetros (como esta), necesitamos 4,192 derivadas.
# Calcularlas a mano sería imposible. El autograd lo hace automáticamente.
#
# ¿Cómo funciona? La Regla de la Cadena (Chain Rule)
# ────────────────────────────────────────────────────
# La regla de la cadena del cálculo dice que si y = f(g(x)), entonces:
#   dy/dx = (dy/dg) × (dg/dx)
#
# El autograd construye un GRAFO COMPUTACIONAL durante el forward pass:
# cada operación (suma, multiplicación, etc.) registra sus "hijos" y las
# "derivadas locales" con respecto a ellos.
#
# Luego, en el backward pass, recorre el grafo al revés (de la pérdida hacia
# los parámetros) multiplicando gradientes en cadena.
#
# Analogía: Imagina una cadena de montaje en una fábrica.
#   Forward:  Materia prima → Producto terminado (con el precio final = loss)
#   Backward: ¿Cuánto contribuyó cada máquina al precio final? (gradientes)
#
# Esta clase `Value` es una versión simplificada de `torch.Tensor`:
#   - torch.Tensor maneja millones de números en paralelo (tensores)
#   - Value maneja UN número a la vez (escalar)
#   - La matemática es la misma

# Let there be Autograd to recursively apply the chain rule through a computation graph
class Value:
    # ────────────────────────────────────────────────────────────────────────────
    # __slots__ es una optimización de Python:
    # En lugar de usar un diccionario interno (__dict__) para almacenar atributos,
    # Python reserva espacios fijos en memoria. Resultado: menos memoria usada,
    # acceso más rápido. Importante cuando tienes miles de objetos Value.
    # ────────────────────────────────────────────────────────────────────────────
    __slots__ = ('data', 'grad', '_children', '_local_grads') # Python optimization for memory usage

    def __init__(self, data, children=(), local_grads=()):
        # data:         El valor escalar de este nodo (un número float)
        #               Calculado durante el FORWARD PASS
        self.data = data
        
        # grad:         La derivada de la pérdida con respecto a este nodo: ∂loss/∂self
        #               Calculada durante el BACKWARD PASS
        #               Empieza en 0; se acumula durante backprop
        self.grad = 0
        
        # _children:    Los nodos de entrada que produjeron este nodo
        #               Ejemplo: si z = x + y, entonces z._children = (x, y)
        #               Esto construye la estructura del grafo computacional
        self._children = children
        
        # _local_grads: Las derivadas parciales de este nodo respecto a cada hijo
        #               Ejemplo: si z = x + y:
        #                 ∂z/∂x = 1  →  local_grad para x
        #                 ∂z/∂y = 1  →  local_grad para y
        #               Si z = x * y:
        #                 ∂z/∂x = y  →  local_grad para x
        #                 ∂z/∂y = x  →  local_grad para y
        self._local_grads = local_grads

    # ──────────────────────────────────────────────────────────────────────────────
    # Operaciones matemáticas con sus derivadas locales
    # ──────────────────────────────────────────────────────────────────────────────
    #
    # Cada operación crea un NUEVO Value con:
    #   - El resultado de la operación como .data
    #   - Los operandos como ._children
    #   - Las derivadas locales como ._local_grads
    #
    # Las derivadas locales son las derivadas de la SALIDA respecto a cada ENTRADA.
    # Son las piezas que se ensamblan con la regla de la cadena en backward().

    def __add__(self, other):
        # z = self + other
        # ∂z/∂self  = 1  (si self sube 1, z sube 1)
        # ∂z/∂other = 1  (si other sube 1, z sube 1)
        # Nota: `other if isinstance(other, Value) else Value(other)` permite
        # sumar Value + número_normal, convirtiendo el número en Value automáticamente
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):
        # z = self * other
        # ∂z/∂self  = other.data  (la derivada de x*y respecto a x es y)
        # ∂z/∂other = self.data   (la derivada de x*y respecto a y es x)
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other):
        # z = self ** other  (self elevado a la potencia other)
        # ∂z/∂self = other * self.data^(other-1)  (regla de la potencia)
        # Nota: `other` aquí es un float/int regular, no un Value
        return Value(self.data**other, (self,), (other * self.data**(other-1),))

    def log(self):
        # z = ln(self)  (logaritmo natural)
        # ∂z/∂self = 1/self.data
        # Esto se usa para calcular la pérdida (cross-entropy loss = -log(prob))
        return Value(math.log(self.data), (self,), (1/self.data,))

    def exp(self):
        # z = e^self  (exponencial)
        # ∂z/∂self = e^self  (la derivada de e^x es e^x, eso es lo que hace especial al número e)
        # Esto se usa en softmax para convertir logits en probabilidades
        return Value(math.exp(self.data), (self,), (math.exp(self.data),))

    def relu(self):
        # z = max(0, self)  (Rectified Linear Unit, función de activación)
        # ∂z/∂self = 1 si self > 0, si no 0
        # float(self.data > 0) convierte el booleano True/False a 1.0/0.0
        #
        # ReLU es la función de activación más popular en redes profundas.
        # Introduce NO-LINEALIDAD: sin esto, apilar capas lineales sería
        # equivalente a una sola capa lineal (el modelo nunca aprendería nada complejo)
        return Value(max(0, self.data), (self,), (float(self.data > 0),))

    # ──────────────────────────────────────────────────────────────────────────────
    # Operaciones "de conveniencia" definidas en términos de las anteriores
    # ──────────────────────────────────────────────────────────────────────────────
    #
    # Python llama a estos "métodos mágicos" (dunder methods = double underscore).
    # Permiten usar operadores estándar (+, -, *, /) con objetos Value.
    #
    # __neg__:   -x   → x * -1
    # __radd__:  5+x  → x+5  (cuando el VALUE está a la derecha del operador)
    # __sub__:   x-y  → x + (-y)
    # __rsub__:  5-x  → 5 + (-x)
    # __rmul__:  5*x  → x*5
    # __truediv__:  x/y → x * y^(-1)  (división como multiplicación del inverso)
    # __rtruediv__: 5/x → 5 * x^(-1)

    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other**-1
    def __rtruediv__(self, other): return other * self**-1

    def backward(self):
        # ──────────────────────────────────────────────────────────────────────────
        # El BACKWARD PASS: aquí ocurre la magia del aprendizaje
        # ──────────────────────────────────────────────────────────────────────────
        #
        # Este método toma el nodo de PÉRDIDA (loss) y propaga gradientes hacia atrás
        # a través de toda la red, desde la salida hasta cada parámetro de entrada.
        #
        # PASO 1: Ordenamiento Topológico
        # ─────────────────────────────────
        # Para aplicar la regla de la cadena correctamente, necesitamos procesar
        # los nodos en el orden correcto: primero los nodos de salida,
        # luego los de entrada. Esto se llama "orden topológico inverso".
        #
        # Usamos DFS (Depth-First Search) para construir este orden.
        # El algoritmo:
        #   1. Visita el nodo actual
        #   2. Primero visita recursivamente TODOS sus hijos
        #   3. Solo entonces agrega el nodo actual a la lista
        # Resultado: los nodos más "profundos" (parámetros) aparecen al principio,
        # los nodos más "superficiales" (pérdida) al final
        
        topo = []       # Lista de nodos en orden topológico
        visited = set() # Conjunto de nodos ya visitados (evita procesar dos veces)
        
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child) # Primero los hijos (DFS recursivo)
                topo.append(v)       # Luego yo mismo
        
        build_topo(self) # Construye el orden desde la pérdida hacia los parámetros

        # PASO 2: Inicializar el gradiente de la pérdida
        # ──────────────────────────────────────────────
        # El gradiente de la pérdida respecto a SÍ MISMA siempre es 1:
        # ∂loss/∂loss = 1
        # Este es el "punto de partida" de la retropropagación.
        self.grad = 1

        # PASO 3: Retropropagar gradientes en orden inverso
        # ──────────────────────────────────────────────────
        # Recorremos la lista en orden INVERSO (de pérdida → parámetros)
        # Para cada nodo v, actualizamos el .grad de sus hijos.
        #
        # La regla de la cadena dice:
        #   ∂loss/∂hijo = ∂loss/∂v × ∂v/∂hijo
        #               = v.grad    × local_grad
        #
        # Usamos += porque un nodo puede tener MÚLTIPLES padres en el grafo.
        # Los gradientes de múltiples rutas se SUMAN (regla de la suma).
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad


# ══════════════════════════════════════════════════════════════════════════════════
# [5] PARÁMETROS DEL MODELO — La "memoria" del GPT
# ══════════════════════════════════════════════════════════════════════════════════
#
# Los parámetros (también llamados "pesos" o "weights") son los números que
# el modelo aprende durante el entrenamiento. Son la "memoria" de la red neuronal.
#
# Al inicio, se inicializan con valores aleatorios pequeños (Gaussiana con std=0.08).
# Después del entrenamiento, contienen los patrones aprendidos del dataset.
#
# Hiperparámetros (configuración de la arquitectura):
# ─────────────────────────────────────────────────────
# Estos son como los "planos arquitectónicos" del modelo. Se eligen ANTES
# del entrenamiento y definen la capacidad y estructura de la red.

# n_layer = 1:
#   Número de "capas Transformer" (profundidad de la red)
#   GPT-2 small: 12 capas | GPT-3: 96 capas | GPT-4: estimado ~120+ capas
#   Aquí: 1 capa (mínimo para demostrar el concepto)

# n_embd = 16:
#   Dimensión del embedding (ancho de la red)
#   Cuántos números representan cada token internamente
#   GPT-2 small: 768 | GPT-3: 12,288 | LLaMA-3 70B: 8,192
#   Aquí: 16 (mínimo funcional)

# block_size = 16:
#   Longitud máxima del contexto (ventana de atención)
#   ¿Cuántos tokens puede "ver" el modelo al mismo tiempo?
#   GPT-2: 1024 | GPT-4: 128,000 | Claude: 200,000
#   Aquí: 16 (suficiente porque el nombre más largo tiene 15 caracteres)
#   ¡El comentario en el código lo confirma!

# n_head = 4:
#   Número de "cabezas de atención" (multi-head attention)
#   Más cabezas = el modelo puede atender a diferentes aspectos simultáneamente
#   GPT-2 small: 12 cabezas | GPT-3: 96 cabezas
#   Aquí: 4 cabezas

# head_dim = n_embd // n_head = 16 // 4 = 4:
#   Dimensión de cada cabeza de atención
#   Cada cabeza opera en un subespacio de 4 dimensiones del embedding de 16

# Initialize the parameters, to store the knowledge of the model
n_layer = 1     # depth of the transformer neural network (number of layers)
n_embd = 16     # width of the network (embedding dimension)
block_size = 16 # maximum context length of the attention window (note: the longest name is 15 characters)
n_head = 4      # number of attention heads
head_dim = n_embd // n_head # derived dimension of each head


# ──────────────────────────────────────────────────────────────────────────────────
# Función `matrix`: Inicializador de matrices de pesos
# ──────────────────────────────────────────────────────────────────────────────────
#
# Crea una matriz de `nout` x `nin` objetos Value con valores aleatorios.
# Cada valor se inicializa con una distribución Normal(0, 0.08):
#   - media = 0 (pesos centrados en cero)
#   - std = 0.08 (pesos pequeños para evitar gradientes explosivos al inicio)
#
# ¿Por qué inicializar con valores pequeños?
# Si los pesos fueran muy grandes, las activaciones crecerían exponencialmente
# a través de las capas (gradientes explosivos). Si fueran muy pequeños (cercanos
# a 0), los gradientes se harían prácticamente 0 (gradientes que se desvanecen).
# std=0.08 es un buen balance empírico.
#
# Esta es una lambda (función anónima) que devuelve una lista de listas de Values:
#   matrix(2, 3) → [[V, V, V],   (fila 0)
#                    [V, V, V]]   (fila 1)
# donde V son objetos Value con valores aleatorios

matrix = lambda nout, nin, std=0.08: [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]


# ──────────────────────────────────────────────────────────────────────────────────
# state_dict: El diccionario de estado del modelo
# ──────────────────────────────────────────────────────────────────────────────────
#
# En PyTorch, `model.state_dict()` devuelve todos los parámetros del modelo.
# Aquí lo implementamos manualmente como un diccionario Python.
#
# Los parámetros del modelo GPT-2 style:
#
#  'wte' (Word Token Embedding): [vocab_size × n_embd]
#    → Matriz de embeddings de tokens. Convierte un id de token (un entero)
#      en un vector de n_embd dimensiones.
#    → Ejemplo: token 'a' (id=0) → vector de 16 números
#    → Parámetros: 27 × 16 = 432
#
#  'wpe' (Word Position Embedding): [block_size × n_embd]
#    → Matriz de embeddings de posición. Convierte una posición (0, 1, 2...)
#      en un vector de n_embd dimensiones.
#    → ¿Por qué? La atención es invariante al orden por defecto.
#      Sin posición, el modelo no sabría si 'e' está en posición 0 o 5.
#    → Parámetros: 16 × 16 = 256
#
#  'lm_head' (Language Model Head): [vocab_size × n_embd]
#    → Capa final que convierte el estado oculto (n_embd dimensiones) en
#      "logits" (vocab_size dimensiones = una puntuación por cada token posible)
#    → Parámetros: 27 × 16 = 432
#    → Nota: Comparte la misma forma que 'wte'. En GPT-2 moderno, los pesos
#      se comparten (weight tying). Aquí son matrices separadas.
#
#  Por cada capa i (layer{i}):
#    - attn_wq: [n_embd × n_embd] → Proyección de Query en atención
#    - attn_wk: [n_embd × n_embd] → Proyección de Key en atención
#    - attn_wv: [n_embd × n_embd] → Proyección de Value en atención
#    - attn_wo: [n_embd × n_embd] → Proyección de salida de atención
#    - mlp_fc1: [4*n_embd × n_embd] → Primera capa del MLP (expande)
#    - mlp_fc2: [n_embd × 4*n_embd] → Segunda capa del MLP (comprime)
#    → Parámetros por capa: 4*(16*16) + 2*(4*16*16) = 1024 + 2048 = 3072
#
# TOTAL: 432 + 256 + 432 + 3072 = 4,192 parámetros
# (¡GPT-4 tiene estimadamente ~1.8 BILLONES de parámetros!)

state_dict = {'wte': matrix(vocab_size, n_embd), 'wpe': matrix(block_size, n_embd), 'lm_head': matrix(vocab_size, n_embd)}
for i in range(n_layer):
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)


# ──────────────────────────────────────────────────────────────────────────────────
# params: Lista plana de todos los parámetros
# ──────────────────────────────────────────────────────────────────────────────────
#
# Aplana el state_dict (diccionario de matrices) en una lista unidimensional
# de todos los objetos Value. Esto facilita el loop del optimizador, que
# necesita iterar sobre TODOS los parámetros para actualizarlos.
#
# La comprensión de listas anidada:
#   - for mat in state_dict.values()  → cada matriz en el diccionario
#   - for row in mat                  → cada fila de la matriz
#   - for p in row                    → cada Value en la fila
# Resultado: una lista plana [p1, p2, p3, ...] con todos los 4,192 parámetros

params = [p for mat in state_dict.values() for row in mat for p in row] # flatten params into a single list[Value]
print(f"num params: {len(params)}")


# ══════════════════════════════════════════════════════════════════════════════════
# [6] ARQUITECTURA GPT — Las funciones matemáticas del Transformer
# ══════════════════════════════════════════════════════════════════════════════════
#
# Aquí están los "bloques de Lego" que construyen el Transformer.
# Cada función realiza una operación matemática específica.
#
# Nota sobre el comentario del código:
# "Follow GPT-2, blessed among the GPTs, with minor differences:
#  layernorm -> rmsnorm, no biases, GeLU -> ReLU"
#
# Las diferencias respecto a GPT-2 original:
#   - LayerNorm → RMSNorm (más simple, misma función práctica)
#   - No biases (sin términos de sesgo, simplifica sin perder expresividad)
#   - GeLU → ReLU (la función de activación más simple posible)

# Define the model architecture: a function mapping tokens and parameters to logits over what comes next
# Follow GPT-2, blessed among the GPTs, with minor differences: layernorm -> rmsnorm, no biases, GeLU -> ReLU

def linear(x, w):
    # ────────────────────────────────────────────────────────────────────────────
    # Transformación Lineal (también llamada "capa fully-connected" o "dense")
    # ────────────────────────────────────────────────────────────────────────────
    #
    # Computa: output = W × x
    # Donde:
    #   x: vector de entrada de dimensión nin (lista de Values)
    #   w: matriz de pesos de dimensión nout × nin (lista de listas de Values)
    #   output: vector de salida de dimensión nout (lista de Values)
    #
    # Para cada fila `wo` de la matriz W:
    #   output_i = Σ (wo_j × x_j) para j en 0..nin
    # Esto es un PRODUCTO PUNTO entre cada fila de W y el vector x.
    #
    # En PyTorch esto sería simplemente: torch.matmul(x, w.T)
    # Aquí lo implementamos con bucles Python explícitos. La matemática es la misma.
    #
    # ¿Por qué importa?
    # Las transformaciones lineales son las operaciones más básicas en redes
    # neuronales. Toda la información fluye a través de ellas. Lo interesante está
    # en los pesos: son los valores que el modelo aprende para representar conocimiento.
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]


def softmax(logits):
    # ────────────────────────────────────────────────────────────────────────────
    # Softmax: Convierte puntuaciones (logits) en probabilidades
    # ────────────────────────────────────────────────────────────────────────────
    #
    # Entrada: lista de logits (números reales, pueden ser negativos o muy grandes)
    # Salida: lista de probabilidades (números entre 0 y 1 que suman 1)
    #
    # Fórmula: softmax(x_i) = exp(x_i) / Σ exp(x_j)
    #
    # Truco de estabilidad numérica: `max_val`
    # ─────────────────────────────────────────
    # Si aplicamos exp() directamente a números grandes (ej: exp(1000)),
    # obtenemos OVERFLOW (infinito matemático). La solución: restar el máximo.
    # exp(x_i - max) / Σ exp(x_j - max) = exp(x_i) / Σ exp(x_j)
    # (matemáticamente equivalente, pero numéricamente estable)
    #
    # Nota: max_val usa .data para acceder al valor numérico raw, no al Value.
    # Esto porque max() necesita comparar números reales, no nodos del grafo.
    #
    # ¿Dónde se usa?
    # 1. En los pesos de atención: determina cuánto "atiende" cada token a otros
    # 2. En la salida: convierte logits en probabilidades sobre el vocabulario
    
    max_val = max(val.data for val in logits)       # Truco de estabilidad numérica
    exps = [(val - max_val).exp() for val in logits] # exp(x_i - max): operación diferenciable
    total = sum(exps)                                 # Denominador: Σ exp(x_j - max)
    return [e / total for e in exps]                 # Normaliza para obtener probabilidades


def rmsnorm(x):
    # ────────────────────────────────────────────────────────────────────────────
    # RMSNorm: Root Mean Square Normalization
    # ────────────────────────────────────────────────────────────────────────────
    #
    # La normalización es CRUCIAL para entrenar redes profundas.
    # Sin ella, los valores dentro de la red pueden crecer o decrecer
    # exponencialmente al pasar por cada capa, haciendo el entrenamiento inestable.
    #
    # RMSNorm (Zhang & Sennrich, 2019) normaliza dividiendo por la raíz cuadrada
    # del promedio de los cuadrados (la "norma RMS"):
    #
    #   ms = Σ(x_i²) / n          ← promedio de cuadrados ("mean square")
    #   scale = 1 / sqrt(ms + ε)   ← inverso de la norma RMS (1e-5 evita división por 0)
    #   output = x * scale         ← cada elemento normalizado
    #
    # Diferencia con LayerNorm (que usa GPT-2 original):
    #   LayerNorm: normaliza por (x - μ) / σ  (resta la media primero)
    #   RMSNorm:   normaliza por x / RMS(x)   (no resta la media)
    # RMSNorm es más simple y en la práctica funciona igual de bien.
    # Modelos modernos como LLaMA y Mistral usan RMSNorm.
    #
    # ¿Por qué se aplica ANTES de cada sub-bloque? (Pre-Norm)
    # Esta arquitectura "Pre-Norm" (normalizar ANTES de cada operación) es más
    # estable que "Post-Norm" (normalizar DESPUÉS). GPT-2 original usa Post-Norm;
    # los modelos modernos (GPT-3, LLaMA, etc.) usan Pre-Norm.
    #
    # Nota del comentario: "not redundant due to backward pass via the residual connection"
    # Aunque pueda parecer redundante normalizar antes de la primera capa
    # (los embeddings ya son valores razonables), es necesario porque durante
    # el backward pass, los gradientes fluyen tanto por la ruta directa COMO
    # por la conexión residual, y ambos necesitan escala similar.
    
    ms = sum(xi * xi for xi in x) / len(x)  # Media de cuadrados: Σ(x_i²)/n
    scale = (ms + 1e-5) ** -0.5             # Inverso de la norma RMS: 1/sqrt(ms + ε)
    return [xi * scale for xi in x]          # Normaliza cada elemento


def gpt(token_id, pos_id, keys, values):
    # ────────────────────────────────────────────────────────────────────────────
    # La función principal del Transformer: Forward Pass completo
    # ────────────────────────────────────────────────────────────────────────────
    #
    # Toma UN token a la vez (autoregresivo) y produce logits sobre el siguiente token.
    # Esta función se llama en un loop, token por token, durante entrenamiento
    # e inferencia.
    #
    # Entradas:
    #   token_id: Entero — el id del token de entrada (ej: 0 para 'a')
    #   pos_id:   Entero — la posición en la secuencia (0, 1, 2, ...)
    #   keys:     Lista de listas — caché de Keys para cada capa [n_layer][t][n_embd]
    #   values:   Lista de listas — caché de Values para cada capa [n_layer][t][n_embd]
    #
    # Salida:
    #   logits: Lista de n_embd Values — puntuaciones sin normalizar para cada token
    #
    # ¿Por qué un caché de keys y values?
    # ─────────────────────────────────────
    # Este es el KV Cache (Key-Value Cache). En atención causal (como GPT),
    # el token en posición t puede atender a TODOS los tokens anteriores (0..t-1).
    # En lugar de recomputar las Keys y Values de todos los tokens anteriores,
    # las guardamos en un caché. Esta es la optimización fundamental que hace
    # viables los LLMs en inferencia.
    # En producción, este caché puede ocupar GBs de memoria.

    # ─── EMBEDDING LOOKUP ──────────────────────────────────────────────────────
    #
    # Convierte el token_id y pos_id en vectores densos (embeddings).
    # Cada id es simplemente un ÍNDICE en la tabla de embeddings.
    # Piensa en wte como un "diccionario": dado el id del token, devuelve su vector.
    #
    # tok_emb: "¿QUÉ es este token?" → vector semántico del token
    # pos_emb: "¿DÓNDE está este token?" → vector de posición
    # x:       "¿Qué token en qué posición?" → combinación de ambos (suma)
    
    tok_emb = state_dict['wte'][token_id] # token embedding
    pos_emb = state_dict['wpe'][pos_id] # position embedding
    x = [t + p for t, p in zip(tok_emb, pos_emb)] # joint token and position embedding
    
    # Pre-Norm antes del primer bloque de atención
    # (ver comentario extendido en rmsnorm() sobre por qué no es redundante)
    x = rmsnorm(x) # note: not redundant due to backward pass via the residual connection

    for li in range(n_layer):
        # ─── BLOQUE 1: MULTI-HEAD SELF-ATTENTION ──────────────────────────────
        #
        # La atención es el mecanismo más importante del Transformer.
        # Permite que cada token "mire" a otros tokens en el contexto
        # y decida qué información tomar de cada uno.
        #
        # La analogía clásica: una base de datos con Query, Key, Value:
        #   Query (Q): "¿Qué estoy buscando?" (el token actual preguntando)
        #   Key   (K): "¿De qué trato yo?" (cada token anunciándose)
        #   Value (V): "¿Qué información tengo?" (el contenido de cada token)
        #
        # Para encontrar información relevante:
        #   1. Compara Q con cada K (dot product → puntuaciones de atención)
        #   2. Normaliza con softmax → pesos de atención (probabilidades)
        #   3. Promedio ponderado de los V → nueva representación del token
        #
        # "Multi-Head": en lugar de hacer esto UNA vez con vectores de n_embd,
        # lo hace n_head VECES con vectores de head_dim = n_embd/n_head.
        # Cada cabeza puede "especializarse" en diferentes aspectos:
        #   Cabeza 1: relaciones gramaticales
        #   Cabeza 2: referencias semánticas
        #   Cabeza 3: dependencias de larga distancia
        #   etc.

        # Guarda x antes de la atención para la conexión residual
        x_residual = x
        x = rmsnorm(x)  # Pre-Norm antes de atención
        
        # Proyecciones lineales Q, K, V
        # Convierte el embedding de entrada en vectores de Query, Key y Value
        # Cada proyección "re-expresa" x desde un ángulo diferente
        q = linear(x, state_dict[f'layer{li}.attn_wq'])  # Query: ¿qué busco?
        k = linear(x, state_dict[f'layer{li}.attn_wk'])  # Key:   ¿de qué trato?
        v = linear(x, state_dict[f'layer{li}.attn_wv'])  # Value: ¿qué información tengo?
        
        # Agrega la Key y Value de este token al caché
        # Después de esto: keys[li] contiene las Keys de todos los tokens hasta ahora
        keys[li].append(k)
        values[li].append(v)
        
        x_attn = []  # Acumula la salida de cada cabeza de atención
        
        for h in range(n_head):
            # ── Extrae los vectores correspondientes a la cabeza h ──────────────
            # Cada cabeza opera en un "subespacio" diferente del embedding.
            # Si n_embd=16 y n_head=4, entonces head_dim=4.
            # Cabeza 0: dimensiones 0-3
            # Cabeza 1: dimensiones 4-7
            # Cabeza 2: dimensiones 8-11
            # Cabeza 3: dimensiones 12-15
            hs = h * head_dim  # Índice de inicio del subespacio de la cabeza h
            
            q_h = q[hs:hs+head_dim]              # Query de la cabeza h: vector de head_dim dims
            k_h = [ki[hs:hs+head_dim] for ki in keys[li]]    # Keys de TODOS los tokens previos, cabeza h
            v_h = [vi[hs:hs+head_dim] for vi in values[li]]  # Values de TODOS los tokens previos, cabeza h
            
            # ── Puntuaciones de atención (Scaled Dot-Product Attention) ────────
            # Para cada token t en el historial:
            #   attn_logit_t = Q · K_t / sqrt(head_dim)
            # El producto punto mide la "similitud" entre Q y K_t.
            # Dividir por sqrt(head_dim) previene que los productos punto sean
            # muy grandes (lo que haría softmax demasiado "extremo", casi binario)
            # Esto se llama "scaling", la "S" en "Scaled Dot-Product Attention"
            attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5 for t in range(len(k_h))]
            
            # ── Pesos de atención ───────────────────────────────────────────────
            # Convierte las puntuaciones en probabilidades con softmax.
            # El token actual solo puede atender a tokens ANTERIORES (incluyéndose).
            # Esto es "causal attention" o "masked attention", necesario para que
            # el modelo no "haga trampa" viendo tokens futuros durante el entrenamiento.
            # Aquí se logra automáticamente porque solo tenemos los tokens hasta
            # la posición actual en el caché (keys[li] tiene t+1 elementos).
            attn_weights = softmax(attn_logits)
            
            # ── Salida de la cabeza: promedio ponderado de Values ───────────────
            # Para cada dimensión j del head_dim:
            #   head_out_j = Σ (attn_weights_t × V_t_j) para todos los t
            # Esto es un promedio ponderado donde los pesos son los attn_weights.
            # El resultado: una representación del token que incorpora información
            # contextual de los tokens más relevantes.
            head_out = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h))) for j in range(head_dim)]
            x_attn.extend(head_out)  # Concatena la salida de esta cabeza
        
        # ── Proyección de salida de atención ────────────────────────────────────
        # Mezcla la información de todas las cabezas con una transformación lineal.
        # Convierte la concatenación de cabezas [n_embd] → [n_embd]
        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
        
        # ── Conexión Residual (Skip Connection) ─────────────────────────────────
        # x = atención(x) + x_original
        # Esta es la innovación de ResNet (He et al., 2016) aplicada a Transformers.
        # Dos beneficios cruciales:
        #   1. Gradientes fluyen directamente de la salida a las primeras capas
        #      (resuelve el problema del gradiente que se desvanece)
        #   2. El modelo puede "optarse por no aprender" si una capa no ayuda
        #      (aprende la identidad si los pesos son ~0)
        x = [a + b for a, b in zip(x, x_residual)]
        
        # ─── BLOQUE 2: MLP (Multi-Layer Perceptron) ───────────────────────────
        #
        # El bloque MLP es la "red de alimentación hacia adelante" que sigue
        # al bloque de atención en cada capa Transformer.
        #
        # Mientras que la atención MEZCLA información entre posiciones (tokens),
        # el MLP TRANSFORMA la representación de cada posición INDEPENDIENTEMENTE.
        #
        # Arquitectura: Expandir → Activar → Comprimir
        #   x → [n_embd → 4*n_embd] → ReLU → [4*n_embd → n_embd] → salida
        #
        # El factor de expansión es 4× (convención de GPT-2).
        # Esto crea un "cuello de botella inverso": primero expande la representación
        # en un espacio de mayor dimensión (más rico en características),
        # aplica no-linealidad (ReLU), luego comprime de vuelta.
        # Se cree que el MLP almacena "conocimiento factual" del modelo.
        
        x_residual = x
        x = rmsnorm(x)  # Pre-Norm antes del MLP
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])  # Expande: 16 → 64 dimensiones
        x = [xi.relu() for xi in x]                       # No-linealidad: max(0, x)
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])  # Comprime: 64 → 16 dimensiones
        x = [a + b for a, b in zip(x, x_residual)]        # Otra conexión residual

    # ─── LM HEAD: Proyección final a logits ────────────────────────────────────
    # Convierte el estado oculto final (n_embd = 16 dimensiones) en logits
    # (vocab_size = 27 dimensiones, una puntuación por cada token del vocabulario)
    # Después de softmax, cada puntuación se convierte en la probabilidad de que
    # ese token sea el siguiente en la secuencia.
    logits = linear(x, state_dict['lm_head'])
    return logits


# ══════════════════════════════════════════════════════════════════════════════════
# [7] OPTIMIZADOR ADAM — Cómo el modelo aprende
# ══════════════════════════════════════════════════════════════════════════════════
#
# Adam (Adaptive Moment Estimation) es el optimizador más usado en Deep Learning.
# Propuesto por Kingma & Ba en 2014, es la base del entrenamiento de prácticamente
# todos los LLMs modernos (GPT, BERT, LLaMA, Claude...).
#
# ¿Por qué no usar SGD (Stochastic Gradient Descent) simple?
# ────────────────────────────────────────────────────────────
# SGD simple actualiza: parámetro -= learning_rate * gradiente
# Problemas:
#   1. La misma tasa de aprendizaje para TODOS los parámetros (no adaptativa)
#   2. Sensible al learning rate: muy alto → diverge; muy bajo → muy lento
#   3. Oscila en direcciones de alta curvatura
#
# Adam soluciona esto con DOS momentos que se adaptan por parámetro:
#
# Primer momento (m): media exponencial de gradientes (momento de velocidad).
#   m = β₁ × m_anterior + (1 - β₁) × gradiente
#   Si los gradientes son consistentemente positivos → m acumula impulso positivo
#   Si los gradientes oscilan → m se promedia y el impulso es suave
#   Análogía: un coche que va acelerando en la misma dirección (momentum)
#
# Segundo momento (v): media exponencial de gradientes al cuadrado (varianza).
#   v = β₂ × v_anterior + (1 - β₂) × gradiente²
#   Parámetros con gradientes grandes → v grande → actualización pequeña (no "salta")
#   Parámetros con gradientes pequeños → v pequeño → actualización relativa mayor
#   Analógía: "frena" automáticamente cuando hay mucha incertidumbre
#
# Actualización de Adam:
#   parámetro -= lr × (m̂ / (√v̂ + ε))
# Donde m̂ y v̂ son versiones "corregidas por sesgo" (bias-corrected) de m y v.
# La corrección es necesaria porque al inicio (step=1), m y v están inicializados
# en 0 y subestimarían los momentos verdaderos.
#
# Valores de hiperparámetros (estándar de la comunidad desde 2015):
#   learning_rate = 0.01  → Qué tan grandes son los pasos de actualización
#   beta1 = 0.85          → "Inercia" del primer momento (estándar: 0.9)
#   beta2 = 0.99          → "Inercia" del segundo momento (estándar: 0.999)
#   eps_adam = 1e-8       → Evita división por cero

# Let there be Adam, the blessed optimizer and its buffers
learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
m = [0.0] * len(params) # first moment buffer
v = [0.0] * len(params) # second moment buffer


# ══════════════════════════════════════════════════════════════════════════════════
# [8] LOOP DE ENTRENAMIENTO
# ══════════════════════════════════════════════════════════════════════════════════
#
# Este es el corazón del aprendizaje. Se repite num_steps veces.
# Cada iteración es un "paso" de entrenamiento.
#
# El ciclo de vida de cada paso:
#   1. SAMPLE    → Toma un documento del dataset
#   2. TOKENIZE  → Convierte el texto en tokens
#   3. FORWARD   → Pasa los tokens por el GPT (construye el grafo computacional)
#   4. LOSS      → Calcula qué tan mal se equivocó el modelo (cross-entropy)
#   5. BACKWARD  → Propaga gradientes hacia atrás (chain rule automática)
#   6. UPDATE    → Actualiza parámetros con Adam
#   7. ZERO GRAD → Reinicia gradientes para el próximo paso
#
# ¿Por qué 1000 pasos?
# Con 32,000 nombres y 1000 pasos, el modelo ve ~1000 ejemplos diferentes.
# Para entrenamiento real, se harían millones o billones de pasos.

# Repeat in sequence
num_steps = 1000 # number of training steps
for step in range(num_steps):

    # ─── TOKENIZACIÓN ──────────────────────────────────────────────────────────
    #
    # `step % len(docs)` cicla sobre todos los documentos en orden.
    # Cuando llega al final del dataset, vuelve al principio.
    #
    # La tokenización del documento 'emma':
    #   'emma' → [BOS, 4, 12, 12, 0, BOS]
    #            [BOS, e,  m,  m, a, BOS]
    #
    # Los dos BOS sirven:
    #   - BOS inicial: señal de "empieza a generar"
    #   - BOS final:   señal de "ya terminé de generar" (es el target del último char)
    #
    # min(block_size, len(tokens) - 1):
    #   - No puede ser más largo que el contexto máximo del modelo (block_size=16)
    #   - -1 porque siempre necesitamos un token "objetivo" (el siguiente token)
    #     para cada posición de entrada
    
    # Take single document, tokenize it, surround it with BOS special token on both sides
    doc = docs[step % len(docs)]
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    n = min(block_size, len(tokens) - 1)

    # ─── FORWARD PASS + CÁLCULO DE LOSS ────────────────────────────────────────
    #
    # Procesamos el documento token por token, de izquierda a derecha.
    # Para cada posición pos_id:
    #   - token_id: el token de entrada en esta posición
    #   - target_id: el token que DEBERÍA predecirse (el siguiente token)
    #
    # Ejemplo con 'emma':
    #   tokens = [BOS, e, m, m, a, BOS]
    #   pos=0: input=BOS,  target=e   → el modelo debe predecir 'e' dado BOS
    #   pos=1: input=e,    target=m   → el modelo debe predecir 'm' dado BOS,e
    #   pos=2: input=m,    target=m   → el modelo debe predecir 'm' dado BOS,e,m
    #   pos=3: input=m,    target=a   → el modelo debe predecir 'a' dado BOS,e,m,m
    #   pos=4: input=a,    target=BOS → el modelo debe predecir BOS dado BOS,e,m,m,a
    #
    # La PÉRDIDA (Cross-Entropy Loss):
    # ──────────────────────────────────
    # loss_t = -log(prob[target_id])
    #
    # Esta es la función de pérdida más usada en clasificación y en LLMs.
    # Intuición:
    #   - Si prob[target]=1.0 (predicción perfecta): loss = -log(1) = 0 (sin error)
    #   - Si prob[target]=0.5 (algo incierto):       loss = -log(0.5) ≈ 0.69
    #   - Si prob[target]=0.01 (muy equivocado):     loss = -log(0.01) ≈ 4.6
    #   - Si prob[target]→0 (completamente mal):     loss → ∞
    #
    # El modelo aprende a MAXIMIZAR la probabilidad del token correcto,
    # equivalentemente, a MINIMIZAR -log(prob[target]).
    #
    # La pérdida final es el PROMEDIO sobre toda la secuencia.
    # Un loss inicial típico: log(vocab_size) = log(27) ≈ 3.3 (distribución uniforme)
    # Un loss bien entrenado: notoriamente más bajo, el modelo "sabe" qué sigue.

    # Forward the token sequence through the model, building up the computation graph all the way to the loss
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    losses = []
    for pos_id in range(n):
        token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax(logits)
        loss_t = -probs[target_id].log()
        losses.append(loss_t)
    loss = (1 / n) * sum(losses) # final average loss over the document sequence. May yours be low.

    # ─── BACKWARD PASS ─────────────────────────────────────────────────────────
    #
    # Una sola llamada a loss.backward() calcula el gradiente de la pérdida
    # con respecto a TODOS los 4,192 parámetros del modelo simultáneamente.
    #
    # Esto es la retropropagación (backpropagation), el algoritmo propuesto por
    # Rumelhart, Hinton & Williams en 1986 que hizo posible el Deep Learning.
    #
    # Internamente, backward() recorre el grafo computacional en orden inverso
    # (de pérdida → parámetros) multiplicando gradientes en cadena.
    # Al finalizar, cada p en params tiene p.grad = ∂loss/∂p
    
    # Backward the loss, calculating the gradients with respect to all model parameters
    loss.backward()

    # ─── ACTUALIZACIÓN DE PARÁMETROS CON ADAM ──────────────────────────────────
    #
    # Learning rate decay lineal:
    #   lr_t = lr × (1 - step/num_steps)
    #   Paso 0:    lr_t = 0.01 (tasa máxima)
    #   Paso 500:  lr_t = 0.005 (mitad)
    #   Paso 999:  lr_t ≈ 0 (casi cero)
    #
    # ¿Por qué reducir el lr con el tiempo?
    # Al inicio: queremos pasos grandes para "llegar rápido" a una buena zona
    # Al final:  queremos pasos pequeños para "afinar" sin saltarnos la solución
    # Es como cuando vas manejando: aceleras en la carretera, frenas al llegar.
    #
    # Las fórmulas de Adam (explicadas arriba en la sección del optimizador):
    #   m_hat = m / (1 - β₁^(step+1))  ← corrección de sesgo del 1er momento
    #   v_hat = v / (1 - β₂^(step+1))  ← corrección de sesgo del 2do momento
    #   p.data -= lr × m_hat / (√v_hat + ε)
    #
    # Al final: p.grad = 0 (reseteamos gradientes para el siguiente paso)
    # Si no reseteamos, los gradientes se ACUMULARÍAN entre pasos (error clásico).
    
    # Adam optimizer update: update the model parameters based on the corresponding gradients
    lr_t = learning_rate * (1 - step / num_steps) # linear learning rate decay
    for i, p in enumerate(params):
        m[i] = beta1 * m[i] + (1 - beta1) * p.grad
        v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2
        m_hat = m[i] / (1 - beta1 ** (step + 1))
        v_hat = v[i] / (1 - beta2 ** (step + 1))
        p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
        p.grad = 0

    # Imprime el progreso. \r sobreescribe la misma línea (sin spam en la consola)
    print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.data:.4f}", end='\r')


# ══════════════════════════════════════════════════════════════════════════════════
# [9] INFERENCIA — Generar texto nuevo con el modelo entrenado
# ══════════════════════════════════════════════════════════════════════════════════
#
# Después del entrenamiento, el modelo puede GENERAR texto nuevo que sigue los
# patrones aprendidos. Para un modelo entrenado en nombres, generará nombres nuevos.
#
# ¿Cómo funciona la generación?
# ──────────────────────────────
# La generación es AUTOREGRESIVA: genera un token a la vez, y cada token
# generado se convierte en la entrada para generar el siguiente.
#
#   BOS → [GPT] → probabilidades → muestreo → token1
#   token1 → [GPT] → probabilidades → muestreo → token2
#   token2 → [GPT] → probabilidades → muestreo → token3
#   ...hasta que se genera BOS (fin del nombre)
#
# ¿Qué es la temperatura?
# ────────────────────────
# La temperatura controla la "creatividad" o "aleatoriedad" de la generación.
# Se aplica dividiendo los logits ANTES del softmax:
#   - temperatura = 1.0: distribución original del modelo
#   - temperatura < 1.0: distribución más "aguda" (más determinista, más predecible)
#   - temperatura > 1.0: distribución más "plana" (más aleatoria, más creativa)
#
# Con temperatura = 0.5 (la que usa este código):
# Los logits se dividen por 0.5, es decir se DUPLICAN.
# Esto hace que las probabilidades altas sean MUCHO más altas relativamente,
# y las bajas mucho más bajas. El modelo es más "confiado" y conservador.
#
# Analogía: como pedir café en una cafetería desconocida.
#   Temperatura alta: ordenas algo aleatorio que nunca probaste (creativo)
#   Temperatura baja: ordenas lo que siempre pides (conservador, predecible)
#
# random.choices(): muestreo de la distribución de probabilidad
# Selecciona aleatoriamente un token con probabilidad proporcional a sus pesos.
# Es como girar una ruleta donde cada sección tiene tamaño proporcional a su prob.

# Inference: may the model babble back to us
temperature = 0.5 # in (0, 1], control the "creativity" of generated text, low to high
print("\n--- inference (new, hallucinated names) ---")
for sample_idx in range(20):
    # Para cada muestra, reinicia el caché de KV (sin contexto previo)
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    token_id = BOS   # Comenzamos siempre con el token BOS
    sample = []
    
    for pos_id in range(block_size):
        # Forward pass para obtener logits del siguiente token
        logits = gpt(token_id, pos_id, keys, values)
        
        # Aplica temperatura dividiendo logits, luego convierte a probabilidades
        # Nota: `l / temperature` usa __rtruediv__ de Value (l es un Value, temperature es float)
        probs = softmax([l / temperature for l in logits])
        
        # Muestreo: selecciona el próximo token según las probabilidades
        # `range(vocab_size)`: los posibles tokens (0, 1, ..., vocab_size-1)
        # `weights=[p.data for p in probs]`: extrae los valores float de los Values
        token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
        
        # Si se genera BOS, el nombre terminó (condición de parada)
        if token_id == BOS:
            break
        
        # Agrega el carácter generado a la muestra
        sample.append(uchars[token_id])
    
    print(f"sample {sample_idx+1:2d}: {''.join(sample)}")


"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RESUMEN: ¿POR QUÉ ESTE CÓDIGO ES TAN IMPRESIONANTE?
════════════════════════════════════════════════════════════════════════════════

1. COMPLETO Y AUTOCONTENIDO
   ~200 líneas de Python puro implementan todo lo necesario para entrenar y
   hacer inferencia con un LLM: tokenizer, autograd, arquitectura Transformer,
   optimizador Adam, loop de entrenamiento, generación de texto.
   No falta nada. Todo lo demás (PyTorch, CUDA, etc.) es solo eficiencia.

2. PUEDES LEER CADA OPERACIÓN
   En producción, el mismo código vive en millones de líneas de C++/CUDA.
   Aquí puedes ver cada operación matemática individualmente.
   Es el nivel más atómico posible del aprendizaje automático.

3. LA MISMA ARQUITECTURA QUE LOS MODELOS MÁS AVANZADOS
   microgpt tiene 4,192 parámetros.
   GPT-4 tiene estimadamente ~1.8 billones de parámetros.
   La arquitectura matemática es prácticamente idéntica.
   Solo cambia la escala.

4. AÑOS DE TRABAJO DESTILADOS EN 200 LÍNEAS
   Karpathy lo construyó destilando años de investigación y docencia:
   micrograd → makemore → nanoGPT → microgpt
   La forma más simple que encontró de explicar cómo funciona un LLM.

5. LA COMUNIDAD LO ADOPTÓ DE INMEDIATO
   En 2 semanas de publicado ya había ports a:
   Rust, C++, Go, JavaScript (corriendo en el navegador), C, y más.
   Eso dice bastante del código.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LECTURAS RECOMENDADAS (para continuar aprendiendo):
═══════════════════════════════════════════════════════

  • Blog de Karpathy sobre microgpt: karpathy.github.io/2026/02/12/microgpt/
  • "Attention Is All You Need" (paper original del Transformer, 2017)
  • "Language Models are Few-Shot Learners" (paper de GPT-3, 2020)
  • Curso de Karpathy en YouTube: "Neural Networks: Zero to Hero"
  • micrograd (el autograd mínimo original de Karpathy)
  • nanoGPT (versión eficiente con PyTorch del mismo Karpathy)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""