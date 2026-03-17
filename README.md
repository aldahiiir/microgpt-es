# microgpt en Español 🇲🇽

Adaptación educativa del [Gist original de Andrej Karpathy](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95). El código es exactamente el mismo; lo que se agregó es documentación exhaustiva en Español para que cualquier persona hispanohablante pueda entender cómo funciona un LLM desde adentro, línea por línea.

> *"The most atomic way to train and run inference for a GPT in pure, dependency-free Python.
> This file is the complete algorithm. Everything else is just efficiency."*
> — @karpathy

---

## ¿Qué es esto?

Un GPT completo, como ChatGPT, implementado en Python puro sin ninguna dependencia externa. Ni NumPy, ni PyTorch, ni TensorFlow. Solo `os`, `math` y `random` de la librería estándar de Python.

## ¿Qué es un GPT?

GPT significa **Generative Pre-trained Transformer**. Es la arquitectura de red neuronal detrás de modelos de lenguaje como ChatGPT, Claude, Gemini, etc. Un GPT aprende patrones en texto y genera texto nuevo que sigue esos patrones.

---

## ¿Quién es Andrej Karpathy?

Una de las figuras más importantes en la historia de la IA moderna:

- Miembro fundador de **OpenAI** (la empresa que creó ChatGPT)
- Ex-Director de **IA y Autopilot en Tesla**
- Profesor de **Stanford** (CS231n, el curso de Deep Learning más conocido del mundo)
- Fundador de **Eureka Labs** (AI-native education)
- Creador de `micrograd`, `makemore`, `nanoGPT`, y ahora `microgpt`

Cuando alguien con ese currículum dice "esto es todo lo que necesitas para entender un LLM", hay que escuchar.

---

## ¿Por qué se llama "microgpt"?

Es parte de una serie de proyectos educativos "micro" de Karpathy:

| Proyecto | Descripción |
|----------|-------------|
| `micrograd` | Motor de autograd en ~100 líneas |
| `makemore` | Modelos de lenguaje simples (bigramas, MLPs, RNNs) |
| `nanoGPT` | GPT en PyTorch, más eficiente pero menos educativo |
| **`microgpt`** | **Todo junto, sin dependencias, en ~200 líneas.** |

Karpathy lo describió como un "proyecto de arte" el 11 de febrero de 2026:
> *"No puedo simplificarlo más. Esto es el contenido algorítmico completo de lo que se necesita. Todo lo demás es solo eficiencia."*

---

## ¿Por qué es sorprendente?

GPT-2 (OpenAI, 2019) tenía **1.5 billones de parámetros** y miles de líneas en múltiples archivos con docenas de dependencias.

Este `microgpt` tiene:

| Característica | microgpt | GPT-4 (estimado) |
|----------------|----------|-----------------|
| Parámetros | **4,192** | ~1.8 billones |
| Líneas de código | **~200** | Millones |
| Dependencias | **CERO** | Decenas |
| Arquitectura | Transformer completo | Transformer completo |

La misma física, el mismo principio, pero tan despojado de todo lo innecesario que puedes ver exactamente cómo funciona cada parte.

Lo comparan con *"The C Programming Language"* de Kernighan & Ritchie: la expresión mínima y canónica de un concepto fundamental.

---

## ¿Qué aprenderás leyendo este archivo?

Al terminar el tutorial entenderás:

1. **Tokenización** — cómo se convierte texto en números
2. **Autograd** — diferenciación automática (el corazón de toda IA)
3. **Transformer con Atención Multi-Cabeza** — cómo fluye la información
4. **Optimizador Adam** — el que usa ChatGPT para aprender
5. **Loop de entrenamiento de un LLM** — el ciclo completo
6. **Inferencia** — cómo genera texto nuevo un LLM

---

## Cómo ejecutarlo

```bash
python microgpt.py
```

No requiere instalación de dependencias. Al ejecutarlo:

1. Descarga automáticamente ~32,000 nombres de personas (dataset de Karpathy)
2. Entrena el modelo durante 1000 pasos
3. Genera 20 nombres nuevos "alucinados" por el modelo

Ejemplo de salida:
```
num docs: 32033
vocab size: 27
num params: 4192
step 1000 / 1000 | loss 2.1234

--- inference (new, hallucinated names) ---
sample  1: elena
sample  2: mara
sample  3: olivia
...
```

---

## Estructura del archivo

```
microgpt.py
│
├── [1] IMPORTS           — os, math, random (solo 3, sin dependencias externas)
├── [2] DATASET           — ~32k nombres en inglés descargados de GitHub
├── [3] TOKENIZER         — Traduce caracteres ↔ enteros (vocabulario de ~27 tokens)
├── [4] AUTOGRAD          — Clase Value: motor de diferenciación automática
├── [5] PARÁMETROS        — state_dict: 4,192 pesos inicializados aleatoriamente
├── [6] ARQUITECTURA GPT  — linear, softmax, rmsnorm, gpt (el Transformer completo)
├── [7] OPTIMIZADOR ADAM  — Actualización adaptativa de parámetros
├── [8] ENTRENAMIENTO     — Loop de 1000 pasos: forward → loss → backward → update
└── [9] INFERENCIA        — Generación autoregresiva con temperatura
```

---

## Componentes documentados

### [3] Tokenizer — Character-level

Convierte cada carácter único del dataset en un entero:

```
'a' → 0, 'b' → 1, ..., 'z' → 25, BOS → 26
vocab_size = 27 (26 letras + 1 token especial BOS)
```

El token **BOS** (Beginning of Sequence) marca el inicio y fin de cada nombre.
GPT-4 usa `cl100k_base` con 100,277 tokens; aquí son 27.

---

### [4] Autograd — Clase `Value`

El corazón del aprendizaje. Implementa diferenciación automática mediante la **regla de la cadena**:

- Cada operación (`+`, `*`, `**`, `log`, `exp`, `relu`) registra sus derivadas locales
- `loss.backward()` recorre el grafo computacional en orden inverso
- Al finalizar, cada parámetro `p` tiene `p.grad = ∂loss/∂p`

Operaciones implementadas:

| Operación | Derivada local |
|-----------|---------------|
| `z = x + y` | `∂z/∂x = 1`, `∂z/∂y = 1` |
| `z = x * y` | `∂z/∂x = y`, `∂z/∂y = x` |
| `z = x ** n` | `∂z/∂x = n * x^(n-1)` |
| `z = log(x)` | `∂z/∂x = 1/x` |
| `z = exp(x)` | `∂z/∂x = exp(x)` |
| `z = relu(x)` | `∂z/∂x = 1 si x>0, si no 0` |

Es básicamente un `torch.Tensor` escalar. La matemática es la misma.

---

### [5] Parámetros del modelo

**Hiperparámetros de arquitectura:**

| Parámetro | Valor | GPT-3 (comparación) |
|-----------|-------|---------------------|
| `n_layer` | 1 | 96 |
| `n_embd` | 16 | 12,288 |
| `block_size` | 16 | 2,048 |
| `n_head` | 4 | 96 |

**Desglose de parámetros (total: 4,192):**

| Tensor | Forma | Parámetros |
|--------|-------|-----------|
| `wte` (Word Token Embedding) | 27 × 16 | 432 |
| `wpe` (Word Position Embedding) | 16 × 16 | 256 |
| `lm_head` (Language Model Head) | 27 × 16 | 432 |
| `attn_wq/wk/wv/wo` (por capa) | 16 × 16 × 4 | 1,024 |
| `mlp_fc1/fc2` (por capa) | (64×16 + 16×64) | 2,048 |
| **TOTAL** | | **4,192** |

---

### [6] Arquitectura GPT — El Transformer

Cada token pasa por el siguiente flujo:

```
token_id, pos_id
     │
     ▼
[Token Embedding + Position Embedding]
     │
     ▼
  RMSNorm
     │
     ├──────────── Para cada capa: ────────────┐
     │                                          │
     ▼                                          │
[Multi-Head Self-Attention]               (Residual)
  ├─ Q = linear(x, wq)                         │
  ├─ K = linear(x, wk)  → KV Cache             │
  ├─ V = linear(x, wv)  → KV Cache             │
  ├─ attn_logits = Q·K / √head_dim             │
  ├─ attn_weights = softmax(attn_logits)        │
  └─ output = Σ(attn_weights × V)              │
     │                                          │
     ▼                                          │
  + Residual ◄─────────────────────────────────┘
     │
     ▼
  RMSNorm
     │
     ▼
[MLP: linear → ReLU → linear]
     │
     ▼
  + Residual
     │
     ▼
[lm_head: linear → logits]
     │
     ▼
  softmax → probabilidades sobre vocabulario
```

**Diferencias respecto a GPT-2 original:**

| GPT-2 original | microgpt |
|----------------|----------|
| LayerNorm | RMSNorm (más simple) |
| Tiene biases | Sin biases |
| GeLU | ReLU |
| Post-Norm | Pre-Norm |

---

### [7] Optimizador Adam

**Adam** (Adaptive Moment Estimation, Kingma & Ba 2014) es el optimizador estándar de todos los LLMs modernos.

Mantiene dos momentos por parámetro:

```python
# Primer momento: velocidad/impulso
m = β₁ × m + (1 - β₁) × gradiente

# Segundo momento: varianza
v = β₂ × v + (1 - β₂) × gradiente²

# Actualización corregida por sesgo
p -= lr × (m̂ / (√v̂ + ε))
```

| Hiperparámetro | Valor | Descripción |
|----------------|-------|-------------|
| `learning_rate` | 0.01 | Tamaño de paso base |
| `beta1` | 0.85 | Inercia del 1er momento |
| `beta2` | 0.99 | Inercia del 2do momento |
| `eps_adam` | 1e-8 | Evita división por cero |

También se usa **learning rate decay lineal**: `lr_t = lr × (1 - step/num_steps)`

---

### [8] Loop de entrenamiento

Por cada documento (nombre), el modelo hace:

```
1. TOKENIZE  → 'emma' → [BOS, e, m, m, a, BOS]
2. FORWARD   → Por cada posición, predice el siguiente token
3. LOSS      → Cross-Entropy: loss_t = -log(prob[token_correcto])
4. BACKWARD  → loss.backward() calcula los 4,192 gradientes
5. UPDATE    → Adam actualiza todos los parámetros
6. ZERO GRAD → p.grad = 0 (listo para el próximo paso)
```

**Cross-Entropy Loss:**

| `prob[target]` | `loss = -log(prob)` | Significado |
|----------------|---------------------|-------------|
| 1.0 | 0.0 | Predicción perfecta |
| 0.5 | 0.69 | Algo incierto |
| 0.01 | 4.6 | Muy equivocado |
| ~0 | → ∞ | Completamente mal |

---

### [9] Inferencia — Generación autoregresiva

La generación funciona token por token:

```
BOS → [GPT] → probs → muestreo → token1
token1 → [GPT] → probs → muestreo → token2
...hasta que se genera BOS (fin del nombre)
```

**Temperatura** controla la creatividad (`temperature = 0.5` por defecto):

| Temperatura | Efecto |
|-------------|--------|
| `< 1.0` | Más conservador, predecible |
| `= 1.0` | Distribución original del modelo |
| `> 1.0` | Más aleatorio, creativo |

---

## Por qué este código importa

1. **Es autocontenido.** ~200 líneas implementan TODO: tokenizer, autograd, Transformer, Adam, entrenamiento e inferencia. No falta nada.

2. **Puedes leer cada operación.** En producción, el mismo código vive en millones de líneas de C++/CUDA. Aquí lo ves tal cual.

3. **La arquitectura es la misma que GPT-4.** 4,192 parámetros vs ~1.8 billones. Solo cambia la escala.

4. **Karpathy lleva una década destilando esto.** `micrograd → makemore → nanoGPT → microgpt`. Es el resultado de años buscando la forma más simple de explicar un LLM.

5. **En dos semanas ya había ports en Rust, C++, Go, JavaScript y C.** Eso dice algo.

---

## Lecturas recomendadas

- Blog de Karpathy: [karpathy.github.io/2026/02/12/microgpt/](https://karpathy.github.io/2026/02/12/microgpt/)
- Paper original del Transformer: [*"Attention Is All You Need"* (2017)](https://arxiv.org/abs/1706.03762)
- Paper de GPT-3: [*"Language Models are Few-Shot Learners"* (2020)](https://arxiv.org/abs/2005.14165)
- Curso de Karpathy: [Neural Networks: Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)
- [micrograd](https://github.com/karpathy/micrograd) — El autograd mínimo original
- [nanoGPT](https://github.com/karpathy/nanoGPT) — Versión eficiente con PyTorch
