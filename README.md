# Lector de cartas para Blackjack con visión por computadora

**Visión por Computadora II · CEIA-FIUBA (2026)**: Cristhian Pettico, Rodolfo Di Chiazza, Julian Blanco

Este prototipo permite **jugar al Blackjack de verdad** con una cámara cenital sobre la mesa:

- **Cartas:** la cámara reconoce las cartas que se reparten y las asigna a la **Casa** (mitad superior de la imagen)
  o al **Jugador** (mitad inferior).
- **Gestos:** el Jugador pide carta o se planta con **gestos de la mano**, como en una mesa real.
- **Reglas:** un motor aplica las reglas del Blackjack. Le indica al crupier qué hacer, decide quién gana y lleva el
  saldo.

También funciona con una sola foto o un video, para leer una mano sin jugar.

El trabajo compara dos formas de reconocer las cartas a partir del **índice de la esquina** (valor y palo). Como en
Blackjack el palo no importa, alcanza con distinguir **13 valores**:

| Pipeline | Cómo funciona | Modelos |
|---|---|---|
| **A: un solo paso** | Un detector encuentra cada esquina y clasifica su valor | YOLO11s con 13 clases (**YOLO-13**) |
| **B: dos etapas** | Un detector solo localiza la esquina, se recorta y una CNN clasifica el valor | YOLO11s con 1 clase (**YOLO-1**) + ResNet-18 (**CNN-13**) |

Sobre la salida de cualquiera de los dos pipelines trabajan tres módulos:

- **Identificador:** une las dos esquinas de cada carta y reparte las cartas entre Casa y Jugador.
- **Seguimiento:** estabiliza la lectura en video.
- **Motor de reglas:** junto con el reconocedor de gestos (MediaPipe), lleva adelante la partida.

```
imagen ─┬─ A: YOLO-13 ──────────────────────────────┐
        └─ B: YOLO-1 ─► recorte ─► CNN-13 ───────────┤─► esquinas con valor ─► emparejar esquinas ─► cartas
                                                     │                          (distancia y ángulo   │
                                                     │                           de la diagonal)      ▼
                                                     └──────────── Casa / Jugador ─► puntaje ─► ganador
video:  esquinas ─► ByteTrack + votación + memoria ─► mano estable ─┐
mano del Jugador ─► MediaPipe (21 puntos) ─► pose ─► gesto ────────┴─► motor de reglas ─► instrucción / resultado / saldo
```

---

## Estructura del repositorio

```
├── Identificador_Cartas/        aplicación: detección -> cartas -> Blackjack
│   ├── identificador_cartas.py  punto de entrada (imagen, cámara o video)
│   ├── detectores.py            pipelines A y B con la misma interfaz
│   ├── class_corner.py          esquina detectada + geometría para emparejar
│   ├── class_carta.py           carta (1 o 2 esquinas) + emparejamiento global
│   ├── class_player.py          Casa / Jugador, puntaje y ganador
│   ├── seguimiento.py           video: ByteTrack + votación por track + memoria + mediana por carta
│   ├── video.py                 grabación en H.264 (WhatsApp, PowerPoint) y conversión de videos
│   ├── blackjack.py             motor de reglas (etapas de la mano, Casa pide hasta 17, pagos, saldo)
│   └── juego.py                 partida completa: detector + seguimiento + reglas + gestos
├── Gestos/                      control por gestos de la mano
│   ├── gestos.py                MediaPipe HandLandmarker + reglas de pose + lógica temporal
│   ├── evaluar_gestos.py        exactitud de las poses en HaGRID (baja la muestra)
│   ├── simular_gestos.py        gestos en video con manos reales de HaGRID
│   └── modelos/                 modelos de MediaPipe (se bajan solos la primera vez; no van al repo)
├── YOLO - 13/                   pipeline A: train_yolo_13.py, dataset y pesos (runs/)
├── YOLO - 1/                    pipeline B (detector): train_yolo_1.py, dataset y pesos (runs/)
├── CNN - 13/                    pipeline B (clasificador): train_cnn_13.py, dataset de recortes y pesos (runs/)
├── Dataset Real/                fotos reales: teogopk/ (público), roboflow/ (propio), extraer_cuadros.py, prelabel.py y guía
├── Experimentos/                evaluación en fotos reales y fine-tuning
│   ├── evaluar_real.py
│   ├── finetune_real.py
│   ├── evaluar_seguimiento.py   experimento de seguimiento en video
│   ├── generar_video_prueba.py  video de "cámara en mano" a partir de una foto
│   ├── simular_partida.py       partidas completas simuladas con cartas y manos reales
│   ├── clases.py                normaliza nombres de clase de otros datasets ('10h', 'AS'...) a valores
│   └── resultados/              CSV con las métricas
├── videos/                      videos de prueba generados (se regeneran con generar_video_prueba.py)
├── tests/                       tests unitarios (reglas y gestos): python -m pytest tests
├── *.jpeg                       4 fotos de prueba del mazo propio (Bicycle Dragon)
├── COMO_JUGAR.md                guía para jugar una partida
└── requirements.txt
```

## Instalación

Requiere Python 3.12.

```bash
git clone https://github.com/dichiazzar/Vision_Computadora_2_TP_2.git
cd Vision_Computadora_2_TP_2
python -m venv .venv
.venv\Scripts\activate          # Windows   (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt
pip install --no-deps mediapipe==1.0.1   # gestos: sin dependencias, porque pide opencv-contrib (choca con opencv-python)
python -m pytest tests                   # 43 tests de reglas y gestos
```

- **GPU NVIDIA:** instalar la versión de PyTorch con CUDA desde [pytorch.org](https://pytorch.org).
- **Mac Apple Silicon:** se usa `mps` automáticamente.
- **Ventanas que no abren:** si aparece `The function is not implemented` al abrir una ventana, es porque hay dos
  OpenCV instalados (`opencv-python` junto con `opencv-python-headless` u `opencv-contrib-python`). Desinstalar
  todos y reinstalar solo `opencv-python`.

Los pesos entrenados ya están en el repositorio, así que se puede usar la aplicación sin entrenar nada.

## Uso rápido

Ejecutar desde la raíz del repositorio:

```bash
# una foto (abre una ventana con el resultado)
python Identificador_Cartas/identificador_cartas.py --imagen prueba.jpeg
python Identificador_Cartas/identificador_cartas.py --imagen ACES_TAPADOS.jpeg --pipeline B

# cámara en vivo (q sale, r reinicia la mano) o un video, guardando el resultado
python Identificador_Cartas/identificador_cartas.py --camara 0 --conf 0.3
python Identificador_Cartas/identificador_cartas.py --video partida.mp4 --conf 0.3 --guardar salida.mp4

# video de prueba sin cámara: se genera a partir de una foto y se mide contra la mano real
python Experimentos/generar_video_prueba.py --foto prueba.jpeg --salida videos/prueba.mp4
python Identificador_Cartas/identificador_cartas.py --video videos/prueba.mp4 --conf 0.3 --esperado "Q|7 K"

# sin ventana: solo imprime y guarda
python Identificador_Cartas/identificador_cartas.py --imagen prueba.jpeg --sin-ventana --guardar out.jpg
```

Ejemplo de salida para `prueba.jpeg`:
```
[A: YOLO-13] prueba.jpeg: 6 esquinas
  Casa: Q = 10
  Jugador: 7 K = 17
  Gana el Jugador
```

En video y cámara el **seguimiento está activado por defecto** (se apaga con `--sin-seguimiento`). Cada esquina
muestra su id de track (`#12 Q 0.91`), las esquinas recordadas aparecen en gris y arriba a la derecha se ve qué
porcentaje de los últimos cuadros coincide con la mano mostrada. Con seguimiento conviene `--conf 0.3`, porque
ByteTrack aprovecha también las detecciones de baja confianza.

Por defecto se usan los pesos ajustados con fotos propias del escenario de la demo, si existen (`--pesos auto`). Con
`--pesos base|real|propio` se eligen a mano (ver experimento 3b).

Para mejorar la detección: luz pareja, cámara a 30–60 cm y cenital, y la esquina con el índice visible.

### Jugar una partida

La guía paso a paso, con la preparación de la cámara, las teclas y qué hacer si algo falla, está en
**[COMO_JUGAR.md](COMO_JUGAR.md)**. En resumen:

```bash
python Identificador_Cartas/identificador_cartas.py --camara 0 --conf 0.3 --juego --gestos
```

1. **Reparto:** el crupier reparte 2 cartas boca arriba al Jugador (abajo) y 2 a la Casa (arriba), la segunda boca
   abajo. Las instrucciones aparecen abajo a la izquierda y el estado de la partida en el panel de la derecha.
2. **Turno del Jugador:**
   - **Pedir carta:** levantar solo el índice, apuntando, durante ~0,5 s. Se ve "APUNTAR -> PEDIR" y una barra de
     avance.
   - **Plantarse:** mostrar la **palma abierta quieta durante ~1,2 s**, o **pasar la mano abierta** sobre las cartas
     (más rápido). Antes de hacer otro gesto hay que retirar la mano.
   - **Teclado:** `p` pide y `l` se planta, como respaldo o sin `--gestos`.
3. **Turno de la Casa:** el crupier da vuelta su carta y pide mientras el sistema indique "debe pedir carta". La Casa
   pide hasta 17, y con `--pide-17-blando` pide también con 17 blando.
4. **Resultado:** el sistema anuncia el resultado y suma el saldo (apuesta fija de 10; el blackjack paga 3:2). Al
   levantar las cartas empieza otra mano. `n` reinicia la mano en curso (conserva el saldo), `r` empieza una
   partida nueva (saldo en 0) y `q` sale.

Con `--guardar-crudo crudo.mp4` se graba además la imagen limpia de la cámara, sin dibujos, para juntar fotos
de entrenamiento (ver [Dataset Real/README.md](Dataset%20Real/README.md)). Con `--guardar partida.mp4` la partida se graba y al salir se convierte a **H.264**, el formato que aceptan
WhatsApp, PowerPoint y los navegadores (usa el ffmpeg de `imageio-ffmpeg`). Para convertir un video grabado antes:
`python Identificador_Cartas/video.py partida.mp4`. Para elegir la cámara, el número se ve con
`python -c "import cv2; [print(i, cv2.VideoCapture(i).read()[0]) for i in range(4)]"`; con el celular como webcam
(Vínculo a Windows, DroidCam o Camo) suele ser la 1. `--resolucion` fija la resolución (por defecto 1280x720).

Los gestos solo se reconocen en la mitad del Jugador, solo en su turno, y no mientras espera la carta que pidió. Así
la mano del crupier repartiendo no dispara nada.

---

## Datasets

| Dataset | Uso | Origen | Tamaño |
|---|---|---|---|
| **Dataset YOLO 13** | Entrenar YOLO-13 | Roboflow [`julian-segundo-blanco/dataset-a-marzb` v1](https://app.roboflow.com/julian-segundo-blanco/dataset-a-marzb/1) (dominio público) | 10.100 imágenes 512×512: 7.070 / 2.020 / 1.010 |
| **Dataset YOLO - 1** | Entrenar YOLO-1 | Roboflow `julian-segundo-blanco/dataset-b-5f2kh` v1 (proyecto privado; el dataset completo está en este repo) | Mismas imágenes, clase única `card-corner` |
| **Dataset CNN-13** | Entrenar CNN-13 | Recortes de las cajas del Dataset YOLO 13 con 10% de margen (`YOLO - 13/Dataset YOLO 13/create_cnn_dataset.py`) | 28.280 / 8.080 / 4.040 recortes, balanceado |
| **Dataset Real (teogopk)** | Test real y fine-tuning | [TeogopK/Playing-Cards-Object-Detection](https://github.com/TeogopK/Playing-Cards-Object-Detection), `data/real_dataset` (CC0) | 98 fotos reales 416×416: 69 / 18 / 11 |
| **Dataset Real (propio)** | Test y fine-tuning en el escenario de la demo | Cuadros de 4 sesiones de juego grabadas con `--guardar-crudo`, etiquetados en Roboflow ([`rodolfo-di-chiazza/tp2-vision-por-computadora-ii` v2](https://universe.roboflow.com/rodolfo-di-chiazza/tp2-vision-por-computadora-ii/dataset/2), CC BY 4.0) | 130 cuadros 508×720, 1.688 esquinas: 79 / 21 / 30 |

**Datasets sintéticos:** los dos datasets de entrenamiento se armaron a partir del dataset sintético
[Playing Cards (Augmented Startups)](https://universe.roboflow.com/augmented-startups/playing-cards-ow27d), que tiene
cartas generadas y pegadas sobre fondos, con las etiquetas reducidas a 13 valores o a una sola clase. Todos los
datasets etiquetan **el índice de la esquina**, no la carta entera.

**Datasets reales:**
- **teogopk:** son fotos reales de cartas de corazones. Tiene tres limitaciones: un solo palo, un mazo con índice en
  las cuatro esquinas, y pocas K. Ver [Dataset Real/teogopk/FUENTE.txt](Dataset%20Real/teogopk/FUENTE.txt).
- **Fotos propias:** [Dataset Real/README.md](Dataset%20Real/README.md) explica cómo sacar y etiquetar fotos propias.
  Incluye `prelabel.py`, que pre-etiqueta las fotos con YOLO-13 para corregirlas en Roboflow.

**Todos los datasets están incluidos en el repositorio**, con imágenes y etiquetas en formato YOLO, así que no hace
falta descargar nada para reentrenar o evaluar:

| Dataset | Carpeta |
|---|---|
| YOLO 13 | `YOLO - 13/Dataset YOLO 13/` |
| YOLO - 1 | `YOLO - 1/Dataset YOLO - 1/` |
| CNN-13 | `CNN - 13/Dataset CNN-13/` (también se regenera con `create_cnn_dataset.py`) |
| teogopk | `Dataset Real/teogopk/` |
| propio | `Dataset Real/roboflow/` |

Los enlaces de Roboflow quedan como referencia de dónde se armaron los datasets. El del dataset de YOLO-1 es un
proyecto privado, por eso no lleva enlace, pero su contenido completo está en `YOLO - 1/Dataset YOLO - 1/`.

---

## Entrenamiento

| Modelo | Script | Configuración | Tiempo |
|---|---|---|---|
| YOLO-13 | `YOLO - 13/train_yolo_13.py` | YOLO11s preentrenado en COCO, 50 épocas, 640 px, batch 16, *patience* 15, aumentos por defecto de ultralytics | 5,6 h (Apple MPS) |
| YOLO-1 | `YOLO - 1/train_yolo_1.py` | Igual que YOLO-13, con una sola clase | 5,0 h (Apple MPS) |
| CNN-13 | `CNN - 13/train_cnn_13.py` | ResNet-18 preentrenada en ImageNet, se ajusta toda la red, 96×96, AdamW + OneCycle (lr 1e-3), *label smoothing* 0,05, corte anticipado | 30 min (CPU, 7 épocas) |

En la CNN se aplican rotaciones de ±15°, traslación, escala, cambios de color y desenfoque. **No se usan espejados**,
porque un índice espejado no es una carta válida.

Los tres scripts se pueden correr desde cualquier carpeta y eligen solos el dispositivo (GPU NVIDIA, Apple MPS o
CPU), por ejemplo `python "YOLO - 13/train_yolo_13.py"`. Los pesos preentrenados (`yolo11s.pt`) los baja
ultralytics la primera vez. Los recortes de la CNN se regeneran con
`python "YOLO - 13/Dataset YOLO 13/create_cnn_dataset.py"`. El script de la CNN también acepta `--eval-only` para
evaluar en test sin reentrenar.

---

## Cómo se arman las cartas (identificador)

1. **Detección:** el pipeline elegido devuelve las esquinas, cada una con su valor y su confianza. Se usa
   *NMS agnóstico a la clase*: si dos cajas casi coinciden, queda solo la más confiable aunque tengan valores
   distintos. Por defecto, YOLO solo suprime cajas del mismo valor. En una partida real, la esquina invertida y medio
   tapada de un 3♥ salía dos veces, como "3" y como "5", y la "5" contaba como una carta de más. En el test propio
   había 4 casos así (8/9, 8/9, 5/3, 6/5), y con NMS agnóstico no queda ninguno.
2. **Emparejamiento:** las dos esquinas con índice de una carta están en vértices opuestos. Un par se acepta si cumple
   tres condiciones:
   - Las dos esquinas tienen **el mismo valor**. Hay una excepción: si la geometría es casi perfecta (costo ≤ 0,5),
     se unen aunque el detector les haya dado valores distintos, y la carta toma el valor de la esquina más
     confiable. Así se corrige la esquina invertida y desenfocada que se lee mal. Los pares del mismo valor siempre
     tienen prioridad.
   - La distancia entre sus centros, **relativa al tamaño de la esquina**, está entre 4,9 y 7,0 (en las fotos reales
     da 5,5–6,1). Al ser relativa, no depende de la resolución ni de la altura de la cámara.
   - El ángulo de la diagonal es cercano a **50°** si la carta está vertical, o a **140°** si está rotada 90°. La
     orientación se deduce de la forma del índice: más alto que ancho, o al revés.

   Se evalúan todos los pares posibles y se aceptan de mejor a peor, sin reutilizar esquinas. Así, con varias cartas
   del mismo valor juntas, cada esquina se une con su pareja más probable.
3. **Esquinas sueltas:** una esquina sin pareja cuenta como carta, porque puede ser una carta tapada a medias.
4. **Reparto y puntaje:** cada carta va a la Casa o al Jugador según su centro. J, Q y K valen 10, y el As vale 11 si
   no pasa de 21 (si no, 1). Luego se determina el resultado: alguien se pasa, blackjack, gana uno o empate.

Con las 4 fotos de prueba, en total 16 variantes (original, rotada 90°, rotada 180° y escala 50%), el emparejamiento
arma bien las cartas detectadas en los 16 casos. Las fallas que quedan vienen de esquinas que el detector no encuentra.

### En video: seguimiento (`seguimiento.py`)

Cada cuadro detectado por separado "parpadea": una esquina se pierde por desenfoque o se lee mal un instante. Se
estabiliza en tres niveles:

1. **Votación por track:** ByteTrack (incluido en ultralytics, con filtro de Kalman e IoU) le da a cada esquina un id
   estable. Su valor es el más votado, ponderado por confianza, en sus últimos 15 cuadros.
2. **Memoria de esquinas:** una esquina vista hace menos de 10 cuadros (unos 0,7 s) que en este cuadro no se detectó
   se mantiene en su última posición. No se aplica si se superpone con una detección actual, que sería la misma
   esquina con otro id.
3. **Mediana por carta:** la mano mostrada se arma carta por carta. Para cada lado y cada valor se toma la mediana de
   cuántas cartas de ese valor hubo en los últimos 15 cuadros. Una esquina perdida en un cuadro y un error de lectura
   en otro no se combinan en una mano equivocada, cosa que sí pasaría eligiendo la mano completa más frecuente.

---

## Reglas del Blackjack (`blackjack.py`)

El motor no inventa cartas: **la cámara es la fuente de verdad**. Recibe en cada cuadro la mano estabilizada por el
seguimiento y los gestos, y recorre las etapas `ESPERANDO → REPARTO → JUGADOR → CASA → FIN`:

- **Reparto:** 2 cartas al Jugador y 1 visible a la Casa. La cámara no ve la carta boca abajo, que aparece cuando se
  da vuelta. Con un blackjack natural, el Jugador no juega: la Casa solo da vuelta su carta.
- **Turno del Jugador:**
  - Pedir: el motor queda esperando una carta nueva en la zona del Jugador.
  - Con más de 21 pierde.
  - Con 21 se planta automáticamente.
- **Turno de la Casa:** pide hasta 17. Con 17 blando, se configura si pide (H17) o no (S17, por defecto). Como la
  carta oculta no se ve, la Casa no puede "espiarla" para cantar blackjack temprano. Su blackjack se resuelve al darla
  vuelta, como en la variante europea.
- **Pagos:** ganar +1, blackjack +1,5, perder −1, empate 0. Se lleva el saldo y el historial de todas las manos.

**Robustez frente a la visión:**
- **Cambios confirmados:** un cambio en la mesa se acepta recién si se repite 5 cuadros seguidos.
- **Cartas tapadas:** durante una mano las cartas solo se agregan, así que una carta que deja de verse un rato (la
  tapa una mano) se mantiene.
- **Lecturas corregidas:** si una carta cambia de valor con la misma cantidad de cartas, se toma como corrección.
- **Irregularidades:** el motor registra, por ejemplo, una carta que nadie pidió o una carta para la Casa en el turno
  del Jugador.
- **Fin de la mano:** termina cuando la mesa queda vacía.

Hay **23 tests** con secuencias de mano, cartas tapadas, ruido de un cuadro, lecturas corregidas, gestos fuera de
turno y saldo: `python -m pytest tests`.

## Gestos de la mano (`Gestos/gestos.py`)

| Acción | Gesto | Detalle |
|---|---|---|
| **Pedir carta** | **APUNTAR**: solo el índice extendido | Sostenido 6 cuadros (~0,5 s) |
| **Plantarse** | **PALMA**: los 4 dedos extendidos | Sostenida 12 cuadros (~1,2 s) **o** desplazada ≥ 1 tamaño de palma (barrido) |

1. **Detección de la mano:** [MediaPipe HandLandmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)
   está preentrenado, corre en CPU en tiempo real y da 21 puntos por mano. Es estimación de pose de la mano.
2. **Clasificación de la pose:** se hace con reglas sobre esos puntos. Un dedo está extendido si
   `dist(muñeca, punta) / dist(muñeca, nudillo) > 1,45`, y doblado si es `< 1,25`. Los umbrales se calibraron con
   HaGRID. Son cocientes de distancias, así que **no dependen de la rotación ni del tamaño de la mano**, algo clave con
   cámara cenital.
3. **Lógica temporal:** pide un tiempo mínimo para cada gesto y deja una espera de 20 cuadros entre gestos.

**Por qué reglas propias y no el reconocedor de MediaPipe:** el Gesture Recognizer preentrenado de MediaPipe
(`Pointing_Up`, `Open_Palm`) **no reconoce nada cuando la mano está rotada 90° o 180°**, mientras que las reglas
propias funcionan igual en cualquier orientación (experimento 6).

**Dos formas de plantarse.** En la primera versión solo valía el barrido. La razón: en HaGRID, la mayoría de los
falsos "PALMA" eran la **otra mano** de la persona, relajada y abierta, y en la mesa eso sería un jugador que apoya la
mano y se planta sin querer.

En la primera prueba con cámara real, el jugador intentó plantarse **mostrando la palma quieta** sobre sus cartas.
Al analizar el video se vio que la pose PALMA se reconocía en 324 cuadros, pero la mano se movía menos de 0,2 palmas
y el gesto nunca se disparaba. Por eso ahora valen las dos formas:
- **Palma sostenida:** hay que mantenerla 1,2 s, más que una mano apoyada de paso.
- **Barrido:** sigue disponible y es más rápido.

Además, después de un gesto **hay que soltar la pose** para que se dispare otro. Con `--plantarse barrido` o
`--plantarse estatico` se usa una sola de las dos formas.

Con el video de esa primera partida, la versión nueva detecta el primer intento de plantarse 1,2 s después de abrir
la palma, y cada intento una sola vez; la versión anterior no detectaba ninguno.

## Experimentos y resultados

### 1. Datos sintéticos (test de cada dataset)

| Modelo | Métrica | Valor |
|---|---|---|
| YOLO-13 | mAP50 / mAP50-95 (valid) | 0,995 / 0,989 |
| YOLO-1 | mAP50 / mAP50-95 (valid) | 0,995 / 0,978 |
| CNN-13 | Exactitud / F1 macro (test, 4.040 recortes) | 1,000 / 1,000 |

Con datos sintéticos el problema queda prácticamente resuelto, así que estos números no dicen cómo funciona el sistema
con cartas reales.

### 2. Brecha entre datos sintéticos y reales

`Experimentos/evaluar_real.py` mide los dos pipelines sobre fotos reales que **ningún modelo vio al entrenar**. Una
detección cuenta como acierto si coincide con la caja real (IoU ≥ 0,5) **y** tiene el valor correcto. Las métricas
por esquina (P, R, F1) se calculan con confianza ≥ 0,5.

Resultados con las 98 fotos reales de teogopk, imágenes originales. Ninguna se usó para entrenar los modelos base:

| Split | Imágenes | Esquinas | A (YOLO-13): mAP50 / F1 | B (YOLO-1 + CNN-13): mAP50 / F1 |
|---|---|---|---|---|
| train | 69 | 238 | 0,77 / 0,73 | 0,82 / 0,81 |
| valid | 18 | 49 | 0,79 / 0,79 | 0,78 / 0,75 |
| test | 11 | 37 | 0,69 / 0,71 | 0,68 / 0,72 |

**Con datos sintéticos los modelos dan 0,99–1,00; con fotos reales caen a 0,7–0,8.** Esta brecha entre lo sintético
y lo real es el resultado central del trabajo, y lo que la motiva es lo siguiente: los datasets públicos de cartas
son casi todos sintéticos, y el 100% en el test sintético no anticipa el rendimiento real.

La CNN-13 también cae: clasifica bien el 83,7% de los recortes reales del valid, contra el 100% en el test sintético.

### 3. Fine-tuning con fotos reales

`Experimentos/finetune_real.py` parte de los pesos ya entrenados. En cada época mezcla 400 imágenes sintéticas al
azar (4.000 recortes en el caso de la CNN) con el train real repetido 5 veces, para que las fotos reales pesen más
sin olvidar lo aprendido con lo sintético. El mejor modelo se elige con el valid real, y el **test real no se usa
hasta la evaluación final**.

Resultado en el **test real** (11 fotos que no se usaron ni para entrenar ni para elegir el modelo), promedio de las
4 variantes (original, rotada 90°, rotada 180° y al 50%):

| Configuración | mAP50 | F1 | R_loc | Latencia (CPU) |
|---|---|---|---|---|
| A: YOLO-13 | 0,69 | 0,62 | 0,60 | 98 ms |
| **A_ft: YOLO-13 con fine-tuning** | **0,89** | **0,85** | 0,83 | 95 ms |
| B: YOLO-1 + CNN-13 | 0,57 | 0,62 | 0,62 | 109 ms |
| B_ft: ambos con fine-tuning | 0,70 | 0,71 | **0,89** | 119 ms |

- **A mejora mucho:** el mAP50 pasa de 0,69 a 0,89, y en la imagen original de 0,69 a 0,90.
- **En B, el detector mejora pero el clasificador no alcanza:** YOLO-1 con fine-tuning es el que mejor ubica las
  esquinas (R_loc 0,89), pero la CNN sigue limitando el resultado. En el valid real pasa de 83,7% a 89,8% de
  exactitud, lejos de lo que logra YOLO-13.

**Pero el fine-tuning no se generaliza a otro mazo.** Las 4 fotos de prueba (mazo Bicycle Dragon, un diseño distinto
al de teogopk) son la prueba:

| Modelo | 4 fotos de prueba: manos correctas /16 (conf 0,5 / 0,3 / 0,2) | Test sintético: mAP50-95 |
|---|---|---|
| A base | **15 / 15 / 16** | **0,989** |
| A_ft | 12 / 14 / 15 | 0,957 |

Con fine-tuning, A pierde las esquinas medio tapadas de `ACES_TAPADOS` y tiene confianzas más bajas con ese mazo.
Además olvida un poco el dominio sintético. La conclusión es que **ajustar con fotos reales sirve, pero hay que
hacerlo con fotos del mazo y la mesa donde se va a usar el sistema**: con 69 fotos de otro mazo, el modelo se
especializa en ese mazo. Esto motivó el experimento siguiente.

### 3b. Fine-tuning con fotos propias (el escenario de la demo)

Se grabaron 4 sesiones de juego con la cámara de la demo (un celular en vertical, como webcam), usando
`--guardar-crudo`. De ahí se extrajeron 130 cuadros con `Dataset Real/extraer_cuadros.py`: uno cada 1,5 s, sin
repetidos ni movidos y sin las franjas negras. Los cuadros se pre-etiquetaron con YOLO-13 y se corrigieron a mano en
Roboflow.

El dataset propio quedó con **1.688 esquinas**:
- **train:** 79 cuadros, de 2 sesiones.
- **valid:** 21 cuadros, de 1 sesión.
- **test:** 30 cuadros, de **otra sesión**, con 412 esquinas.

Son mesas con muchas cartas, superpuestas y medio tapadas, en `Dataset Real/roboflow/`. El fine-tuning
(`--dataset propio`) usa la misma receta que con teogopk.

Resultados en el **test propio** (30 cuadros de una sesión que no se usó para entrenar), imagen original:

| Configuración | mAP50 | Precisión | Recall | F1 | Latencia (CPU) |
|---|---|---|---|---|---|
| A: YOLO-13 base | 0,86 | 0,95 | 0,66 | 0,78 | 117 ms |
| A_ft: ajustado con teogopk | 0,91 | 0,93 | 0,67 | 0,78 | 108 ms |
| **A_propio: ajustado con fotos propias** | **0,99** | **0,96** | **0,96** | **0,96** | 111 ms |
| B: YOLO-1 + CNN-13 base | 0,82 | 0,99 | 0,64 | 0,78 | 144 ms |
| B_ft: ajustado con teogopk | 0,90 | 0,95 | 0,73 | 0,83 | 148 ms |
| B_propio: ajustado con fotos propias | 0,94 | 0,92 | 0,93 | 0,93 | 155 ms |

Promedio de las 4 variantes: A_propio 0,95 de mAP50 y 0,91 de F1, contra 0,86 y 0,78 del base. B_propio queda en 0,91
y 0,88.

- **Con fotos del escenario real, el recall pasa de 0,66 a 0,96.** El modelo base casi no se equivoca
  (precisión 0,95), pero en mesas con muchas cartas amontonadas pierde un tercio de las esquinas. Con fotos propias
  las encuentra casi todas, sin perder precisión. El ajuste con teogopk, otro mazo, casi no cambia el recall
  (0,66 → 0,67).
- **A sigue siendo mejor que B** también después de ajustar los dos con los mismos datos, y es ~40 ms más rápido.
  La CNN-13 ya clasificaba bien este mazo (97,9% de exactitud en el valid propio antes de ajustar, 98,7% después).
  La mejora de B viene sobre todo del detector.
- **Especialización en la orientación:** todas las fotos propias se grabaron con el celular en vertical. Con la
  imagen rotada 90°, A_propio baja a 0,82 de F1, un poco por debajo del base (0,84). Para jugar con la cámara en
  horizontal habría que sumar una sesión grabada así.
- **Fuera del escenario no empeora:** con las 4 fotos de prueba del mazo Bicycle Dragon, A_propio empata con el base
  (15/16 manos correctas, con confianza 0,5).
- **Manos (`cartas_ok`):** pasan de 7% a 30% en la imagen original. Siguen bajas porque en estos cuadros hay 8 a 12
  cartas por mesa y basta una esquina mal para fallar la mano. En una mano real de Blackjack (2 a 5 cartas por lado),
  el sistema anda bien, como se vio al jugar.

**Decisión:** la aplicación usa por defecto los pesos ajustados con fotos propias si existen (`--pesos auto`), y si
no, los base. También se pueden elegir con `--pesos base|real|propio`.

### 4. Robustez (fotos propias del mazo Bicycle Dragon)

Se evaluaron las 4 fotos de prueba en sus 4 variantes (original, rotada 90°, rotada 180° y al 50%), contando una
imagen como correcta si las cartas de la Casa y del Jugador son exactamente las reales:

| Pipeline | Manos correctas | Latencia (CPU) |
|---|---|---|
| A: YOLO-13 | 15/16 | 59 ms/imagen |
| B: YOLO-1 + CNN-13 | 10/16 → **13/16** uniendo esquinas de valores distintos | 77 ms/imagen |

El B falla sobre todo porque la CNN confunde el índice del A♦ de este mazo con un "4", aunque en el test sintético
tenga 100% de exactitud. La CNN solo ve un recorte chico y aprendió el estilo del mazo sintético. YOLO-13 ve más
contexto y generaliza mejor. Unir esquinas con geometría casi perfecta aunque tengan valores distintos recupera 3 de
esos casos: la carta toma el valor de la esquina más confiable.

### 5. Seguimiento en video

`Experimentos/evaluar_seguimiento.py` genera, a partir de cada foto de prueba, un video de 10 s a 15 FPS que simula
una cámara en mano: rotación de ±6°, zoom de ±8%, traslación de ±4%, desenfoque por movimiento, cambios de luz, ruido y
compresión JPEG. Cada video se procesa con y sin seguimiento, con confianza 0,3. La mano real es la misma en todo el
video.

| Pipeline | Seguimiento | Mano correcta (promedio de los 4 videos) | Cambios de mano en 10 s |
|---|---|---|---|
| A: YOLO-13 | no | 50,5% | 47,3 |
| A: YOLO-13 | **sí** | **76,2%** | **4,3** |
| B: YOLO-1 + CNN-13 | no | 24,0% | 78,0 |
| B: YOLO-1 + CNN-13 | sí | 34,8% | 17,5 |

Detalle por video del pipeline A (mano correcta sin → con seguimiento): `prueba` 61% → **90%**, `ACES` 70% → **91%**,
`aces torcidos` 62% → 81% y `ACES_TAPADOS` 9% → 43%. En este último, el A♣ que tiene una sola esquina visible, junto
a la caja del mazo, se detecta en apenas 27 de los 150 cuadros: si una esquina falta la mayor parte del tiempo, el
seguimiento no la puede recuperar. El resultado completo está en `Experimentos/resultados/seguimiento_videos.csv`.

### 6. Poses de la mano en imágenes reales (HaGRID)

`Gestos/evaluar_gestos.py` usa 40 fotos por clase de [HaGRID](https://github.com/hukenovs/hagrid) (10 clases, 400
imágenes): "one" debe dar APUNTAR, "palm", "stop" y "four" deben dar PALMA, y el resto no debe disparar nada. Cada
imagen se prueba original y rotada 90° y 180°:

| Método | Rotación | Recall APUNTAR | Recall PALMA | Falsos disparos | Sin mano detectada |
|---|---|---|---|---|---|
| **Reglas propias** | original | 0,80 | 0,89 | 10,0% | 9,2% |
| **Reglas propias** | 90° | 0,78 | 0,92 | 10,4% | 10,0% |
| **Reglas propias** | 180° | 0,83 | 0,91 | 9,2% | 9,5% |
| MediaPipe Gesture Recognizer | original | 0,80 | 0,52 | 0,4% | 9,2% |
| MediaPipe Gesture Recognizer | 90° / 180° | **0,00** | **0,00** | — | ~10% |

Cuando MediaPipe encuentra la mano, la regla clasifica bien casi siempre. Los errores se reparten así:
- **Manos no detectadas:** cerca del 9%.
- **Falsos "PALMA":** salen de la otra mano abierta de la foto. Por eso la palma tiene que sostenerse 1,2 s o
  moverse.

### 7. Gestos en video

`Gestos/simular_gestos.py` arma clips de 20 cuadros con manos reales de HaGRID pegadas sobre la mesa, con rotación al
azar, quietas o barriendo, y los pasa por el reconocedor completo:

| Caso | Clips | Resultado |
|---|---|---|
| Palma / stop / four **barriendo** → PLANTARSE | 45 | 37 detectados (82%) |
| Palma **quieta** (sostenida 20 cuadros) → PLANTARSE | 15 | 13 detectados (87%) |
| Índice quieto → PEDIR | 15 | 4 detectados (27%) |
| Sin gesto (puño, pulgar arriba/abajo, paz, teléfono, tres; quietas o barriendo) | 120 | 5 falsos disparos (4,2%) |

Con `--plantarse barrido` (palma quieta no vale), los falsos disparos bajan a **0,7%** (1 de 135, contando también
las palmas quietas). Es el costo de aceptar la palma sostenida. En la práctica los gestos solo cuentan en el turno
del Jugador y en su mitad de la mesa.

La tasa baja de PEDIR se explica por el clip sintético: MediaPipe no encuentra la mano en esos recortes, que son
chicos (~100 px) y agrandados. Cuando la detecta, la pose sale APUNTAR en el 100% de los cuadros. En imágenes HaGRID
completas, APUNTAR tiene un recall de 0,80 (experimento 6). Este número tiene que confirmarse con cámara real.

### 8. Partidas completas simuladas

`Experimentos/simular_partida.py` arma la mesa cuadro a cuadro con **recortes de cartas reales** del mazo Bicycle
Dragon sobre un fondo de mármol: reparto, la carta oculta de la Casa que después se da vuelta y cartas que se
levantan. Agrega movimiento de cámara y ruido. Con `--gestos`, los gestos los hace una **mano real de HaGRID** sobre
la mesa y los reconoce el sistema. Las 3 manos de prueba son:

1. El Jugador pide y gana 18 a 17.
2. Blackjack del Jugador.
3. El Jugador pide con 20 y se pasa.

| Gestos | Manos con el resultado correcto | Saldo final (esperado +15) | Irregularidades |
|---|---|---|---|
| Inyectados | 3/3 | +15 | 0 |
| **Reconocidos con MediaPipe** | **3/3** | **+15** | 0 |

Los videos quedan en `videos/partida_simulada*.mp4`.

### Cómo reproducir

```bash
python Experimentos/evaluar_real.py --dataset teogopk                 # pesos base (y fine-tunings si existen)
python Experimentos/finetune_real.py --dataset teogopk --modelo todos # pesos *-finetune-real (--device mps en la Mac)
python Experimentos/evaluar_real.py --dataset propio                  # test con fotos del mazo de la demo
python Experimentos/finetune_real.py --dataset propio --modelo todos  # pesos *-finetune-propio
python Experimentos/evaluar_real.py --dataset <otro export YOLO> --split test
python Experimentos/evaluar_seguimiento.py                # genera los videos si faltan
python Gestos/evaluar_gestos.py                           # baja HaGRID (400 imágenes) la primera vez
python Gestos/simular_gestos.py
python Experimentos/simular_partida.py [--gestos]
python -m pytest tests
```

---

## Limitaciones y trabajo futuro

- **Dataset real chico:** el test real de teogopk tiene 11 fotos, de un solo palo y con un mazo de índices en las
  cuatro esquinas. Por eso las métricas de "manos" (`cartas_ok`) no son confiables con ese dataset; sirven las
  métricas por esquina. El test propio (30 cuadros, 412 esquinas) es más representativo.
- **Fine-tuning con otro mazo:** el ajuste con teogopk (un solo palo, otro diseño) no se generaliza a otro mazo. El
  ajuste con fotos propias sí funciona (recall de 0,66 a 0,96), pero queda especializado en ese mazo, esa mesa y la
  cámara en vertical.
- **Test propio de una sola sesión:** las 30 fotos de test son de una sesión distinta a las de entrenamiento, pero
  del mismo mazo y la misma mesa. Faltaría medir con otra mesa y otra iluminación.
- **Espejado en YOLO:** los YOLO se entrenaron con espejado horizontal (`fliplr=0.5`, el valor por defecto), que
  genera índices espejados que no existen en un mazo real. Probar `fliplr=0` es una mejora pendiente.
- **Cartas tapadas:** si las dos esquinas de una carta quedan tapadas, la carta no se cuenta. Si una carta tiene las
  dos esquinas visibles pero no se emparejan, se cuenta dos veces.
- **Reparto Casa / Jugador:** se hace por la mitad de la imagen. Una mesa real necesitaría zonas configurables.
- **Videos de prueba simulados:** se generaron a partir de fotos, con movimiento de cámara simulado. Falta medir con
  video real del mazo de la demo, con cartas que entran, salen y se superponen.
- **Contar cartas jugadas:** el seguimiento estabiliza la mano actual, pero todavía no cuenta las cartas que ya salieron
  del mazo, que sería útil para la estrategia.
- **Gestos sin probar en cámara real:** se evaluaron con fotos y clips de HaGRID. Falta medirlos con cámara real y un
  jugador, sobre todo PEDIR. Doblar, dividir y apostar no están implementados.
- **Un solo Jugador:** el reparto entre Casa y Jugador es por mitades de la imagen. Varios jugadores necesitarían una
  zona por jugador.

## Créditos y licencias

- **Dataset sintético base:** Playing Cards, de Augmented Startups (Roboflow Universe, dominio público).
- **Dataset real:** Teodor Kostadinov, [TeogopK/Playing-Cards-Object-Detection](https://github.com/TeogopK/Playing-Cards-Object-Detection) (CC0).
- **Modelos preentrenados:** YOLO11 de [Ultralytics](https://github.com/ultralytics/ultralytics) (AGPL-3.0) y ResNet-18 de torchvision (ImageNet).
- **Gestos:** [MediaPipe](https://github.com/google-ai-edge/mediapipe) HandLandmarker y GestureRecognizer (Apache-2.0).
  [HaGRID](https://github.com/hukenovs/hagrid) (Kapitanov et al., CC BY-SA 4.0), muestra `cj-mills/hagrid-sample-30k-384p`
  de Hugging Face, usada solo para evaluar.
