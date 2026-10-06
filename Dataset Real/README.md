# Dataset Real: fotos reales de cartas

Entrenamos los tres modelos con imágenes sintéticas, y en ese dominio el problema queda prácticamente resuelto
(mAP50 0,995 y 100% de exactitud de la CNN-13). Para saber cómo funciona el sistema con cartas de verdad necesitábamos
fotos reales, y las usamos para dos cosas:

1. **Test real:** medir los pipelines A y B en el dominio donde se usa el sistema.
2. **Fine-tuning:** ajustar los modelos con fotos reales para achicar la brecha entre lo sintético y lo real.

En esta carpeta hay dos datasets reales:

| Carpeta | Origen | Tamaño |
|---|---|---|
| `teogopk/` | Público: [TeogopK/Playing-Cards-Object-Detection](https://github.com/TeogopK/Playing-Cards-Object-Detection), `data/real_dataset` (CC0). Ver `teogopk/FUENTE.txt` | 98 fotos 416×416: 69 / 18 / 11 |
| `roboflow/` | Propio: cuadros de nuestras partidas, etiquetados en Roboflow ([`rodolfo-di-chiazza/tp2-vision-por-computadora-ii` v2](https://universe.roboflow.com/rodolfo-di-chiazza/tp2-vision-por-computadora-ii/dataset/2), CC BY 4.0) | 130 cuadros 508×720, 1.688 esquinas: 79 / 21 / 30 |

Empezamos con teogopk, pero tiene tres limitaciones: un solo palo (corazones), un mazo con índices en las cuatro
esquinas y pocas K. Además, el fine-tuning con esas fotos no mejoró el rendimiento con nuestro mazo. Por eso armamos
un dataset propio en el escenario de la demo: nuestra mesa, nuestro mazo y la misma cámara con la que jugamos.

## Cómo armamos el dataset propio

### 1. Grabación

En lugar de sacar fotos sueltas, grabamos **4 sesiones de juego** con la cámara de la demo: un celular cenital usado
como webcam, en vertical. Usamos `--guardar-crudo`, que graba la imagen limpia de la cámara, sin las cajas ni los
textos que dibuja la aplicación (esos dibujos contaminarían el entrenamiento):

```
python Identificador_Cartas/identificador_cartas.py --camara 1 --conf 0.3 --juego --gestos --guardar-crudo videos/crudo_sesion1.mp4
```

Entre una sesión y otra fuimos cambiando las cartas para cubrir los 13 valores. Las mesas tienen muchas cartas
superpuestas y medio tapadas, que es justamente lo difícil.

### 2. Extracción de cuadros y separación por sesión

Con `extraer_cuadros.py` tomamos un cuadro cada 1,5 s. El script descarta los cuadros casi iguales al anterior
(compara por bloques, porque agregar una carta cambia solo una parte de la imagen) y los movidos o desenfocados, y
recorta las franjas negras que agrega el celular al grabar en vertical.

Separamos los splits **por sesión, no al azar**: si dos cuadros casi iguales quedan uno en train y otro en test, la
métrica de test sale inflada. Todos los cuadros de un mismo video van al mismo split:

| Split | Sesiones | Cuadros |
|---|---|---|
| train | sesiones 1 y 2 | 38 + 41 = 79 |
| valid | sesión 3 | 21 |
| test | sesión 4 | 30 (412 esquinas) |

```
python "Dataset Real/extraer_cuadros.py" videos/crudo_sesion1.mp4 videos/crudo_sesion2.mp4 --split train
python "Dataset Real/extraer_cuadros.py" videos/crudo_sesion3.mp4 --split valid
python "Dataset Real/extraer_cuadros.py" videos/crudo_sesion4.mp4 --split test
```

Los cuadros quedan en `fotos/{train,valid,test}/`. Esa carpeta no va al repositorio, porque las imágenes finales
están en `roboflow/`.

### 3. Pre-etiquetado

Etiquetar 1.688 esquinas a mano desde cero lleva mucho tiempo. Por eso `prelabel.py` detecta las esquinas con
YOLO-13 y genera `para_roboflow/{train,valid,test}/` con las imágenes y las etiquetas propuestas, en formato YOLO:

```
python "Dataset Real/prelabel.py"
```

### 4. Corrección en Roboflow

Subimos cada carpeta de `para_roboflow/` a un proyecto de Object Detection en Roboflow, eligiendo el split a mano
("Upload to: Train / Valid / Test") para que Roboflow no los repartiera al azar. Después **revisamos todas las cajas**:
agregamos las esquinas que faltaban, borramos las detecciones falsas y corregimos los valores mal asignados. Usamos
el mismo criterio que los datasets sintéticos: la caja encierra el índice de la esquina (valor y palo), y se
etiquetan las dos esquinas de cada carta si se ven.

Esta revisión es importante: sin ella, el test real quedaría sesgado a favor del pipeline A, porque las etiquetas
las propuso YOLO-13.

Finalmente generamos una versión **sin aumentos de datos y sin redimensionar**, la exportamos en formato YOLOv11 y la
descomprimimos en `roboflow/` (con `data.yaml`, `train/`, `valid/` y `test/`).

## Cómo lo usamos

Desde la raíz del repositorio:

```
python Experimentos/evaluar_real.py --dataset propio                      # evalúa todas las configuraciones en el test propio
python Experimentos/finetune_real.py --dataset propio --modelo todos      # fine-tuning (en la Mac: --device mps)
```

El fine-tuning usa solo train y valid; el test no se toca hasta la evaluación. Los pesos quedan en las carpetas
`*-finetune-propio`, separados de los ajustados con teogopk (`*-finetune-real`). Los resultados quedan en
`Experimentos/resultados/` y están resumidos en el [README principal](../README.md) (experimentos 3 y 3b).

Con 79 cuadros propios, el recall del pipeline A en el test propio pasó de 0,66 a 0,96. Por eso la aplicación usa
estos pesos por defecto (`--pesos auto`).

## Cómo ampliarlo

Para sumar sesiones (por ejemplo con otra mesa, otra luz o la cámara en horizontal) repetimos los mismos pasos:
grabar con `--guardar-crudo`, extraer cuadros con todos los de una sesión en un mismo split, pre-etiquetar, corregir
en Roboflow, exportar una versión nueva en `roboflow/` y volver a correr el fine-tuning y la evaluación.
