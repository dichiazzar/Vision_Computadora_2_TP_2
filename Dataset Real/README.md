# Dataset Real: fotos del mazo físico

## Para qué sirve
Los modelos se entrenaron con imágenes sintéticas. Con fotos reales del mazo Bicycle Dragon, el pipeline A acierta
15 de 16 manos, pero el pipeline B solo 10 de 16. La CNN-13 confunde el índice del A♦ con un "4", aunque en el test
sintético tenga 100% de exactitud. Este dataset sirve para dos cosas:

1. **Test real** para el paper: medir A y B en el dominio donde se usa el sistema.
2. **Fine-tuning** con fotos reales para achicar la brecha entre lo sintético y lo real.

## Qué fotos sacar (unas 100 en total)
Sacarlas desde arriba, como se vería la mesa en la demo, con la carta entera dentro del cuadro.

- **Cubrir los 13 valores de forma pareja**, de distintos palos. Que aparezca cada valor unas 15 veces o más en total.
- **De 2 a 6 cartas por foto**, algunas con dos cartas del mismo valor juntas.
- **Variar lo siguiente:**
  - Fondos: madera, mantel, mesa clara y oscura, paño verde.
  - Luz: natural, lámpara, con algo de sombra.
  - Distancia de la cámara: de 30 a 60 cm.
  - Rotación: cartas derechas, giradas de 10 a 45° y a 90°.
- **Incluir casos difíciles:**
  - Cartas con una esquina tapada.
  - Cartas superpuestas.
  - Cartas boca abajo (el dorso no se etiqueta, sirve como ejemplo negativo).
  - Algo de desenfoque.

## Cómo separar las fotos (importante)
Separar **por sesión**, no al azar. Si dos fotos casi iguales quedan una en train y otra en test, la métrica de test
sale inflada. Una sesión es un mismo día, fondo y luz.

| Carpeta | Contenido | Cantidad aprox. |
|---|---|---|
| `fotos/train/` | 2 o 3 sesiones | 60 |
| `fotos/valid/` | 1 sesión | 20 |
| `fotos/test/` | 1 sesión **con un fondo que no aparezca en train** | 20 |

Si las fotos están en formato .heic (iPhone), exportarlas como .jpg.

## Opción recomendada: sacar las fotos de video
En lugar de sacar las fotos una por una, se puede **grabar mientras se juega o se reparte**, con el mismo escenario
de la demo (cámara, mesa y luz), y extraer cuadros de ahí.

1. Grabar la imagen **cruda**, sin cajas ni textos dibujados, que contaminarían el entrenamiento. Usar una sesión por
   video, por ejemplo una para train, otra para valid y otra para test, cambiando algo entre sesiones (luz, hora,
   mantel):
   ```
   python Identificador_Cartas/identificador_cartas.py --camara 1 --conf 0.3 --juego --gestos --guardar-crudo videos/crudo_sesion1.mp4
   ```
   Los videos anotados (`--guardar`) **no sirven** para esto.
2. Extraer cuadros. El script toma uno cada 1,5 s y descarta los repetidos y los movidos. **Todos los cuadros de un
   mismo video van al mismo split:**
   ```
   python "Dataset Real/extraer_cuadros.py" videos/crudo_sesion1.mp4 videos/crudo_sesion2.mp4 --split train
   python "Dataset Real/extraer_cuadros.py" videos/crudo_sesion3.mp4 --split valid
   python "Dataset Real/extraer_cuadros.py" videos/crudo_sesion4.mp4 --split test
   ```
3. Seguir con el pre-etiquetado, igual que con fotos sueltas.

Unos 10 minutos de juego dan del orden de 150–250 cuadros útiles. Conviene ir cambiando las cartas para cubrir los
13 valores.

## Pre-etiquetado y corrección
1. Correr el pre-etiquetado desde la carpeta del repo:
   ```
   python "Dataset Real/prelabel.py"
   ```
   El script detecta las esquinas con YOLO-13 y genera `para_roboflow/{train,valid,test}/` con las imágenes y sus etiquetas.
2. En Roboflow, crear un proyecto nuevo de **Object Detection**.
3. Subir cada carpeta de `para_roboflow/` eligiendo su split. Es la opción "Upload to: Train / Valid / Test", y hay que
   usarla para **no dejar que Roboflow las reparta al azar**.
4. **Revisar todas las cajas**:
   - Agregar las esquinas que faltan.
   - Borrar las detecciones falsas.
   - Corregir los valores mal asignados.
   - Usar el mismo criterio que el dataset original: la caja encierra el índice de la esquina (valor y palo), y se
     etiquetan las dos esquinas de cada carta si se ven.

   Sin esta revisión, el test real queda sesgado a favor del pipeline A, porque las etiquetas las propuso YOLO-13.
5. Generar una versión **sin aumentos de datos y sin redimensionar**, y exportarla en formato **YOLOv11**.
6. Descomprimirla en `Dataset Real/roboflow/`, que tiene que quedar con `data.yaml`, `train/`, `valid/` y `test/`.

## Lo que sigue (después de tener el dataset)
Correr desde la carpeta del repo:

1. **Evaluar los modelos base en el test real** (línea de base para el paper):
   ```
   python Experimentos/evaluar_real.py
   ```
2. **Fine-tuning** de los tres modelos con train real y sintético mezclados. Solo usa train y valid; el test no se toca:
   ```
   python Experimentos/finetune_real.py --modelo todos            # en la Mac: --device mps
   ```
3. **Volver a evaluar.** Ahora aparecen también `A_ft` y `B_ft`, comparados sobre el mismo test real:
   ```
   python Experimentos/evaluar_real.py
   ```
Los resultados quedan en `Experimentos/resultados/` (CSV por configuración y variante, AP por valor y detalle por imagen).
