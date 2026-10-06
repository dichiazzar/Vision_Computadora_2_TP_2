# Cómo jugar

Para jugar necesitamos tres cosas: una cámara mirando la mesa, un mazo y alguien que haga de crupier (puede ser la
misma persona que juega).

## 1. Preparación

- **Cámara cenital, a unos 40–60 cm de la mesa.** Nosotros usamos el celular como webcam (Vínculo a Windows / Phone
  Link; también sirven DroidCam o Camo), sostenido arriba de la mesa con un soporte. Recomendamos ponerlo **en
  horizontal**: en vertical la imagen queda angosta y se pierde resolución.
- **La imagen se divide en dos:** la mitad de **arriba es la Casa** y la de **abajo, el Jugador**, separadas por una
  línea roja.
- **Luz pareja y sin reflejos**, con la esquina de cada carta (el valor) visible.
- **Número de la cámara:** suele ser 0 para la de la notebook y 1 para el celular. Para ver cuáles hay:
  ```
  python -c "import cv2; [print(i, cv2.VideoCapture(i).read()[0]) for i in range(4)]"
  ```

## 2. Arrancar

```
python Identificador_Cartas/identificador_cartas.py --camara 1 --conf 0.3 --juego --gestos --guardar videos/partida.mp4
```

- **`--guardar`** es opcional: graba la partida en H.264, el formato que aceptan WhatsApp y PowerPoint.
- **Panel de la derecha:** muestra la etapa, los puntajes, el saldo y los últimos eventos.
- **Abajo a la izquierda:** aparece la instrucción de cada momento.

## 3. Una mano

1. **Reparto.** El panel indica "Repartir: 2 cartas al Jugador y 1 a la Casa".
   - 2 cartas **boca arriba** en la zona del Jugador.
   - 1 carta boca arriba en la zona de la Casa y **otra boca abajo** al lado.
   - Conviene repartir de a una y sin apuro: cada carta tarda ~1 s en confirmarse.
2. **Turno del Jugador** (etapa JUGADOR):
   - **Pedir carta:** el Jugador levanta solo el índice en la mitad de abajo durante ~0,5 s. Cuando el panel indica
     "pidió carta: repartirle una", el crupier deja una carta nueva boca arriba en la zona del Jugador. Mientras el
     sistema espera esa carta ignora los gestos, así que la mano que reparte no dispara nada.
   - **Plantarse:** el Jugador muestra la **palma abierta quieta ~1,2 s** sobre sus cartas, o **pasa la mano
     abierta** por encima (más rápido).
   - **Entre un gesto y otro hay que retirar la mano.**
   - Con más de 21 el Jugador pierde al instante. Con 21 se planta solo.
3. **Turno de la Casa:**
   - El crupier da vuelta la carta oculta.
   - Mientras el panel diga **"Casa: debe pedir carta"**, agrega cartas en la zona de la Casa.
   - Cuando diga "se planta", sale el resultado.
4. **Resultado:** aparece abajo y se suma al saldo. La apuesta es fija, de 10, y el blackjack paga 15.
5. **Mano nueva:** se levantan todas las cartas. Cuando la mesa queda vacía unos segundos, empieza otra mano.

## 4. Teclas

Las teclas funcionan con la ventana del juego seleccionada.

| Tecla | Acción |
|---|---|
| `p` / `l` | Pedir / plantarse, por si un gesto no sale |
| `n` | Reiniciar la mano en curso (conserva el saldo) |
| `r` | Partida nueva (saldo en 0) |
| `q` | Salir. La terminal muestra el resumen con todos los eventos |

## 5. Problemas que encontramos y cómo los resolvimos

- **"LA CÁMARA NO ENVÍA IMAGEN":** la cámara del celular se transmite a una sola app a la vez. Nos pasó con Phone
  Link abierto: hay que cerrar Phone Link, la app Cámara o la vista previa de *Configuración → Dispositivos móviles*,
  y aceptar la notificación de Vínculo a Windows en el celular, que tiene que estar desbloqueado.
- **Una carta no se reconoce:** acercar la cámara o mejorar la luz, y evitar que alguna carta cruce la línea roja.
- **Aparece una "irregularidad" en el panel:** el sistema vio algo fuera de orden, por ejemplo una carta que nadie
  pidió. La acepta igual y la deja registrada.
- **Plantarse se dispara solo:** con `--plantarse barrido` solo vale pasar la mano, y la palma quieta deja de contar.
- **Se trabó una mano:** tecla `n`.
