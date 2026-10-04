# Cómo jugar

Para jugar hacen falta tres cosas: una cámara mirando la mesa, un mazo y alguien que haga de crupier (puede ser la
misma persona que juega).

## 1. Preparación

- **Cámara cenital, a unos 40–60 cm de la mesa.** Conviene usar el celular como webcam (Vínculo a Windows / Phone
  Link, DroidCam o Camo), **en horizontal**, sostenido arriba de la mesa con algún soporte. En vertical la imagen
  queda angosta y se pierde resolución.
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

1. **Reparto.** El panel dice "Repartir: 2 cartas al Jugador y 1 a la Casa".
   - 2 cartas **boca arriba** en la zona del Jugador.
   - 1 carta boca arriba en la zona de la Casa y **otra boca abajo** al lado.
   - Repartir de a una y sin apuro: cada carta tarda ~1 s en confirmarse.
2. **Turno del Jugador** (etapa JUGADOR):
   - **Pedir carta:** levantar solo el índice en la mitad de abajo durante ~0,5 s. Cuando el panel dice "pidió
     carta: repartirle una", dejar una carta nueva boca arriba en la zona del Jugador. Mientras se espera esa carta,
     los gestos se ignoran, así que la mano que reparte no dispara nada.
   - **Plantarse:** mostrar la **palma abierta quieta ~1,2 s** sobre las cartas, o **pasar la mano abierta** por
     encima (más rápido).
   - **Antes de hacer otro gesto, retirar la mano.**
   - Con más de 21 se pierde al instante. Con 21, el Jugador se planta solo.
3. **Turno de la Casa:**
   - Dar vuelta la carta oculta.
   - Mientras el panel diga **"Casa: debe pedir carta"**, agregar cartas en la zona de la Casa.
   - Cuando diga "se planta", sale el resultado.
4. **Resultado:** aparece abajo y se suma al saldo. La apuesta es fija, de 10, y el blackjack paga 15.
5. **Mano nueva:** levantar todas las cartas. Cuando la mesa queda vacía unos segundos, empieza otra mano.

## 4. Teclas

Las teclas funcionan con la ventana del juego seleccionada.

| Tecla | Acción |
|---|---|
| `p` / `l` | Pedir / plantarse, por si un gesto no sale |
| `n` | Reiniciar la mano en curso (conserva el saldo) |
| `r` | Partida nueva (saldo en 0) |
| `q` | Salir. La terminal muestra el resumen con todos los eventos |

## 5. Si algo falla

- **"LA CÁMARA NO ENVÍA IMAGEN":** la cámara del celular se transmite a una sola app a la vez. Cerrar Phone Link, la
  app Cámara o la vista previa de *Configuración → Dispositivos móviles*, y aceptar la notificación de Vínculo a
  Windows en el celular, que tiene que estar desbloqueado.
- **Una carta no se reconoce:** acercar la cámara o mejorar la luz. Que ninguna carta cruce la línea roja.
- **Aparece una "irregularidad" en el panel:** el sistema vio algo fuera de orden, por ejemplo una carta que nadie
  pidió. La acepta igual y la deja registrada.
- **Plantarse se dispara solo:** con `--plantarse barrido` solo vale pasar la mano, y la palma quieta deja de contar.
- **Se trabó una mano:** tecla `n`.
