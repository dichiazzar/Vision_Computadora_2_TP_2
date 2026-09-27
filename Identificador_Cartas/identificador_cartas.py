'''Vamos a crear la clase esquina. 
Atributos: coordenadas y valor. Deribamos el centroide
Los almacenamos en un contenedor y si detecta 2 del mismo valor llama a los metodos: inferir si es la misma carta
Si es la misma carta se agranda el bbox
Si no es la misma carta no se hace nada
Al final nos indica que valores hay en la mesa por frame

Se va a tener la clase jugador que es el que va a determinar las cartas de cada uno
'''

from class_corner import Corner
from class_player import Entidad
from class_carta import Carta
from ultralytics import YOLO
import cv2

modelo = YOLO("/Users/julianblanco/Downloads/BlackJack/YOLO - 13/runs/detect/blackjack-cv/yolo13-baseline-3/weights/best.pt")
img_path = "/Users/julianblanco/Downloads/BlackJack/ACES_TAPADOS.jpeg"
frame = cv2.imread(img_path)
height, width = frame.shape[:2]

def main():

    player = Entidad("Jugador")
    dealer = Entidad("Casa")
    resultados = modelo(img_path)
    resultado = resultados[0]
    data = resultado.boxes.data
    for deteccion in data.cpu().numpy():
        # Desencapsular los 6 valores en una sola línea
        xmin, ymin, xmax, ymax, conf, class_id = deteccion.tolist()
        rank = modelo.names[int(class_id)]
        esquina = Corner([xmin, ymin, xmax, ymax], rank)

        if esquina.centroid[1] < height / 2:
            dealer.agregar_esquina(esquina)
        else:
            player.agregar_esquina(esquina)

    frame_resultado = dibujar_bboxes(frame, dealer, player)

    cv2.imshow("Detector de cartas", frame_resultado)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # player.determinar_estado()
    # if player.estado == 1:
    #     print("El Jugador se pasó")
    # else: 
    #     dealer.determinar_estado()
    #     if dealer.estado == 1:
    #             print("La casa se pasó")
    #     else: 
    #         if player.estado < dealer.estado:
    #             print("Ganó el Jugador")
    #         else:
    #             print("Ganó el Dealer")


def dibujar_bboxes(frame, dealer, player):

    entidades = [dealer, player]

    for entidad in entidades:

        # ====================================================
        # DIBUJAR DIAGONALES ENTRE ESQUINAS
        # ====================================================

        for esquina1, esquina2, distancia, angulo in entidad.diagonales:

            cx1, cy1 = esquina1.centroid
            cx2, cy2 = esquina2.centroid

            # Línea entre centroides
            cv2.line(
                frame,
                (int(cx1), int(cy1)),
                (int(cx2), int(cy2)),
                (0, 255, 255),
                2
            )

            # Punto medio de la diagonal
            medio_x = int((cx1 + cx2) / 2)
            medio_y = int((cy1 + cy2) / 2)

            # Texto con distancia
            cv2.putText(
                frame,
                f"d={distancia:.1f}px",
                (medio_x, medio_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2
            )

            # Texto con ángulo
            cv2.putText(
                frame,
                f"a={angulo:.1f}deg",
                (medio_x, medio_y + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2
            )

        # ====================================================
        # 1. DIBUJAR ESQUINAS
        # ====================================================

        for rank, esquinas in entidad.esquinas.items():

            for esquina in esquinas:

                x1, y1 = esquina.sup_izq
                x2, y2 = esquina.inf_der

                # Bounding box de la esquina
                cv2.rectangle(
                    frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (0, 255, 0),
                    2
                )

                # Centroide
                cx, cy = esquina.centroid

                cv2.circle(
                    frame,
                    (int(cx), int(cy)),
                    4,
                    (0, 255, 0),
                    -1
                )

                # Rank detectado
                cv2.putText(
                    frame,
                    str(esquina.rank),
                    (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )


        # ====================================================
        # 2. DIBUJAR CARTAS RECONSTRUIDAS
        # ====================================================

        for carta in entidad.cartas:

            x1, y1 = carta.sup_izq
            x2, y2 = carta.inf_der

            cv2.rectangle(
                frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (255, 0, 0),
                3
            )

            cv2.putText(
                frame,
                f"Carta: {carta.rank}",
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2
            )


    # ========================================================
    # 3. LINEA PLAYER / DEALER
    # ========================================================

    height, width = frame.shape[:2]

    cv2.line(
        frame,
        (0, height // 2),
        (width, height // 2),
        (0, 0, 255),
        2
    )

    cv2.putText(
        frame,
        "DEALER",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2
    )

    cv2.putText(
        frame,
        "PLAYER",
        (20, height // 2 + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2
    )

    return frame


if __name__ == "__main__":

    main()