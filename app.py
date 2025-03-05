from flask import Flask, render_template, Response
import cv2
from ultralytics import YOLO

app = Flask(__name__)

# Caminho para o modelo treinado
model_path = "C:\\Users\\Luis Carlos\\Documents\\yolo_web_app\\model\\best.pt"

# Carregar o modelo YOLO
model = YOLO(model_path)

# Configurações da câmera IP
ip_cam = "192.168.1.66"
port = "4747"
stream_url = f"http://{ip_cam}:{port}/video"

# Definir a confiança mínima para exibir detecções
CONFIDENCE_THRESHOLD = 0.4  # Ajuste esse valor conforme necessário

def generate_frames():
    # Inicializar a captura de vídeo
    cap = cv2.VideoCapture(stream_url)

    if not cap.isOpened():
        print("Erro ao acessar a câmera IP!")
        return

    while True:
        ret, frame = cap.read()

        # Verificar se o frame foi capturado corretamente
        if not ret:
            print("Erro ao capturar o frame!")
            break

        # Fazer a detecção no frame usando o modelo YOLO
        results = model(frame)

        # Obter as detecções
        detections = results[0].boxes

        # Desenhar as bounding boxes e labels no frame
        for det in detections:
            confidence = det.conf[0].item()  # Obter o valor de confiança como número normal

            # Filtrar detecções pela confiança mínima
            if confidence < CONFIDENCE_THRESHOLD:
                continue  # Ignorar detecções com confiança abaixo do limite

            # Obter as coordenadas da caixa delimitadora
            x1, y1, x2, y2 = map(int, det.xyxy[0])

            # Obter a classe detectada
            class_id = int(det.cls[0])
            label = results[0].names[class_id]

            # Formatar a string do rótulo com a classe e a confiança
            label_with_confidence = f"{label} {confidence*100:.1f}%"

            # Configurar cor e espessura da caixa e do texto
            color = (0, 255, 0)  # Verde
            box_thickness = 2  # Espessura da caixa
            font_scale = 0.6  # Tamanho do texto
            font_thickness = 2  # Espessura do texto

            # Desenhar a caixa delimitadora
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_thickness)

            # Ajustar a posição do texto para não sair da tela
            label_size = cv2.getTextSize(label_with_confidence, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
            label_x = x1
            label_y = y1 - 10 if y1 - 10 > 10 else y1 + label_size[1] + 10

            # Desenhar o rótulo
            cv2.putText(frame, label_with_confidence, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness)

        # Codificar o frame para envio via stream
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            print("Erro ao codificar o frame.")
            break
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    # Liberar a câmera
    cap.release()
    print("Stream de vídeo encerrado.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Permite que a aplicação seja acessada por outros dispositivos na rede local
    app.run(host='0.0.0.0', port=5000, debug=True) 

