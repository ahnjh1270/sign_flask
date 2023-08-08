import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, Response, jsonify, request, url_for, redirect, session
from PIL import ImageFont, ImageDraw, Image
import mediapipe as mp
import speech_recognition as sr

app = Flask(__name__)
app.secret_key = "123"
text=""


mp_drawing_styles = mp.solutions.drawing_styles
actions = ['주먹', '안녕', '감사합니다']
seq_length = 30

model = tf.keras.models.load_model('models/model.h5')
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)


@app.route('/')
def index():
    return redirect(url_for('webpage'))

@app.route('/webpage')
def webpage():
    headers = {'Cache-Control': 'no-cache'}
    return render_template('webpage.html'), 200, headers

@app.route('/developer')
def about_us():
    return render_template('developer.html')

@app.route('/ksl')
def ksl():
    headers = {'Cache-Control': 'no-cache'}
    return render_template('ksl_program.html'), 200, headers

@app.route("/stt", methods=["GET", "POST"])
def stt():
    if "transcript" not in session:
        session["transcript"] = ""

    if request.method == "POST":
        r = sr.Recognizer()
        with sr.Microphone() as source:
            print("Say something!")
            audio = r.listen(source)

        try:
            transcript = r.recognize_google(audio, language="ko-KR")
            print("Google Speech Recognition thinks you said " + transcript)
            session["transcript"] += transcript + "\n"  # 이전 입력과 현재 입력을 합쳐 세션에 저장합니다.
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
        except sr.RequestError as e:
            print("Could not request results from Google Speech Recognition service; {0}".format(e))

    return render_template('stt.html', transcript=session["transcript"].split("\n"))

@app.route("/delete", methods=["POST"])
def delete_latest():
    if "transcript" in session:
        transcript = session["transcript"].split("\n")
        if transcript:
            transcript.pop()  # 가장 최근의 입력을 삭제
        session["transcript"] = "\n".join(transcript)
    return {"transcript": session["transcript"].split("\n")}

def calculate_angles(joints):
    v1 = joints[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :3]  # 부모 관절
    v2 = joints[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :3]  # 자식 관절
    v = v2 - v1
    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]
    angles = np.arccos(np.einsum('nt,nt->n', v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                 v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))
    return np.degrees(angles)

def generate_frames():
    cap = cv2.VideoCapture(0)
    seq = []
    action_seq = []
    last_action = None
    global text
    
    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break

        img = cv2.flip(img, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if result.multi_hand_landmarks is not None:
            for res in result.multi_hand_landmarks:
                joint = np.zeros((21, 4))
                for j, lm in enumerate(res.landmark):
                    joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

                angles = calculate_angles(joint)
                d = np.concatenate([joint.flatten(), angles])

                seq.append(d)
                mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS,
                                          mp_drawing_styles.get_default_hand_landmarks_style(),
                                          mp_drawing_styles.get_default_hand_connections_style())

                if len(seq) < seq_length:
                    continue

                input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)

                y_pred = model.predict(input_data).squeeze()

                i_pred = int(np.argmax(y_pred))
                conf = y_pred[i_pred]

                if conf < 0.9:
                    continue

                action = actions[i_pred]
                action_seq.append(action)

                if len(action_seq) < 10:
                    continue

                this_action = '?'
                if all(action == action_seq[-1] for action in action_seq[-7:]):
                    this_action = action

                    if last_action != this_action:
                        if this_action == '주먹':
                            text = '주먹'
                        elif this_action == '안녕':
                            text = '안녕하세요'
                        elif this_action == '감사합니다':
                            text = '감사합니다'
                        last_action = this_action

                

        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_text')
def get_text():
    global text
    # Return the 'text' variable as a JSON response
    return jsonify({'text': text})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
