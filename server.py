from flask import Flask,render_template,request
import cv2
import dlib
from flask_cors import CORS, cross_origin
import cv2
import tensorflow as tf
from typing import List
from moviepy.editor import VideoFileClip
import speech_recognition as sr

def convert_mp4_to_mpg(input_path, output_path):
    video = VideoFileClip(input_path)
    video.write_videofile(output_path, codec='mpeg1video')

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
tf.config.list_physical_devices('GPU')
physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)

def extract_audio(video_path, audio_path):
    video = VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile(audio_path,codec='pcm_s16le')
    audio.close()

def convert_audio_to_text(audio_path):
    recognizer = sr.Recognizer()
    audio_file = sr.AudioFile(audio_path)

    with audio_file as source:
        audio_data = recognizer.record(source)

    text = recognizer.recognize_google(audio_data)
    return text

def CTCLoss(y_true, y_pred):
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

    loss = tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    return loss

model=tf.keras.models.load_model("model",custom_objects={'CTCLoss':CTCLoss})
def calculate_multiple_faces_percentage(video_file: str) -> float:
    cap = cv2.VideoCapture(video_file)
    fl = []
    detector = dlib.get_frontal_face_detector()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        fl.append(len(faces))

        for i, face in enumerate(faces, start=1):
            x, y = face.left(), face.top()
            x1, y1 = face.right(), face.bottom()
            cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)
            cv2.putText(frame, f'face num{i}', (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cap.release()
    cv2.destroyAllWindows()
    if fl.count(0) == len(fl):
        return 100
    count = len([num for num in fl if num > 1])
    percentage = (count / len(fl)) * 100 if fl else 0
    return percentage

def load_data(path: str): 
    path = bytes.decode(path.numpy())
    file_name = path.split('\\')[-1].split('.')[0]
    frames = load_video(file_name+".mpg") 
    return frames

def load_video(path:str) -> List[float]:
    cap = cv2.VideoCapture(path)
    frames = []
    for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))): 
        ret, frame = cap.read()
        if frame is not None:
            frame = tf.image.rgb_to_grayscale(frame)
            frames.append(frame[190:236,80:220,:])
    cap.release()
    
    mean = tf.math.reduce_mean(frames)
    std = tf.math.reduce_std(tf.cast(frames, tf.float32))
    return tf.cast((frames - mean), tf.float32) / std

def lipsync(filename,text):
    sample = load_data(tf.convert_to_tensor(filename))
    yhat = model.predict(tf.expand_dims(sample, axis=0))
    decoded = tf.keras.backend.ctc_decode(yhat, input_length=[75], greedy=True)[0][0].numpy()
    lips=[tf.strings.reduce_join([num_to_char(word) for word in sentence]) for sentence in decoded][0].numpy().decode("utf-8")
    extract_audio(filename, "tempaudio.wav")
    text_result = convert_audio_to_text("tempaudio.wav")
    if lips.lower() == text_result.lower() and lips.lower() == text.lower():
        return "Real"
    else:
        return "Fake"

@app.route('/',methods=['GET','POST'])
@cross_origin()
def hello():
    if request.method =='GET':
        return render_template("frontend.html")
    text=request.form.get("text")
    f=request.files['file']
    filename=f.filename
    f.save(filename)
    if not filename.endswith(".mpg"):
        convert_mp4_to_mpg(filename,"testfile.mpg")
    if calculate_multiple_faces_percentage("testfile.mpg") ==0:
        return lipsync('.\\testfile.mpg',text)
    else:
        return "Fake"
    return "Error"

if __name__ == '__main__':
    app.run(debug=True)