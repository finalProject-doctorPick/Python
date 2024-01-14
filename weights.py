import tensorflow as tf
from flask import Flask, render_template, request, jsonify
import json

app = Flask(__name__)

# 모델 정의 (예시 모델)
def create_model(input_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model

# 모델 로드
model_weights_path = './weights'
# 예시로 input_dim을 100으로 가정
input_dim = 100
model = create_model(input_dim)
model.load_weights(model_weights_path)

# 모델 구조를 JSON 형식으로 출력
model_json = model.to_json()
print(json.dumps(json.loads(model_json), indent=2))  # JSON 형식으로 예쁘게 출력

# 다른 Flask 함수 및 라우트 정의
# ...

if __name__ == '__main__':
    app.run(debug=True)
