import tensorflow as tf

# 모델 로드
model_dir = './pb'
# 모델 불러오기
loaded_model = tf.saved_model.load(model_dir)

# "serving_default" 서명의 입력 및 출력 텐서 이름 확인
input_tensor_name = loaded_model.signatures["serving_default"].inputs[0].name
output_tensor_name = loaded_model.signatures["serving_default"].outputs[0].name

# 결과 출력
print(f"Input Tensor Name: {input_tensor_name}")
print(f"Output Tensor Name: {output_tensor_name}")
