import tempfile
from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import easyocr
import matplotlib.pyplot as plt
import re
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

def clean_text(raw_text):
    # 정규 표현식을 사용하여 특수문자 제거
    cleaned_text = re.sub('[^a-zA-Z0-9가-힣\s]', '', raw_text)
    return cleaned_text

def extract_color(image, color_name, lower_bound, upper_bound):
    # 색상 공간 변환 (BGR -> HSV)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 색상 범위에 대한 마스크 생성
    mask = cv2.inRange(hsv_image, np.array(lower_bound), np.array(upper_bound))

    # 원본 이미지에 색상 마스크를 적용하여 색상 영역 추출
    color_extracted_image = cv2.bitwise_and(image, image, mask=mask)

    # 추출된 색상 영역에서 픽셀 수 계산
    pixel_count = cv2.countNonZero(mask)

    # 결과 이미지 및 픽셀 수 반환
    return color_extracted_image, pixel_count

# 글자 추출만을 위한 이미지 전처리
def preprocess_image(image):
    # 전경과 배경 분리 (GrabCut 알고리즘 사용)
    mask = np.zeros(image.shape[:2], np.uint8)
    rect = (10, 10, image.shape[1]-10, image.shape[0]-10)  # 전경이 있는 사각형 지정
    cv2.grabCut(image, mask, rect, None, None, 10, cv2.GC_INIT_WITH_RECT)

    # 분리된 이미지 얻기
    foreground = np.where((mask == 2) | (mask == 0), 0, 255).astype('uint8')

    # 글자 강조 (CLAHE를 사용하여 지역적인 대비 향상)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # clipLimit를 조정하여 대비를 낮춤
    clahe = cv2.createCLAHE(clipLimit=28.0, tileGridSize=(10, 10))
    enhanced_image = clahe.apply(gray_image)

    # 전경에 대해서만 CLAHE 적용
    enhanced_image = cv2.bitwise_and(enhanced_image, enhanced_image, mask=foreground)

    # 이진화를 통한 글자와 배경 분리
    _, binary_image = cv2.threshold(enhanced_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 샤프닝 적용
    kernel_sharpening = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened_image = cv2.filter2D(binary_image, -1, kernel_sharpening)

    # 모폴로지 연산을 적용하여 작은 잡음 제거 -- 조정 가능
    kernel = np.ones((3, 3), np.uint8)
    morph_image = cv2.morphologyEx(sharpened_image, cv2.MORPH_CLOSE, kernel, iterations=2)

    return morph_image

def identify_pill_properties(image_path, ocr_languages=['en']):
    # 이미지 로드 (기본 BGR 형식)
    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # 이미지가 성공적으로 로드되었는지 확인
    if image_bgr is None:
        print(f"이미지를 로드하는 중 오류가 발생했습니다: {image_path}")
        return

    # 이미지를 그레이스케일로 변환
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # 가우시안 블러 적용
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # 허프 원 변환 적용
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=30, minRadius=30, maxRadius=100
    )

    # 원이 검출되었을 때
    if circles is not None:
        circles = np.uint16(np.around(circles))

        # 결과 이미지를 화면에 표시
        plt.figure(figsize=(12, 6))
        for i in range(circles.shape[1]):
            # 마스크 생성
            mask = np.zeros_like(gray)
            cv2.circle(mask, (circles[0, i, 0], circles[0, i, 1]), circles[0, i, 2], (255, 255, 255), thickness=-1)

            # 마스크를 이용하여 배경 제거
            result_image = cv2.bitwise_and(image_bgr, image_bgr, mask=mask)

            enlargement_factor = 2  # 조절 가능한 상수
            x1, y1 = circles[0, i, 0] - circles[0, i, 2] * enlargement_factor, circles[0, i, 1] - circles[
                0, i, 2] * enlargement_factor
            x2, y2 = circles[0, i, 0] + circles[0, i, 2] * enlargement_factor, circles[0, i, 1] + circles[
                0, i, 2] * enlargement_factor

            # 원을 감싸는 사각형에 맞춰 이미지 자르기
            cropped_image = result_image[int(y1):int(y2), int(x1):int(x2)]

            # 전처리된 이미지 얻기
            processed_image = preprocess_image(cropped_image)

            # EasyOCR 리더 생성
            reader = easyocr.Reader(ocr_languages)

            # 이미지에서 텍스트 읽기
            ocr_result = reader.readtext(processed_image)

            concatenated_text = ""
            confidence_scores = []

            for detection in ocr_result:
                # 특수문자 처리를 추가하여 텍스트 정제
                cleaned_text = clean_text(detection[1])

                # 각 텍스트를 연결
                concatenated_text += cleaned_text.replace(" ", "")

                # 텍스트 및 위치 출력
                print(f"알약 텍스트: {cleaned_text}")
                print(f"경계 상자 좌표: {detection[0]}")
                print(f"신뢰도: {detection[2]}")
                print("-" * 20)

                confidence_scores.append(detection[2])

            # 전체 텍스트 출력
            print("전체 텍스트:", concatenated_text)

            # 평균 신뢰도 계산
            average_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0

            # 원본 이미지 표시
            plt.subplot(1, 2, 1)
            plt.imshow(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
            plt.title('Processed Image')

            # 텍스트를 추출한 이미지 표시
            extracted_text_image = cv2.cvtColor(processed_image.copy(), cv2.COLOR_BGR2RGB)
            for detection in ocr_result:
                box = detection[0]
                cv2.rectangle(extracted_text_image, (int(box[0][0]), int(box[0][1])), (int(box[2][0]), int(box[2][1])),
                              (0, 255, 0), 2)

            plt.subplot(1, 2, 2)

            # 원본 파일 이름에서 확장자를 제외한 부분을 추출
            original_filename = os.path.splitext(os.path.basename(image_path))[0]

            # 이미지를 원본 파일 이름을 기반으로 저장
            result_path = f'static/{original_filename}_extracted_text_image.jpg'
            plt.imshow(extracted_text_image)
            plt.title('Extracted Text')
            plt.savefig(result_path)
            plt.close()

            # 저장된 이미지의 경로를 반환
            image_url = f'/static/{original_filename}_extracted_text_image.jpg'

            return concatenated_text, image_url, average_confidence
    else:
        print("원이 검출되지 않았습니다.")


def identify_pill_color(image_path):
    # 이미지 불러오기
    original_image = cv2.imread(image_path)

    color_ranges = {
        '하양': ([0, 0, 200], [180, 30, 255]),
        '노랑': ([20, 100, 100], [40, 255, 255]),
        '주황': ([10, 100, 100], [20, 255, 255]),
        '분홍': ([150, 50, 50], [180, 255, 255]),
        '빨강': ([0, 100, 100], [10, 255, 255]),
        '갈색': ([0, 50, 50], [20, 150, 150]),
        '초록': ([40, 50, 50], [80, 255, 255]),
        '파랑': ([100, 50, 50], [140, 255, 255]),
        '남색': ([140, 50, 50], [160, 255, 255]),
        '보라': ([160, 50, 50], [180, 255, 255]),
        '회색': ([0, 0, 80], [180, 30, 200])
    }

    # 각 색상에 대한 픽셀 수 계산
    color_pixel_counts = {}
    for color_name, (lower_bound, upper_bound) in color_ranges.items():
        _, pixel_count = extract_color(original_image, color_name, lower_bound, upper_bound)
        color_pixel_counts[color_name] = pixel_count

    # 가장 많은 픽셀 수를 가진 색상 추출
    max_color = max(color_pixel_counts, key=color_pixel_counts.get)

    # 텍스트 및 색상 출력
    print(f"알약 색상: {max_color}")
    return max_color

@app.route('/')
def index():
    return render_template('file_upload2.html')

@app.route('/api/flask/ocr', methods=['POST'])
def ocr():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'})

    image = request.files['image']

    # 이미지를 임시 디렉터리에 저장
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
        temp_filename = temp_file.name
        image.save(temp_filename)

    # 원본 파일 이름 저장
    original_filename = image.filename

    # 알약 텍스트, 색상, 경계상자 좌표 식별
    pill_text, image_url, confidence = identify_pill_properties(temp_filename)
    pill_color = identify_pill_color(temp_filename)

    # JSON 응답 생성
    response_data = {
        'pill_text': pill_text,
        'pill_color': pill_color,
        'image_url': image_url,
        'confidence': confidence
    }

    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True)
