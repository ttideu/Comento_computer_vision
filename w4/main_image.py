# main_image.py (웹캠 없을 때)
import cv2
import numpy as np
from ultralytics import YOLO
from calorie_calc import CalorieCalculator

def main():
    # 1. 설정
    # 테스트할 이미지 파일 이름 (폴더에 이 파일이 있어야 함!)
    image_path = "test_image.jpg" 
    
    print(f"이미지 로드 중: {image_path}")
    
    # 모델 및 계산기 로드
    model = YOLO("yolov8n.pt") 
    calc = CalorieCalculator()
    
    # 이미지 읽기
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"오류: '{image_path}' 파일을 찾을 수 없습니다.")
        print("구글에서 음식 사진을 다운받아 같은 폴더에 넣어주세요.")
        return

    # 이미지가 너무 크면 보기 불편하므로 화면에 맞게 조정 (가로 800px 기준)
    if frame.shape[1] > 800:
        scale = 800 / frame.shape[1]
        frame = cv2.resize(frame, None, fx=scale, fy=scale)

    # 2. 객체 탐지 (웹캠과 달리 가이드 박스 없이 전체 화면 탐지)
    results = model(frame)
    detected_foods = []
    
    print("\n=== 분석 결과 ===")
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # 좌표 및 클래스 정보
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf = float(box.conf[0])

            # 신뢰도가 너무 낮은 건 제외 (0.3 이상만)
            if conf > 0.3:
                # 영양 정보 가져오기
                nutri = calc.get_nutrition(cls_name)
                kcal = nutri[0]
                detected_foods.append(cls_name)
                
                print(f"- 감지됨: {cls_name} (정확도: {conf:.2f}) -> {kcal}kcal")

                # 시각화 (빨간 박스 + 텍스트)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
                label = f"{cls_name} {kcal}kcal"
                # 텍스트 배경
                (w_text, h_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(frame, (x1, y1 - 25), (x1 + w_text, y1), (0, 0, 255), -1)
                cv2.putText(frame, label, (x1, y1 - 8), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # 3. 결과 리포트 오버레이 (좌측 상단)
    total_cal, t_carb, t_prot, t_fat = calc.calculate_total(detected_foods)
    
    # 반투명 배경 박스 생성
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (350, 160), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
    
    # 텍스트 출력
    cv2.putText(frame, "[ Diet Analysis ]", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"Total: {total_cal} kcal", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    cv2.putText(frame, f"Carbs: {t_carb}g", (30, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(frame, f"Protein: {t_prot}g", (30, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(frame, f"Fat: {t_fat}g", (30, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # 4. 이미지 띄우기
    cv2.imshow("Result Image", frame)
    
    print("\n아무 키나 누르면 종료됩니다.")
    cv2.waitKey(0) # 아무 키나 누를 때까지 대기
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
