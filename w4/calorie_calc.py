# calorie_calc.py
import datetime

class CalorieCalculator:
    def __init__(self):
        """
        데이터베이스 구축 기준: 메이저 브랜드 인기 메뉴 평균값 (2024)
        """
        self.nutrition_db = {
            # [칼로리(kcal), 탄수화물(g), 단백질(g), 지방(g)]
            
            # 1. 햄버거 (맥도날드, 버거킹, 롯데리아 대표 메뉴 평균)
            'hamburger': [560, 48, 26, 28], 
            'burger': [560, 48, 26, 28],
            
            # 2. 피자 (피자헛, 도미노, 파파존스 R사이즈 1조각 평균)
            'pizza': [276, 32, 12, 10],
            
            # 3. 치킨 (BBQ, BHC, 교촌 닭다리 1조각 평균)
            'chicken': [290, 15, 22, 18],
            'fried chicken': [290, 15, 22, 18],

            # 4. 기타 일반 식품 (USDA 데이터 참조)
            'hot dog': [290, 25, 10, 18],
            'sandwich': [350, 45, 15, 12],
            'apple': [52, 14, 0.3, 0.2],    
            'orange': [47, 12, 0.9, 0.1],
            'donut': [250, 30, 3, 14],      
            'cake': [370, 55, 5, 15],       
            'banana': [89, 23, 1, 0.3],

            # 5. 기본값 (인식된 음식이 DB에 없을 경우)
            'food': [300, 30, 10, 10]
        }
        self.current_meal = [] 

    def get_nutrition(self, food_name):
        # 소문자로 변환하여 DB 검색, 없으면 'food' 반환
        key = food_name.lower()
        return self.nutrition_db.get(key, self.nutrition_db['food'])

    def calculate_total(self, detected_items):
        """현재 인식된 음식들의 총 영양소 계산"""
        total_cal = 0
        total_carbs = 0
        total_prot = 0
        total_fat = 0
        
        self.current_meal = detected_items
        
        for item in detected_items:
            nutri = self.get_nutrition(item)
            total_cal += nutri[0]
            total_carbs += nutri[1]
            total_prot += nutri[2]
            total_fat += nutri[3]
            
        return total_cal, total_carbs, total_prot, total_fat

    def generate_report(self):
        """현재 식단 분석 보고서 텍스트 생성"""
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        t_cal, t_carb, t_prot, t_fat = self.calculate_total(self.current_meal)
        
        report = f"\n=== [Diet Analysis Report] ===\n"
        report += f"Date: {now}\n"
        report += "* Based on Major Brand Averages\n"
        report += "-"*40 + "\n"
        
        for item in self.current_meal:
            kcal = self.get_nutrition(item)[0]
            report += f"- {item.capitalize():<15} : {kcal} kcal\n"
            
        report += "-"*40 + "\n"
        report += f"Total Calories : {t_cal} kcal\n"
        report += f"Carbs          : {t_carb} g\n"
        report += f"Protein        : {t_prot} g\n"
        report += f"Fat            : {t_fat} g\n"
        report += "="*40 + "\n"
        
        return report
