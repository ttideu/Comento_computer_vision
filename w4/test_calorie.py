# test_calorie.py
import pytest
from calorie_calc import CalorieCalculator

def test_nutrition_db_check():
    """DB에 주요 메뉴가 있는지 확인"""
    calc = CalorieCalculator()
    burger = calc.get_nutrition('hamburger')
    assert burger[0] == 560 # 평균값 확인

def test_calculation_logic():
    """칼로리 합산 로직 확인"""
    calc = CalorieCalculator()
    # 햄버거(560) + 사과(52) = 612
    meal = ['hamburger', 'apple']
    total_cal, _, _, _ = calc.calculate_total(meal)
    assert total_cal == 612

def test_unknown_food():
    """없는 음식은 기본값 처리"""
    calc = CalorieCalculator()
    unknown = calc.get_nutrition('kimchi')
    assert unknown[0] == 300
