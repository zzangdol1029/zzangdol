# 구구단 출력 함수
def gugudan():
    for i in range(2, 10):  # 2단부터 9단까지
        print(f"*** {i}단 ***")
        for j in range(1, 10):  # 각 단의 1~9 곱셈
            print(f"{i} x {j} = {i * j}")
        print() #테스트 단 사이에 간격 추가


# 함수 호출
gugudan()
