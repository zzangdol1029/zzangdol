# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import json
import time

# API URL
BASE_URL = "https://sd.solideos.com/api/org/user/sort/list"

# 인증 정보
HEADERS = {
    'Content-Type': 'application/json',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

COOKIES = {
    'GOSSOcookie': '82e2d0c2-d230-45bc-a308-ce676d54f147',
    'IsCookieActived': 'true',
    'OcxUpload': 'off',
    'PRW': 'show',
    'PUW': 'show',
    'fileDownload': 'true',
    'isAttachAreaFolded': 'false',
    'isDoCareLoungeCloseClick': 'true',
    'reflash_time': '-1'
}

def classify_position(position):
    """직급을 담당, 책임, 수석, 팀장, 기타로 분류"""
    if not position or position.strip() == '':
        return '미분류'
    
    position = position.strip()
    
    # 정확한 매칭
    if position in ['담당', '책임', '수석', '팀장']:
        return position
    
    # 포함된 키워드로 분류
    if '담당' in position:
        return '담당'
    elif '책임' in position:
        return '책임'
    elif '수석' in position:
        return '수석'
    elif '팀장' in position or '팀장' in position:
        return '팀장'
    else:
        return '기타'

def get_all_employees_comprehensive():
    """여러 방법으로 전체 직원 조회"""
    print("=== 전체 직원 데이터 조회 시작 ===")
    
    # 방법 1: 큰 offset으로 한 번에 조회
    print("\n방법 1: 큰 offset으로 전체 조회...")
    all_employees = []
    
    for offset in [5000, 3000, 2000, 1000, 800]:
        params = {
            "keyword": "",
            "page": 0,
            "offset": offset,
            "nodeType": "org"
        }
        
        try:
            response = requests.get(
                BASE_URL,
                params=params,
                headers=HEADERS,
                cookies=COOKIES,
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                if data and "data" in data and data["data"]:
                    employees = data["data"]
                    print(f"Offset {offset}: {len(employees)}명 조회 성공")
                    if len(employees) > len(all_employees):
                        all_employees = employees
                        if len(employees) > 500:  # 충분히 많으면 이 방법 사용
                            print(f"충분한 데이터 확보: {len(employees)}명")
                            return all_employees
            
            time.sleep(1)
            
        except Exception as e:
            print(f"Offset {offset} 오류: {e}")
            continue
    
    # 방법 2: 페이징으로 조회 (방법 1이 실패했을 경우)
    if len(all_employees) < 100:
        print("\n방법 2: 페이징으로 전체 조회...")
        page = 0
        max_pages = 100
        
        while page < max_pages:
            params = {
                "keyword": "",
                "page": page,
                "offset": 100,
                "nodeType": "org"
            }
            
            try:
                response = requests.get(
                    BASE_URL,
                    params=params,
                    headers=HEADERS,
                    cookies=COOKIES,
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if data and "data" in data and data["data"]:
                        employees = data["data"]
                        all_employees.extend(employees)
                        print(f"페이지 {page}: +{len(employees)}명 (총 {len(all_employees)}명)")
                        
                        if len(employees) < 100:
                            print("마지막 페이지 도달")
                            break
                    else:
                        break
                else:
                    print(f"페이지 {page}: HTTP {response.status_code}")
                    break
                
                page += 1
                time.sleep(0.3)
                
            except Exception as e:
                print(f"페이지 {page} 오류: {e}")
                break
    
    # 방법 3: 초성별 검색 (방법 2도 실패했을 경우)
    if len(all_employees) < 200:
        print("\n방법 3: 초성별 검색으로 조회...")
        search_terms = ['김', '이', '박', '최', '정', '강', '조', '윤', '장', '임', '한', '오', '서', '신', '권', '황', '안', '송', '류', '전']
        seen_ids = set()
        
        for term in search_terms:
            params = {
                "keyword": term,
                "page": 0,
                "offset": 1000,
                "nodeType": "org"
            }
            
            try:
                response = requests.get(
                    BASE_URL,
                    params=params,
                    headers=HEADERS,
                    cookies=COOKIES,
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if data and "data" in data:
                        employees = data["data"]
                        new_count = 0
                        for emp in employees:
                            emp_id = emp.get('userId') or emp.get('empNo') or emp.get('userName')
                            if emp_id not in seen_ids:
                                seen_ids.add(emp_id)
                                all_employees.append(emp)
                                new_count += 1
                        
                        print(f"'{term}' 검색: {len(employees)}명 중 신규 {new_count}명 (총 {len(all_employees)}명)")
                
                time.sleep(0.2)
                
            except Exception as e:
                print(f"'{term}' 검색 오류: {e}")
                continue
    
    print(f"\n=== 최종 결과: {len(all_employees)}명 조회 완료 ===")
    return all_employees

def analyze_positions(employees):
    """직급별 통계 분석"""
    print("\n=== 직급별 통계 분석 시작 ===")
    
    # 원본 직급 통계
    original_position_stats = defaultdict(int)
    # 분류된 직급 통계
    classified_position_stats = defaultdict(int)
    # 팀별 분류된 직급 통계
    team_position_stats = defaultdict(lambda: defaultdict(int))
    
    print("\n직원 정보 (처음 50명):")
    print("-" * 80)
    
    for i, emp in enumerate(employees):
        original_position = emp.get('position', '미분류')
        classified_position = classify_position(original_position)
        team = emp.get('deptName', '미분류')
        name = emp.get('userName', '무명')
        
        # 통계 업데이트
        original_position_stats[original_position] += 1
        classified_position_stats[classified_position] += 1
        team_position_stats[team][classified_position] += 1
        
        # 처음 50명만 출력
        if i < 50:
            print(f"{name}: {original_position} → {classified_position} ({team})")
        elif i == 50:
            print("... (나머지 직원 정보는 생략)")
    
    # 원본 직급 통계 출력
    print("\n" + "="*60)
    print("원본 직급별 통계 (분류 전)")
    print("="*60)
    sorted_original = sorted(original_position_stats.items(), key=lambda x: x[1], reverse=True)
    for position, count in sorted_original:
        print(f"{position}: {count}명")
    
    return dict(classified_position_stats), dict(team_position_stats), dict(original_position_stats)

def create_position_chart(position_stats, total_employees):
    """직급별 통계 차트 생성"""
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False
    
    # 데이터 준비 (순서: 담당, 책임, 수석, 팀장, 기타, 미분류)
    desired_order = ['담당', '책임', '수석', '팀장', '기타', '미분류']
    positions = []
    counts = []
    
    for pos in desired_order:
        if pos in position_stats:
            positions.append(pos)
            counts.append(position_stats[pos])
    
    # 색상 설정
    colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FFB366', '#D3D3D3']
    
    # 막대 그래프
    plt.figure(figsize=(12, 8))
    bars = plt.bar(positions, counts, color=colors[:len(positions)])
    
    # 막대 위에 숫자와 비율 표시
    for bar, count in zip(bars, counts):
        percentage = (count / total_employees) * 100
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                f'{count}명\n({percentage:.1f}%)', ha='center', va='bottom', fontweight='bold')
    
    plt.xlabel('직급')
    plt.ylabel('인원수')
    plt.title(f'전체 직급별 인원 통계 (총 {total_employees}명)\n담당/책임/수석/팀장/기타')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('position_stats_classified.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 파이 차트
    plt.figure(figsize=(10, 8))
    # 미분류 제외하고 파이 차트 생성
    pie_positions = [pos for pos in positions if pos != '미분류']
    pie_counts = [position_stats[pos] for pos in pie_positions]
    pie_colors = colors[:len(pie_positions)]
    
    plt.pie(pie_counts, labels=pie_positions, autopct='%1.1f%%', startangle=90, colors=pie_colors)
    plt.title(f'전체 직급 분포 (총 {sum(pie_counts)}명)\n담당/책임/수석/팀장/기타')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('position_pie_classified.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_team_position_chart(team_position_stats):
    """팀별 직급 통계 차트 생성"""
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False
    
    # 인원이 많은 상위 15개 팀 선택
    team_totals = {team: sum(positions.values()) for team, positions in team_position_stats.items()}
    top_teams = sorted(team_totals.items(), key=lambda x: x[1], reverse=True)[:15]
    
    if not top_teams:
        print("팀별 데이터가 없습니다.")
        return
    
    # 직급 순서
    position_order = ['담당', '책임', '수석', '팀장', '기타', '미분류']
    colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FFB366', '#D3D3D3']
    
    # 팀별 직급 분포 막대 그래프
    fig, ax = plt.subplots(figsize=(18, 10))
    
    teams = [team for team, _ in top_teams]
    x = np.arange(len(teams))
    width = 0.12  # 막대 너비
    
    for i, position in enumerate(position_order):
        counts = [team_position_stats[team].get(position, 0) for team in teams]
        offset = (i - len(position_order)/2 + 0.5) * width
        ax.bar(x + offset, counts, width, label=position, color=colors[i])
    
    ax.set_xlabel('팀/부서')
    ax.set_ylabel('인원수')
    ax.set_title('팀별 직급 분포 (상위 15개 팀)\n담당/책임/수석/팀장/기타')
    ax.set_xticks(x)
    ax.set_xticklabels(teams, rotation=45, ha='right')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('team_position_stats_classified.png', dpi=300, bbox_inches='tight')
    plt.show()

def save_to_excel(employees, classified_stats, team_stats, original_stats):
    """결과를 엑셀로 저장"""
    with pd.ExcelWriter('employee_statistics_classified.xlsx', engine='openpyxl') as writer:
        # 전체 직원 데이터
        df_employees = pd.DataFrame(employees)
        df_employees.to_excel(writer, sheet_name='전체직원', index=False)
        
        # 분류된 직급 통계
        df_classified = pd.DataFrame(list(classified_stats.items()), columns=['분류된직급', '인원수'])
        df_classified = df_classified.sort_values('인원수', ascending=False)
        df_classified.to_excel(writer, sheet_name='분류된직급통계', index=False)
        
        # 원본 직급 통계
        df_original = pd.DataFrame(list(original_stats.items()), columns=['원본직급', '인원수'])
        df_original = df_original.sort_values('인원수', ascending=False)
        df_original.to_excel(writer, sheet_name='원본직급통계', index=False)
        
        # 팀별 직급 통계
        team_data = []
        for team, positions in team_stats.items():
            for position, count in positions.items():
                team_data.append({'팀': team, '직급': position, '인원수': count})
        
        df_team_positions = pd.DataFrame(team_data)
        df_team_positions.to_excel(writer, sheet_name='팀별직급통계', index=False)

def print_statistics(classified_stats, team_stats, original_stats):
    """통계 결과를 콘솔에 출력"""
    total_employees = sum(classified_stats.values())
    
    print("\n" + "="*60)
    print("분류된 직급별 통계 (담당/책임/수석/팀장/기타)")
    print("="*60)
    
    # 순서대로 출력
    order = ['담당', '책임', '수석', '팀장', '기타', '미분류']
    for position in order:
        if position in classified_stats:
            count = classified_stats[position]
            percentage = (count / total_employees) * 100
            print(f"{position}: {count}명 ({percentage:.1f}%)")
    
    print(f"\n총 직원 수: {total_employees}명")
    
    print("\n" + "="*60)
    print("팀별 직급 통계 (인원 많은 순 상위 20개 팀)")
    print("="*60)
    
    team_totals = {team: sum(positions.values()) for team, positions in team_stats.items()}
    sorted_teams = sorted(team_totals.items(), key=lambda x: x[1], reverse=True)
    
    for team, total in sorted_teams[:20]:
        print(f"\n[{team}] 총 {total}명")
        team_positions = sorted(team_stats[team].items(), key=lambda x: x[1], reverse=True)
        for position, count in team_positions:
            print(f"  - {position}: {count}명")

# 메인 실행
if __name__ == "__main__":
    print("=" * 80)
    print("전체 직원 데이터 조회 및 직급별 통계 분석")
    print("직급 분류: 담당/책임/수석/팀장/기타")
    print("=" * 80)
    
    # 전체 직원 데이터 조회
    employees = get_all_employees_comprehensive()
    
    if not employees:
        print("직원 데이터를 가져올 수 없습니다.")
        exit()
    
    print(f"\n총 {len(employees)}명의 직원 데이터를 조회했습니다.")
    
    # 통계 분석
    classified_stats, team_stats, original_stats = analyze_positions(employees)
    
    # 결과 출력
    print_statistics(classified_stats, team_stats, original_stats)
    
    # 차트 생성
    print("\n차트 생성 중...")
    create_position_chart(classified_stats, len(employees))
    create_team_position_chart(team_stats)
    
    # 엑셀 저장
    print("\n엑셀 파일 저장 중...")
    save_to_excel(employees, classified_stats, team_stats, original_stats)
    
    print("\n" + "="*80)
    print("완료! 다음 파일들이 생성되었습니다:")
    print("- employee_statistics_classified.xlsx: 전체 통계 데이터")
    print("- position_stats_classified.png: 분류된 직급별 막대 그래프")
    print("- position_pie_classified.png: 분류된 직급별 파이 차트")
    print("- team_position_stats_classified.png: 팀별 직급 분포 차트")
    print("="*80)