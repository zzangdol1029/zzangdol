import json
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta

# 상수 정의
BATCH_TIMES = [9, 12, 15, 18]
SENT_FILE = "sent_notifications.json"
USERS_FILE = "users.json"

def load_environment():
    """환경변수 로딩"""
    load_dotenv()
    return {
        'service_key': os.getenv('SERVICE_KEY'),
        'coolsms_api_key': os.getenv('COOLSMS_API_KEY'),
        'coolsms_api_secret': os.getenv('COOLSMS_API_SECRET'),
        'coolsms_sender': os.getenv('COOLSMS_SENDER')
    }

def get_batch_time_ranges(now):
    """배치 시간대 설정 함수"""
    valid_times = [h for h in BATCH_TIMES if h <= now.hour]

    # case 1: 현재 시각이 가장 이른 배치 이전일 경우
    if not valid_times:
        prev_day = now - timedelta(days=1)
        # 전날 마지막 배치 시각부터 오늘 첫 배치 시각까지
        bgn = prev_day.replace(hour=BATCH_TIMES[-1], minute=0, second=0, microsecond=0)
        end = now.replace(hour=BATCH_TIMES[0], minute=0, second=0, microsecond=0)
        return bgn.strftime("%Y%m%d%H%M"), end.strftime("%Y%m%d%H%M")

    # case 2: 현재 시각이 배치 시간 이후일 경우
    # 가장 가까운 이전 배치 구간 반환
    prev_batch_hour = max(valid_times)
    idx = BATCH_TIMES.index(prev_batch_hour)

    if idx == 0:
        # 첫 번째 배치인 경우: 전날 마지막 배치 시각부터 오늘 첫 배치 시각까지
        bgn = now.replace(hour=BATCH_TIMES[-1], minute=0, second=0, microsecond=0) - timedelta(days=1)
    else:
        # 일반 배치 구간 (이전 배치 시각 → 현재 배치 시각)
        bgn = now.replace(hour=BATCH_TIMES[idx-1], minute=0, second=0, microsecond=0)

    end = now.replace(hour=prev_batch_hour, minute=0, second=0, microsecond=0)
    return bgn.strftime("%Y%m%d%H%M"), end.strftime("%Y%m%d%H%M")

def make_sms_text(prefix: str, content_name: str) -> str:
    """SMS 메시지 내용 생성"""
    max_length = 30
    if len(content_name) > max_length:
        content_name = content_name[:max_length - 3] + "..."
    return prefix + content_name

def load_sent_data():
    """발송 이력 로딩"""
    if os.path.exists(SENT_FILE):
        with open(SENT_FILE, 'r', encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_sent_data(sent_data):
    """발송 이력 저장"""
    with open(SENT_FILE, 'w', encoding="utf-8") as f:
        json.dump(sent_data, f, indent=2, ensure_ascii=False)

def load_users():
    """사용자 정보 로딩"""
    with open(USERS_FILE, 'r', encoding="utf-8") as f:
        return json.load(f)