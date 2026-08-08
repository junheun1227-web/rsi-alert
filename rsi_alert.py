#!/usr/bin/env python3
"""
매그니피센트7 RSI 알림 시스템
- RSI(14, Wilder's smoothing)가 30 이하 / 25 이하로 떨어지면 카카오워크 봇으로 알림
- 장중 여부와 상관없이 스케줄 실행 시마다 최신가를 반영해 RSI를 재계산
- 같은 날 같은 종목·같은 임계값에 대해서는 중복 알림을 보내지 않음 (state 파일로 추적)

카카오워크는 Slack류의 "incoming webhook"이 아니라 App Key 인증 기반 REST API를 사용합니다.
  1) users.find_by_email  : 받을 사람의 이메일 -> user_id 조회
  2) conversations.open   : user_id -> 1:1 대화방(conversation_id) 열기
  3) messages.send        : conversation_id로 실제 메시지 전송
conversation_id는 최초 1회만 조회하고 state 파일에 캐시해 재사용합니다.

개인 카카오톡("나에게 보내기")은 Kakao Developers 앱의 REST API 키 + OAuth refresh_token을 사용합니다.
  1) refresh_token으로 access_token을 매번 새로 발급 (access_token은 몇 시간 후 만료되므로)
  2) access_token으로 /v2/api/talk/memo/default/send 호출해 "나와의 채팅"으로 메시지 전송
refresh_token 발급은 최초 1회 OAuth 로그인 동의가 필요하며 kakao_oauth_setup.yml로 진행합니다.

필요한 환경변수:
  KAKAOWORK_APP_KEY    : 카카오워크 관리자 > 봇 관리에서 발급받은 App Key
  KAKAOWORK_EMAIL      : 알림을 받을 카카오워크 계정 이메일
  KAKAO_REST_API_KEY   : Kakao Developers 앱의 REST API 키
  KAKAO_REFRESH_TOKEN  : OAuth 동의 후 발급받은 refresh_token (kakao_oauth_setup.yml 참고)

사용법:
  python rsi_alert.py            # 체크 후 조건 충족 시 카카오워크 + 개인 카톡으로 알림 전송
  python rsi_alert.py --test     # 두 채널 모두 연결 테스트 메시지만 전송
  python rsi_alert.py --dry-run  # 알림 전송 없이 현재 RSI만 출력
"""

import json
import os
import sys
from datetime import datetime, date

import requests

import analysis

# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------

TICKERS = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "GOOGL": "Alphabet",
    "AMZN": "Amazon",
    "META": "Meta",
    "NVDA": "NVIDIA",
    "TSLA": "Tesla",
}

RSI_PERIOD = 14
THRESHOLDS = [30, 25]  # 낮은 값이 더 심각하므로 둘 다 체크 (30 먼저, 25는 더 급락 시)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(SCRIPT_DIR, "rsi_alert_state.json")
LATEST_RSI_FILE = os.path.join(SCRIPT_DIR, "latest_rsi.json")

KAKAOWORK_APP_KEY = os.environ.get("KAKAOWORK_APP_KEY", "")
KAKAOWORK_EMAIL = os.environ.get("KAKAOWORK_EMAIL", "")
KAKAOWORK_API_BASE = "https://api.kakaowork.com/v1"

KAKAO_REST_API_KEY = os.environ.get("KAKAO_REST_API_KEY", "")
KAKAO_REFRESH_TOKEN = os.environ.get("KAKAO_REFRESH_TOKEN", "")


# ---------------------------------------------------------------------------
# 종목 분석 (RSI + 5개 추가 지표는 analysis.py에서 계산)
# ---------------------------------------------------------------------------

def get_analysis(ticker: str, name: str) -> dict:
    """RSI, 볼린저밴드, 이동평균, MACD, 거래량, 일목균형표를 종합한 매수/매도 분석 결과."""
    return analysis.analyze_ticker(ticker, name)


# ---------------------------------------------------------------------------
# 알림 상태 관리 (중복 알림 방지)
# ---------------------------------------------------------------------------

def load_state() -> dict:
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {}


def save_state(state: dict) -> None:
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def already_alerted(state: dict, ticker: str, threshold: int, today: str) -> bool:
    key = f"{ticker}_{threshold}"
    return state.get(key) == today


def mark_alerted(state: dict, ticker: str, threshold: int, today: str) -> None:
    key = f"{ticker}_{threshold}"
    state[key] = today


def save_latest_rsi(analyses: dict, ts: str) -> None:
    """카카오 챗봇 스킬 서버가 읽어갈 수 있도록 종목별 전체 분석 결과를 캐시 파일로 저장.
    챗봇 쪽에서 매번 야후 파이낸스를 호출하지 않고 이 캐시만 읽으면 되므로 응답이 빠르고 안정적임."""
    payload = {"updated_at": ts, "tickers": analyses}
    with open(LATEST_RSI_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# 카카오워크 알림 전송
# ---------------------------------------------------------------------------

def _kw_headers() -> dict:
    return {
        "Authorization": f"Bearer {KAKAOWORK_APP_KEY}",
        "Content-Type": "application/json;charset=utf-8",
    }


def get_conversation_id(state: dict) -> str | None:
    """카카오워크 1:1 대화방 ID를 가져온다. 최초 1회만 API 호출하고 이후엔 state에 캐시된 값을 재사용."""
    cached = state.get("kakaowork_conversation_id")
    if cached:
        return cached

    if not KAKAOWORK_APP_KEY or not KAKAOWORK_EMAIL:
        return None

    # 1) 이메일로 user_id 조회
    resp = requests.get(
        f"{KAKAOWORK_API_BASE}/users.find_by_email",
        headers=_kw_headers(),
        params={"email": KAKAOWORK_EMAIL},
        timeout=10,
    )
    data = resp.json()
    if not data.get("success"):
        print(f"[에러] users.find_by_email 실패: {data.get('error')}")
        return None
    user_id = data["user"]["id"]

    # 2) 1:1 대화방 열기
    resp = requests.post(
        f"{KAKAOWORK_API_BASE}/conversations.open",
        headers=_kw_headers(),
        json={"user_id": user_id},
        timeout=10,
    )
    data = resp.json()
    if not data.get("success"):
        print(f"[에러] conversations.open 실패: {data.get('error')}")
        return None

    conversation_id = data["conversation"]["id"]
    state["kakaowork_conversation_id"] = conversation_id
    return conversation_id


def send_kakaowork_message(text: str, state: dict) -> bool:
    if not KAKAOWORK_APP_KEY or not KAKAOWORK_EMAIL:
        print("[경고] KAKAOWORK_APP_KEY / KAKAOWORK_EMAIL이 설정되지 않아 메시지를 보내지 않습니다.")
        print("--- 전송 예정 메시지 ---")
        print(text)
        return False

    conversation_id = get_conversation_id(state)
    if not conversation_id:
        print("[에러] 대화방 ID를 확보하지 못해 메시지를 보내지 않습니다.")
        return False

    resp = requests.post(
        f"{KAKAOWORK_API_BASE}/messages.send",
        headers=_kw_headers(),
        json={"conversation_id": conversation_id, "text": text},
        timeout=10,
    )
    data = resp.json()
    ok = bool(data.get("success"))
    if not ok:
        print(f"[에러] 카카오워크 전송 실패: {data.get('error')}")
    return ok


# ---------------------------------------------------------------------------
# 개인 카카오톡("나에게 보내기") 알림 전송
# ---------------------------------------------------------------------------

def _kakao_get_access_token() -> str | None:
    if not KAKAO_REST_API_KEY or not KAKAO_REFRESH_TOKEN:
        return None

    resp = requests.post(
        "https://kauth.kakao.com/oauth/token",
        data={
            "grant_type": "refresh_token",
            "client_id": KAKAO_REST_API_KEY,
            "refresh_token": KAKAO_REFRESH_TOKEN,
        },
        timeout=10,
    )
    data = resp.json()
    if "access_token" not in data:
        print(f"[에러] 카카오 access_token 갱신 실패: {data}")
        return None
    return data["access_token"]


def send_kakaotalk_memo(text: str) -> bool:
    """Kakao Developers 앱의 '나에게 보내기'(talk_message scope)로 본인 카카오톡에 메시지 전송."""
    if not KAKAO_REST_API_KEY or not KAKAO_REFRESH_TOKEN:
        print("[경고] KAKAO_REST_API_KEY / KAKAO_REFRESH_TOKEN이 설정되지 않아 개인 카톡 전송을 건너뜁니다.")
        return False

    access_token = _kakao_get_access_token()
    if not access_token:
        return False

    template_object = {
        "object_type": "text",
        "text": text,
        "link": {
            "web_url": "https://finance.yahoo.com",
            "mobile_web_url": "https://finance.yahoo.com",
        },
    }

    resp = requests.post(
        "https://kapi.kakao.com/v2/api/talk/memo/default/send",
        headers={"Authorization": f"Bearer {access_token}"},
        data={"template_object": json.dumps(template_object)},
        timeout=10,
    )
    data = resp.json()
    ok = data.get("result_code") == 0
    if not ok:
        print(f"[에러] 개인 카톡 전송 실패: {data}")
    return ok


# ---------------------------------------------------------------------------
# 메인 로직
# ---------------------------------------------------------------------------

def run(dry_run: bool = False) -> None:
    today = date.today().isoformat()
    state = load_state()
    triggered = []
    analyses = {}

    for ticker, name in TICKERS.items():
        try:
            result = get_analysis(ticker, name)
        except Exception as e:
            print(f"[에러] {ticker} 조회 실패: {e}")
            continue

        analyses[ticker] = result
        rsi, price = result["rsi"], result["price"]

        # 25가 30보다 더 급락한 상태이므로 25부터 체크해서 더 심각한 것만 알림
        for threshold in sorted(THRESHOLDS):  # [25, 30]
            if rsi <= threshold and not already_alerted(state, ticker, threshold, today):
                triggered.append((ticker, name, rsi, price, threshold))
                mark_alerted(state, ticker, threshold, today)
                break  # 더 낮은 임계값 하나만 알림 (25가 30을 포함)

    # 현재 상태 로그 출력
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] 분석 결과")
    for ticker, result in analyses.items():
        flag = " <-- RSI 임계값 이하!" if result["rsi"] <= max(THRESHOLDS) else ""
        print(
            f"  {ticker:6s} ({result['name']:10s}) RSI={result['rsi']:6.2f}  "
            f"가격=${result['price']:,.2f}  판단={result['emoji']}{result['label']} "
            f"(매수 {result['buy_score']}/60, 매도 {result['sell_score']}/60){flag}"
        )

    save_latest_rsi(analyses, ts)  # 챗봇 질의응답용 캐시 갱신 (항상 저장)

    if not triggered:
        print("알림 조건에 해당하는 종목 없음.")
        save_state(state)  # conversation_id 캐시 등 갱신분 저장
        return

    if dry_run:
        print(f"[dry-run] {len(triggered)}건 알림 대상이지만 실제 전송은 생략합니다.")
        return

    for ticker, name, rsi, price, threshold in triggered:
        msg = (
            f"⚠️ RSI 알림: {name}({ticker})\n"
            f"RSI(14) = {rsi:.2f}  (기준: {threshold} 이하)\n"
            f"현재가: ${price:,.2f}\n"
            f"시각: {ts}"
        )
        kw_sent = send_kakaowork_message(msg, state)
        kt_sent = send_kakaotalk_memo(msg)
        print(f"  -> {ticker} 카카오워크 {'성공' if kw_sent else '실패/미설정'} / 개인 카톡 {'성공' if kt_sent else '실패/미설정'}")

    save_state(state)


if __name__ == "__main__":
    if "--test" in sys.argv:
        _state = load_state()
        kw_ok = send_kakaowork_message("✅ 카카오워크 RSI 알림 봇 연결 테스트입니다.", _state)
        kt_ok = send_kakaotalk_memo("✅ 개인 카톡(나에게 보내기) RSI 알림 연결 테스트입니다.")
        save_state(_state)
        print(f"카카오워크: {'성공' if kw_ok else '실패'} / 개인 카톡: {'성공' if kt_ok else '실패'}")
    else:
        run(dry_run="--dry-run" in sys.argv)
