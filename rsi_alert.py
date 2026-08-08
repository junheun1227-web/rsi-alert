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

필요한 환경변수:
  KAKAOWORK_APP_KEY   : 카카오워크 관리자 > 봇 관리에서 발급받은 App Key
  KAKAOWORK_EMAIL     : 알림을 받을 카카오워크 계정 이메일

사용법:
  python rsi_alert.py            # 체크 후 조건 충족 시 카카오워크로 알림 전송
  python rsi_alert.py --test     # 카카오워크 연결 테스트 메시지만 전송
  python rsi_alert.py --dry-run  # 알림 전송 없이 현재 RSI만 출력
"""

import json
import os
import sys
from datetime import datetime, date

import requests
import yfinance as yf
import pandas as pd

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

KAKAOWORK_APP_KEY = os.environ.get("KAKAOWORK_APP_KEY", "")
KAKAOWORK_EMAIL = os.environ.get("KAKAOWORK_EMAIL", "")
KAKAOWORK_API_BASE = "https://api.kakaowork.com/v1"


# ---------------------------------------------------------------------------
# RSI 계산 (Wilder's smoothing, 표준 방식)
# ---------------------------------------------------------------------------

def calc_rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """표준 Wilder's RSI: 최초 평균은 단순평균(SMA), 이후는 Wilder 스무딩으로 재귀 계산.
    (증권사 HTS/트레이딩뷰 등에서 쓰는 방식과 동일하게 맞춤)"""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    avg_gain = avg_gain.copy()
    avg_loss = avg_loss.copy()
    # index `period`는 rolling()이 이미 올바른 초기 SMA를 채웠으므로 그대로 두고,
    # 그 다음 값부터 Wilder 재귀 스무딩을 적용한다.
    for i in range(period + 1, len(gain)):
        avg_gain.iloc[i] = (avg_gain.iloc[i - 1] * (period - 1) + gain.iloc[i]) / period
        avg_loss.iloc[i] = (avg_loss.iloc[i - 1] * (period - 1) + loss.iloc[i]) / period

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def get_current_rsi(ticker: str) -> tuple[float, float]:
    """일봉 기준 RSI(14)와 최신 가격을 반환.
    마지막 바(오늘)는 장중 실시간가로 대체되어, 장중에도 최신 RSI를 반영함."""
    data = yf.Ticker(ticker).history(period="3mo", interval="1d")
    if data.empty or len(data) < RSI_PERIOD + 1:
        raise ValueError(f"{ticker}: 데이터 부족")

    # 장중 실시간가로 마지막 종가를 갱신 (fast_info 사용)
    try:
        last_price = yf.Ticker(ticker).fast_info["last_price"]
        if last_price:
            data.loc[data.index[-1], "Close"] = last_price
    except Exception:
        pass  # 실패 시 마지막 일봉 종가 그대로 사용

    rsi_series = calc_rsi(data["Close"])
    return float(rsi_series.iloc[-1]), float(data["Close"].iloc[-1])


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
# 메인 로직
# ---------------------------------------------------------------------------

def run(dry_run: bool = False) -> None:
    today = date.today().isoformat()
    state = load_state()
    triggered = []
    results = []

    for ticker, name in TICKERS.items():
        try:
            rsi, price = get_current_rsi(ticker)
        except Exception as e:
            print(f"[에러] {ticker} 조회 실패: {e}")
            continue

        results.append((ticker, name, rsi, price))

        # 25가 30보다 더 급락한 상태이므로 25부터 체크해서 더 심각한 것만 알림
        for threshold in sorted(THRESHOLDS):  # [25, 30]
            if rsi <= threshold and not already_alerted(state, ticker, threshold, today):
                triggered.append((ticker, name, rsi, price, threshold))
                mark_alerted(state, ticker, threshold, today)
                break  # 더 낮은 임계값 하나만 알림 (25가 30을 포함)

    # 현재 상태 로그 출력
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] RSI 체크 결과")
    for ticker, name, rsi, price in results:
        flag = " <-- 임계값 이하!" if rsi <= max(THRESHOLDS) else ""
        print(f"  {ticker:6s} ({name:10s}) RSI={rsi:6.2f}  가격=${price:,.2f}{flag}")

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
        sent = send_kakaowork_message(msg, state)
        print(f"  -> {ticker} 알림 전송 {'성공' if sent else '실패/미설정'}")

    save_state(state)


if __name__ == "__main__":
    if "--test" in sys.argv:
        _state = load_state()
        ok = send_kakaowork_message("✅ 카카오워크 RSI 알림 봇 연결 테스트입니다.", _state)
        save_state(_state)
        print("전송 성공" if ok else "전송 실패 - APP_KEY / EMAIL을 확인하세요.")
    else:
        run(dry_run="--dry-run" in sys.argv)
