"""
6개 기술적 지표 기반 매수/매도 판단 엔진.

사용자가 정의한 채점 기준(RSI/볼린저밴드/이동평균/MACD/거래량/일목균형표, 각 10점 만점, 총 60점)에
따라 종목별 매수 점수·매도 점수·최종 판단을 계산한다.

주의: "근접", "반등" 등 사용자가 정확한 수치를 명시하지 않은 항목은 아래 코드에 합리적인 기준을
직접 정의해 사용했다 (각 함수의 주석 참고). RSI 30/25 push 알림(rsi_alert.py의 THRESHOLDS)과는
별개의 로직이며, 이 모듈은 매수/매도 "점수화 모델"만 담당한다.

투자 판단의 참고용이며 투자 권유가 아니다. 기술적 분석은 미래 수익을 보장하지 않는다.
"""

import yfinance as yf
import pandas as pd

RSI_PERIOD = 14


# ---------------------------------------------------------------------------
# 데이터 조회
# ---------------------------------------------------------------------------

def ticker_exists(ticker: str) -> bool:
    """짧은 기간 데이터만 가져와 해당 티커가 실제로 존재하는지 저비용으로 확인 (한국 종목코드의
    .KS/.KQ 판별 등에 사용). fetch_ohlcv와 달리 최소 데이터 길이를 요구하지 않는다."""
    try:
        data = yf.Ticker(ticker).history(period="5d", interval="1d")
        return not data.empty
    except Exception:
        return False


def fetch_ohlcv(ticker: str, period: str = "2y") -> pd.DataFrame:
    """120일 이동평균, 일목균형표(52+26) 계산에 충분한 기간을 확보하기 위해 2년치 일봉을 가져온다."""
    data = yf.Ticker(ticker).history(period=period, interval="1d")
    if data.empty or len(data) < 130:
        raise ValueError(f"{ticker}: 데이터 부족 (분석 데이터 부족)")

    # 장중 실시간가로 마지막 종가를 갱신
    try:
        last_price = yf.Ticker(ticker).fast_info["last_price"]
        if last_price:
            data.loc[data.index[-1], "Close"] = last_price
    except Exception:
        pass

    return data


# ---------------------------------------------------------------------------
# 공용 유틸
# ---------------------------------------------------------------------------

def calc_rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """표준 Wilder's RSI (최초 평균은 SMA, 이후 재귀 스무딩)."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=period).mean().copy()
    avg_loss = loss.rolling(window=period).mean().copy()

    for i in range(period + 1, len(gain)):
        avg_gain.iloc[i] = (avg_gain.iloc[i - 1] * (period - 1) + gain.iloc[i]) / period
        avg_loss.iloc[i] = (avg_loss.iloc[i - 1] * (period - 1) + loss.iloc[i]) / period

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _trend(series: pd.Series, lookback: int = 5) -> str:
    """최근 lookback일 대비 변화율로 상승/횡보/하락 판정 (기준: ±0.5%)."""
    if len(series) <= lookback:
        return "횡보"
    now, prev = series.iloc[-1], series.iloc[-1 - lookback]
    if pd.isna(now) or pd.isna(prev) or prev == 0:
        return "횡보"
    change = (now - prev) / abs(prev)
    if change > 0.005:
        return "상승"
    if change < -0.005:
        return "하락"
    return "횡보"


# ---------------------------------------------------------------------------
# 지표별 채점 (각 함수는 (매수점수, 매도점수, 설명문자열)을 반환)
# ---------------------------------------------------------------------------

def score_rsi(close: pd.Series):
    rsi = float(calc_rsi(close).iloc[-1])

    if rsi <= 25:
        buy = 10
    elif rsi <= 30:
        buy = 9
    elif rsi <= 35:
        buy = 6
    elif rsi <= 40:
        buy = 3
    else:
        buy = 0

    if rsi >= 75:
        sell = 10
    elif rsi >= 71:
        sell = 9
    elif rsi >= 66:
        sell = 7
    elif rsi >= 61:
        sell = 4
    elif rsi >= 51:
        sell = 2
    else:
        sell = 0

    return buy, sell, f"{rsi:.1f}", rsi


def score_bollinger(close: pd.Series):
    """근접 기준: 상/하단 밴드로부터 밴드폭의 25% 이내를 '근접'으로 정의."""
    mid = close.rolling(20).mean()
    std = close.rolling(20).std()
    upper = mid + 2 * std
    lower = mid - 2 * std

    c0, c1 = close.iloc[-1], close.iloc[-2]
    u0, u1 = upper.iloc[-1], upper.iloc[-2]
    l0, l1 = lower.iloc[-1], lower.iloc[-2]
    m0 = mid.iloc[-1]

    band_width = (u0 - l0) or 1e-9
    position = (c0 - l0) / band_width  # 0=하단, 0.5=중심, 1=상단

    if c1 < l1 and c0 >= l0:
        buy, buy_desc = 10, "하단 이탈 후 복귀"
    elif c0 <= l0:
        buy, buy_desc = 8, "하단밴드 터치"
    elif position <= 0.25:
        buy, buy_desc = 6, "하단 근접"
    elif c0 <= m0:
        buy, buy_desc = 3, "중심선 아래"
    else:
        buy, buy_desc = 0, "중심선 위"

    if c1 > u1 and c0 <= u0:
        sell, sell_desc = 10, "상단 이탈 후 복귀"
    elif c0 >= u0:
        sell, sell_desc = 8, "상단밴드 터치"
    elif position >= 0.75:
        sell, sell_desc = 6, "상단 근접"
    elif c0 >= m0:
        sell, sell_desc = 3, "중심선 위"
    else:
        sell, sell_desc = 0, "중심선 아래"

    desc = buy_desc if buy >= sell else sell_desc
    return buy, sell, desc


def score_ma(close: pd.Series):
    """'근처에서 반등' 기준: 최근 3일 최저 종가가 20일선 대비 1% 이내로 접근한 뒤 현재가가 20일선 위."""
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()
    ma120 = close.rolling(120).mean()

    c0, c1 = close.iloc[-1], close.iloc[-2]
    ma20_0, ma20_1 = ma20.iloc[-1], ma20.iloc[-2]

    trend20 = _trend(ma20)
    trend60 = _trend(ma60)
    trend120 = _trend(ma120)
    all_down = trend20 == "하락" and trend60 == "하락" and trend120 == "하락"
    all_up = trend20 == "상승" and trend60 == "상승" and trend120 == "상승"

    recent_low = close.iloc[-3:].min()
    near_ma20 = (ma20_0 != 0) and (abs(recent_low - ma20_0) / ma20_0 < 0.01)

    crossed_up = c1 < ma20_1 and c0 >= ma20_0
    crossed_down = c1 > ma20_1 and c0 <= ma20_0

    if crossed_up:
        buy = 10
    elif c0 > ma20_0 and near_ma20:
        buy = 8
    elif c0 < ma20_0 and trend20 == "상승":
        buy = 5
    elif trend20 == "횡보":
        buy = 3
    elif trend20 == "하락" and not all_down:
        buy = 2
    else:
        buy = 0

    if crossed_down:
        sell = 10
    elif c0 < ma20_0 and near_ma20:
        sell = 8
    elif c0 > ma20_0 and trend20 == "하락":
        sell = 5
    elif trend20 == "횡보":
        sell = 3
    elif trend20 == "상승" and not all_up:
        sell = 2
    else:
        sell = 0

    desc = (
        f"20일선 {ma20_0:.1f}({trend20}) / 60일선 {ma60.iloc[-1]:.1f}({trend60}) / "
        f"120일선 {ma120.iloc[-1]:.1f}({trend120})"
    )
    return buy, sell, desc


def score_macd(close: pd.Series):
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()

    m0, m1 = macd.iloc[-1], macd.iloc[-2]
    s0, s1 = signal.iloc[-1], signal.iloc[-2]
    rising = m0 > m1

    cross_up = m1 < s1 and m0 >= s0
    cross_down = m1 > s1 and m0 <= s0

    if cross_up and m0 < 0:
        buy = 10
    elif cross_up:
        buy = 8
    elif m0 > s0:
        buy = 6
    elif rising:
        buy = 4
    else:
        buy = 0

    if cross_down and m0 > 0:
        sell = 10
    elif cross_down:
        sell = 8
    elif m0 < s0:
        sell = 6
    elif not rising:
        sell = 4
    else:
        sell = 0

    desc = f"MACD {m0:.2f} / Signal {s0:.2f}"
    return buy, sell, desc, float(m0), float(s0)


def score_volume(close: pd.Series, volume: pd.Series):
    avg20 = volume.shift(1).rolling(20).mean()
    v0 = volume.iloc[-1]
    avg0 = avg20.iloc[-1]
    ratio = (v0 / avg0) if avg0 else 1.0

    price_up = close.iloc[-1] > close.iloc[-2]
    price_down = close.iloc[-1] < close.iloc[-2]

    def tier(r):
        if r >= 2.0:
            return 10
        if r >= 1.5:
            return 8
        if r >= 1.2:
            return 6
        if r >= 0.8:
            return 3
        return 1

    buy = tier(ratio) if price_up else 0
    sell = tier(ratio) if price_down else 0
    desc = f"평균 대비 {ratio * 100:.0f}%"
    return buy, sell, desc


def score_ichimoku(high: pd.Series, low: pd.Series, close: pd.Series):
    conv = (high.rolling(9).max() + low.rolling(9).min()) / 2
    base = (high.rolling(26).max() + low.rolling(26).min()) / 2
    span_a = ((conv + base) / 2).shift(26)
    span_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)

    c0, c1 = close.iloc[-1], close.iloc[-2]
    top0, bot0 = max(span_a.iloc[-1], span_b.iloc[-1]), min(span_a.iloc[-1], span_b.iloc[-1])
    top1, bot1 = max(span_a.iloc[-2], span_b.iloc[-2]), min(span_a.iloc[-2], span_b.iloc[-2])
    conv0, base0 = conv.iloc[-1], base.iloc[-1]
    a0, b0 = span_a.iloc[-1], span_b.iloc[-1]

    cross_up = c1 < top1 and c0 >= top0
    cross_down = c1 > bot1 and c0 <= bot0

    if cross_up:
        buy = 10
    elif c0 > top0:
        buy = 8
    elif conv0 > base0:
        buy = 6
    elif a0 > b0:
        buy = 4
    else:
        buy = 0

    if cross_down:
        sell = 10
    elif c0 < bot0:
        sell = 8
    elif conv0 < base0:
        sell = 6
    elif a0 < b0:
        sell = 4
    else:
        sell = 0

    if c0 > top0:
        position = "구름 위"
    elif c0 < bot0:
        position = "구름 아래"
    else:
        position = "구름 안"
    desc = f"{position} (전환선 {conv0:.1f} / 기준선 {base0:.1f})"
    return buy, sell, desc, bot0


# ---------------------------------------------------------------------------
# 종합 판단
# ---------------------------------------------------------------------------

def classify(score: int) -> str:
    if score >= 50:
        return "강력"
    if score >= 40:
        return "보통"
    if score >= 35:
        return "후보"
    return "없음"


def analyze_ticker(ticker: str, name: str) -> dict:
    data = fetch_ohlcv(ticker)
    close, high, low, volume = data["Close"], data["High"], data["Low"], data["Volume"]

    rsi_buy, rsi_sell, rsi_desc, rsi_value = score_rsi(close)
    bb_buy, bb_sell, bb_desc = score_bollinger(close)
    ma_buy, ma_sell, ma_desc = score_ma(close)
    macd_buy, macd_sell, macd_desc, macd_val, signal_val = score_macd(close)
    vol_buy, vol_sell, vol_desc = score_volume(close, volume)
    ichi_buy, ichi_sell, ichi_desc, cloud_bottom = score_ichimoku(high, low, close)

    buy_total = rsi_buy + bb_buy + ma_buy + macd_buy + vol_buy + ichi_buy
    sell_total = rsi_sell + bb_sell + ma_sell + macd_sell + vol_sell + ichi_sell

    price = float(close.iloc[-1])
    ma120 = float(close.rolling(120).mean().iloc[-1])

    # 매수 금지 조건: 120일선 아래 + MACD<Signal + 구름 아래 동시 충족
    buy_blocked = bool((price < ma120) and (macd_val < signal_val) and (price < cloud_bottom))

    buy_grade = classify(buy_total)
    sell_grade = classify(sell_total)

    if not buy_blocked and buy_total >= 40:
        verdict = "매수"
        emoji = "🟢"
    elif sell_total >= 40:
        verdict = "매도"
        emoji = "🔴"
    else:
        verdict = "관망"
        emoji = "🟡"

    label = {
        ("매수", "강력"): "강력 매수",
        ("매수", "보통"): "매수",
        ("매도", "강력"): "강력 매도",
        ("매도", "보통"): "매도",
    }.get((verdict, buy_grade if verdict == "매수" else sell_grade), verdict)
    if buy_blocked and buy_total >= 35:
        label = "관망 (매수 보류: 하락추세+MACD 약세+구름 아래 동시 충족)"

    items = [
        {"name": "RSI", "detail": rsi_desc, "buy": rsi_buy, "sell": rsi_sell},
        {"name": "볼린저밴드", "detail": bb_desc, "buy": bb_buy, "sell": bb_sell},
        {"name": "이동평균", "detail": ma_desc, "buy": ma_buy, "sell": ma_sell},
        {"name": "MACD", "detail": macd_desc, "buy": macd_buy, "sell": macd_sell},
        {"name": "거래량", "detail": vol_desc, "buy": vol_buy, "sell": vol_sell},
        {"name": "일목균형표", "detail": ichi_desc, "buy": ichi_buy, "sell": ichi_sell},
    ]

    return {
        "name": name,
        "price": round(price, 2),
        "rsi": round(rsi_value, 2),
        "buy_score": buy_total,
        "sell_score": sell_total,
        "verdict": verdict,
        "label": label,
        "emoji": emoji,
        "buy_blocked": buy_blocked,
        "items": items,
    }
