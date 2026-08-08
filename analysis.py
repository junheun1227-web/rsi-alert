"""
기술적 분석 스코어링 엔진 (사용자 지정 규칙 v2).

RSI(14)/볼린저밴드(20,2시그마)/이동평균(5,20,60)/MACD(12,26,9)/거래량/일목균형표(9,26,52)
6개 지표로 매수 점수·매도 점수를 각각 100점 만점으로 산출하고, ADX(14) 장세 필터·하락추세
방어·RSI 게이트 보정을 거쳐 최종 판정을 내린다.

데이터는 야후 파이낸스 실제 시세로 계산한다 (추정치 없음). 각 지표는 계산에 필요한 최소
거래일 수가 확보되지 않으면 "데이터 없음"으로 표시하고 0점 처리하며, 6개 중 3개 이상이
데이터 없음이면 점수를 매기지 않고 데이터 부족을 알린다.

투자 판단의 참고용이며 투자 권유가 아니다. 기술적 분석은 미래 수익을 보장하지 않는다.
"""

import numpy as np
import pandas as pd
import yfinance as yf

RSI_PERIOD = 14


# ---------------------------------------------------------------------------
# 데이터 조회
# ---------------------------------------------------------------------------

def ticker_exists(ticker: str) -> bool:
    try:
        data = yf.Ticker(ticker).history(period="5d", interval="1d")
        return not data.empty
    except Exception:
        return False


def fetch_ohlcv(ticker: str, period: str = "2y") -> pd.DataFrame:
    """일목균형표(52+26)까지 안정적으로 계산되도록 2년치 일봉을 가져온다.
    최근 상장 종목 등으로 데이터가 짧으면 각 지표별로 개별적으로 '데이터 없음' 처리한다."""
    data = yf.Ticker(ticker).history(period=period, interval="1d")
    if data.empty or len(data) < RSI_PERIOD + 1:
        raise ValueError(f"{ticker}: 상장 초기 등으로 분석 가능한 데이터가 부족합니다")

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


def _wilder_smooth(series: pd.Series, period: int) -> pd.Series:
    result = series.rolling(window=period).mean().copy()
    for i in range(period + 1, len(series)):
        prev = result.iloc[i - 1]
        if pd.isna(prev):
            continue
        result.iloc[i] = (prev * (period - 1) + series.iloc[i]) / period
    return result


def calc_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """표준 Wilder's ADX(14)."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=high.index
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=high.index
    )

    atr = _wilder_smooth(tr, period)
    plus_di = 100 * _wilder_smooth(plus_dm, period) / atr
    minus_di = 100 * _wilder_smooth(minus_dm, period) / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return _wilder_smooth(dx, period)


def detect_rsi_divergence(close: pd.Series, rsi: pd.Series, lookback: int = 14):
    """단순화된 다이버전스 감지: 최근 lookback일 구간의 저점/고점 대비
    현재 가격은 갱신됐는데 RSI는 갱신되지 않은 경우를 상승/하락 다이버전스로 판정."""
    window_close = close.iloc[-lookback:]
    window_rsi = rsi.iloc[-lookback:]
    if window_close.isna().any() or window_rsi.isna().any():
        return False, False

    idx_min = window_close.idxmin()
    idx_max = window_close.idxmax()
    last_idx = window_close.index[-1]

    bullish = False
    bearish = False
    if idx_min != last_idx:
        if window_close.iloc[-1] < window_close.loc[idx_min] and rsi.iloc[-1] > window_rsi.loc[idx_min]:
            bullish = True
    if idx_max != last_idx:
        if window_close.iloc[-1] > window_close.loc[idx_max] and rsi.iloc[-1] < window_rsi.loc[idx_max]:
            bearish = True
    return bullish, bearish


# ---------------------------------------------------------------------------
# 지표별 채점 - 각 함수는 dict를 반환: {available, buy, sell, value_str, detail}
# ---------------------------------------------------------------------------

def score_rsi(close: pd.Series) -> dict:
    if len(close) < RSI_PERIOD + 1:
        return {"available": False, "buy": 0, "sell": 0, "value": "데이터 없음", "detail": "데이터 없음"}

    rsi_series = calc_rsi(close)
    rsi = float(rsi_series.iloc[-1])
    if pd.isna(rsi):
        return {"available": False, "buy": 0, "sell": 0, "value": "데이터 없음", "detail": "데이터 없음"}

    buy = 20 if rsi <= 30 else (12 if rsi <= 40 else 0)
    sell = 20 if rsi >= 70 else (12 if rsi >= 60 else 0)

    bullish_div, bearish_div = detect_rsi_divergence(close, rsi_series)
    if bullish_div:
        buy += 5
    if bearish_div:
        sell += 5

    div_note = ""
    if bullish_div:
        div_note = ", 상승 다이버전스 포착"
    elif bearish_div:
        div_note = ", 하락 다이버전스 포착"

    zone = "과매도" if rsi <= 30 else ("과매수" if rsi >= 70 else "중립")
    detail = f"{rsi:.1f} ({zone}{div_note})"
    return {"available": True, "buy": buy, "sell": sell, "value": f"{rsi:.1f}", "detail": detail, "raw_rsi": rsi}


def score_bollinger(close: pd.Series) -> dict:
    if len(close) < 20:
        return {"available": False, "buy": 0, "sell": 0, "value": "데이터 없음", "detail": "데이터 없음"}

    mid = close.rolling(20).mean()
    std = close.rolling(20).std()
    upper = mid + 2 * std
    lower = mid - 2 * std

    c0 = close.iloc[-1]
    u0, l0 = upper.iloc[-1], lower.iloc[-1]
    if pd.isna(u0) or pd.isna(l0) or u0 == l0:
        return {"available": False, "buy": 0, "sell": 0, "value": "데이터 없음", "detail": "데이터 없음"}

    percent_b = (c0 - l0) / (u0 - l0)

    buy = 15 if percent_b <= 0 else (9 if percent_b <= 0.2 else 0)
    sell = 15 if percent_b >= 1 else (9 if percent_b >= 0.8 else 0)

    band_width_pct = (u0 - l0) / mid.iloc[-1] * 100 if mid.iloc[-1] else 0
    detail = f"%B={percent_b:.2f} (밴드폭 {band_width_pct:.1f}%)"
    return {"available": True, "buy": buy, "sell": sell, "value": f"{percent_b:.2f}", "detail": detail}


def score_ma(close: pd.Series) -> dict:
    if len(close) < 61:
        return {"available": False, "buy": 0, "sell": 0, "value": "데이터 없음", "detail": "데이터 없음"}

    ma5 = close.rolling(5).mean()
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()

    if pd.isna(ma60.iloc[-1]):
        return {"available": False, "buy": 0, "sell": 0, "value": "데이터 없음", "detail": "데이터 없음"}

    c0 = close.iloc[-1]
    ma5_0, ma5_1 = ma5.iloc[-1], ma5.iloc[-2]
    ma20_0, ma20_1 = ma20.iloc[-1], ma20.iloc[-2]
    ma60_0 = ma60.iloc[-1]

    golden_cross = ma5_1 <= ma20_1 and ma5_0 > ma20_0
    dead_cross = ma5_1 >= ma20_1 and ma5_0 < ma20_0
    aligned_up = ma5_0 > ma20_0 > ma60_0     # 정배열
    aligned_down = ma5_0 < ma20_0 < ma60_0   # 역배열

    buy = (10 if golden_cross else 0) + (5 if c0 > ma20_0 else 0) + (5 if aligned_up else 0)
    sell = (10 if dead_cross else 0) + (5 if c0 < ma20_0 else 0) + (5 if aligned_down else 0)

    state = "정배열" if aligned_up else ("역배열" if aligned_down else "혼조")
    cross_note = "골든크로스" if golden_cross else ("데드크로스" if dead_cross else "")
    disparity = (c0 - ma20_0) / ma20_0 * 100 if ma20_0 else 0
    detail = f"5/20/60일선 {ma5_0:.1f}/{ma20_0:.1f}/{ma60_0:.1f} ({state}{', ' + cross_note if cross_note else ''}, 20일선 이격도 {disparity:+.1f}%)"
    return {"available": True, "buy": buy, "sell": sell, "value": state, "detail": detail}


def score_macd(close: pd.Series) -> dict:
    if len(close) < 36:
        return {"available": False, "buy": 0, "sell": 0, "value": "데이터 없음", "detail": "데이터 없음"}

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal

    m0, m1 = macd.iloc[-1], macd.iloc[-2]
    s0, s1 = signal.iloc[-1], signal.iloc[-2]
    h0, h1 = hist.iloc[-1], hist.iloc[-2]

    cross_up = m1 <= s1 and m0 > s0
    cross_down = m1 >= s1 and m0 < s0
    hist_flip_up = h1 < 0 and h0 > 0
    hist_flip_down = h1 > 0 and h0 < 0

    buy = (10 if cross_up else 0) + (5 if hist_flip_up else 0) + (5 if m0 > 0 else 0)
    sell = (10 if cross_down else 0) + (5 if hist_flip_down else 0) + (5 if m0 < 0 else 0)

    zero_pos = "0선 위" if m0 > 0 else "0선 아래"
    cross_note = "시그널 상향돌파" if cross_up else ("시그널 하향돌파" if cross_down else "교차 없음")
    detail = f"MACD {m0:.2f} / Signal {s0:.2f} ({zero_pos}, {cross_note})"
    return {"available": True, "buy": buy, "sell": sell, "value": f"{m0:.2f}", "detail": detail}


def score_volume(close: pd.Series, volume: pd.Series) -> dict:
    if len(close) < 21:
        return {"available": False, "buy": 0, "sell": 0, "value": "데이터 없음", "detail": "데이터 없음"}

    avg20 = volume.shift(1).rolling(20).mean()
    v0, avg0 = volume.iloc[-1], avg20.iloc[-1]
    if pd.isna(avg0) or avg0 == 0:
        return {"available": False, "buy": 0, "sell": 0, "value": "데이터 없음", "detail": "데이터 없음"}

    ratio = v0 / avg0
    up_day = close.iloc[-1] > close.iloc[-2]
    down_day = close.iloc[-1] < close.iloc[-2]

    if up_day and ratio >= 1.5:
        buy = 10
    elif up_day and ratio >= 1.2:
        buy = 6
    else:
        buy = 0

    if down_day and ratio >= 1.5:
        sell = 10
    elif up_day and ratio < 0.8:  # 상승 중 거래량 감소 (모멘텀 약화)
        sell = 6
    else:
        sell = 0

    direction = "양봉" if up_day else ("음봉" if down_day else "보합")
    detail = f"20일 평균 대비 {ratio * 100:.0f}% ({direction})"
    return {"available": True, "buy": buy, "sell": sell, "value": f"{ratio * 100:.0f}%", "detail": detail}


def score_ichimoku(high: pd.Series, low: pd.Series, close: pd.Series) -> dict:
    if len(close) < 79:
        return {"available": False, "buy": 0, "sell": 0, "value": "데이터 없음", "detail": "데이터 없음"}

    conv = (high.rolling(9).max() + low.rolling(9).min()) / 2
    base = (high.rolling(26).max() + low.rolling(26).min()) / 2
    span_a = ((conv + base) / 2).shift(26)
    span_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)

    c0, c1 = close.iloc[-1], close.iloc[-2]
    a0, b0 = span_a.iloc[-1], span_b.iloc[-1]
    a1, b1 = span_a.iloc[-2], span_b.iloc[-2]
    if pd.isna(a0) or pd.isna(b0):
        return {"available": False, "buy": 0, "sell": 0, "value": "데이터 없음", "detail": "데이터 없음"}

    top0, bot0 = max(a0, b0), min(a0, b0)
    top1, bot1 = max(a1, b1), min(a1, b1)
    conv0, base0 = conv.iloc[-1], base.iloc[-1]

    cross_up = c1 < top1 and c0 >= top0
    cross_down = c1 > bot1 and c0 <= bot0

    chikou_prev = close.iloc[-27] if len(close) >= 27 else None
    chikou_buy = chikou_prev is not None and c0 > chikou_prev
    chikou_sell = chikou_prev is not None and c0 < chikou_prev

    buy = (6 if cross_up else 0) + (5 if conv0 > base0 else 0) + (4 if chikou_buy else 0)
    sell = (6 if cross_down else 0) + (5 if conv0 < base0 else 0) + (4 if chikou_sell else 0)

    position = "구름 위" if c0 > top0 else ("구름 아래" if c0 < bot0 else "구름 안")
    cloud_thickness = abs(a0 - b0) / c0 * 100 if c0 else 0
    detail = (
        f"{position}, 전환선 {conv0:.1f} / 기준선 {base0:.1f}, "
        f"구름두께 {cloud_thickness:.1f}%, 후행스팬 {'우호적' if chikou_buy else ('비우호적' if chikou_sell else '중립')}"
    )
    return {"available": True, "buy": buy, "sell": sell, "value": position, "detail": detail}


# ---------------------------------------------------------------------------
# 종합 판단
# ---------------------------------------------------------------------------

def analyze_ticker(ticker: str, name: str) -> dict:
    data = fetch_ohlcv(ticker)
    close, high, low, volume = data["Close"], data["High"], data["Low"], data["Volume"]
    ref_date = data.index[-1].strftime("%Y-%m-%d")
    price = float(close.iloc[-1])

    rsi_r = score_rsi(close)
    bb_r = score_bollinger(close)
    ma_r = score_ma(close)
    macd_r = score_macd(close)
    vol_r = score_volume(close, volume)
    ichi_r = score_ichimoku(high, low, close)

    results = {"RSI(14)": rsi_r, "볼린저밴드 %B": bb_r, "이동평균선": ma_r, "MACD": macd_r,
               "거래량": vol_r, "일목균형표": ichi_r}
    unavailable = [k for k, v in results.items() if not v["available"]]

    if len(unavailable) >= 3:
        return {
            "name": name,
            "ref_date": ref_date,
            "price": round(price, 2),
            "insufficient_data": True,
            "missing": unavailable,
            "results": results,
        }

    # --- ADX(14) 장세 필터 ---
    adx_series = calc_adx(high, low, close) if len(close) >= 30 else None
    adx_value = float(adx_series.iloc[-1]) if adx_series is not None and not pd.isna(adx_series.iloc[-1]) else None

    if adx_value is not None and adx_value < 20:
        regime, mean_rev_mult, trend_mult = "횡보장", 1.5, 0.5
    elif adx_value is not None and adx_value > 25:
        regime, mean_rev_mult, trend_mult = "추세장", 0.5, 1.5
    else:
        regime, mean_rev_mult, trend_mult = "중립", 1.0, 1.0

    rsi_buy, rsi_sell = rsi_r["buy"] * mean_rev_mult, rsi_r["sell"] * mean_rev_mult
    bb_buy, bb_sell = bb_r["buy"] * mean_rev_mult, bb_r["sell"] * mean_rev_mult
    ma_buy, ma_sell = ma_r["buy"] * trend_mult, ma_r["sell"] * trend_mult
    macd_buy, macd_sell = macd_r["buy"] * trend_mult, macd_r["sell"] * trend_mult
    vol_buy, vol_sell = vol_r["buy"], vol_r["sell"]
    ichi_buy, ichi_sell = ichi_r["buy"], ichi_r["sell"]

    buy_total = rsi_buy + bb_buy + ma_buy + macd_buy + vol_buy + ichi_buy
    sell_total = rsi_sell + bb_sell + ma_sell + macd_sell + vol_sell + ichi_sell

    adjustments = [f"장세: ADX {adx_value:.1f} ({regime})" if adx_value is not None else "장세: ADX 계산 불가 (중립 적용)"]
    if regime != "중립":
        adjustments.append(f"RSI·볼린저 x{mean_rev_mult}, 이평선·MACD x{trend_mult} 적용")

    # --- 하락추세 방어: 종가 < 120일선이면 매수 -15 ---
    ma120 = close.rolling(120).mean()
    ma120_0 = ma120.iloc[-1] if len(close) >= 120 else None
    if ma120_0 is not None and not pd.isna(ma120_0) and price < ma120_0:
        buy_total -= 15
        adjustments.append("하락추세 방어: 종가<120일선, 매수 -15")
    buy_total = max(0, buy_total)
    sell_total = max(0, sell_total)

    # --- 판정 ---
    if buy_total >= 75:
        verdict = "적극 매수"
    elif buy_total >= 60:
        verdict = "매수"
    elif sell_total >= 75:
        verdict = "전량 매도"
    elif sell_total >= 60:
        verdict = "분할 매도"
    else:
        verdict = "관망"

    # --- RSI 게이트: RSI>=45면 매수 판정 금지 ---
    rsi_value = rsi_r.get("raw_rsi")
    gate_applied = False
    if rsi_value is not None and rsi_value >= 45 and verdict in ("적극 매수", "매수"):
        verdict = "관망"
        gate_applied = True
        adjustments.append(f"RSI 게이트: RSI {rsi_value:.1f} >= 45, 매수 판정 무효화 -> 관망")

    emoji = {"적극 매수": "🟢", "매수": "🟢", "전량 매도": "🔴", "분할 매도": "🔴", "관망": "🟡"}[verdict]

    # 상충 신호 메모
    conflicts = []
    if buy_total >= 40 and sell_total >= 40:
        conflicts.append("매수·매도 점수가 모두 높게 나와 신호가 엇갈립니다")
    if rsi_r["available"] and ma_r["available"]:
        if rsi_r["buy"] > 0 and ma_r["sell"] > 0:
            conflicts.append("RSI는 매수 쪽, 이동평균선은 매도 쪽을 가리킵니다")
        elif rsi_r["sell"] > 0 and ma_r["buy"] > 0:
            conflicts.append("RSI는 매도 쪽, 이동평균선은 매수 쪽을 가리킵니다")

    return {
        "name": name,
        "ref_date": ref_date,
        "price": round(price, 2),
        "insufficient_data": False,
        "results": results,
        "adx": adx_value,
        "regime": regime,
        "adjustments": adjustments,
        "buy_score": round(buy_total, 1),
        "sell_score": round(sell_total, 1),
        "verdict": verdict,
        "emoji": emoji,
        "gate_applied": gate_applied,
        "conflicts": conflicts,
        # 하위 호환(기존 rsi_alert.py의 30/25 push 알림 로직이 참조)
        "rsi": round(rsi_value, 2) if rsi_value is not None else None,
    }
