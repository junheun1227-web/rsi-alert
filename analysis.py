"""
기술적 분석 스코어링 엔진 (사용자 지정 규칙 v4 - 11개 지표).

지표(원점수 배점):
  RSI(14) 15 / 스토캐스틱(14,3,3) 8 / 볼린저밴드 %B(20,2σ) 12 / 이동평균선(5,20,60,120) 15
  MACD(12,26,9) 15 / 거래량·OBV 10 / 일목균형표(9,26,52) 12 / CCI(14) 5 / ATR(14) 5
  캔들 패턴 5 / 지지·저항 8
  (사양서 배점을 그대로 합산하면 상한은 110점이며, 판정 임계값 70/50/35는 이 척도에 적용한다.)

보정 순서: ①ADX 장세 필터(+DI/-DI 하락추세 추가 감점) → ②주봉 다중 시간프레임 일치도
  → ③시장지수 필터 → ④120일선 하락추세 방어 → ⑤실적 발표 임박 → ⑥RSI 게이트(단계 하향)
  → ⑦데이터 신뢰도 보정(단계 하향)

데이터: 야후 파이낸스 실거래 시세(OHLCV)만 사용한다. 추정·창작값은 쓰지 않으며, 계산에
필요한 거래일이 부족한 지표는 "데이터 없음" 표기 후 0점 처리한다. 11개 중 4개 이상이
데이터 없음이면 점수를 내지 않고 부족한 지표를 사용자에게 알린다. 단일 소스만 사용하므로
소스 간 값 상충은 발생하지 않는다(상충 시 낮은 점수 반영 규칙은 해당 없음).
데이터 신뢰도 = 확보 지표 수 ÷ 11 × 100(%).

향후 주가 예측·목표가는 산출하지 않는다.
"""

import re

import numpy as np
import pandas as pd
import requests
import yfinance as yf

RSI_PERIOD = 14
TOTAL_INDICATORS = 11

INDICATOR_ORDER = [
    "RSI(14)", "스토캐스틱", "볼린저 %B", "이동평균선", "MACD", "거래량/OBV",
    "일목균형표", "CCI(14)", "ATR(14)", "캔들 패턴", "지지·저항",
]

# 장세 필터 그룹
MEAN_REV_GROUP = ["RSI(14)", "스토캐스틱", "볼린저 %B", "CCI(14)"]
TREND_GROUP = ["이동평균선", "MACD", "일목균형표"]


def _nodata(msg: str = "계산에 필요한 거래일 수가 부족합니다.") -> dict:
    return {"available": False, "buy": 0, "sell": 0, "value": "데이터 없음",
            "detail": "데이터 없음", "reason": msg}


# ---------------------------------------------------------------------------
# 데이터 조회
# ---------------------------------------------------------------------------

def ticker_exists(ticker: str) -> bool:
    try:
        return not yf.Ticker(ticker).history(period="5d", interval="1d").empty
    except Exception:
        return False


def fetch_ohlcv(ticker: str, period: str = "2y") -> pd.DataFrame:
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
# 공용 계산 유틸
# ---------------------------------------------------------------------------

def calc_rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    delta = close.diff()
    gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean().copy()
    avg_loss = loss.rolling(period).mean().copy()
    for i in range(period + 1, len(gain)):
        avg_gain.iloc[i] = (avg_gain.iloc[i - 1] * (period - 1) + gain.iloc[i]) / period
        avg_loss.iloc[i] = (avg_loss.iloc[i - 1] * (period - 1) + loss.iloc[i]) / period
    return 100 - (100 / (1 + avg_gain / avg_loss))


def _wilder(series: pd.Series, period: int) -> pd.Series:
    out = series.rolling(period).mean().copy()
    for i in range(period + 1, len(series)):
        if pd.isna(out.iloc[i - 1]):
            continue
        out.iloc[i] = (out.iloc[i - 1] * (period - 1) + series.iloc[i]) / period
    return out


def calc_true_range(high, low, close) -> pd.Series:
    pc = close.shift(1)
    return pd.concat([high - low, (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)


def calc_atr(high, low, close, period: int = 14) -> pd.Series:
    return _wilder(calc_true_range(high, low, close), period)


def calc_adx_di(high, low, close, period: int = 14):
    """Wilder's ADX와 +DI/-DI를 함께 반환."""
    tr = calc_true_range(high, low, close)
    up, down = high.diff(), -low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=high.index)
    atr = _wilder(tr, period)
    plus_di = 100 * _wilder(plus_dm, period) / atr
    minus_di = 100 * _wilder(minus_dm, period) / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return _wilder(dx, period), plus_di, minus_di


def calc_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff().fillna(0))
    return (direction * volume).cumsum()


def _recent_cross(fast: pd.Series, slow: pd.Series, lookback: int = 5):
    """최근 lookback영업일 내 상향/하향 교차 여부와 발생 시점(며칠 전)."""
    up = down = False
    days_ago = None
    n = len(fast)
    for i in range(max(1, n - lookback), n):
        if pd.isna(fast.iloc[i - 1]) or pd.isna(slow.iloc[i - 1]):
            continue
        if fast.iloc[i - 1] <= slow.iloc[i - 1] and fast.iloc[i] > slow.iloc[i]:
            up, down, days_ago = True, False, n - 1 - i
        if fast.iloc[i - 1] >= slow.iloc[i - 1] and fast.iloc[i] < slow.iloc[i]:
            down, up, days_ago = True, False, n - 1 - i
    return up, down, days_ago


def _shrinking(series: pd.Series, days: int = 3) -> bool:
    """절댓값이 days일 연속 축소 중인지."""
    if len(series) < days + 1:
        return False
    vals = series.abs().iloc[-(days + 1):]
    if vals.isna().any():
        return False
    return bool(all(vals.iloc[i] < vals.iloc[i - 1] for i in range(1, len(vals))))


# ---------------------------------------------------------------------------
# 지표별 채점
# ---------------------------------------------------------------------------

def score_rsi(close: pd.Series) -> dict:
    if len(close) < RSI_PERIOD + 1:
        return _nodata()
    rsi_s = calc_rsi(close)
    rsi = float(rsi_s.iloc[-1])
    if pd.isna(rsi):
        return _nodata()

    # RSI 50~70대는 상승 추세에서 정상적인 강세 구간이라 매도 신호로 보지 않는다.
    # 진짜 과매수(추격 매수 자제)는 70 이상부터, 극단적 과매수는 80 이상으로 본다.
    buy = 15 if rsi <= 30 else (11 if rsi <= 40 else (6 if rsi <= 45 else 0))
    sell = 15 if rsi >= 80 else (9 if rsi >= 70 else 0)

    # 다이버전스 (최근 14일 저점/고점 대비 가격-RSI 방향 불일치)
    bull = bear = False
    if len(close) >= 20:
        w_c, w_r = close.iloc[-14:], rsi_s.iloc[-14:]
        if not (w_c.isna().any() or w_r.isna().any()):
            i_min, i_max, last = w_c.idxmin(), w_c.idxmax(), w_c.index[-1]
            if i_min != last and w_c.iloc[-1] < w_c.loc[i_min] and rsi_s.iloc[-1] > w_r.loc[i_min]:
                bull = True
            if i_max != last and w_c.iloc[-1] > w_c.loc[i_max] and rsi_s.iloc[-1] < w_r.loc[i_max]:
                bear = True
    if bull:
        buy = min(15, buy + 5)
    if bear:
        sell = min(15, sell + 5)

    zone = ("과매도(≤30)" if rsi <= 30 else "과매수(≥70)" if rsi >= 70
            else "약세권(30~45)" if rsi <= 45 else "강세권(55~70)" if rsi >= 55 else "중립(45~55)")
    prev = rsi_s.iloc[-6] if len(rsi_s) >= 6 and not pd.isna(rsi_s.iloc[-6]) else None
    dir_txt = ("방향성 판단 불가" if prev is None
               else f"5일 전 {prev:.1f}에서 상승" if rsi - prev > 2
               else f"5일 전 {prev:.1f}에서 하락" if prev - rsi > 2
               else f"5일 전 {prev:.1f}과 큰 변화 없음")
    div_txt = "상승 다이버전스 확인" if bull else ("하락 다이버전스 확인" if bear else "다이버전스 없음")
    return {"available": True, "buy": buy, "sell": sell, "value": f"{rsi:.1f}",
            "detail": f"{rsi:.1f} {zone}",
            "reason": f"RSI {rsi:.1f}로 {zone}, {dir_txt}, {div_txt}", "raw_rsi": rsi,
            "bull_div": bull, "bear_div": bear}


def score_stochastic(high, low, close) -> dict:
    if len(close) < 20:
        return _nodata()
    ll, hh = low.rolling(14).min(), high.rolling(14).max()
    rng = (hh - ll).replace(0, np.nan)
    k_fast = 100 * (close - ll) / rng
    k = k_fast.rolling(3).mean()      # slow %K
    d = k.rolling(3).mean()           # %D
    if pd.isna(k.iloc[-1]) or pd.isna(d.iloc[-1]):
        return _nodata()
    k0, d0 = float(k.iloc[-1]), float(d.iloc[-1])
    up, down, _ = _recent_cross(k, d, lookback=2)

    if k0 <= 20 and up:
        buy = 8
    elif k0 <= 20:
        buy = 5
    elif k0 <= 30:
        buy = 3
    else:
        buy = 0
    if k0 >= 80 and down:
        sell = 8
    elif k0 >= 80:
        sell = 5
    elif k0 >= 70:
        sell = 3
    else:
        sell = 0

    zone = "침체권(≤20)" if k0 <= 20 else ("과열권(≥80)" if k0 >= 80 else "중립권")
    cross = "%K가 %D를 상향돌파" if up else ("%K가 %D를 하향돌파" if down else "교차 없음")
    return {"available": True, "buy": buy, "sell": sell, "value": f"%K {k0:.0f}/%D {d0:.0f}",
            "detail": f"%K {k0:.1f} / %D {d0:.1f} ({zone}, {cross})",
            "reason": f"%K {k0:.1f}·%D {d0:.1f}로 {zone}, {cross}"}


def score_bollinger(close: pd.Series) -> dict:
    if len(close) < 20:
        return _nodata()
    mid, std = close.rolling(20).mean(), close.rolling(20).std()
    upper, lower = mid + 2 * std, mid - 2 * std
    c0, u0, l0, m0 = close.iloc[-1], upper.iloc[-1], lower.iloc[-1], mid.iloc[-1]
    if pd.isna(u0) or pd.isna(l0) or u0 == l0:
        return _nodata()
    pb = (c0 - l0) / (u0 - l0)

    buy = 12 if pb <= 0 else (8 if pb <= 0.2 else (4 if pb <= 0.35 else 0))
    sell = 12 if pb >= 1 else (8 if pb >= 0.8 else (4 if pb >= 0.65 else 0))

    bw = (upper - lower) / mid * 100
    bw0 = float(bw.iloc[-1])
    lookback = min(126, len(bw.dropna()))  # 6개월
    bw_min = float(bw.dropna().iloc[-lookback:].min()) if lookback > 0 else bw0
    squeeze = bw0 <= bw_min * 1.05

    pos = ("상단 이탈" if pb >= 1 else "상단 근접" if pb >= 0.8 else "상단권" if pb >= 0.65
           else "하단 이탈" if pb <= 0 else "하단 근접" if pb <= 0.2 else "하단권" if pb <= 0.35 else "중앙부")
    sq_txt = "밴드폭이 6개월 최저 수준인 스퀴즈 상태로 방향성 돌파 대기" if squeeze else f"밴드폭 {bw0:.1f}%"
    return {"available": True, "buy": buy, "sell": sell,
            "value": f"{pb:.2f}" + (" (스퀴즈)" if squeeze else ""),
            "detail": f"%B {pb:.2f} ({pos}), {sq_txt}",
            "reason": f"%B {pb:.2f}로 {pos}, {sq_txt}", "squeeze": squeeze}


def score_ma(close: pd.Series) -> dict:
    if len(close) < 61:
        return _nodata()
    ma5, ma20 = close.rolling(5).mean(), close.rolling(20).mean()
    ma60, ma120 = close.rolling(60).mean(), close.rolling(120).mean()
    if pd.isna(ma60.iloc[-1]):
        return _nodata()
    c0, a5, a20, a60 = close.iloc[-1], ma5.iloc[-1], ma20.iloc[-1], ma60.iloc[-1]
    a120 = ma120.iloc[-1] if len(close) >= 120 else np.nan

    golden, dead, days_ago = _recent_cross(ma5, ma20, lookback=5)
    imminent = bool(abs(a5 - a20) / a20 <= 0.01) if a20 else False
    up_align, down_align = a5 > a20 > a60, a5 < a20 < a60

    # 크로스 발생과 임박은 택일
    buy = (7 if golden else (3 if (imminent and not dead) else 0)) + (4 if c0 > a20 else 0) + (4 if up_align else 0)
    sell = (7 if dead else (3 if (imminent and not golden) else 0)) + (4 if c0 < a20 else 0) + (4 if down_align else 0)

    state = "정배열" if up_align else ("역배열" if down_align else "혼조")
    cross = (f"{days_ago}일 전 골든크로스" if golden else f"{days_ago}일 전 데드크로스" if dead
             else "5·20일선 이격 1% 이내로 크로스 임박" if imminent else "최근 크로스 없음")
    disp = (c0 - a20) / a20 * 100 if a20 else 0
    d120 = f", 120일선 이격 {(c0 - a120) / a120 * 100:+.1f}%" if not pd.isna(a120) else ""
    return {"available": True, "buy": buy, "sell": sell, "value": state,
            "detail": f"{state}, {cross}, 20일선 이격 {disp:+.1f}%{d120}",
            "reason": f"{state} 구조에서 {cross}, 20일선 이격도 {disp:+.1f}%{d120}",
            "ma120": None if pd.isna(a120) else float(a120)}


def score_macd(close: pd.Series) -> dict:
    if len(close) < 36:
        return _nodata()
    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    m0, s0, h0 = macd.iloc[-1], signal.iloc[-1], hist.iloc[-1]
    up, down, days_ago = _recent_cross(macd, signal, lookback=5)
    shrink = _shrinking(hist, days=3)

    # 돌파와 히스토그램 축소는 택일
    buy = (8 if up else (4 if (shrink and h0 < 0) else 0)) + (4 if m0 > 0 else 0)
    sell = (8 if down else (4 if (shrink and h0 > 0) else 0)) + (4 if m0 < 0 else 0)

    zero = "0선 위" if m0 > 0 else "0선 아래"
    ev = (f"{days_ago}일 전 시그널선 상향돌파" if up else f"{days_ago}일 전 시그널선 하향돌파" if down
          else "히스토그램 3일 연속 축소" if shrink else "교차·축소 신호 없음")
    return {"available": True, "buy": buy, "sell": sell, "value": f"{m0:.2f}",
            "detail": f"MACD {m0:.2f}/Signal {s0:.2f} ({zero}, {ev})",
            "reason": f"MACD {m0:.2f}·시그널 {s0:.2f}로 {zero}, {ev}"}


def score_volume(close, volume) -> dict:
    if len(close) < 21:
        return _nodata()
    avg20 = volume.shift(1).rolling(20).mean()
    v0, a0 = volume.iloc[-1], avg20.iloc[-1]
    if pd.isna(a0) or a0 == 0:
        return _nodata()
    ratio = v0 / a0
    up_day = close.iloc[-1] > close.iloc[-2]
    down_day = close.iloc[-1] < close.iloc[-2]

    buy = 6 if (up_day and ratio >= 1.5) else (4 if (up_day and ratio >= 1.2) else 0)
    sell = 6 if (down_day and ratio >= 1.5) else (4 if (up_day and ratio < 0.8) else 0)

    # OBV 다이버전스 (최근 20일)
    obv = calc_obv(close, volume)
    obv_bull = obv_bear = False
    if len(close) >= 25:
        w_c, w_o = close.iloc[-20:], obv.iloc[-20:]
        i_min, i_max, last = w_c.idxmin(), w_c.idxmax(), w_c.index[-1]
        if i_min != last and w_c.iloc[-1] <= w_c.loc[i_min] * 1.01 and w_o.iloc[-1] > w_o.loc[i_min]:
            obv_bull = True
        if i_max != last and w_c.iloc[-1] >= w_c.loc[i_max] * 0.99 and w_o.iloc[-1] < w_o.loc[i_max]:
            obv_bear = True
    if obv_bull:
        buy += 4
    if obv_bear:
        sell += 4

    direction = "양봉" if up_day else ("음봉" if down_day else "보합")
    obv_txt = "OBV 상승 다이버전스" if obv_bull else ("OBV 하락 다이버전스" if obv_bear else "OBV 다이버전스 없음")
    return {"available": True, "buy": min(buy, 10), "sell": min(sell, 10),
            "value": f"{ratio * 100:.0f}%",
            "detail": f"20일 평균 대비 {ratio:.2f}배 ({direction}), {obv_txt}",
            "reason": f"거래량이 20일 평균의 {ratio:.2f}배이고 당일은 {direction}, {obv_txt}"}


def score_ichimoku(high, low, close) -> dict:
    if len(close) < 79:
        return _nodata()
    conv = (high.rolling(9).max() + low.rolling(9).min()) / 2
    base = (high.rolling(26).max() + low.rolling(26).min()) / 2
    span_a = ((conv + base) / 2).shift(26)
    span_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
    c0, a0, b0 = close.iloc[-1], span_a.iloc[-1], span_b.iloc[-1]
    if pd.isna(a0) or pd.isna(b0):
        return _nodata()
    top, bot = max(a0, b0), min(a0, b0)
    conv0, base0 = conv.iloc[-1], base.iloc[-1]
    above, below = c0 > top, c0 < bot

    thickness = abs(a0 - b0) / c0 * 100 if c0 else 0
    thick_series = (span_a - span_b).abs() / close * 100
    thinning = bool(_shrinking(thick_series, days=3)) and thickness < 3

    # 구름 돌파와 전환 임박은 택일
    buy = (5 if above else (2 if (thinning and not below) else 0)) + (4 if conv0 > base0 else 0) + \
          (3 if c0 > close.iloc[-27] else 0)
    sell = (5 if below else (2 if (thinning and not above) else 0)) + (4 if conv0 < base0 else 0) + \
           (3 if c0 < close.iloc[-27] else 0)

    pos = "구름 위" if above else ("구름 아래" if below else "구름 안")
    return {"available": True, "buy": buy, "sell": sell, "value": pos,
            "detail": f"{pos}, 전환선 {conv0:.1f}/기준선 {base0:.1f}, 구름두께 {thickness:.1f}%"
                      + (", 구름 얇아지며 전환 임박" if thinning else ""),
            "reason": f"주가는 {pos}, 전환선{'>' if conv0 > base0 else '<'}기준선, "
                      f"후행스팬 {'우위' if c0 > close.iloc[-27] else '열위'}, 구름두께 {thickness:.1f}%"}


def score_cci(high, low, close) -> dict:
    if len(close) < 20:
        return _nodata()
    tp = (high + low + close) / 3
    sma = tp.rolling(14).mean()
    md = tp.rolling(14).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    cci = (tp - sma) / (0.015 * md.replace(0, np.nan))
    if pd.isna(cci.iloc[-1]):
        return _nodata()
    c0, c1 = float(cci.iloc[-1]), float(cci.iloc[-2])

    if c0 <= -100 and c0 > c1:
        buy, btxt = 5, "-100 이하에서 반등 시작"
    elif c0 <= -100:
        buy, btxt = 3, "-100 이하 유지"
    else:
        buy, btxt = 0, ""
    if c0 >= 100 and c0 < c1:
        sell, stxt = 5, "+100 이상에서 꺾임 시작"
    elif c0 >= 100:
        sell, stxt = 3, "+100 이상 유지"
    else:
        sell, stxt = 0, ""

    state = btxt or stxt or "±100 밴드 내 중립"
    return {"available": True, "buy": buy, "sell": sell, "value": f"{c0:.0f}",
            "detail": f"CCI {c0:.1f} ({state})", "reason": f"CCI {c0:.1f}, {state}"}


def score_atr(high, low, close) -> dict:
    if len(close) < 36:
        return _nodata()
    atr = calc_atr(high, low, close)
    a0 = atr.iloc[-1]
    avg = atr.rolling(20).mean().iloc[-1]
    if pd.isna(a0) or pd.isna(avg) or avg == 0:
        return _nodata()
    ratio = a0 / avg
    down_day = close.iloc[-1] < close.iloc[-2]

    if ratio <= 0.85:
        buy, btxt = 5, "변동성 수렴(20일 평균 대비 축소)"
    elif ratio <= 1.15:
        buy, btxt = 2, "변동성 평균 수준"
    else:
        buy, btxt = 0, ""
    if ratio >= 1.5 and down_day:
        sell, stxt = 5, "변동성 급확대 + 음봉(패닉 구간)"
    elif ratio > 1.15:
        sell, stxt = 2, "변동성 확대 중"
    else:
        sell, stxt = 0, ""

    state = stxt or btxt or "변동성 특이사항 없음"
    pct = a0 / close.iloc[-1] * 100
    return {"available": True, "buy": buy, "sell": sell, "value": f"{ratio:.2f}배",
            "detail": f"ATR {a0:.2f}(주가의 {pct:.1f}%), 20일 평균의 {ratio:.2f}배 — {state}",
            "reason": f"ATR은 20일 평균의 {ratio:.2f}배로 {state}"}


def score_candle(o, h, l, c) -> dict:
    if len(c) < 5:
        return _nodata()
    o0, h0, l0, c0 = o.iloc[-1], h.iloc[-1], l.iloc[-1], c.iloc[-1]
    o1, c1 = o.iloc[-2], c.iloc[-2]
    o2, c2 = o.iloc[-3], c.iloc[-3]
    body = abs(c0 - o0) or (h0 - l0) * 0.001 or 1e-9
    upper_sh, lower_sh = h0 - max(o0, c0), min(o0, c0) - l0
    bull, bear = c0 > o0, c0 < o0

    lo20 = l.iloc[-20:].min() if len(l) >= 20 else l.min()
    hi20 = h.iloc[-20:].max() if len(h) >= 20 else h.max()
    at_low = c0 <= lo20 * 1.03
    at_high = c0 >= hi20 * 0.97

    buy = sell = 0
    pattern = "특이 패턴 없음"
    # 매수 패턴
    if at_low and bull and lower_sh >= 2 * body and upper_sh <= body:
        buy, pattern = 5, "저점권 망치형"
    elif bull and c1 < o1 and c0 >= o1 and o0 <= c1:
        buy, pattern = 5, "상승장악형"
    elif bull and abs(c1 - o1) < body * 0.5 and c2 < o2 and c0 > (o2 + c2) / 2:
        buy, pattern = 5, "샛별형"
    elif bull and lower_sh >= body:
        buy, pattern = 3, "아랫꼬리 긴 양봉"
    # 매도 패턴
    elif at_high and upper_sh >= 2 * body and lower_sh <= body:
        sell, pattern = 5, "고점권 유성형"
    elif bear and c1 > o1 and c0 <= o1 and o0 >= c1:
        sell, pattern = 5, "하락장악형"
    elif bear and abs(c1 - o1) < body * 0.5 and c2 > o2 and c0 < (o2 + c2) / 2:
        sell, pattern = 5, "석별형"
    elif bear and upper_sh >= body:
        sell, pattern = 3, "윗꼬리 긴 음봉"

    return {"available": True, "buy": buy, "sell": sell, "value": pattern,
            "detail": f"{pattern} ({'양봉' if bull else '음봉' if bear else '보합'})",
            "reason": f"직전 캔들은 {'양봉' if bull else '음봉' if bear else '보합'}이며 {pattern} 확인"}


def score_support_resistance(high, low, close) -> dict:
    if len(close) < 61:
        return _nodata()
    c0 = close.iloc[-1]
    # 피봇 (직전 20거래일 기준)
    ph, pl, pc = high.iloc[-21:-1].max(), low.iloc[-21:-1].min(), close.iloc[-2]
    p = (ph + pl + pc) / 3
    s1, s2 = 2 * p - ph, p - (ph - pl)
    r1, r2 = 2 * p - pl, p + (ph - pl)
    prev_low = low.iloc[-61:-1].min()
    prev_high = high.iloc[-61:-1].max()
    ma60 = close.rolling(60).mean().iloc[-1]

    supports = [x for x in [s1, s2, prev_low, ma60] if x and not pd.isna(x) and x <= c0 * 1.01]
    resists = [x for x in [r1, r2, prev_high] if x and not pd.isna(x) and x >= c0 * 0.99]

    sup_d = min((abs(c0 - x) / c0 * 100 for x in supports), default=None)
    res_d = min((abs(x - c0) / c0 * 100 for x in resists), default=None)

    buy = 8 if (sup_d is not None and sup_d <= 3) else (5 if (sup_d is not None and sup_d <= 5) else 0)
    sell = 8 if (res_d is not None and res_d <= 3) else (5 if (res_d is not None and res_d <= 5) else 0)

    parts = []
    if sup_d is not None:
        parts.append(f"최근접 지지선까지 {sup_d:.1f}%")
    if res_d is not None:
        parts.append(f"최근접 저항선까지 {res_d:.1f}%")
    txt = ", ".join(parts) if parts else "주요 레벨과 거리 있음"
    return {"available": True, "buy": buy, "sell": sell,
            "value": (f"지지 {sup_d:.1f}%" if sup_d is not None else "지지 -") +
                     (f" / 저항 {res_d:.1f}%" if res_d is not None else " / 저항 -"),
            "detail": txt, "reason": txt}


# ---------------------------------------------------------------------------
# 보정용 보조 데이터
# ---------------------------------------------------------------------------

def weekly_confirmation(data: pd.DataFrame):
    """일봉 데이터를 주봉으로 리샘플해 주봉 RSI와 20주선 방향을 확인.
    별도 네트워크 호출 없이 계산하며, 데이터가 부족하면 None을 반환한다."""
    try:
        wk = data.resample("W-FRI").agg({"Open": "first", "High": "max", "Low": "min",
                                         "Close": "last", "Volume": "sum"}).dropna()
        if len(wk) < 25:
            return None
        wc = wk["Close"]
        w_rsi = float(calc_rsi(wc).iloc[-1])
        ma20w = wc.rolling(20).mean()
        if pd.isna(w_rsi) or pd.isna(ma20w.iloc[-1]) or pd.isna(ma20w.iloc[-4]):
            return None
        rising = bool(ma20w.iloc[-1] > ma20w.iloc[-4])
        bullish = bool(w_rsi >= 50 and rising)
        bearish = bool(w_rsi < 50 and not rising)
        verdict = "상승" if bullish else ("하락" if bearish else "중립")
        return {"rsi": w_rsi, "ma20w_rising": rising, "verdict": verdict}
    except Exception:
        return None


def market_index_state(ticker: str):
    """종목 소속 시장 지수(KOSPI/NASDAQ)가 20일선 위/아래이고 상승/하락 중인지."""
    symbol = "^KS11" if ticker.upper().endswith((".KS", ".KQ")) else "^IXIC"
    name = "KOSPI" if symbol == "^KS11" else "NASDAQ"
    try:
        idx = yf.Ticker(symbol).history(period="6mo", interval="1d")
        if idx.empty or len(idx) < 25:
            return None
        c = idx["Close"]
        ma20 = c.rolling(20).mean()
        if pd.isna(ma20.iloc[-1]) or pd.isna(ma20.iloc[-4]):
            return None
        above = bool(c.iloc[-1] > ma20.iloc[-1])
        rising = bool(ma20.iloc[-1] > ma20.iloc[-4])
        return {"name": name, "above_ma20": above, "rising": rising}
    except Exception:
        return None


def earnings_imminent(ticker: str):
    """실적 발표일이 3거래일 이내인지. 조회 실패 시 None(보정 생략)."""
    try:
        df = yf.Ticker(ticker).get_earnings_dates(limit=8)
        if df is None or df.empty:
            return None
        now = pd.Timestamp.now(tz=df.index.tz) if df.index.tz else pd.Timestamp.now()
        future = df.index[df.index >= now]
        if len(future) == 0:
            return False
        return bool((future.min() - now).days <= 5)  # 3거래일 ≈ 달력 5일
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 뉴스 헤드라인 (참고용 — 단순 키워드 매칭이라 점수 산정에는 반영하지 않는다)
# ---------------------------------------------------------------------------

_NEWS_POS_WORDS = [
    "surge", "surges", "beat", "beats", "record", "upgrade", "upgraded", "raises",
    "raised", "growth", "profit", "soar", "soars", "jump", "jumps", "rally", "rallies",
    "outperform", "expands", "wins", "win", "approval", "approved", "partnership",
    "buyback", "dividend hike", "strong demand", "탄력", "호조", "급등", "상향",
    "실적 호조", "신고가", "수주", "특허", "승인", "협력", "자사주", "배당", "흑자전환",
]
_NEWS_NEG_WORDS = [
    "plunge", "plunges", "miss", "misses", "cut", "cuts", "downgrade", "downgraded",
    "lawsuit", "sues", "sued", "probe", "recall", "recalls", "loss", "losses", "warns",
    "warning", "decline", "declines", "drop", "drops", "falls", "investigation",
    "fraud", "fine", "fined", "delay", "delays", "layoff", "layoffs", "급락", "하향",
    "적자", "리콜", "소송", "조사", "벌금", "부진", "경고", "감원", "구조조정", "제재",
]


def _tag_headline(title: str) -> str:
    t = title.lower()
    pos = any(w in t for w in _NEWS_POS_WORDS)
    neg = any(w in t for w in _NEWS_NEG_WORDS)
    if pos and not neg:
        return "호재"
    if neg and not pos:
        return "악재"
    return "중립"


def _is_korean(text: str) -> bool:
    return any("가" <= ch <= "힣" for ch in text)


def _translate_to_ko(text: str) -> str:
    """구글 번역 비공식 엔드포인트(API 키 불필요)로 영문 제목을 한국어로 옮긴다.
    비공식 엔드포인트라 언제든 실패할 수 있으므로, 실패 시 원문 제목을 그대로 반환해
    응답 자체가 끊기지 않게 한다."""
    if not text or _is_korean(text):
        return text
    try:
        resp = requests.get(
            "https://translate.googleapis.com/translate_a/single",
            params={"client": "gtx", "sl": "auto", "tl": "ko", "dt": "t", "q": text},
            timeout=3,
        )
        data = resp.json()
        return "".join(seg[0] for seg in data[0])
    except Exception:
        return text


def _mentions(text_l: str, symbol: str) -> bool:
    """텍스트에 티커 심볼이 '단어'로서 등장하는지 확인한다(예: 'F' 같은 짧은 심볼이
    아무 단어에나 부분일치하는 오탐을 막기 위해 단어 경계로 매칭)."""
    if len(symbol) < 2:
        return False
    return re.search(rf"\b{re.escape(symbol.lower())}\b", text_l) is not None


def _fetch_news_kr(ticker: str, limit: int) -> list:
    """국내(.KS/.KQ) 종목은 야후보다 네이버 금융의 종목별 뉴스 탭을 쓴다. 네이버가
    종목코드로 직접 태깅해 배포하는 뉴스라 관련성이 훨씬 높고, 원문이 이미 한국어라
    번역이 필요 없다. 야후의 국내 종목 뉴스는 거의 비어 있거나 무관한 경우가 많아
    아예 이 경로로 대체한다."""
    from bs4 import BeautifulSoup

    code = ticker.split(".")[0]
    resp = requests.get(
        "https://finance.naver.com/item/news_news.naver",
        params={"code": code, "page": 1, "sm": "title_entity_id.basic"},
        headers={
            "User-Agent": "Mozilla/5.0",
            "Referer": f"https://finance.naver.com/item/main.naver?code={code}",
        },
        timeout=5,
    )
    resp.encoding = resp.apparent_encoding
    soup = BeautifulSoup(resp.text, "html.parser")

    out = []
    for row in soup.select("table.type5 tr"):
        a = row.select_one("td.title a")
        if not a:
            continue
        title = a.get_text(strip=True)
        if not title:
            continue
        info_td = row.select_one("td.info")
        publisher = info_td.get_text(strip=True) if info_td else ""
        out.append({"title": title, "publisher": publisher, "tag": _tag_headline(title)})
        if len(out) >= limit:
            break
    return out


def _fetch_news_yahoo(ticker: str, name: str, limit: int) -> list:
    """야후 파이낸스 최신 뉴스 헤드라인을 가져와 (1) 실제로 이 종목과 관련된 기사만 추리고
    (2) 제목에 담긴 키워드로 호재/악재/중립을 태깅한 뒤 (3) 한국어로 번역해 반환한다.
    관련성 판정: 야후가 명시한 관련 티커에 포함되거나, 제목·요약에 티커 심볼(단어 단위)이나
    회사명이 실제로 등장하는 기사만 채택한다 — 이걸로도 걸러지지 않으면 그냥 제외한다
    (억지로 채워서 무관한 기사를 보여주지 않는다)."""
    items = yf.Ticker(ticker).news or []

    base_symbol = ticker.split(".")[0]
    name_l = (name or "").strip().lower()

    out = []
    for it in items:
        content = it.get("content") if isinstance(it.get("content"), dict) else it
        title = (content or {}).get("title") or it.get("title")
        summary = (content or {}).get("summary") or it.get("summary") or ""
        if not title:
            continue

        related = it.get("relatedTickers") or (content or {}).get("relatedTickers") or []
        related_syms = {str(r).upper().split(".")[0] for r in related} if isinstance(related, list) else set()
        text_l = f"{title} {summary}".lower()

        is_related = (
            base_symbol.upper() in related_syms
            or _mentions(text_l, base_symbol)
            or (name_l and name_l in text_l)
        )
        if not is_related:
            continue

        provider = (content or {}).get("provider") if isinstance((content or {}).get("provider"), dict) else None
        publisher = (provider or {}).get("displayName") or it.get("publisher") or ""
        tag = _tag_headline(title)  # 태깅은 번역 전 원문 기준(키워드 목록이 영/한 혼합이라 원문이 더 안정적)
        out.append({"title": _translate_to_ko(title.strip()), "publisher": publisher, "tag": tag})
        if len(out) >= limit:
            break
    return out


def fetch_news(ticker: str, name: str = "", limit: int = 3) -> list:
    """국내 종목(.KS/.KQ)은 네이버 금융, 그 외는 야후 파이낸스에서 뉴스를 가져온다.
    태깅은 기사 본문을 읽고 판단하는 게 아니라 단순 단어 매칭이라 반어법·복합 맥락은
    반영하지 못하는 참고용 정보이며, 매수/매도 점수 산정에는 포함하지 않는다.
    조회 실패 시(네트워크 오류, 페이지 구조 변경 등) 전체 응답이 죽지 않도록 빈 리스트를
    반환한다."""
    try:
        if ticker.upper().endswith((".KS", ".KQ")):
            return _fetch_news_kr(ticker, limit)
        return _fetch_news_yahoo(ticker, name, limit)
    except Exception:
        return []


# ---------------------------------------------------------------------------
# 판정 사다리
# ---------------------------------------------------------------------------

# 기존엔 35~50점이 "관심"으로 통째로 묶여, 40점대 초중반처럼 꽤 뚜렷한 강세도
# 근소한 관심 신호와 구분이 안 됐다. 20/35/50/70 네 경계로 5단계씩 세분화한다.
BUY_LADDER = ["적극 매수", "매수", "비중확대 검토(매수 관심)", "관심(매수 워치)", "관망"]
SELL_LADDER = ["전량 매도", "분할 매도", "비중축소 검토(매도 관심)", "관심(매도 워치)", "관망"]
EMOJI = {"적극 매수": "🟢🟢", "매수": "🟢", "비중확대 검토(매수 관심)": "🟢🟡", "관심(매수 워치)": "🟡",
         "전량 매도": "🔴🔴", "분할 매도": "🔴", "비중축소 검토(매도 관심)": "🔴🟡", "관심(매도 워치)": "🟡",
         "혼조(신호 상충, 관망)": "🟡", "관망": "⚪"}


def _downgrade(verdict: str) -> str:
    for ladder in (BUY_LADDER, SELL_LADDER):
        if verdict in ladder:
            i = ladder.index(verdict)
            return ladder[min(i + 1, len(ladder) - 1)]
    return verdict


def classify(buy: float, sell: float) -> str:
    if buy >= 50 and sell >= 50:
        return "혼조(신호 상충, 관망)"
    if buy >= 70:
        return "적극 매수"
    if buy >= 50:
        return "매수"
    if sell >= 70:
        return "전량 매도"
    if sell >= 50:
        return "분할 매도"
    if buy >= 35:
        return "비중확대 검토(매수 관심)"
    if sell >= 35:
        return "비중축소 검토(매도 관심)"
    if buy >= 20:
        return "관심(매수 워치)"
    if sell >= 20:
        return "관심(매도 워치)"
    return "관망"


# ---------------------------------------------------------------------------
# 종합 분석
# ---------------------------------------------------------------------------

def analyze_ticker(ticker: str, name: str) -> dict:
    data = fetch_ohlcv(ticker)
    o, h, l, c, v = data["Open"], data["High"], data["Low"], data["Close"], data["Volume"]
    ref_date = data.index[-1].strftime("%Y-%m-%d")
    price = float(c.iloc[-1])

    results = {
        "RSI(14)": score_rsi(c),
        "스토캐스틱": score_stochastic(h, l, c),
        "볼린저 %B": score_bollinger(c),
        "이동평균선": score_ma(c),
        "MACD": score_macd(c),
        "거래량/OBV": score_volume(c, v),
        "일목균형표": score_ichimoku(h, l, c),
        "CCI(14)": score_cci(h, l, c),
        "ATR(14)": score_atr(h, l, c),
        "캔들 패턴": score_candle(o, h, l, c),
        "지지·저항": score_support_resistance(h, l, c),
    }
    missing = [k for k, r in results.items() if not r["available"]]
    confidence = round((TOTAL_INDICATORS - len(missing)) / TOTAL_INDICATORS * 100)

    if len(missing) >= 4:
        return {"name": name, "ref_date": ref_date, "price": round(price, 2),
                "insufficient_data": True, "missing": missing,
                "confidence": confidence, "results": results, "order": INDICATOR_ORDER}

    corrections = []

    # ① ADX 장세 필터
    adx_s, pdi_s, mdi_s = calc_adx_di(h, l, c)
    adx = float(adx_s.iloc[-1]) if not pd.isna(adx_s.iloc[-1]) else None
    pdi = float(pdi_s.iloc[-1]) if not pd.isna(pdi_s.iloc[-1]) else None
    mdi = float(mdi_s.iloc[-1]) if not pd.isna(mdi_s.iloc[-1]) else None

    if adx is not None and adx < 20:
        regime, mr, tr = "횡보장", 1.4, 0.6
    elif adx is not None and adx > 25:
        regime, mr, tr = "추세장", 0.6, 1.4
    else:
        regime, mr, tr = "중립(20~25)", 1.0, 1.0

    for label, r in results.items():
        m = mr if label in MEAN_REV_GROUP else (tr if label in TREND_GROUP else 1.0)
        r["buy"] = round(r["buy"] * m, 1)
        r["sell"] = round(r["sell"] * m, 1)

    buy = round(sum(r["buy"] for r in results.values()), 1)
    sell = round(sum(r["sell"] for r in results.values()), 1)
    weight_desc = (f"평균회귀형(RSI·스토캐스틱·볼린저·CCI) x{mr}, 추세형(이평선·MACD·일목) x{tr}, 나머지 x1.0")
    corrections.append(f"①장세 가중치 적용 → 매수 {buy:g} / 매도 {sell:g}")

    di_dir = "중립"
    if pdi is not None and mdi is not None:
        di_dir = "+DI 우위(상승)" if pdi > mdi else "-DI 우위(하락)"
        if adx is not None and adx > 25 and mdi > pdi:
            buy = round(buy * 0.8, 1)
            corrections.append(f"①-1 추세장 하락추세(-DI>+DI) → 매수 x0.8 = {buy:g}")

    # ② 주봉 다중 시간프레임
    wk = weekly_confirmation(data)
    if wk is None:
        tf_txt = "주봉 데이터 확보 실패 → 보정 생략"
        corrections.append("②주봉 확인 불가로 보정 생략")
    else:
        daily_side = "상승" if buy > sell else ("하락" if sell > buy else "중립")
        if wk["verdict"] == "중립" or daily_side == "중립":
            tf_txt = f"일봉 {daily_side} / 주봉 {wk['verdict']} → 판단 보류(보정 없음)"
            corrections.append("②주봉 방향 중립 → 보정 없음")
        elif wk["verdict"] == daily_side:
            if daily_side == "상승":
                buy = round(buy * 1.15, 1)
            else:
                sell = round(sell * 1.15, 1)
            tf_txt = f"일봉 {daily_side} / 주봉 {wk['verdict']} → 일치"
            corrections.append(f"②주봉 일치 → 해당 점수 x1.15 (매수 {buy:g} / 매도 {sell:g})")
        else:
            if daily_side == "상승":
                buy = round(buy * 0.85, 1)
            else:
                sell = round(sell * 0.85, 1)
            tf_txt = f"일봉 {daily_side} / 주봉 {wk['verdict']} → 상충"
            corrections.append(f"②주봉 상충 → 해당 점수 x0.85 (매수 {buy:g} / 매도 {sell:g})")

    # ③ 시장지수 필터 — "추세를 거스르지 마라"는 원칙은 반대 방향을 깎는 것뿐 아니라
    # 같은 방향을 소폭 밀어주는 것도 포함한다(비중은 절반으로, 과도한 가산은 피함).
    idx = market_index_state(ticker)
    if idx is None:
        corrections.append("③시장지수 조회 실패 → 보정 생략")
    elif not idx["above_ma20"] and not idx["rising"]:
        buy = round(buy - 8, 1)
        sell = round(sell + 4, 1)
        corrections.append(f"③{idx['name']} 20일선 아래+하락 → 매수 -8 / 매도 +4 = 매수 {buy:g} / 매도 {sell:g}")
    elif idx["above_ma20"] and idx["rising"]:
        sell = round(sell - 8, 1)
        buy = round(buy + 4, 1)
        corrections.append(f"③{idx['name']} 20일선 위+상승 → 매도 -8 / 매수 +4 = 매수 {buy:g} / 매도 {sell:g}")
    else:
        corrections.append(f"③{idx['name']} 중립 → 보정 없음")

    # ④ 120일선 하락추세 방어 (매수만)
    ma120 = results["이동평균선"].get("ma120")
    if ma120:
        gap = (price - ma120) / ma120 * 100
        if gap < 0:
            penalty = 5 if gap >= -5 else (10 if gap >= -10 else 15)
            buy = round(buy - penalty, 1)
            corrections.append(f"④120일선 이격 {gap:+.1f}% → 매수 -{penalty} = {max(0, buy):g}")
        else:
            corrections.append(f"④120일선 이격 {gap:+.1f}% (위) → 보정 없음")
    else:
        corrections.append("④120일선 데이터 부족 → 보정 생략")

    buy, sell = round(max(0, buy), 1), round(max(0, sell), 1)

    # ⑤ 실적 발표 임박
    earn = earnings_imminent(ticker)
    if earn is True:
        buy, sell = round(buy * 0.8, 1), round(sell * 0.8, 1)
        corrections.append(f"⑤실적 발표 3거래일 이내 → 양쪽 x0.8 (매수 {buy:g} / 매도 {sell:g})")
    elif earn is False:
        corrections.append("⑤실적 발표 임박 아님 → 보정 없음")
    else:
        corrections.append("⑤실적 일정 조회 실패 → 보정 생략")

    verdict = classify(buy, sell)

    # ⑥ RSI 게이트 — RSI 50~70대는 상승 추세의 정상적인 강세 구간이므로 그 자체로는
    # 매수 판정을 깎지 않는다. 진짜 "추격 매수/매도 자제" 신호로 보는 경우만 강등한다:
    # 극단적 과매수/과매도(80 이상 / 20 이하), 또는 70대·30대이면서 반대 방향 다이버전스가
    # 동반돼 고점·저점 소진 정황이 함께 확인될 때만 적용한다.
    rsi_val = results["RSI(14)"].get("raw_rsi")
    rsi_bear_div = results["RSI(14)"].get("bear_div", False)
    rsi_bull_div = results["RSI(14)"].get("bull_div", False)
    if rsi_val is not None:
        buy_gate = rsi_val >= 80 or (rsi_val >= 70 and rsi_bear_div)
        sell_gate = rsi_val <= 20 or (rsi_val <= 30 and rsi_bull_div)
        if buy_gate and verdict in BUY_LADDER[:-1]:
            new = _downgrade(verdict)
            tag = "극단적 과매수" if rsi_val >= 80 else "과매수+하락다이버전스"
            corrections.append(f"⑥RSI {rsi_val:.1f} {tag} → 매수 판정 '{verdict}'→'{new}'")
            verdict = new
        elif sell_gate and verdict in SELL_LADDER[:-1]:
            new = _downgrade(verdict)
            tag = "극단적 과매도" if rsi_val <= 20 else "과매도+상승다이버전스"
            corrections.append(f"⑥RSI {rsi_val:.1f} {tag} → 매도 판정 '{verdict}'→'{new}'")
            verdict = new
        else:
            corrections.append(f"⑥RSI {rsi_val:.1f} → 게이트 미발동")

    # ⑦ 신뢰도 보정 — 11개 지표가 모두 확보된 경우(우량주 대다수)엔 매번 "보정 없음"만
    # 찍혀 형식적인 줄이 되므로, 실제로 지표가 빠져 신뢰도가 깎였을 때만 기록한다.
    low_conf = confidence < 70
    if low_conf and verdict != "관망":
        new = _downgrade(verdict)
        corrections.append(f"⑦데이터 신뢰도 {confidence}%<70 → '{verdict}'→'{new}'")
        verdict = new
    elif missing:
        corrections.append(f"⑦데이터 신뢰도 {confidence}% ({', '.join(missing)} 없음, 70% 이상이라 보정 없음)")

    emoji = EMOJI.get(verdict, "⚪")

    # 상충 신호
    conflicts = []
    if buy >= 50 and sell >= 50:
        conflicts.append("매수·매도 점수가 모두 50 이상으로 방향이 정리되지 않았습니다")
    mom = sum(results[k]["buy"] - results[k]["sell"] for k in MEAN_REV_GROUP if results[k]["available"])
    trd = sum(results[k]["buy"] - results[k]["sell"] for k in TREND_GROUP if results[k]["available"])
    if mom > 0 > trd:
        conflicts.append("모멘텀 지표는 매수, 추세 지표는 매도를 가리켜 서로 엇갈립니다")
    elif trd > 0 > mom:
        conflicts.append("추세 지표는 매수, 모멘텀 지표는 매도를 가리켜 서로 엇갈립니다")
    if wk is not None and "상충" in tf_txt:
        conflicts.append("일봉과 주봉 방향이 반대입니다")
    if not conflicts:
        conflicts.append("지표 간 뚜렷한 상충 없이 같은 방향을 가리킵니다")

    # 관찰 포인트
    watch = []
    ma_r = results["이동평균선"]
    if ma_r["available"] and "임박" in ma_r["detail"]:
        watch.append("5·20일선 크로스가 실제로 확정되면 이평선 점수가 7점으로 확대됩니다")
    if results["볼린저 %B"].get("squeeze"):
        watch.append("볼린저 스퀴즈 해소 방향(상단·하단 중 어디를 뚫는지)이 다음 신호가 됩니다")
    if rsi_val is not None and 45 < rsi_val < 55:
        watch.append(f"RSI가 현재 {rsi_val:.1f}로 게이트 경계(45/55) 근처이며, 이탈 시 판정 단계가 바뀝니다")
    if adx is not None and 20 <= adx <= 25:
        watch.append(f"ADX {adx:.1f}가 20/25선을 넘으면 장세 가중치가 바뀌어 점수가 재배분됩니다")
    if not watch:
        watch.append("현재 임계값 근처의 지표가 없어 단기간 내 판정 변화 요인은 제한적입니다")

    return {
        "name": name, "ref_date": ref_date, "price": round(price, 2),
        "insufficient_data": False, "results": results, "order": INDICATOR_ORDER,
        "confidence": confidence, "missing": missing,
        "adx": adx, "plus_di": pdi, "minus_di": mdi, "di_dir": di_dir,
        "regime": regime, "weight_desc": weight_desc,
        "timeframe": tf_txt, "corrections": corrections, "conflicts": conflicts, "watch": watch,
        "buy_score": buy, "sell_score": sell, "verdict": verdict, "emoji": emoji,
        "low_confidence": low_conf,
        # 하위 호환 (rsi_alert.py의 RSI 30/25 알림)
        "rsi": round(rsi_val, 2) if rsi_val is not None else None,
    }
