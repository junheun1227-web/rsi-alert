"""
기술적 분석 스코어링 엔진 (사용자 지정 규칙 v3).

RSI(14)/볼린저밴드(20,2시그마)/이동평균(5,20,60)/MACD(12,26,9)/거래량/일목균형표(9,26,52)
6개 지표로 매수 점수·매도 점수를 각각 100점 만점으로 산출하고, ADX(14) 장세 필터·하락추세
방어·RSI 게이트 보정을 거쳐 최종 판정을 내린다.

각 지표는 표 표기용 `현재값`(value)과 2~3문장 분량의 `판정 근거`(reason)를 함께 반환한다.
지표별 표시 점수에는 장세(ADX) 가중치가 이미 반영돼 있어, 표의 점수를 그대로 더하면 소계가
나온다. 여기에 하락추세 방어 페널티를 적용한 값이 최종 총점이다.

데이터는 야후 파이낸스 실제 시세로 계산한다 (추정치·예측 없음). 각 지표는 계산에 필요한
최소 거래일 수가 확보되지 않으면 "데이터 없음"으로 표시하고 0점 처리하며, 6개 중 3개 이상이
데이터 없음이면 점수를 매기지 않고 데이터 부족을 알린다.

향후 주가 예측이나 목표가는 산출하지 않는다. 투자 판단의 참고용이며 투자 권유가 아니다.
"""

import numpy as np
import pandas as pd
import yfinance as yf

RSI_PERIOD = 14

NO_DATA = {"available": False, "buy": 0, "sell": 0, "value": "데이터 없음",
           "detail": "데이터 없음", "reason": "계산에 필요한 거래일 수가 부족해 산출하지 못했습니다."}


def _nodata() -> dict:
    return dict(NO_DATA)


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


def _recent_cross(fast: pd.Series, slow: pd.Series, lookback: int = 5):
    """최근 lookback영업일 이내 골든/데드 크로스 발생 여부와 발생 시점(며칠 전)."""
    golden, dead, days_ago = False, False, None
    n = len(fast)
    for i in range(max(1, n - lookback), n):
        if pd.isna(fast.iloc[i - 1]) or pd.isna(slow.iloc[i - 1]):
            continue
        if fast.iloc[i - 1] <= slow.iloc[i - 1] and fast.iloc[i] > slow.iloc[i]:
            golden, dead, days_ago = True, False, n - 1 - i
        if fast.iloc[i - 1] >= slow.iloc[i - 1] and fast.iloc[i] < slow.iloc[i]:
            dead, golden, days_ago = True, False, n - 1 - i
    return golden, dead, days_ago


def _cross_imminent(fast: pd.Series, slow: pd.Series, lookback: int = 5) -> bool:
    """두 선의 간격이 최근 lookback일 동안 계속 좁혀지고 있으면 '크로스 임박'으로 본다."""
    if len(fast) < lookback + 1:
        return False
    gap = (fast - slow).abs()
    recent = gap.iloc[-(lookback + 1):]
    if recent.isna().any():
        return False
    return bool(recent.iloc[-1] < recent.iloc[0] * 0.5)


# ---------------------------------------------------------------------------
# 지표별 채점
# 반환: {available, buy, sell, value(표 표기용), detail(한 줄 요약), reason(2~3문장 근거)}
# ---------------------------------------------------------------------------

def score_rsi(close: pd.Series) -> dict:
    if len(close) < RSI_PERIOD + 1:
        return _nodata()

    rsi_series = calc_rsi(close)
    rsi = float(rsi_series.iloc[-1])
    if pd.isna(rsi):
        return _nodata()

    buy = 20 if rsi <= 30 else (12 if rsi <= 40 else 0)
    sell = 20 if rsi >= 70 else (12 if rsi >= 60 else 0)

    bullish_div, bearish_div = detect_rsi_divergence(close, rsi_series)
    if bullish_div:
        buy += 5
    if bearish_div:
        sell += 5

    zone = "과매도(30 이하)" if rsi <= 30 else (
        "과매수(70 이상)" if rsi >= 70 else (
            "약세 중립(30~40)" if rsi <= 40 else (
                "강세 중립(60~70)" if rsi >= 60 else "중립(40~60)")))

    prev = rsi_series.iloc[-6] if len(rsi_series) >= 6 and not pd.isna(rsi_series.iloc[-6]) else None
    if prev is None:
        direction = "최근 방향성은 판단할 데이터가 부족합니다"
    elif rsi - prev > 2:
        direction = f"5일 전 {prev:.1f}에서 상승해 방향은 위로 향하고 있습니다"
    elif prev - rsi > 2:
        direction = f"5일 전 {prev:.1f}에서 하락해 방향은 아래로 향하고 있습니다"
        direction = f"5일 전 {prev:.1f}에서 하락해 방향은 아래로 향하고 있습니다"
    else:
        direction = f"5일 전 {prev:.1f} 대비 거의 변화가 없어 방향성은 뚜렷하지 않습니다"

    if bullish_div:
        div = "가격은 저점을 낮췄지만 RSI는 저점을 높인 상승 다이버전스가 관측되어 매수 근거를 더합니다"
    elif bearish_div:
        div = "가격은 고점을 높였지만 RSI는 고점을 낮춘 하락 다이버전스가 관측되어 매도 근거를 더합니다"
    else:
        div = "다이버전스는 관측되지 않습니다"

    if buy > 0:
        side = "따라서 RSI는 매수 근거로 작용합니다"
    elif sell > 0:
        side = "따라서 RSI는 매도 근거로 작용합니다"
    else:
        side = "중립 구간이라 RSI 단독으로는 매수·매도 어느 쪽 근거도 되지 않습니다"

    return {
        "available": True, "buy": buy, "sell": sell,
        "value": f"{rsi:.1f}",
        "detail": f"{rsi:.1f} ({zone})",
        "reason": f"현재 RSI는 {rsi:.1f}로 {zone} 구간입니다. {direction}. {div}. {side}.",
        "raw_rsi": rsi,
    }


def score_bollinger(close: pd.Series) -> dict:
    if len(close) < 20:
        return _nodata()

    mid = close.rolling(20).mean()
    std = close.rolling(20).std()
    upper = mid + 2 * std
    lower = mid - 2 * std

    c0 = close.iloc[-1]
    u0, l0, m0 = upper.iloc[-1], lower.iloc[-1], mid.iloc[-1]
    if pd.isna(u0) or pd.isna(l0) or u0 == l0:
        return _nodata()

    percent_b = (c0 - l0) / (u0 - l0)
    buy = 15 if percent_b <= 0 else (9 if percent_b <= 0.2 else 0)
    sell = 15 if percent_b >= 1 else (9 if percent_b >= 0.8 else 0)

    width = (u0 - l0) / m0 * 100 if m0 else 0
    width_series = (upper - lower) / mid * 100
    prev_width = width_series.iloc[-21] if len(width_series) >= 21 and not pd.isna(width_series.iloc[-21]) else None

    if percent_b >= 1:
        pos = "상단 밴드를 이탈한 상태로 단기 과열 신호입니다"
    elif percent_b >= 0.8:
        pos = "상단 밴드에 근접해 있어 과열 구간에 들어섰습니다"
    elif percent_b <= 0:
        pos = "하단 밴드를 이탈한 상태로 단기 과매도 신호입니다"
    elif percent_b <= 0.2:
        pos = "하단 밴드에 근접해 있어 과매도 구간에 들어섰습니다"
    else:
        pos = "밴드 중앙부에 위치해 이탈 신호는 없습니다"

    if prev_width is None:
        trend = f"밴드폭은 {width:.1f}%이며 추이를 비교할 데이터는 부족합니다"
    elif width > prev_width * 1.15:
        trend = f"밴드폭은 {width:.1f}%로 20일 전 {prev_width:.1f}% 대비 확대되어 변동성이 커지는 중입니다"
    elif width < prev_width * 0.85:
        trend = f"밴드폭은 {width:.1f}%로 20일 전 {prev_width:.1f}% 대비 축소(스퀴즈)되어 변동성이 줄어드는 중입니다"
    else:
        trend = f"밴드폭은 {width:.1f}%로 20일 전 {prev_width:.1f}%와 큰 차이가 없습니다"

    side = ("이 위치는 매수 근거입니다" if buy > 0
            else ("이 위치는 매도 근거입니다" if sell > 0
                  else "매수·매도 어느 쪽 근거도 되지 않습니다"))

    return {
        "available": True, "buy": buy, "sell": sell,
        "value": f"{percent_b:.2f}",
        "detail": f"%B={percent_b:.2f} (밴드폭 {width:.1f}%)",
        "reason": f"%B는 {percent_b:.2f}로 {pos}. {trend}. {side}.",
    }


def score_ma(close: pd.Series) -> dict:
    if len(close) < 61:
        return _nodata()

    ma5 = close.rolling(5).mean()
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()
    if pd.isna(ma60.iloc[-1]):
        return _nodata()

    c0 = close.iloc[-1]
    ma5_0, ma20_0, ma60_0 = ma5.iloc[-1], ma20.iloc[-1], ma60.iloc[-1]

    aligned_up = ma5_0 > ma20_0 > ma60_0
    aligned_down = ma5_0 < ma20_0 < ma60_0
    golden, dead, days_ago = _recent_cross(ma5, ma20, lookback=5)
    imminent = _cross_imminent(ma5, ma20, lookback=5)

    buy = (10 if aligned_up else 0) + (5 if c0 > ma20_0 else 0) + (5 if golden else 0)
    sell = (10 if aligned_down else 0) + (5 if c0 < ma20_0 else 0) + (5 if dead else 0)

    state = "정배열" if aligned_up else ("역배열" if aligned_down else "혼조")
    disparity = (c0 - ma20_0) / ma20_0 * 100 if ma20_0 else 0

    if state == "정배열":
        state_txt = "5일선>20일선>60일선의 정배열로 중기 상승 구조입니다"
    elif state == "역배열":
        state_txt = "5일선<20일선<60일선의 역배열로 중기 하락 구조입니다"
    else:
        state_txt = "정배열도 역배열도 아닌 혼조 구조로 방향성이 정리되지 않았습니다"

    if golden:
        cross_txt = f"{days_ago}영업일 전 5일선이 20일선을 상향 돌파하는 골든크로스가 발생했습니다"
    elif dead:
        cross_txt = f"{days_ago}영업일 전 5일선이 20일선을 하향 돌파하는 데드크로스가 발생했습니다"
    elif imminent:
        cross_txt = "5일선과 20일선의 간격이 빠르게 좁혀지고 있어 크로스가 임박한 상태입니다"
    else:
        cross_txt = "최근 5영업일 내 5일선·20일선 간 크로스는 없었습니다"

    side = ("종합적으로 매수 근거가 우세합니다" if buy > sell
            else ("종합적으로 매도 근거가 우세합니다" if sell > buy else "매수·매도 근거가 팽팽합니다"))

    return {
        "available": True, "buy": buy, "sell": sell,
        "value": state,
        "detail": f"5/20/60일선 {ma5_0:.1f}/{ma20_0:.1f}/{ma60_0:.1f} ({state}, 이격도 {disparity:+.1f}%)",
        "reason": f"{state_txt}. {cross_txt}. 종가는 20일선 대비 이격도 {disparity:+.1f}%입니다. {side}.",
    }


def score_macd(close: pd.Series) -> dict:
    if len(close) < 36:
        return _nodata()

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal

    m0, s0 = macd.iloc[-1], signal.iloc[-1]
    h0, h5 = hist.iloc[-1], hist.iloc[-6] if len(hist) >= 6 else hist.iloc[0]
    golden, dead, days_ago = _recent_cross(macd, signal, lookback=5)

    buy = (10 if m0 > s0 else 0) + (5 if m0 > 0 else 0) + (5 if golden else 0)
    sell = (10 if m0 < s0 else 0) + (5 if m0 < 0 else 0) + (5 if dead else 0)

    zero_pos = "0선 위로 중기 모멘텀은 상승 쪽" if m0 > 0 else "0선 아래로 중기 모멘텀은 하락 쪽"
    rel = "MACD가 시그널선 위에 있어 단기 모멘텀은 개선 방향" if m0 > s0 else "MACD가 시그널선 아래에 있어 단기 모멘텀은 둔화 방향"

    if golden:
        cross_txt = f"{days_ago}영업일 전 시그널선 상향 돌파가 발생했습니다"
    elif dead:
        cross_txt = f"{days_ago}영업일 전 시그널선 하향 돌파가 발생했습니다"
    elif abs(h0) < abs(h5) * 0.5:
        cross_txt = "히스토그램이 빠르게 축소되고 있어 교차가 임박한 상태입니다"
    else:
        cross_txt = "최근 5영업일 내 시그널선 교차는 없었습니다"

    conflict = ""
    if (m0 > s0 and m0 < 0) or (m0 < s0 and m0 > 0):
        conflict = " 다만 0선 위치와 시그널선 관계가 서로 다른 방향을 가리켜 신호가 상충합니다."

    return {
        "available": True, "buy": buy, "sell": sell,
        "value": f"{m0:.2f}",
        "detail": f"MACD {m0:.2f} / Signal {s0:.2f} (히스토그램 {h0:+.2f})",
        "reason": f"MACD {m0:.2f}, 시그널 {s0:.2f}로 {zero_pos}이며, {rel}입니다. {cross_txt}.{conflict}",
    }


def score_volume(close: pd.Series, volume: pd.Series) -> dict:
    if len(close) < 21:
        return _nodata()

    avg20 = volume.shift(1).rolling(20).mean()
    v0, avg0 = volume.iloc[-1], avg20.iloc[-1]
    if pd.isna(avg0) or avg0 == 0:
        return _nodata()

    ratio = v0 / avg0
    up_day = close.iloc[-1] > close.iloc[-2]
    down_day = close.iloc[-1] < close.iloc[-2]

    buy = 10 if (up_day and ratio >= 1.5) else (6 if (up_day and ratio >= 1.2) else 0)
    sell = 10 if (down_day and ratio >= 1.5) else (6 if (up_day and ratio < 0.8) else 0)

    direction = "양봉(전일 대비 상승)" if up_day else ("음봉(전일 대비 하락)" if down_day else "보합")
    level = ("평균을 크게 웃도는 대량 거래" if ratio >= 1.5
             else ("평균을 다소 웃도는 수준" if ratio >= 1.2
                   else ("평균에 못 미치는 한산한 거래" if ratio < 0.8 else "평균 수준")))

    if up_day and ratio >= 1.2:
        match = "주가 상승과 거래량 증가가 같은 방향으로 일치해 상승에 힘이 실린 모습이며 매수 근거입니다"
    elif up_day and ratio < 0.8:
        match = "주가는 올랐지만 거래량이 따라주지 않아 상승 동력이 약하다는 불일치 신호이며 매도 근거로 봅니다"
    elif down_day and ratio >= 1.5:
        match = "하락에 대량 거래가 동반돼 매도 압력이 실질적이라는 뜻이며 매도 근거입니다"
    else:
        match = "주가 방향과 거래량 사이에 뚜렷한 신호가 없어 어느 쪽 근거도 되지 않습니다"

    return {
        "available": True, "buy": buy, "sell": sell,
        "value": f"{ratio * 100:.0f}%",
        "detail": f"20일 평균 대비 {ratio * 100:.0f}% ({direction})",
        "reason": f"당일 거래량은 20일 평균의 {ratio:.2f}배로 {level}입니다. 당일 캔들은 {direction}입니다. {match}.",
    }


def score_ichimoku(high: pd.Series, low: pd.Series, close: pd.Series) -> dict:
    if len(close) < 79:
        return _nodata()

    conv = (high.rolling(9).max() + low.rolling(9).min()) / 2
    base = (high.rolling(26).max() + low.rolling(26).min()) / 2
    span_a = ((conv + base) / 2).shift(26)
    span_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)

    c0 = close.iloc[-1]
    a0, b0 = span_a.iloc[-1], span_b.iloc[-1]
    if pd.isna(a0) or pd.isna(b0):
        return _nodata()

    top0, bot0 = max(a0, b0), min(a0, b0)
    conv0, base0 = conv.iloc[-1], base.iloc[-1]
    above, below = c0 > top0, c0 < bot0

    chikou_prev = close.iloc[-27]
    chikou_buy = c0 > chikou_prev
    chikou_sell = c0 < chikou_prev

    buy = (6 if above else 0) + (5 if conv0 > base0 else 0) + (4 if chikou_buy else 0)
    sell = (6 if below else 0) + (5 if conv0 < base0 else 0) + (4 if chikou_sell else 0)

    position = "구름 위" if above else ("구름 아래" if below else "구름 안")
    thickness = abs(a0 - b0) / c0 * 100 if c0 else 0

    if above:
        pos_txt = "주가가 구름대 위에 있어 중기 추세는 상승 우위이며 구름 상단이 지지선 역할을 합니다"
    elif below:
        pos_txt = "주가가 구름대 아래에 있어 중기 추세는 하락 우위이며 구름 하단이 저항선 역할을 합니다"
        pos_txt = "주가가 구름대 아래에 있어 중기 추세는 하락 우위이며 구름 하단이 저항선 역할을 합니다"
    else:
        pos_txt = "주가가 구름대 안에 있어 방향성이 정리되지 않은 관망 구간입니다"

    cb_txt = ("전환선이 기준선 위에 있어 단기 흐름은 우호적입니다" if conv0 > base0
              else ("전환선이 기준선 아래에 있어 단기 흐름은 비우호적입니다" if conv0 < base0
                    else "전환선과 기준선이 겹쳐 단기 방향성이 없습니다"))
    ch_txt = ("후행스팬이 26일 전 주가보다 위에 있어 매수 근거를 더합니다" if chikou_buy
              else ("후행스팬이 26일 전 주가보다 아래에 있어 매도 근거를 더합니다" if chikou_sell
                    else "후행스팬은 26일 전 주가와 같은 수준입니다"))
    thick_txt = (f"구름 두께는 주가 대비 {thickness:.1f}%로 두꺼워 돌파 시 저항·지지가 강합니다"
                 if thickness >= 5 else
                 f"구름 두께는 주가 대비 {thickness:.1f}%로 얇아 돌파가 비교적 쉬운 상태입니다")

    return {
        "available": True, "buy": buy, "sell": sell,
        "value": position,
        "detail": f"{position}, 전환선 {conv0:.1f}/기준선 {base0:.1f}, 구름두께 {thickness:.1f}%",
        "reason": f"{pos_txt}. {cb_txt}. {ch_txt}. {thick_txt}.",
    }


# ---------------------------------------------------------------------------
# 종합 판단
# ---------------------------------------------------------------------------

INDICATOR_ORDER = ["RSI(14)", "볼린저밴드 %B", "이동평균선", "MACD", "거래량", "일목균형표"]


def analyze_ticker(ticker: str, name: str) -> dict:
    data = fetch_ohlcv(ticker)
    close, high, low, volume = data["Close"], data["High"], data["Low"], data["Volume"]
    ref_date = data.index[-1].strftime("%Y-%m-%d")
    price = float(close.iloc[-1])

    results = {
        "RSI(14)": score_rsi(close),
        "볼린저밴드 %B": score_bollinger(close),
        "이동평균선": score_ma(close),
        "MACD": score_macd(close),
        "거래량": score_volume(close, volume),
        "일목균형표": score_ichimoku(high, low, close),
    }
    unavailable = [k for k, v in results.items() if not v["available"]]

    if len(unavailable) >= 3:
        return {
            "name": name, "ref_date": ref_date, "price": round(price, 2),
            "insufficient_data": True, "missing": unavailable, "results": results,
        }

    # --- ADX(14) 장세 필터 ---
    adx_series = calc_adx(high, low, close) if len(close) >= 30 else None
    adx_value = (float(adx_series.iloc[-1])
                 if adx_series is not None and not pd.isna(adx_series.iloc[-1]) else None)

    if adx_value is not None and adx_value < 20:
        regime, mean_rev_mult, trend_mult = "횡보장", 1.5, 0.5
    elif adx_value is not None and adx_value > 25:
        regime, mean_rev_mult, trend_mult = "추세장", 0.5, 1.5
    else:
        regime, mean_rev_mult, trend_mult = "중립", 1.0, 1.0

    # 표시 점수 = 가중치 적용 후 점수 (표의 점수를 더하면 소계가 나오도록)
    mult_map = {"RSI(14)": mean_rev_mult, "볼린저밴드 %B": mean_rev_mult,
                "이동평균선": trend_mult, "MACD": trend_mult,
                "거래량": 1.0, "일목균형표": 1.0}
    for label, r in results.items():
        m = mult_map[label]
        r["buy"] = round(r["buy"] * m, 1)
        r["sell"] = round(r["sell"] * m, 1)

    buy_subtotal = round(sum(r["buy"] for r in results.values()), 1)
    sell_subtotal = round(sum(r["sell"] for r in results.values()), 1)

    weight_desc = (f"평균회귀형(RSI·볼린저) x{mean_rev_mult}, 추세형(이평선·MACD) x{trend_mult}, "
                   f"거래량·일목균형표 x1.0")

    buy_total, sell_total = buy_subtotal, sell_subtotal
    corrections = []

    # --- 하락추세 방어: 종가가 120일선 대비 -3% 이하이면 매수 -10 ---
    ma120 = close.rolling(120).mean()
    ma120_0 = ma120.iloc[-1] if len(close) >= 120 else None
    if ma120_0 is not None and not pd.isna(ma120_0) and ma120_0 > 0:
        gap120 = (price - ma120_0) / ma120_0 * 100
        if gap120 <= -3:
            buy_total -= 10
            corrections.append(
                f"하락추세 방어 적용(종가가 120일선 대비 {gap120:+.1f}%) → 매수 {buy_subtotal:.1f} - 10 = {max(0, buy_total):.1f}점")

    buy_total = round(max(0, buy_total), 1)
    sell_total = round(max(0, sell_total), 1)

    # --- 판정 ---
    if buy_total >= 70:
        verdict = "적극 매수"
    elif buy_total >= 55:
        verdict = "매수"
    elif sell_total >= 70:
        verdict = "전량 매도"
    elif sell_total >= 55:
        verdict = "분할 매도"
    else:
        verdict = "관망"

    # --- RSI 게이트: 과매수(RSI>=70)에서는 매수 판정 무효화 ---
    rsi_value = results["RSI(14)"].get("raw_rsi")
    gate_applied = False
    if rsi_value is not None and rsi_value >= 70 and verdict in ("적극 매수", "매수"):
        corrections.append(
            f"RSI 게이트 발동(RSI {rsi_value:.1f} ≥ 70 과매수) → '{verdict}'에서 '관망'으로 다운그레이드")
        verdict = "관망"
        gate_applied = True

    if not corrections:
        corrections.append("추가 보정 없음 (하락추세 방어·RSI 게이트 모두 미적용)")

    emoji = {"적극 매수": "🟢", "매수": "🟢", "전량 매도": "🔴", "분할 매도": "🔴", "관망": "🟡"}[verdict]

    # --- 상충 신호 ---
    conflicts = []
    if buy_total >= 35 and sell_total >= 35:
        conflicts.append("매수·매도 점수가 모두 높게 나와 방향이 한쪽으로 정리되지 않았습니다")
    ma_r, rsi_r, macd_r, ichi_r = results["이동평균선"], results["RSI(14)"], results["MACD"], results["일목균형표"]
    if rsi_r["available"] and ma_r["available"]:
        if rsi_r["buy"] > 0 and ma_r["sell"] > 0:
            conflicts.append("RSI는 과매도 반등(매수)을, 이동평균선은 하락 구조(매도)를 가리켜 서로 엇갈립니다")
        elif rsi_r["sell"] > 0 and ma_r["buy"] > 0:
            conflicts.append("RSI는 과열(매도)을, 이동평균선은 상승 구조(매수)를 가리켜 서로 엇갈립니다")
    if macd_r["available"] and ichi_r["available"] and macd_r["buy"] > 0 and ichi_r["sell"] > 0:
        conflicts.append("MACD는 단기 개선을, 일목균형표는 중기 하락 우위를 나타내 시간축 간 신호가 다릅니다")
    if not conflicts:
        conflicts.append("지표 간 뚜렷한 상충 없이 대체로 같은 방향을 가리키고 있습니다")

    return {
        "name": name,
        "ref_date": ref_date,
        "price": round(price, 2),
        "insufficient_data": False,
        "results": results,
        "order": INDICATOR_ORDER,
        "adx": adx_value,
        "regime": regime,
        "weight_desc": weight_desc,
        "buy_subtotal": buy_subtotal,
        "sell_subtotal": sell_subtotal,
        "corrections": corrections,
        "conflicts": conflicts,
        "buy_score": buy_total,
        "sell_score": sell_total,
        "verdict": verdict,
        "emoji": emoji,
        "gate_applied": gate_applied,
        # 하위 호환(rsi_alert.py의 RSI 30/25 push 알림 로직이 참조)
        "rsi": round(rsi_value, 2) if rsi_value is not None else None,
        "adjustments": [f"ADX {adx_value:.1f} ({regime})" if adx_value is not None else "ADX 계산 불가"] + corrections,
    }
