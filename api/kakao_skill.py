"""
카카오 i 오픈빌더 스킬 서버 (Vercel Python 서버리스 함수)

두 가지 경로로 응답합니다.
1) 매그니피센트7(AAPL/MSFT/GOOGL/AMZN/META/NVDA/TSLA): GitHub Actions가 15분마다 갱신해둔
   latest_rsi.json 캐시를 읽어 즉시 답변 (빠름, 수백ms, 카카오 응답 제한시간 걱정 없음)
2) 그 외 임의의 한국/미국 종목: 야후 파이낸스 검색 API로 종목명 -> 티커를 실시간으로 찾은 뒤
   analysis.py로 즉석에서 6개 지표를 계산해 답변 (느림, 2~5초, 드물게 카카오 응답 제한시간
   초과로 실패할 수 있음 - 이 경우 사용자가 다시 물어보면 됨)

배포: 이 저장소를 Vercel에 연결하면 이 파일이 자동으로
  https://<프로젝트>.vercel.app/api/kakao_skill
경로의 엔드포인트가 됩니다. 이 URL을 카카오 i 오픈빌더의 스킬 URL로 등록하세요.

환경변수(Vercel 프로젝트 설정 > Environment Variables):
  LATEST_RSI_URL : latest_rsi.json의 raw GitHub URL
                   예) https://raw.githubusercontent.com/{계정}/{저장소}/main/latest_rsi.json
"""

from http.server import BaseHTTPRequestHandler
import json
import os
import sys
import urllib.request
import urllib.parse
from datetime import datetime

# analysis.py는 저장소 루트에 있음(GitHub Actions 스크립트와 공유). Vercel이 api/ 를
# 진입점으로 실행할 때도 import 되도록 저장소 루트를 sys.path에 명시적으로 추가한다.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import analysis

LATEST_RSI_URL = os.environ.get(
    "LATEST_RSI_URL",
    "https://raw.githubusercontent.com/junheun1227-web/rsi-alert/main/latest_rsi.json",
)
KR_STOCKS_URL = os.environ.get(
    "KR_STOCKS_URL",
    "https://raw.githubusercontent.com/junheun1227-web/rsi-alert/main/kr_stocks.json",
)

# 매그니피센트7 종목명/티커 별칭 매칭 (캐시 경로, 빠름)
ALIASES = {
    "AAPL": ["aapl", "애플", "apple", "아이폰"],
    "MSFT": ["msft", "마이크로소프트", "microsoft", " ms "],
    "GOOGL": ["googl", "구글", "google", "알파벳", "alphabet"],
    "AMZN": ["amzn", "아마존", "amazon"],
    "META": ["meta", "메타", "페이스북", "facebook"],
    "NVDA": ["nvda", "엔비디아", "nvidia"],
    "TSLA": ["tsla", "테슬라", "tesla"],
}

# 국내 대형주(코스피/코스닥) 직접 매핑 - 야후 검색 API에 의존하지 않고 바로 실시간 조회하기 위함
# (야후 검색 API가 한글 질의를 잘 못 찾거나 클라우드 IP에서 막히는 경우가 있어 보강용으로 추가)
KR_STOCKS = {
    "005930.KS": ("삼성전자", ["삼성전자", "samsung electronics", "005930"]),
    "000660.KS": ("SK하이닉스", ["sk하이닉스", "하이닉스", "000660"]),
    "035420.KS": ("NAVER", ["네이버", "naver", "035420"]),
    "035720.KS": ("카카오", ["카카오", "kakao", "035720"]),
    "005380.KS": ("현대차", ["현대차", "현대자동차", "005380"]),
    "000270.KS": ("기아", ["기아", "기아차", "000270"]),
    "373220.KS": ("LG에너지솔루션", ["lg에너지솔루션", "엘지에너지솔루션", "373220"]),
    "207940.KS": ("삼성바이오로직스", ["삼성바이오로직스", "삼성바이오", "207940"]),
    "006400.KS": ("삼성SDI", ["삼성sdi", "006400"]),
    "051910.KS": ("LG화학", ["lg화학", "엘지화학", "051910"]),
    "068270.KS": ("셀트리온", ["셀트리온", "068270"]),
    "005490.KS": ("POSCO홀딩스", ["포스코홀딩스", "포스코", "posco", "005490"]),
    "012330.KS": ("현대모비스", ["현대모비스", "012330"]),
    "066570.KS": ("LG전자", ["lg전자", "엘지전자", "066570"]),
    "096770.KS": ("SK이노베이션", ["sk이노베이션", "096770"]),
    "323410.KS": ("카카오뱅크", ["카카오뱅크", "323410"]),
    "018260.KS": ("삼성에스디에스", ["삼성에스디에스", "samsung sds", "018260"]),
    "105560.KS": ("KB금융", ["kb금융", "국민은행", "105560"]),
    "055550.KS": ("신한지주", ["신한지주", "신한은행", "055550"]),
    "015760.KS": ("한국전력", ["한국전력", "한전", "015760"]),
    "010130.KS": ("고려아연", ["고려아연", "010130"]),
    "011200.KS": ("HMM", ["hmm", "011200"]),
    "042700.KS": ("한미반도체", ["한미반도체", "042700"]),
    "247540.KQ": ("에코프로비엠", ["에코프로비엠", "247540"]),
    "086520.KQ": ("에코프로", ["에코프로", "086520"]),
    "196170.KQ": ("알테오젠", ["알테오젠", "196170"]),
}

# 질문 문장에서 종목명만 추출하기 위해 제거할 잡단어
FILLER_WORDS = [
    "어떄", "어때", "어떠니", "알려줘", "알려주세요", "분석해줘", "분석",
    "매수해도", "매도해도", "매수", "매도", "해도", "될까요", "될까", "가능할까",
    "가능해", "지금", "현재", "좀", "혹시", "궁금해", "궁금", "rsi", "RSI",
    "종목", "주식", "?", "!", ".", ",",
]


def find_ticker(utterance: str):
    text = f" {utterance.lower()} "
    for ticker, keywords in ALIASES.items():
        for kw in keywords:
            if kw in text:
                return ticker
    return None


def find_kr_ticker(query: str):
    text = f" {query.lower()} "
    for ticker, (name, keywords) in KR_STOCKS.items():
        for kw in keywords:
            if kw in text:
                return ticker, name
    return None, None


def fetch_kr_stocks() -> dict:
    """GitHub Actions가 KRX(코스피+코스닥) 전체 상장사 명단을 받아 저장해둔 kr_stocks.json 캐시.
    회사명(정확히 일치) -> 티커코드(.KS/.KQ) 매핑, 약 2,000~2,600개 종목을 포함한다."""
    req = urllib.request.Request(
        KR_STOCKS_URL + f"?_={os.urandom(4).hex()}",
        headers={"Cache-Control": "no-cache"},
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        return json.loads(resp.read().decode("utf-8"))


def find_kr_ticker_full(query: str):
    """KRX 전체 상장사 명단에서 회사명으로 검색 (정확 일치 우선, 안 되면 부분 일치)."""
    try:
        data = fetch_kr_stocks()
    except Exception:
        return None, None

    stocks = data.get("stocks", {})
    q = query.strip()
    if not q:
        return None, None

    # 1) 정확히 일치
    if q in stocks:
        return stocks[q], q

    # 2) 공백 제거 후 정확히 일치 (예: "국도 화학" -> "국도화학")
    q_nospace = q.replace(" ", "")
    for name, ticker in stocks.items():
        if name.replace(" ", "") == q_nospace:
            return ticker, name

    # 3) 부분 일치 (질문에 회사명이 포함되거나, 회사명에 질문이 포함되는 경우)
    candidates = [
        (name, ticker) for name, ticker in stocks.items()
        if name in q or (len(q_nospace) >= 2 and q_nospace in name.replace(" ", ""))
    ]
    if candidates:
        # 가장 이름이 긴(더 구체적인) 매칭을 우선
        name, ticker = max(candidates, key=lambda x: len(x[0]))
        return ticker, name

    return None, None


def resolve_us_ticker_direct(query: str):
    """사용자가 QQQ, TQQQ, AAPL처럼 정확한 미국 티커(알파벳 1~5자, ETF 포함)를 입력한 경우
    야후 검색 API를 거치지 않고 바로 존재 여부만 확인해 즉시 매칭한다.
    (검색 API가 한글/일부 환경에서 불안정한 것과 무관하게 ETF·개별종목 티커를 안정적으로 잡기 위함)"""
    q = query.strip().upper()
    if not (1 <= len(q) <= 5 and q.isalpha()):
        return None, None
    if analysis.ticker_exists(q):
        return q, q
    return None, None


def resolve_kr_code(query: str):
    """6자리 숫자 종목코드를 입력한 경우 코스피(.KS)/코스닥(.KQ) 여부를 직접 확인."""
    digits = query.strip()
    if not (digits.isdigit() and len(digits) == 6):
        return None, None
    for suffix in (".KS", ".KQ"):
        ticker = digits + suffix
        if analysis.ticker_exists(ticker):
            return ticker, digits
    return None, None


def extract_query(utterance: str) -> str:
    text = utterance
    for w in FILLER_WORDS:
        text = text.replace(w, " ")
    return " ".join(text.split()).strip()


def fetch_latest_rsi() -> dict:
    req = urllib.request.Request(
        LATEST_RSI_URL + f"?_={os.urandom(4).hex()}",  # 캐시 우회용 더미 쿼리
        headers={"Cache-Control": "no-cache"},
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        return json.loads(resp.read().decode("utf-8"))


def resolve_ticker(query: str):
    """야후 파이낸스 검색 API로 종목명 -> (티커, 종목명) 실시간 해석.
    한국(KRX, .KS/.KQ)·미국 등 야후에 상장된 대부분 종목을 지원한다."""
    if not query:
        return None, None
    try:
        url = (
            "https://query2.finance.yahoo.com/v1/finance/search"
            f"?q={urllib.parse.quote(query)}&quotesCount=5&newsCount=0"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=4) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        for q in data.get("quotes", []):
            symbol = q.get("symbol")
            # ETF(QQQ, TQQQ 등)도 검색되도록 EQUITY 외에 ETF 타입도 허용
            if symbol and q.get("quoteType") in (None, "EQUITY", "ETF"):
                name = q.get("shortname") or q.get("longname") or symbol
                return symbol, name
    except Exception:
        pass
    return None, None


def build_kakao_response(text: str) -> dict:
    return {
        "version": "2.0",
        "template": {"outputs": [{"simpleText": {"text": text}}]},
    }


def format_analysis(ticker: str, info: dict, updated_at: str) -> str:
    """6개 지표(RSI/볼린저/이평선/MACD/거래량/일목균형표) 종합 매수·매도 분석을
    사용자 지정 출력 형식(종목/기준일/지표별 근거표/종합판정)으로 정리.
    카카오톡 simpleText는 마크다운 표·굵게를 렌더링하지 않으므로 일반 텍스트로 정렬한다."""
    currency = "₩" if ticker.upper().endswith((".KS", ".KQ")) else "$"

    def fmt_price(p):
        return f"{currency}{p:,.0f}" if currency == "₩" else f"{currency}{p:,.2f}"

    if info.get("insufficient_data"):
        missing = ", ".join(info.get("missing", []))
        return (
            f"{info['name']}({ticker}) — 점수 산출 불가\n"
            f"기준일: {info.get('ref_date', '-')} 종가 | 종가: {fmt_price(info['price'])} | "
            f"데이터 신뢰도: {info.get('confidence', 0)}%\n\n"
            f"데이터 없음 지표(4개 이상): {missing}\n"
            f"위 지표값을 알려주시면 그 값으로 점수를 산출하겠습니다.\n"
            f"기준시각: {updated_at}"
        )

    results = info["results"]
    order = info.get("order", list(results.keys()))

    def R(label):
        return results.get(label, {"reason": "데이터 없음", "available": False})

    conf = info.get("confidence", 0)
    conf_tag = " (데이터 부족)" if info.get("low_confidence") else ""

    # --- 헤더 ---
    lines = [
        f"{info['name']}({ticker}) {info['emoji']} **{info['verdict']}**{conf_tag} | "
        f"매수 {info['buy_score']:g}점 / 매도 {info['sell_score']:g}점",
        f"기준일: {info.get('ref_date', '')} 종가 | 종가: {fmt_price(info['price'])} | 데이터 신뢰도: {conf}%",
        "",
        "## 지표 요약",
        "| 지표 | 현재값 | 매수 | 매도 |",
        "|---|---|---|---|",
    ]
    for label in order:
        r = results[label]
        lines.append(f"| {label} | {r['value']} | {r['buy']:g} | {r['sell']:g} |")

    # --- 판정 근거 (5개 그룹) ---
    lines += [
        "",
        "## 판정 근거",
        f"- **모멘텀(RSI·스토캐스틱·CCI)**: {R('RSI(14)')['reason']}. "
        f"{R('스토캐스틱')['reason']}. {R('CCI(14)')['reason']}.",
        f"- **추세(이평선·MACD·일목)**: {R('이동평균선')['reason']}. "
        f"{R('MACD')['reason']}. {R('일목균형표')['reason']}.",
        f"- **변동성(볼린저·ATR)**: {R('볼린저 %B')['reason']}. {R('ATR(14)')['reason']}.",
        f"- **수급(거래량·OBV)**: {R('거래량/OBV')['reason']}.",
        f"- **가격 구조(지지저항·캔들)**: {R('지지·저항')['reason']}. {R('캔들 패턴')['reason']}.",
    ]

    # --- 종합 ---
    adx = info.get("adx")
    adx_txt = f"{adx:.1f}" if adx is not None else "계산 불가"
    lines += [
        "",
        "## 종합",
        f"- 장세: ADX {adx_txt} / DI {info.get('di_dir', '-')} → {info.get('regime', '-')} → {info.get('weight_desc', '')}",
        f"- 시간프레임: {info.get('timeframe', '-')}",
        f"- 보정 내역: {' / '.join(info.get('corrections', []))}",
        f"- 상충 신호: {' '.join(info.get('conflicts', []))}",
        f"- 관찰 포인트: {' '.join(info.get('watch', []))}",
        "",
        f"기준시각: {updated_at}",
    ]
    return "\n".join(lines)


def handle_utterance(utterance: str) -> str:
    # 1) 매그니피센트7: 캐시 경로 (빠름)
    ticker = find_ticker(utterance)
    if ticker:
        try:
            data = fetch_latest_rsi()
        except Exception as e:
            return f"분석 데이터를 불러오지 못했어요 ({e}). 잠시 후 다시 시도해주세요."

        info = data.get("tickers", {}).get(ticker)
        if not info:
            return f"{ticker} 분석 데이터 부족: 아직 준비되지 않았어요. 잠시 후 다시 시도해주세요."

        updated_at = data.get("updated_at", "알 수 없음")
        return format_analysis(ticker, info, f"{updated_at} (최대 15분 캐시)")

    # 2) 그 외 임의 종목(한국/미국 등): 실시간 조회 (느릴 수 있음)
    query = extract_query(utterance)
    if not query:
        return "어떤 종목인지 못 찾았어요. 예) 'AAPL 어때', '삼성전자 매수 매도', '005930'"

    # 2-1) 국내 대형주 직접 매핑 (네트워크 호출 없이 즉시 매칭, 최다 조회 종목 위주)
    resolved_ticker, resolved_name = find_kr_ticker(query)
    # 2-2) 6자리 숫자 종목코드: 코스피/코스닥 직접 판별
    if not resolved_ticker:
        resolved_ticker, resolved_name = resolve_kr_code(query)
    # 2-3) 알파벳 1~5자 정확한 미국 티커(개별종목·ETF 모두 포함, 예: QQQ/TQQQ/AAPL) 직접 확인
    if not resolved_ticker:
        resolved_ticker, resolved_name = resolve_us_ticker_direct(query)
    # 2-4) KRX 전체 상장사 명단(코스피+코스닥 약 2,000개 이상)에서 회사명 검색
    if not resolved_ticker:
        resolved_ticker, resolved_name = find_kr_ticker_full(query)
    # 2-5) 그 외: 야후 파이낸스 검색 API로 종목명 -> 티커 해석 (ETF 포함 미국 등 그 외 종목)
    if not resolved_ticker:
        resolved_ticker, resolved_name = resolve_ticker(query)

    if not resolved_ticker:
        return f"'{query}' 종목을 찾지 못했어요. 정확한 종목명이나 티커로 다시 시도해주세요."

    try:
        info = analysis.analyze_ticker(resolved_ticker, resolved_name)
    except Exception as e:
        return f"'{resolved_name}'({resolved_ticker}) 분석 데이터 부족: {e}"

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return format_analysis(resolved_ticker, info, f"{now} (실시간 조회)")


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0) or 0)
            body = self.rfile.read(length) if length else b"{}"
            payload = json.loads(body or b"{}")
            utterance = payload.get("userRequest", {}).get("utterance", "")
            reply = handle_utterance(utterance)
        except Exception as e:
            reply = f"오류가 발생했어요: {e}"

        response = build_kakao_response(reply)
        out = json.dumps(response, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def do_GET(self):
        out = json.dumps(
            {"status": "ok", "message": "RSI Kakao skill server is running"},
            ensure_ascii=False,
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)
