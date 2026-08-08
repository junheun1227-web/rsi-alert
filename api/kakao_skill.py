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
            if symbol and q.get("quoteType") in (None, "EQUITY"):
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
    """6개 지표 종합 매수/매도 분석을 사용자가 지정한 출력 형식으로 정리.
    (카카오톡 simpleText는 마크다운을 렌더링하지 않으므로 **/### 기호 대신 일반 텍스트로 구성)"""
    verdict = info["verdict"]  # "매수" / "매도" / "관망"
    use_buy = verdict != "매도"  # 매도 판단일 때만 매도점수 기준 근거를 보여줌

    lines = [
        f"[{info['name']}({ticker})]",
        f"현재 판단: {info['emoji']} {info['label']}",
        f"매수점수: {info['buy_score']}/60   매도점수: {info['sell_score']}/60",
        "",
        "근거",
    ]
    for item in info["items"]:
        score = item["buy"] if use_buy else item["sell"]
        lines.append(f"- {item['name']}: {item['detail']} → {score}점")

    key_item = max(info["items"], key=lambda it: it["buy"] if use_buy else it["sell"])
    lines += [
        "",
        "핵심 판단",
        f"→ {key_item['name']} 지표({key_item['detail']})가 가장 강한 근거이며, "
        f"종합적으로 {info['label']} 의견입니다.",
    ]
    if info.get("buy_blocked"):
        lines.append("※ 매수 보류 조건 충족: 120일선 아래 + MACD 약세 + 일목구름 아래 동시 충족")

    currency = "₩" if ticker.upper().endswith((".KS", ".KQ")) else "$"
    price_str = f"{currency}{info['price']:,.2f}" if currency == "$" else f"{currency}{info['price']:,.0f}"
    lines += [
        "",
        f"현재가: {price_str}  |  기준시각: {updated_at}",
        "※ 투자 권유가 아닌 정의된 기술적 분석 모델 기반 참고 정보입니다.",
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
