"""
카카오 i 오픈빌더 스킬 서버 (Vercel Python 서버리스 함수)

카카오톡 채널 챗봇에서 "AAPL RSI", "테슬라 RSI 알려줘" 같은 질문이 오면
GitHub Actions가 주기적으로 갱신해둔 latest_rsi.json 캐시를 읽어서 즉시 답변합니다.
매번 야후 파이낸스를 직접 호출하지 않기 때문에 응답이 빠르고(수백ms), 카카오의 응답
제한시간(약 5초)에 걸릴 걱정이 없습니다.

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
import urllib.request

LATEST_RSI_URL = os.environ.get(
    "LATEST_RSI_URL",
    "https://raw.githubusercontent.com/junheun1227-web/rsi-alert/main/latest_rsi.json",
)

# 종목명/티커 별칭 매칭 (한글 질문도 인식하도록)
ALIASES = {
    "AAPL": ["aapl", "애플", "apple", "아이폰"],
    "MSFT": ["msft", "마이크로소프트", "microsoft", " ms "],
    "GOOGL": ["googl", "구글", "google", "알파벳", "alphabet"],
    "AMZN": ["amzn", "아마존", "amazon"],
    "META": ["meta", "메타", "페이스북", "facebook"],
    "NVDA": ["nvda", "엔비디아", "nvidia"],
    "TSLA": ["tsla", "테슬라", "tesla"],
}


def find_ticker(utterance: str):
    text = f" {utterance.lower()} "
    for ticker, keywords in ALIASES.items():
        for kw in keywords:
            if kw in text:
                return ticker
    return None


def fetch_latest_rsi() -> dict:
    req = urllib.request.Request(
        LATEST_RSI_URL + f"?_={os.urandom(4).hex()}",  # 캐시 우회용 더미 쿼리
        headers={"Cache-Control": "no-cache"},
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        return json.loads(resp.read().decode("utf-8"))


def build_kakao_response(text: str) -> dict:
    return {
        "version": "2.0",
        "template": {"outputs": [{"simpleText": {"text": text}}]},
    }


def handle_utterance(utterance: str) -> str:
    ticker = find_ticker(utterance)
    if not ticker:
        available = ", ".join(ALIASES.keys())
        return (
            "어떤 종목인지 못 찾았어요. 예) 'AAPL RSI', '테슬라 RSI 알려줘'\n"
            f"지원 종목: {available}"
        )

    try:
        data = fetch_latest_rsi()
    except Exception as e:
        return f"RSI 데이터를 불러오지 못했어요 ({e}). 잠시 후 다시 시도해주세요."

    info = data.get("tickers", {}).get(ticker)
    if not info:
        return f"{ticker} 데이터가 아직 준비되지 않았어요. 잠시 후 다시 시도해주세요."

    updated_at = data.get("updated_at", "알 수 없음")
    rsi = info["rsi"]
    price = info["price"]
    name = info["name"]

    status = ""
    if rsi <= 25:
        status = " → 25 이하: 과매도 심화"
    elif rsi <= 30:
        status = " → 30 이하: 과매도"

    return (
        f"{name}({ticker})\n"
        f"RSI(14) = {rsi}{status}\n"
        f"현재가: ${price:,.2f}\n"
        f"기준시각: {updated_at} (최대 30분 이내 캐시값)"
    )


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
