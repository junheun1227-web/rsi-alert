# 매그니피센트7 RSI 카카오워크 알림 봇

RSI(14)가 30 이하 / 25 이하로 떨어지면 카카오워크로 알림을 보내는 자동화입니다.
GitHub Actions에서 15분마다 실행되며, 같은 날 같은 종목·같은 임계값에 대해서는 중복 알림을 보내지 않습니다.

대상 종목: AAPL, MSFT, GOOGL, AMZN, META, NVDA, TSLA

## 왜 GitHub Actions인가

카카오 쪽으로 메시지를 "전송(POST)"하는 요청은 일반적인 클라우드/개인 서버 환경에서만 가능합니다.
GitHub Actions는 무료로 완전한 인터넷 접근이 가능한 실행 환경을 제공하므로, 이 워크플로가 스케줄에 따라
주가를 조회하고 카카오워크로 메시지를 보내는 역할을 합니다.

## 설정 순서

### 1. GitHub 저장소 만들기

- github.com에서 새 저장소 생성 (Public 권장 — Public 저장소는 Actions 실행 시간이 무제한 무료입니다.
  Private로 하면 무료 계정 기준 월 2,000분 한도가 있어 15분 간격 스케줄은 한도를 넘길 수 있습니다.)
- 이 폴더의 4개 파일을 그대로 저장소에 업로드:
  - `rsi_alert.py`
  - `requirements.txt`
  - `.github/workflows/rsi_alert.yml`
  - `README.md` (선택)

  업로드는 GitHub 웹의 "Add file → Upload files"로 드래그 앤 드롭하면 됩니다. `.github/workflows/` 폴더 구조는
  그대로 유지되어야 합니다.

### 2. 카카오워크 봇 App Key 발급받기

1. 카카오워크 관리자 페이지(admin.kakaowork.com)에 회사/조직 계정으로 로그인
   - 아직 카카오워크 워크스페이스가 없다면 무료로 새로 만들 수 있습니다 (소규모 팀 무료 플랜 있음)
2. `앱 관리` → `봇 앱 만들기`에서 새 봇 생성
3. 생성된 봇의 **App Key**를 복사해둡니다
4. 알림을 받을 본인 계정의 **카카오워크 로그인 이메일**을 확인해둡니다 (이 이메일로 1:1 대화방을 자동으로 엽니다)

> 참고: 기존에 만들어두신 Kakao Developers 앱(developers.kakao.com)은 카카오워크와는 별개의 서비스라
> 이번 알림에는 사용되지 않습니다. 카카오워크 관리자 페이지에서 별도로 봇을 만들어야 합니다.

### 3. GitHub 저장소에 Secret 등록

저장소의 `Settings → Secrets and variables → Actions → New repository secret`에서 2개 등록:

| Secret 이름 | 값 |
|---|---|
| `KAKAOWORK_APP_KEY` | 2번에서 발급받은 App Key |
| `KAKAOWORK_EMAIL` | 알림 받을 카카오워크 계정 이메일 |

### 4. 동작 확인

- 저장소의 `Actions` 탭 → `RSI Alert (Magnificent 7)` 워크플로 선택 → `Run workflow`로 수동 실행
- 처음 실행 시 카카오워크 대화방 ID를 자동으로 조회해 `rsi_alert_state.json`에 캐시하고, 이후 실행부터는 재조회하지 않습니다
- 실행 로그에서 종목별 RSI 값과 알림 발송 여부를 확인할 수 있습니다
- 정상 동작 확인 후에는 `on.schedule`의 cron이 15분마다 자동으로 실행합니다

### 5. 주기 / 임계값 조정

- 체크 주기: `.github/workflows/rsi_alert.yml`의 `cron: "*/15 * * * *"` 값 수정 (예: 30분마다 → `*/30 * * * *`)
- 임계값 · 종목: `rsi_alert.py` 상단의 `TICKERS`, `THRESHOLDS` 값 수정

## 로컬 테스트

카카오워크 연결만 확인하고 싶다면:

```bash
pip install -r requirements.txt
export KAKAOWORK_APP_KEY=xxx
export KAKAOWORK_EMAIL=you@example.com
python rsi_alert.py --test
```

RSI 값만 보고 싶다면 (알림 전송 없이):

```bash
python rsi_alert.py --dry-run
```
