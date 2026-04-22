# ARIA Engine

> **A**gentic **R**easoning and **I**nformation **A**ccess

범용 AI 추론 엔진. LLM 종속 없이 자율적으로 사고하고 검색하고 판단하는 시스템.

## Architecture

```
ARIA Engine
├── Intent Analyzer     → 다층 의도 분석 (Multi-Level Intent Parsing)
├── Knowledge Router    → 검색 소스 자동 선택 (Vector DB / Web / Tools)
├── ReAct Agent Loop    → Think → Act → Observe → Reflect 반복
├── Provider Abstraction → LiteLLM (Claude/GPT/Gemini/DeepSeek/Local)
└── Vector Store        → Qdrant + FastEmbed (로컬 임베딩)
```

## Quick Start

```bash
# 1. Qdrant 시작
docker compose up -d

# 2. Python 가상환경 세팅
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# 3. 환경변수 설정
cp .env.example .env
# .env 파일에서 API 키 입력

# 4. 서버 시작
python run.py

# 5. 테스트
pytest tests/ -v
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/v1/health` | 헬스 체크 |
| POST | `/v1/query` | 에이전트에게 질문 |
| POST | `/v1/knowledge` | 지식 추가 |
| POST | `/v1/knowledge/{collection}/search` | 벡터 검색 |
| GET | `/v1/cost` | 비용 현황 |
| GET | `/v1/collections` | 컬렉션 목록 |

## Design Principles

1. **Provider Agnostic** - LLM/벡터DB 언제든 교체 가능
2. **Product Agnostic** - 어떤 제품이든 REST API로 연결
3. **Cost Aware** - 모든 호출의 비용 추적 + KillSwitch
4. **Solo Operator** - 1인 운영 가능한 복잡도
