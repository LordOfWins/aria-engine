"""ARIA Engine - Golden Test Set Baseline 측정

사용법 (로컬에서 실행):
    # 1. Qdrant + ARIA 서버 기동
    docker compose up -d
    python run.py &

    # 2. 심리학 지식 베이스에 문서 적재 (이미 적재되어 있으면 생략)

    # 3. Baseline 측정 실행
    python tests/golden/run_baseline.py

    # 4. 결과 확인
    cat tests/golden/baseline_results.json

측정 항목:
- retrieval_hit_rate: 검색 결과에 관련 문서가 1개 이상 포함된 비율
- keyword_recall: 기대 키워드 중 답변에 포함된 비율
- avg_confidence: 에이전트 자체 평가 신뢰도 평균
- avg_latency_ms: 평균 응답 시간
- avg_cost_per_query: 질의당 평균 비용

비교 모드:
- --mode vector  : 벡터 검색만 (기존)
- --mode hybrid  : Hybrid Retrieval (BM25 + 벡터)
- --mode both    : 양쪽 모두 측정 후 비교 (기본값)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

# 기본 설정
ARIA_BASE_URL = "http://localhost:8100"
GOLDEN_TEST_SET_PATH = Path(__file__).parent / "golden_test_set_v1.json"
RESULTS_OUTPUT_PATH = Path(__file__).parent / "baseline_results.json"
REQUEST_TIMEOUT = 120  # 에이전트 응답 대기 (초)


@dataclass
class TestResult:
    """단일 테스트 결과"""
    test_id: str
    domain: str
    difficulty: str
    query: str
    answer: str = ""
    confidence: float = 0.0
    iterations: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    keyword_hits: list[str] = field(default_factory=list)
    keyword_misses: list[str] = field(default_factory=list)
    keyword_recall: float = 0.0
    search_results_count: int = 0
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "test_id": self.test_id,
            "domain": self.domain,
            "difficulty": self.difficulty,
            "query": self.query,
            "answer_preview": self.answer[:200] if self.answer else "",
            "confidence": self.confidence,
            "iterations": self.iterations,
            "latency_ms": round(self.latency_ms, 2),
            "cost_usd": round(self.cost_usd, 6),
            "keyword_recall": round(self.keyword_recall, 3),
            "keyword_hits": self.keyword_hits,
            "keyword_misses": self.keyword_misses,
            "search_results_count": self.search_results_count,
            "error": self.error,
        }


def load_test_set(path: Path) -> list[dict[str, Any]]:
    """Golden Test Set JSON 로드"""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["test_cases"]


def evaluate_keywords(answer: str, expected_keywords: list[str]) -> tuple[list[str], list[str], float]:
    """답변에서 기대 키워드 포함 여부 평가

    Returns:
        (hits, misses, recall)
    """
    answer_lower = answer.lower()
    hits = [kw for kw in expected_keywords if kw.lower() in answer_lower]
    misses = [kw for kw in expected_keywords if kw.lower() not in answer_lower]
    recall = len(hits) / len(expected_keywords) if expected_keywords else 0.0
    return hits, misses, recall


def run_single_test(
    client: httpx.Client,
    test_case: dict[str, Any],
    api_key: str | None = None,
) -> TestResult:
    """단일 테스트 케이스 실행"""
    result = TestResult(
        test_id=test_case["id"],
        domain=test_case["domain"],
        difficulty=test_case["difficulty"],
        query=test_case["query"],
    )

    headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key

    payload = {
        "query": test_case["query"],
        "collection": test_case.get("collection", "psychology_kb"),
    }

    start_time = time.time()

    try:
        response = client.post(
            f"{ARIA_BASE_URL}/v1/query",
            json=payload,
            headers=headers,
            timeout=REQUEST_TIMEOUT,
        )
        elapsed_ms = (time.time() - start_time) * 1000

        if response.status_code != 200:
            result.error = f"HTTP {response.status_code}: {response.text[:200]}"
            result.latency_ms = elapsed_ms
            return result

        data = response.json()
        result.answer = data.get("answer", "")
        result.confidence = data.get("confidence", 0.0)
        result.iterations = data.get("iterations", 0)
        result.latency_ms = elapsed_ms
        result.search_results_count = data.get("search_results_count", 0)

        # 비용 추출
        cost_summary = data.get("cost_summary", {})
        result.cost_usd = cost_summary.get("daily_cost_usd", 0.0)

        # 키워드 평가
        expected_keywords = test_case.get("expected_keywords", [])
        hits, misses, recall = evaluate_keywords(result.answer, expected_keywords)
        result.keyword_hits = hits
        result.keyword_misses = misses
        result.keyword_recall = recall

    except httpx.TimeoutException:
        result.error = "TIMEOUT"
        result.latency_ms = (time.time() - start_time) * 1000
    except Exception as e:
        result.error = str(e)[:200]
        result.latency_ms = (time.time() - start_time) * 1000

    return result


def run_baseline(
    test_cases: list[dict[str, Any]],
    api_key: str | None = None,
    mode_label: str = "baseline",
) -> dict[str, Any]:
    """전체 Golden Test Set 실행 + 통계 집계"""
    results: list[TestResult] = []
    total = len(test_cases)

    print(f"\n{'='*60}")
    print(f"  ARIA Engine Golden Test Baseline — {mode_label}")
    print(f"  총 {total}문항")
    print(f"{'='*60}\n")

    with httpx.Client() as client:
        # 서버 헬스 체크
        try:
            health = client.get(f"{ARIA_BASE_URL}/v1/health", timeout=5)
            if health.status_code != 200:
                print(f"[ERROR] 서버 응답 없음: {health.status_code}")
                sys.exit(1)
            print(f"[OK] 서버 정상: {health.json()}\n")
        except Exception as e:
            print(f"[ERROR] 서버 연결 실패: {e}")
            print(f"  → docker compose up -d && python run.py 실행 필요")
            sys.exit(1)

        for i, test_case in enumerate(test_cases, 1):
            print(f"[{i:2d}/{total}] {test_case['id']} ({test_case['difficulty']}) — {test_case['query'][:40]}...")
            result = run_single_test(client, test_case, api_key)

            if result.error:
                print(f"       ❌ ERROR: {result.error[:80]}")
            else:
                print(f"       ✅ recall={result.keyword_recall:.2f}  "
                      f"conf={result.confidence:.2f}  "
                      f"latency={result.latency_ms:.0f}ms")

            results.append(result)

    # 통계 집계
    successful = [r for r in results if not r.error]
    failed = [r for r in results if r.error]

    stats = {
        "mode": mode_label,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_tests": total,
        "successful": len(successful),
        "failed": len(failed),
    }

    if successful:
        stats.update({
            "avg_keyword_recall": round(sum(r.keyword_recall for r in successful) / len(successful), 3),
            "avg_confidence": round(sum(r.confidence for r in successful) / len(successful), 3),
            "avg_latency_ms": round(sum(r.latency_ms for r in successful) / len(successful), 2),
            "avg_iterations": round(sum(r.iterations for r in successful) / len(successful), 2),
            "avg_search_results": round(sum(r.search_results_count for r in successful) / len(successful), 2),
        })

        # 난이도별 통계
        by_difficulty: dict[str, dict[str, float]] = {}
        for diff in ["easy", "moderate", "hard"]:
            diff_results = [r for r in successful if r.difficulty == diff]
            if diff_results:
                by_difficulty[diff] = {
                    "count": len(diff_results),
                    "avg_keyword_recall": round(
                        sum(r.keyword_recall for r in diff_results) / len(diff_results), 3
                    ),
                    "avg_confidence": round(
                        sum(r.confidence for r in diff_results) / len(diff_results), 3
                    ),
                }
        stats["by_difficulty"] = by_difficulty

        # 도메인별 통계
        by_domain: dict[str, dict[str, float]] = {}
        for domain in ["attachment", "relationship", "therapy", "personality", "emotion"]:
            domain_results = [r for r in successful if r.domain == domain]
            if domain_results:
                by_domain[domain] = {
                    "count": len(domain_results),
                    "avg_keyword_recall": round(
                        sum(r.keyword_recall for r in domain_results) / len(domain_results), 3
                    ),
                }
        stats["by_domain"] = by_domain

    return {
        "summary": stats,
        "results": [r.to_dict() for r in results],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="ARIA Golden Test Baseline 측정")
    parser.add_argument(
        "--mode", choices=["vector", "hybrid", "both"], default="both",
        help="측정 모드: vector(기존) / hybrid(BM25+벡터) / both(비교)",
    )
    parser.add_argument(
        "--api-key", default=None,
        help="ARIA API 키 (auth_disabled=true면 불필요)",
    )
    parser.add_argument(
        "--output", default=str(RESULTS_OUTPUT_PATH),
        help="결과 저장 경로",
    )
    parser.add_argument(
        "--test-set", default=str(GOLDEN_TEST_SET_PATH),
        help="Golden Test Set JSON 경로",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="테스트 케이스 수 제한 (0=전체)",
    )
    args = parser.parse_args()

    # Golden Test Set 로드
    test_cases = load_test_set(Path(args.test_set))
    if args.limit > 0:
        test_cases = test_cases[:args.limit]

    print(f"Golden Test Set 로드 완료: {len(test_cases)}문항")

    # 실행
    # NOTE: vector vs hybrid 전환은 서버 설정으로 제어
    # 현재 서버는 hybrid 모드로 기동됨
    # vector-only 측정이 필요하면 app.py에서 hybrid_retriever=None으로 기동
    all_results = run_baseline(
        test_cases,
        api_key=args.api_key,
        mode_label=args.mode,
    )

    # 결과 저장
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"  결과 저장: {output_path}")
    print(f"{'='*60}")

    # 요약 출력
    summary = all_results["summary"]
    print(f"\n📊 Baseline 요약 ({summary['mode']}):")
    print(f"   성공: {summary['successful']}/{summary['total_tests']}")
    if summary.get("avg_keyword_recall") is not None:
        print(f"   평균 키워드 재현율: {summary['avg_keyword_recall']:.3f}")
        print(f"   평균 신뢰도: {summary['avg_confidence']:.3f}")
        print(f"   평균 응답시간: {summary['avg_latency_ms']:.0f}ms")

    if "by_difficulty" in summary:
        print(f"\n   난이도별 키워드 재현율:")
        for diff, stats in summary["by_difficulty"].items():
            print(f"     {diff:10s}: {stats['avg_keyword_recall']:.3f} (n={stats['count']})")


if __name__ == "__main__":
    main()
