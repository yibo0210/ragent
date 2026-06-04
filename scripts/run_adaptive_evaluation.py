"""Adaptive GraphRAG evaluation script.

Evaluates: query type classification accuracy + graph utility decision quality.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.agent.query_profiler import QueryProfiler
from backend.rag.retrieval_planner import get_retrieval_planner
from backend.rag.graph_utility_estimator import get_graph_utility_estimator

ADAPTIVE_BENCHMARK = [
    # --- factoid (6) ---
    {"question": "What is Redis?", "expected_query_type": "factoid", "expected_use_graph": False,
     "ground_truth": "Redis is an in-memory key-value data store used as a cache and message broker."},
    {"question": "What is Kafka used for?", "expected_query_type": "factoid", "expected_use_graph": False,
     "ground_truth": "Kafka is a distributed event streaming platform for high-throughput data pipelines."},
    {"question": "Define FastAPI.", "expected_query_type": "factoid", "expected_use_graph": False,
     "ground_truth": "FastAPI is a modern Python web framework for building APIs with automatic OpenAPI docs."},
    {"question": "Explain what a database index does.", "expected_query_type": "factoid", "expected_use_graph": False,
     "ground_truth": "A database index speeds up query performance by reducing the amount of data scanned."},
    {"question": "What is Python?", "expected_query_type": "factoid", "expected_use_graph": False,
     "ground_truth": "Python is a high-level interpreted programming language known for its readability."},
    {"question": "What does CPU stand for?", "expected_query_type": "factoid", "expected_use_graph": False,
     "ground_truth": "CPU stands for Central Processing Unit, the primary processor in a computer."},

    # --- entity_relation (4) ---
    {"question": "Who founded OpenAI?", "expected_query_type": "entity_relation", "expected_use_graph": True,
     "ground_truth": "OpenAI was founded by Sam Altman, Greg Brockman, Elon Musk and others."},
    {"question": "Which companies did Tencent invest in?", "expected_query_type": "entity_relation", "expected_use_graph": True,
     "ground_truth": "Tencent has investments in Epic Games, Riot Games, Tesla, Snap and many others."},
    {"question": "Who is the CEO of Microsoft?", "expected_query_type": "entity_relation", "expected_use_graph": True,
     "ground_truth": "Satya Nadella has been the CEO of Microsoft since 2014."},
    {"question": "What products does Apple sell?", "expected_query_type": "entity_relation", "expected_use_graph": True,
     "ground_truth": "Apple sells iPhone, iPad, Mac, Apple Watch, AirPods and services like iCloud and Apple Music."},

    # --- multi_hop (4) ---
    {"question": "Which company acquired the startup that developed Kubernetes?", "expected_query_type": "multi_hop",
     "expected_use_graph": True,
     "ground_truth": "Google acquired the startup that developed Kubernetes before donating it to CNCF."},
    {"question": "Trace the investment chain from SoftBank to ByteDance.", "expected_query_type": "multi_hop",
     "expected_use_graph": True,
     "ground_truth": "SoftBank invested in Alibaba which has partnerships in the same ecosystem as ByteDance."},
    {"question": "Find competitors of the company that partnered with our main supplier.", "expected_query_type": "multi_hop",
     "expected_use_graph": True,
     "ground_truth": "This requires multi-hop graph traversal to find competitors through supply chain relations."},
    {"question": "Which organizations collaborated with both Google and Microsoft?", "expected_query_type": "multi_hop",
     "expected_use_graph": True,
     "ground_truth": "Several organizations like OpenAI have partnerships with both Google and Microsoft."},

    # --- global_summary (3) ---
    {"question": "Summarize the key technologies used in this project.", "expected_query_type": "global_summary",
     "expected_use_graph": False, "expected_use_community": True,
     "ground_truth": "The project uses FastAPI, LangGraph, Milvus, Neo4j, MySQL, Redis and Qwen LLM."},
    {"question": "What are the major themes across all documents?", "expected_query_type": "global_summary",
     "expected_use_graph": False, "expected_use_community": True,
     "ground_truth": "Major themes include multi-agent orchestration, knowledge retrieval and enterprise deployment."},
    {"question": "Give me an overview of the system architecture.", "expected_query_type": "global_summary",
     "expected_use_graph": False, "expected_use_community": True,
     "ground_truth": "The system uses Supervisor-Workers pattern with 6 specialized agents for different retrieval tasks."},

    # --- temporal (3) ---
    {"question": "Who was CEO of Microsoft in 2018?", "expected_query_type": "temporal", "expected_use_graph": True,
     "ground_truth": "Satya Nadella was the CEO of Microsoft in 2018."},
    {"question": "What happened in Q3 2023?", "expected_query_type": "temporal", "expected_use_graph": True,
     "ground_truth": "Q3 2023 saw the rollout of incremental graph clustering pipeline and Redis Streams integration."},
    {"question": "Which technology stack was used before 2020?", "expected_query_type": "temporal", "expected_use_graph": True,
     "ground_truth": "Before 2020, the project may have used older tech like REST APIs without GraphRAG capabilities."},

    # --- comparison (3) ---
    {"question": "Compare GraphRAG and vanilla RAG.", "expected_query_type": "comparison", "expected_use_graph": True,
     "ground_truth": "GraphRAG uses knowledge graphs for relationship-aware retrieval while vanilla RAG relies on vector similarity."},
    {"question": "What are the differences between Redis and Memcached?", "expected_query_type": "comparison",
     "expected_use_graph": False,
     "ground_truth": "Redis supports more data types, persistence and clustering while Memcached is simpler and purely in-memory."},
    {"question": "Which is better: FastAPI or Flask?", "expected_query_type": "comparison", "expected_use_graph": False,
     "ground_truth": "FastAPI offers async support and auto-docs; Flask is simpler and has a larger ecosystem."},
]


def evaluate_query_classification():
    profiler = QueryProfiler(use_embedding=False)
    correct = 0
    total = len(ADAPTIVE_BENCHMARK)
    details = []

    for item in ADAPTIVE_BENCHMARK:
        intent = profiler.profile(item["question"])
        predicted = intent.query_type
        expected = item["expected_query_type"]
        details.append({
            "question": item["question"][:80],
            "expected": expected,
            "predicted": predicted,
            "match": predicted == expected,
        })
        if predicted == expected:
            correct += 1

    return {"accuracy": correct / total if total else 0, "correct": correct, "total": total, "details": details}


def evaluate_retrieval_plan():
    planner = get_retrieval_planner()
    correct_graph = 0
    total_graph = 0
    correct_community = 0
    total_community = 0

    for item in ADAPTIVE_BENCHMARK:
        intent = {"query_type": item["expected_query_type"]}
        plan = planner.plan(intent=intent)
        if "expected_use_graph" in item:
            total_graph += 1
            if plan.use_graph == item["expected_use_graph"]:
                correct_graph += 1
        if "expected_use_community" in item:
            total_community += 1
            if plan.use_community == item["expected_use_community"]:
                correct_community += 1

    return {
        "graph_accuracy": correct_graph / total_graph if total_graph else 0,
        "graph_correct": correct_graph, "graph_total": total_graph,
        "community_accuracy": correct_community / total_community if total_community else 0,
        "community_correct": correct_community, "community_total": total_community,
    }


def evaluate_graph_utility():
    estimator = get_graph_utility_estimator()
    correct = 0
    total = 0
    for item in ADAPTIVE_BENCHMARK:
        if "expected_use_graph" not in item:
            continue
        total += 1
        use = estimator.should_use_graph(item["question"], item["expected_query_type"])
        if use == item["expected_use_graph"]:
            correct += 1
    return {"accuracy": correct / total if total else 0, "correct": correct, "total": total}


if __name__ == "__main__":
    print("=" * 60)
    print("Adaptive GraphRAG Evaluation")
    print("=" * 60)

    r1 = evaluate_query_classification()
    print(f"\n1. Query Classification Accuracy: {r1['accuracy']:.1%} ({r1['correct']}/{r1['total']})")
    for d in r1["details"]:
        status = "OK" if d["match"] else f"FAIL (expected {d['expected']})"
        print(f"  {status:35s} predicted={d['predicted']:20s} | {d['question'][:55]}")

    r2 = evaluate_retrieval_plan()
    print(f"\n2. Retrieval Plan — Graph Decision: {r2['graph_accuracy']:.1%} ({r2['graph_correct']}/{r2['graph_total']})")
    if r2["community_total"] > 0:
        print(f"   Retrieval Plan — Community Decision: {r2['community_accuracy']:.1%} ({r2['community_correct']}/{r2['community_total']})")

    r3 = evaluate_graph_utility()
    print(f"\n3. Graph Utility Estimator: {r3['accuracy']:.1%} ({r3['correct']}/{r3['total']})")

    overall = (r1["accuracy"] + r2["graph_accuracy"] + r3["accuracy"]) / 3
    print(f"\n{'=' * 60}")
    print(f"Overall Score: {overall:.1%}")
    print(f"Classification: {r1['accuracy']:.1%} | Plan Graph: {r2['graph_accuracy']:.1%} | Utility Est: {r3['accuracy']:.1%}")
