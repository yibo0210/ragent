"""离线实体消歧脚本。"""
import sys
sys.path.insert(0, ".")
from backend.graph.entity_resolution import run_entity_resolution

if __name__ == "__main__":
    result = run_entity_resolution()
    print(f"候选对: {result['candidates_found']}")
    print(f"LLM 确认: {result['llm_confirmed']}")
    for r in result["merged_pairs"]:
        print(f"  合并: {r['merged_into']} ← {r['removed']}")
