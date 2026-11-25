"""Manual End-to-End Testing Script for Phase 9.

Executes representative natural language queries across all 12 intent classes,
edge cases, safety boundaries, confirmation paths, and error scenarios.
Cleans up all temporary test artifacts upon completion.
"""

import sys
from pathlib import Path
import json

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import AITerminalAssistant
from core.command_generator import OperationType, RiskLevel, GenerationStatus
from core.command_validator import ValidationStatus
from core.executor import ExecutionStatus
import config


def run_manual_tests() -> None:
    print("=" * 80)
    print("AI TERMINAL ASSISTANT — PHASE 9 END-TO-END MANUAL TEST HARNESS")
    print("=" * 80)

    # Initialize assistant with real workspace
    assistant = AITerminalAssistant()
    ws = assistant.workspace_dir
    print(f"Workspace Directory: {ws}\n")

    test_results = []

    def log_test(name: str, query: str, op, val, exc, expected_status: str, notes: str = ""):
        passed = False
        actual_status = "NONE"
        if exc is not None:
            actual_status = exc.execution_status.value
            passed = (actual_status == expected_status)
        elif val is not None:
            actual_status = val.validation_status.value
            passed = (actual_status == expected_status)
        elif op is not None:
            actual_status = op.generation_status.value
            passed = (actual_status == expected_status)

        res = {
            "test_name": name,
            "query": query,
            "intent": op.intent if op else "N/A",
            "operation": op.operation_type.value if op else "N/A",
            "risk": op.risk_level.value if op else "N/A",
            "actual_status": actual_status,
            "expected_status": expected_status,
            "passed": passed,
            "notes": notes,
        }
        test_results.append(res)
        badge = "[PASS]" if passed else "[FAIL]"
        print(f"{badge} {name}: {query} -> Status: {actual_status} (Intent: {res['intent']})")

    # 1. CURRENT_DIRECTORY (LOW risk)
    op, val, exc = assistant.process_query("show current directory")
    log_test("1. Current Directory", "show current directory", op, val, exc, "SUCCESS")

    # 2. LIST_FILES (LOW risk)
    op, val, exc = assistant.process_query("list all files")
    log_test("2. List Files", "list all files", op, val, exc, "SUCCESS")

    # 3. CREATE_FILE (MEDIUM risk - approved)
    test_file = ws / "manual_test_doc.txt"
    if test_file.exists():
        test_file.unlink()
    op, val, exc = assistant.process_query("create file manual_test_doc.txt", approved=True)
    log_test("3. Create File (Approved)", "create file manual_test_doc.txt", op, val, exc, "SUCCESS", "File created")

    # 4. TARGET CONFLICT (CREATE_FILE on existing file)
    op, val, exc = assistant.process_query("create file manual_test_doc.txt", approved=True)
    log_test("4. Target Conflict", "create file manual_test_doc.txt", op, val, exc, "CONFLICT_TARGET_EXISTS", "Conflict detected")

    # 5. CREATE_DIRECTORY (MEDIUM risk - declined)
    declined_dir = ws / "declined_demo_folder"
    if declined_dir.exists():
        declined_dir.rmdir()
    op, val, exc = assistant.process_query("create directory declined_demo_folder", approved=False)
    log_test("5. Create Directory (Declined)", "create directory declined_demo_folder", op, val, exc, "CANCELLED", "User cancelled")

    # 6. CREATE_DIRECTORY (MEDIUM risk - approved)
    demo_dir = ws / "demo_folder"
    if demo_dir.exists():
        demo_dir.rmdir()
    op, val, exc = assistant.process_query("create directory demo_folder", approved=True)
    log_test("6. Create Directory (Approved)", "create directory demo_folder", op, val, exc, "SUCCESS", "Folder created")

    # 7. SEARCH_FILES (LOW risk)
    op, val, exc = assistant.process_query("search for txt files")
    log_test("7. Search Files", "search for txt files", op, val, exc, "SUCCESS")

    # 8. CHECK_VERSION (LOW risk - allowlisted tool)
    op, val, exc = assistant.process_query("check python version")
    log_test("8. Check Version (Python)", "check python version", op, val, exc, "SUCCESS")

    # 9. SYSTEM_INFO (LOW risk)
    op, val, exc = assistant.process_query("show system information")
    log_test("9. System Info", "show system information", op, val, exc, "SUCCESS")

    # 10. NETWORK_INFO (LOW risk)
    op, val, exc = assistant.process_query("show network configuration")
    log_test("10. Network Info", "show network configuration", op, val, exc, "SUCCESS")

    # 11. COPY_FILE (MEDIUM risk - approved)
    copy_dst = ws / "manual_test_doc_copy.txt"
    if copy_dst.exists():
        copy_dst.unlink()
    op, val, exc = assistant.process_query("copy manual_test_doc.txt to manual_test_doc_copy.txt", approved=True)
    log_test("11. Copy File", "copy manual_test_doc.txt to manual_test_doc_copy.txt", op, val, exc, "SUCCESS")

    # 12. MOVE_FILE (MEDIUM risk - approved)
    move_dst = ws / "manual_test_doc_moved.txt"
    if move_dst.exists():
        move_dst.unlink()
    op, val, exc = assistant.process_query("move manual_test_doc_copy.txt to manual_test_doc_moved.txt", approved=True)
    log_test("12. Move File", "move manual_test_doc_copy.txt to manual_test_doc_moved.txt", op, val, exc, "SUCCESS")

    # 13. DELETE_FILE with wrong token (HIGH risk - cancelled)
    op, val, exc = assistant.process_query("delete file manual_test_doc.txt", approved=True, deletion_token="wrong_token")
    log_test("13. Delete File (Bad Token)", "delete file manual_test_doc.txt", op, val, exc, "CANCELLED", "Token mismatch")

    # 14. DELETE_FILE with valid token (HIGH risk - approved + DELETE)
    op, val, exc = assistant.process_query("delete file manual_test_doc.txt", approved=True, deletion_token="DELETE")
    log_test("14. Delete File (Valid Token)", "delete file manual_test_doc.txt", op, val, exc, "SUCCESS", "Permanently removed")

    # 15. TRAVERSAL REJECTION (Safety Boundary)
    op, val, exc = assistant.process_query("list files in ../../Windows")
    log_test("15. Traversal Rejection", "list files in ../../Windows", op, val, exc, "UNSAFE_PATH", "Traversal blocked")

    # 16. MISSING PARAMETER (Clarification)
    op, val, exc = assistant.process_query("move file")
    log_test("16. Missing Parameters", "move file", op, val, exc, "NEEDS_CLARIFICATION", "Clarification requested")

    # 17. UNKNOWN INTENT (Out of Scope)
    op, val, exc = assistant.process_query("tell me a joke about robots")
    log_test("17. Unknown Intent", "tell me a joke about robots", op, val, exc, "UNSUPPORTED_INTENT", "Rejected out-of-scope")

    # 18. PARAPHRASE (Semantic Engine test)
    op, val, exc = assistant.process_query("what directory am I in right now")
    log_test("18. Semantic Paraphrase", "what directory am I in right now", op, val, exc, "SUCCESS", "Paraphrase resolved")

    # Clean up test artifacts
    if move_dst.exists():
        move_dst.unlink()
    if demo_dir.exists():
        demo_dir.rmdir()

    print("\n" + "=" * 80)
    passed_count = sum(1 for r in test_results if r["passed"])
    print(f"Manual Test Suite Results: {passed_count}/{len(test_results)} Passed.")
    print("=" * 80)

    # Save results to scratch/manual_test_results.json
    out_file = Path(config.BASE_DIR) / "data" / "manual_test_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(test_results, f, indent=2)
    print(f"Results saved to: {out_file}")


if __name__ == "__main__":
    run_manual_tests()
