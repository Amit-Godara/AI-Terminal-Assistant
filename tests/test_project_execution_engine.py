"""Tests for Phase 13: Controlled Project Command Execution Engine."""

from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest

import config
from core.project_detector import ProjectDetector
from core.project_analyzer import ProjectAnalyzer
from core.project_command_engine import (
    ProjectCommandEngine,
    ProjectCommandRecommendation,
    RecommendedCommand,
    ProjectTask,
    EvidenceCategory,
    CommandRiskLevel,
)
from core.project_execution_engine import (
    ProjectExecutionEngine,
    ExecutionPreview,
    ExecutionApproval,
    ProjectExecutionResult,
    ExecutionStatus,
    ExecutionRiskLevel,
)


@pytest.fixture(scope="module")
def detector() -> ProjectDetector:
    return ProjectDetector()


@pytest.fixture(scope="module")
def analyzer(detector: ProjectDetector) -> ProjectAnalyzer:
    return ProjectAnalyzer(detector=detector)


@pytest.fixture(scope="module")
def cmd_engine() -> ProjectCommandEngine:
    return ProjectCommandEngine()


@pytest.fixture
def exec_engine(tmp_path: Path) -> ProjectExecutionEngine:
    history_file = tmp_path / "test_history.json"
    return ProjectExecutionEngine(timeout_seconds=5, history_file_path=history_file)


def _make_recommendation(
    root: Path,
    command: str,
    wd: Path | None = None,
    evidence_source: str = "main.py",
    evidence_category: EvidenceCategory = EvidenceCategory.CONVENTIONAL,
    risk_level: CommandRiskLevel = CommandRiskLevel.MEDIUM,
    explanation: str = "Run python script",
    task: ProjectTask = ProjectTask.RUN_PROJECT,
) -> ProjectCommandRecommendation:
    return ProjectCommandRecommendation(
        user_request="Run this project",
        detected_task=task,
        project_name=root.name,
        project_root=str(root),
        recommended_commands=[
            RecommendedCommand(
                command=command,
                working_directory=str(wd or root),
                explanation=explanation,
                evidence_source=evidence_source,
                evidence_category=evidence_category,
                confidence=0.9,
                risk_level=risk_level,
            )
        ],
        risk_level=risk_level,
        execution_allowed=False,
    )


# ------------------------------------------------------------------------------
# 1. Recommendation and Preview
# ------------------------------------------------------------------------------

def test_valid_recommendation_creates_preview_without_execution(tmp_path, detector, analyzer, cmd_engine, exec_engine):
    (tmp_path / "requirements.txt").write_text("flask\n", encoding="utf-8")
    (tmp_path / "app.py").write_text("print('executed')\n", encoding="utf-8")
    det = detector.detect(tmp_path)
    ana = analyzer.analyze(tmp_path)
    rec = cmd_engine.recommend("How do I run this project?", det, ana, tmp_path)

    with patch.object(subprocess, "run", side_effect=AssertionError("prepare_execution must not run subprocess")):
        preview = exec_engine.prepare_execution(rec, tmp_path)

    assert preview.status == ExecutionStatus.PREVIEW_CREATED
    assert preview.is_blocked is False
    assert preview.executable == "python"
    assert preview.arguments == ["app.py"]
    assert preview.approval_required is True


def test_invalid_empty_recommendation_blocked(tmp_path, exec_engine):
    rec = ProjectCommandRecommendation(
        user_request="unknown",
        detected_task=ProjectTask.UNKNOWN_PROJECT_TASK,
        project_root=str(tmp_path),
        recommended_commands=[],
        execution_allowed=False,
    )
    preview = exec_engine.prepare_execution(rec, tmp_path)
    assert preview.is_blocked is True
    assert preview.status == ExecutionStatus.BLOCKED


def test_missing_evidence_source_blocked(tmp_path, exec_engine):
    rec = _make_recommendation(tmp_path, "python app.py", evidence_source="")
    preview = exec_engine.prepare_execution(rec, tmp_path)
    assert preview.is_blocked is True
    assert "evidence" in (preview.block_reason or "").lower()


def test_missing_project_root_blocked(tmp_path, exec_engine):
    missing = tmp_path / "does_not_exist"
    rec = _make_recommendation(missing, "python app.py")
    preview = exec_engine.prepare_execution(rec, missing)
    assert preview.is_blocked is True


def test_invalid_working_directory_blocked(tmp_path, exec_engine):
    missing_wd = tmp_path / "nonexistent_sub"
    rec = _make_recommendation(tmp_path, "python app.py", wd=missing_wd)
    preview = exec_engine.prepare_execution(rec, tmp_path)
    assert preview.is_blocked is True


# ------------------------------------------------------------------------------
# 2. Approval Security
# ------------------------------------------------------------------------------

def test_execution_without_approval_rejected(tmp_path, exec_engine):
    (tmp_path / "app.py").write_text("print('ok')\n", encoding="utf-8")
    rec = _make_recommendation(tmp_path, "python app.py")
    preview = exec_engine.prepare_execution(rec, tmp_path)

    res = exec_engine.execute(preview, approval=None)
    assert res.status == ExecutionStatus.REJECTED
    assert res.approval_verified is False


def test_explicit_valid_approval_succeeds(tmp_path, exec_engine):
    (tmp_path / "app.py").write_text("print('hello_phase13')\n", encoding="utf-8")
    rec = _make_recommendation(tmp_path, f"{sys.executable} app.py")
    # Allow sys.executable stem
    exec_engine.allowed_executables.add(Path(sys.executable).stem.lower())
    preview = exec_engine.prepare_execution(rec, tmp_path)
    approval = exec_engine.create_approval(preview, approved=True)

    res = exec_engine.execute(preview, approval)
    assert res.status == ExecutionStatus.COMPLETED
    assert res.approval_verified is True
    assert res.exit_code == 0
    assert "hello_phase13" in res.stdout


def test_expired_approval_rejected(tmp_path, exec_engine):
    (tmp_path / "app.py").write_text("print('ok')\n", encoding="utf-8")
    rec = _make_recommendation(tmp_path, "python app.py")
    preview = exec_engine.prepare_execution(rec, tmp_path)
    preview.expires_at = (datetime.now() - timedelta(seconds=10)).isoformat()
    approval = exec_engine.create_approval(preview, approved=True)

    with patch.object(subprocess, "run", side_effect=AssertionError("Expired approval must not execute")):
        res = exec_engine.execute(preview, approval)
    assert res.status == ExecutionStatus.REJECTED
    assert "expired" in (res.error_message or "").lower()


def test_reused_approval_rejected(tmp_path, exec_engine):
    (tmp_path / "app.py").write_text("print('once')\n", encoding="utf-8")
    exec_engine.allowed_executables.add(Path(sys.executable).stem.lower())
    rec = _make_recommendation(tmp_path, f"{sys.executable} app.py")
    preview = exec_engine.prepare_execution(rec, tmp_path)
    approval = exec_engine.create_approval(preview, approved=True)

    first = exec_engine.execute(preview, approval)
    assert first.status == ExecutionStatus.COMPLETED

    second = exec_engine.execute(preview, approval)
    assert second.status == ExecutionStatus.REJECTED
    assert "already been used" in (second.error_message or "").lower()


def test_modified_command_or_executable_invalidates_approval(tmp_path, exec_engine):
    (tmp_path / "app.py").write_text("print('ok')\n", encoding="utf-8")
    rec = _make_recommendation(tmp_path, "python app.py")
    preview = exec_engine.prepare_execution(rec, tmp_path)
    approval = exec_engine.create_approval(preview, approved=True)

    approval.approved_executable = "node"
    res = exec_engine.execute(preview, approval)
    assert res.status == ExecutionStatus.REJECTED


def test_modified_arguments_invalidates_approval(tmp_path, exec_engine):
    (tmp_path / "app.py").write_text("print('ok')\n", encoding="utf-8")
    rec = _make_recommendation(tmp_path, "python app.py")
    preview = exec_engine.prepare_execution(rec, tmp_path)
    approval = exec_engine.create_approval(preview, approved=True)

    approval.approved_arguments = ["evil.py"]
    res = exec_engine.execute(preview, approval)
    assert res.status == ExecutionStatus.REJECTED


def test_modified_working_directory_invalidates_approval(tmp_path, exec_engine):
    (tmp_path / "app.py").write_text("print('ok')\n", encoding="utf-8")
    rec = _make_recommendation(tmp_path, "python app.py")
    preview = exec_engine.prepare_execution(rec, tmp_path)
    approval = exec_engine.create_approval(preview, approved=True)

    approval.approved_working_directory = str(tmp_path / "other")
    res = exec_engine.execute(preview, approval)
    assert res.status == ExecutionStatus.REJECTED


def test_modified_project_root_invalidates_approval(tmp_path, exec_engine):
    (tmp_path / "app.py").write_text("print('ok')\n", encoding="utf-8")
    rec = _make_recommendation(tmp_path, "python app.py")
    preview = exec_engine.prepare_execution(rec, tmp_path)
    approval = exec_engine.create_approval(preview, approved=True)

    approval.approved_project_root = str(tmp_path.parent)
    res = exec_engine.execute(preview, approval)
    assert res.status == ExecutionStatus.REJECTED


def test_generic_approval_rejected(tmp_path, exec_engine):
    (tmp_path / "app.py").write_text("print('ok')\n", encoding="utf-8")
    rec = _make_recommendation(tmp_path, "python app.py")
    preview = exec_engine.prepare_execution(rec, tmp_path)
    approval = exec_engine.create_approval(preview, approved=True)
    approval.is_generic = True

    res = exec_engine.execute(preview, approval)
    assert res.status == ExecutionStatus.REJECTED


# ------------------------------------------------------------------------------
# 3. Command Parsing & Executable Policy
# ------------------------------------------------------------------------------

def test_argument_array_parsing(tmp_path, exec_engine):
    rec = _make_recommendation(tmp_path, "python -m pytest tests/")
    preview = exec_engine.prepare_execution(rec, tmp_path)
    assert preview.is_blocked is False
    assert preview.executable == "python"
    assert preview.arguments == ["-m", "pytest", "tests/"]


@pytest.mark.parametrize("bad_cmd", [
    "python app.py && echo hacked",
    "python app.py || echo fallback",
    "python app.py | findstr err",
    "python app.py; whoami",
    "python app.py > out.txt",
    "python app.py < in.txt",
    "python $(whoami).py",
    "python `whoami`.py",
    "python ${HOME}/app.py",
])
def test_reject_unsafe_shell_syntax(tmp_path, exec_engine, bad_cmd):
    rec = _make_recommendation(tmp_path, bad_cmd)
    preview = exec_engine.prepare_execution(rec, tmp_path)
    assert preview.is_blocked is True
    assert preview.status == ExecutionStatus.BLOCKED


def test_reject_unsupported_executable(tmp_path, exec_engine):
    rec = _make_recommendation(tmp_path, "curl http://example.com")
    preview = exec_engine.prepare_execution(rec, tmp_path)
    assert preview.is_blocked is True
    assert "allowed executable policy" in (preview.block_reason or "").lower()


@pytest.mark.parametrize("shell_cmd", [
    "cmd.exe /c dir",
    "powershell -Command Get-Date",
    "pwsh -c ls",
    "bash -c whoami",
])
def test_reject_shell_interpreters(tmp_path, exec_engine, shell_cmd):
    rec = _make_recommendation(tmp_path, shell_cmd)
    preview = exec_engine.prepare_execution(rec, tmp_path)
    assert preview.is_blocked is True


@pytest.mark.parametrize("batch_cmd", [
    "gradlew.bat build",
    ".\\mvnw.cmd test",
    "script.bat",
])
def test_reject_unsafe_bat_and_cmd_launchers(tmp_path, exec_engine, batch_cmd):
    rec = _make_recommendation(tmp_path, batch_cmd)
    preview = exec_engine.prepare_execution(rec, tmp_path)
    assert preview.is_blocked is True
    assert "batch launcher" in (preview.block_reason or "").lower()


# ------------------------------------------------------------------------------
# 4. Security Boundaries & Risk Blocking
# ------------------------------------------------------------------------------

def test_path_traversal_protection(tmp_path, exec_engine):
    bad_root = os.path.join(str(tmp_path), "..", "outside")
    rec = _make_recommendation(tmp_path, "python app.py")
    preview = exec_engine.prepare_execution(rec, bad_root)
    assert preview.is_blocked is True


def test_external_working_directory_rejected(tmp_path, exec_engine):
    outside = tmp_path.parent
    rec = _make_recommendation(tmp_path, "python app.py", wd=outside)
    preview = exec_engine.prepare_execution(rec, tmp_path)
    assert preview.is_blocked is True


def test_symlink_boundary_protection(tmp_path, exec_engine):
    outside = tmp_path.parent / f"outside_target_{tmp_path.name}"
    outside.mkdir(exist_ok=True)
    link_dir = tmp_path / "linked_out"
    try:
        link_dir.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("Symlink creation requires elevated privileges on Windows")

    rec = _make_recommendation(tmp_path, "python app.py", wd=link_dir)
    preview = exec_engine.prepare_execution(rec, tmp_path)
    assert preview.is_blocked is True


def test_protected_system_directory_rejected(tmp_path, exec_engine):
    sys_dir = Path(os.environ.get("SystemRoot", r"C:\Windows"))
    if not sys_dir.exists():
        pytest.skip("SystemRoot not found")
    rec = _make_recommendation(sys_dir, "python app.py", wd=sys_dir)
    preview = exec_engine.prepare_execution(rec, sys_dir)
    assert preview.is_blocked is True


def test_high_risk_operation_blocked_even_with_approval(tmp_path, exec_engine):
    rec = _make_recommendation(
        tmp_path,
        "npm run migrate:reset",
        evidence_source="package.json",
        evidence_category=EvidenceCategory.PROJECT_DEFINED,
        risk_level=CommandRiskLevel.HIGH,
    )
    preview = exec_engine.prepare_execution(rec, tmp_path)
    assert preview.is_blocked is True
    assert preview.risk_level == ExecutionRiskLevel.HIGH

    approval = exec_engine.create_approval(preview, approved=True)
    res = exec_engine.execute(preview, approval)
    assert res.status == ExecutionStatus.BLOCKED


def test_project_defined_script_warning(tmp_path, exec_engine):
    rec = _make_recommendation(
        tmp_path,
        "npm run dev",
        evidence_source="package.json",
        evidence_category=EvidenceCategory.PROJECT_DEFINED,
        risk_level=CommandRiskLevel.MEDIUM,
    )
    preview = exec_engine.prepare_execution(rec, tmp_path)
    assert preview.executes_project_code is True
    assert any("project-defined code" in w.lower() for w in preview.warnings)


def test_no_shell_true_and_no_env_secret_leak(tmp_path, exec_engine, monkeypatch):
    monkeypatch.setenv("SUPER_SECRET_API_KEY", "secret-12345")
    script = tmp_path / "env_check.py"
    script.write_text("import os; print(os.environ.get('SUPER_SECRET_API_KEY', 'MISSING'))\n", encoding="utf-8")

    exec_engine.allowed_executables.add(Path(sys.executable).stem.lower())
    rec = _make_recommendation(tmp_path, f"{sys.executable} env_check.py")
    preview = exec_engine.prepare_execution(rec, tmp_path)
    approval = exec_engine.create_approval(preview, approved=True)

    res = exec_engine.execute(preview, approval)
    assert res.status == ExecutionStatus.COMPLETED
    assert "MISSING" in res.stdout
    assert "secret-12345" not in res.stdout


# ------------------------------------------------------------------------------
# 5. Execution Behavior (Controlled Real Mock Scripts)
# ------------------------------------------------------------------------------

def test_nonzero_exit_code_and_stderr_capture(tmp_path, exec_engine):
    script = tmp_path / "fail_script.py"
    script.write_text("import sys; sys.stderr.write('boom_err\\n'); sys.exit(2)\n", encoding="utf-8")

    exec_engine.allowed_executables.add(Path(sys.executable).stem.lower())
    rec = _make_recommendation(tmp_path, f"{sys.executable} fail_script.py")
    preview = exec_engine.prepare_execution(rec, tmp_path)
    approval = exec_engine.create_approval(preview, approved=True)

    res = exec_engine.execute(preview, approval)
    assert res.status == ExecutionStatus.FAILED
    assert res.exit_code == 2
    assert "boom_err" in res.stderr


def test_timeout_handling(tmp_path, exec_engine):
    script = tmp_path / "slow_script.py"
    script.write_text("import time; time.sleep(5)\n", encoding="utf-8")

    exec_engine.allowed_executables.add(Path(sys.executable).stem.lower())
    rec = _make_recommendation(tmp_path, f"{sys.executable} slow_script.py")
    preview = exec_engine.prepare_execution(rec, tmp_path, timeout_seconds=1)
    approval = exec_engine.create_approval(preview, approved=True)

    res = exec_engine.execute(preview, approval)
    assert res.status == ExecutionStatus.TIMED_OUT
    assert res.timed_out is True


def test_output_truncation(tmp_path):
    history_file = tmp_path / "hist.json"
    small_engine = ProjectExecutionEngine(timeout_seconds=5, max_output_chars=50, history_file_path=history_file)
    small_engine.allowed_executables.add(Path(sys.executable).stem.lower())

    script = tmp_path / "loud.py"
    script.write_text("print('A' * 200)\n", encoding="utf-8")

    rec = _make_recommendation(tmp_path, f"{sys.executable} loud.py")
    preview = small_engine.prepare_execution(rec, tmp_path)
    approval = small_engine.create_approval(preview, approved=True)

    res = small_engine.execute(preview, approval)
    assert res.status == ExecutionStatus.COMPLETED
    assert res.output_truncated is True
    assert "OUTPUT TRUNCATED" in res.stdout


def test_execution_exception_handling(tmp_path, exec_engine):
    rec = _make_recommendation(tmp_path, "python app.py")
    preview = exec_engine.prepare_execution(rec, tmp_path)
    approval = exec_engine.create_approval(preview, approved=True)

    with patch.object(subprocess, "Popen", side_effect=OSError("mocked spawn failure")):
        res = exec_engine.execute(preview, approval)
    assert res.status == ExecutionStatus.FAILED
    assert "mocked spawn failure" in (res.error_message or "")


# ------------------------------------------------------------------------------
# 6. Audit Logging & Secret Redaction
# ------------------------------------------------------------------------------

def test_audit_logging_for_completed_failed_and_rejected(tmp_path, exec_engine):
    ok_script = tmp_path / "ok.py"
    ok_script.write_text("print('ok')\n", encoding="utf-8")
    fail_script = tmp_path / "fail.py"
    fail_script.write_text("import sys; sys.stderr.write('API_KEY=supersecret123\\n'); sys.exit(1)\n", encoding="utf-8")

    exec_engine.allowed_executables.add(Path(sys.executable).stem.lower())

    # 1. Completed
    p1 = exec_engine.prepare_execution(_make_recommendation(tmp_path, f"{sys.executable} ok.py"), tmp_path)
    exec_engine.execute(p1, exec_engine.create_approval(p1, True))

    # 2. Failed with secret in stderr
    p2 = exec_engine.prepare_execution(_make_recommendation(tmp_path, f"{sys.executable} fail.py"), tmp_path)
    res2 = exec_engine.execute(p2, exec_engine.create_approval(p2, True))
    assert "supersecret123" not in res2.stderr

    # 3. Rejected
    p3 = exec_engine.prepare_execution(_make_recommendation(tmp_path, f"{sys.executable} ok.py"), tmp_path)
    exec_engine.execute(p3, approval=None)

    history = exec_engine.history_manager.get_history()
    assert len(history) == 3
    statuses = [h["execution_status"] for h in history]
    assert statuses == ["COMPLETED", "FAILED", "REJECTED"]
    assert "supersecret123" not in json.dumps(history)


# ------------------------------------------------------------------------------
# 7. Phase 13.1 Security Hardening Tests
# ------------------------------------------------------------------------------

def test_forged_approval_without_hmac_token_rejected(tmp_path, exec_engine):
    """Forging an ExecutionApproval using only public preview_hash must be rejected."""
    (tmp_path / "app.py").write_text("print('forged')\n", encoding="utf-8")
    rec = _make_recommendation(tmp_path, "python app.py")
    preview = exec_engine.prepare_execution(rec, tmp_path)

    forged = ExecutionApproval(
        preview_id=preview.preview_id,
        preview_hash=preview.preview_hash,
        approved=True,
        approved_executable=preview.executable,
        approved_arguments=list(preview.arguments),
        approved_working_directory=preview.working_directory,
        approved_project_root=preview.authorized_project_root,
        approved_risk_level=preview.risk_level,
        approval_token="0" * 64,
        is_generic=False,
    )
    res = exec_engine.execute(preview, forged)
    assert res.status == ExecutionStatus.REJECTED
    assert "forged" in (res.error_message or "").lower() or "token" in (res.error_message or "").lower()


def test_trusted_callback_authorization_boundary(tmp_path):
    """Engine with require_trusted_callback=True enforces callback confirmation."""
    (tmp_path / "ok.py").write_text("print('trusted_ok')\n", encoding="utf-8")
    hist = tmp_path / "hist_cb.json"

    denied_engine = ProjectExecutionEngine(
        timeout_seconds=5,
        history_file_path=hist,
        user_approval_callback=lambda p: False,
        require_trusted_callback=True,
    )
    denied_engine.allowed_executables.add(Path(sys.executable).stem.lower())
    rec = _make_recommendation(tmp_path, f"{sys.executable} ok.py")
    p1 = denied_engine.prepare_execution(rec, tmp_path)
    a1 = denied_engine.create_approval(p1, approved=True)
    assert a1.approved is False
    res1 = denied_engine.execute(p1, a1)
    assert res1.status == ExecutionStatus.REJECTED

    approved_engine = ProjectExecutionEngine(
        timeout_seconds=5,
        history_file_path=hist,
        user_approval_callback=lambda p: True,
        require_trusted_callback=True,
    )
    approved_engine.allowed_executables.add(Path(sys.executable).stem.lower())
    p2 = approved_engine.prepare_execution(rec, tmp_path)
    a2 = approved_engine.authorize_preview(p2)
    assert a2.approved is True
    assert a2.auth_method == "TRUSTED_CALLBACK"
    res2 = approved_engine.execute(p2, a2)
    assert res2.status == ExecutionStatus.COMPLETED
    assert "trusted_ok" in res2.stdout


def test_path_hijacking_shadowed_executable_blocked(tmp_path, exec_engine):
    """A malicious executable named like an allowed tool inside the project dir must be blocked."""
    (tmp_path / "python.exe").write_text("fake binary", encoding="utf-8")
    (tmp_path / "app.py").write_text("print('hi')\n", encoding="utf-8")

    rec = _make_recommendation(tmp_path, "python app.py")
    preview = exec_engine.prepare_execution(rec, tmp_path)
    assert preview.is_blocked is True
    assert "shadowing" in (preview.block_reason or "").lower() or "hijacking" in (preview.block_reason or "").lower()


def test_resolved_executable_path_outside_project(tmp_path, exec_engine):
    """Bare allowlisted executables resolve to an explicit path outside the project."""
    rec = _make_recommendation(tmp_path, "python --version")
    preview = exec_engine.prepare_execution(rec, tmp_path)
    assert preview.is_blocked is False
    if preview.resolved_executable:
        assert Path(preview.resolved_executable).is_absolute()
        assert tmp_path.resolve() not in Path(preview.resolved_executable).resolve().parents


def test_bounded_streaming_stdout_and_stderr_no_deadlock(tmp_path):
    """Both stdout and stderr exceeding max_output_chars are bounded independently without deadlock."""
    hist = tmp_path / "hist_stream.json"
    bounded_engine = ProjectExecutionEngine(timeout_seconds=10, max_output_chars=256, history_file_path=hist)
    bounded_engine.allowed_executables.add(Path(sys.executable).stem.lower())

    script = tmp_path / "flood.py"
    script.write_text(
        "import sys\n"
        "for _ in range(200):\n"
        "    sys.stdout.write('O' * 1024)\n"
        "    sys.stderr.write('E' * 1024)\n"
        "sys.stdout.flush()\n"
        "sys.stderr.flush()\n",
        encoding="utf-8",
    )

    rec = _make_recommendation(tmp_path, f"{sys.executable} flood.py")
    preview = bounded_engine.prepare_execution(rec, tmp_path)
    approval = bounded_engine.create_approval(preview, approved=True)

    res = bounded_engine.execute(preview, approval)
    assert res.status == ExecutionStatus.COMPLETED
    assert res.output_truncated is True
    assert res.stdout_truncated is True
    assert res.stderr_truncated is True
    assert len(res.stdout) <= 300
    assert len(res.stderr) <= 300


def test_timeout_terminates_child_process_tree(tmp_path, exec_engine):
    """Timed-out parent process that spawns a child process terminates the process tree."""
    pid_file = tmp_path / "child.pid"
    child_script = tmp_path / "child_sleeper.py"
    child_script.write_text("import time; time.sleep(30)\n", encoding="utf-8")

    parent_script = tmp_path / "parent_spawner.py"
    parent_script.write_text(
        "import subprocess, sys, time\n"
        f"p = subprocess.Popen([sys.executable, {repr(str(child_script))}])\n"
        f"open({repr(str(pid_file))}, 'w').write(str(p.pid))\n"
        "time.sleep(30)\n",
        encoding="utf-8",
    )

    exec_engine.allowed_executables.add(Path(sys.executable).stem.lower())
    rec = _make_recommendation(tmp_path, f"{sys.executable} parent_spawner.py")
    preview = exec_engine.prepare_execution(rec, tmp_path, timeout_seconds=1)
    approval = exec_engine.create_approval(preview, approved=True)

    res = exec_engine.execute(preview, approval)
    assert res.status == ExecutionStatus.TIMED_OUT
    assert res.timed_out is True

    if pid_file.exists():
        child_pid = int(pid_file.read_text(encoding="utf-8").strip())
        # Verify child process is no longer alive
        is_alive = True
        try:
            os.kill(child_pid, 0)
        except OSError:
            is_alive = False
        assert is_alive is False


def test_safe_env_preserves_windows_vars_and_strips_secrets(tmp_path, exec_engine, monkeypatch):
    """Sanitized env preserves PATH/SystemRoot while stripping API keys, tokens, and passwords."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test1234567890abcdef")
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_test1234567890abcdef")
    monkeypatch.setenv("DB_PASSWORD", "hunter2")
    monkeypatch.setenv("AUTH_CREDENTIAL", "topsecret")

    script = tmp_path / "check_env.py"
    script.write_text(
        "import os, json\n"
        "print(json.dumps({\n"
        "  'has_path': 'PATH' in os.environ,\n"
        "  'has_sysroot': ('SystemRoot' in os.environ or 'SYSTEMROOT' in os.environ),\n"
        "  'leaked': [k for k in os.environ if any(s in k.upper() for s in ('KEY','SECRET','TOKEN','PASSWORD','CREDENTIAL'))],\n"
        "}))\n",
        encoding="utf-8",
    )

    exec_engine.allowed_executables.add(Path(sys.executable).stem.lower())
    rec = _make_recommendation(tmp_path, f"{sys.executable} check_env.py")
    preview = exec_engine.prepare_execution(rec, tmp_path)
    approval = exec_engine.create_approval(preview, approved=True)

    res = exec_engine.execute(preview, approval)
    assert res.status == ExecutionStatus.COMPLETED
    data = json.loads(res.stdout.strip())
    assert data["has_path"] is True
    if os.name == "nt":
        assert data["has_sysroot"] is True
    assert data["leaked"] == []

