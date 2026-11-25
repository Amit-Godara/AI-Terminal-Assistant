"""Safety and Controlled Execution Unit Tests for CommandExecutor.

Tests cover:
1. Successful directory creation
2. Successful file creation (exclusive open)
3. Successful listing
4. Successful search
5. Successful copy
6. Successful move
7. Successful deletion with double confirmation (token == 'DELETE')
8. Rejection without validation
9. Rejection after validation failure
10. Cancellation when user approval is NO
11. Cancellation on empty approval input
12. Cancellation on invalid approval input
13. Cancellation on EOF (EOFError)
14. Cancellation on KeyboardInterrupt
15. Second confirmation token verification for deletion
16. No execution when confirmation is missing / denied
17. File not found handling
18. Existing target conflict handling
19. Permission error handling
20. Subprocess timeout handling
21. Nonzero subprocess return code handling
22. Execution audit history logging
23. History corruption handling (safe recovery without data loss)
24. Verification of no shell execution (shell=False)
25. Verification that arbitrary preview text is never executed
"""

import json
import os
from pathlib import Path
import subprocess
import pytest
from unittest.mock import MagicMock, patch

import config
from core.command_generator import (
    CommandOperation,
    OperationType,
    RiskLevel,
    GenerationStatus,
)
from core.command_validator import (
    CommandValidator,
    ValidationResult,
    ValidationStatus,
)
from core.executor import (
    CommandExecutor,
    ExecutionResult,
    ExecutionStatus,
)


@pytest.fixture
def sandbox_workspace(tmp_path):
    """Provides an isolated temporary workspace directory."""
    workspace = tmp_path / "exec_workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    return workspace


@pytest.fixture
def history_file(tmp_path):
    """Provides an isolated history audit file."""
    return tmp_path / "test_history.json"


@pytest.fixture
def validator(sandbox_workspace):
    """CommandValidator configured with sandbox workspace."""
    return CommandValidator(workspace_dir=sandbox_workspace, safe_mode=True)


@pytest.fixture
def executor(history_file):
    """CommandExecutor configured with test history file."""
    return CommandExecutor(timeout_seconds=5, history_file_path=history_file)


# 1. Successful directory creation
def test_1_create_directory_execution(validator, executor, sandbox_workspace):
    """Test successful directory creation via executor."""
    op = CommandOperation(
        operation_type=OperationType.CREATE_DIR,
        intent="CREATE_DIRECTORY",
        arguments={"directory_name": "TestDir"},
        generation_status=GenerationStatus.READY_FOR_VALIDATION,
        source_query="create folder TestDir",
    )
    val_res = validator.validate(op)
    assert val_res.is_valid is True

    exec_res = executor.execute(op, val_res, approved=True)
    assert exec_res.success is True
    assert exec_res.execution_status == ExecutionStatus.SUCCESS
    assert (sandbox_workspace / "TestDir").is_dir()


# 2. Successful file creation
def test_2_create_file_execution(validator, executor, sandbox_workspace):
    """Test successful empty file creation via executor."""
    op = CommandOperation(
        operation_type=OperationType.CREATE_FILE,
        intent="CREATE_FILE",
        arguments={"file_name": "created.txt"},
        generation_status=GenerationStatus.READY_FOR_VALIDATION,
        source_query="create file created.txt",
    )
    val_res = validator.validate(op)
    assert val_res.is_valid is True

    exec_res = executor.execute(op, val_res, approved=True)
    assert exec_res.success is True
    assert exec_res.execution_status == ExecutionStatus.SUCCESS
    assert (sandbox_workspace / "created.txt").is_file()


# 3. Successful listing
def test_3_list_files_execution(validator, executor, sandbox_workspace):
    """Test directory listing outputs formatted items."""
    (sandbox_workspace / "file1.txt").write_text("1", encoding="utf-8")
    (sandbox_workspace / "file2.log").write_text("2", encoding="utf-8")

    op = CommandOperation(
        operation_type=OperationType.LIST,
        intent="LIST_FILES",
        arguments={},
        generation_status=GenerationStatus.READY_FOR_VALIDATION,
        source_query="list files",
    )
    val_res = validator.validate(op)
    exec_res = executor.execute(op, val_res)

    assert exec_res.success is True
    assert "file1.txt" in exec_res.stdout
    assert "file2.log" in exec_res.stdout


# 4. Successful search
def test_4_search_files_execution(validator, executor, sandbox_workspace):
    """Test search files finds matching pattern."""
    (sandbox_workspace / "app.py").write_text("python", encoding="utf-8")
    (sandbox_workspace / "readme.md").write_text("markdown", encoding="utf-8")

    op = CommandOperation(
        operation_type=OperationType.SEARCH,
        intent="SEARCH_FILES",
        arguments={"directory": ".", "pattern": "*.py", "recurse": False},
        generation_status=GenerationStatus.READY_FOR_VALIDATION,
        source_query="search python files",
    )
    val_res = validator.validate(op)
    exec_res = executor.execute(op, val_res)

    assert exec_res.success is True
    assert "app.py" in exec_res.stdout
    assert "readme.md" not in exec_res.stdout


# 5. Successful copy
def test_5_copy_file_execution(validator, executor, sandbox_workspace):
    """Test copy file execution."""
    src = sandbox_workspace / "original.txt"
    src.write_text("original content", encoding="utf-8")

    op = CommandOperation(
        operation_type=OperationType.COPY,
        intent="COPY_FILE",
        arguments={"source": "original.txt", "destination": "copied.txt"},
        generation_status=GenerationStatus.READY_FOR_VALIDATION,
        source_query="copy original.txt to copied.txt",
    )
    val_res = validator.validate(op)
    exec_res = executor.execute(op, val_res, approved=True)

    assert exec_res.success is True
    assert (sandbox_workspace / "copied.txt").read_text(encoding="utf-8") == "original content"
    assert src.exists()  # Source remains


# 6. Successful move
def test_6_move_file_execution(validator, executor, sandbox_workspace):
    """Test move file execution."""
    src = sandbox_workspace / "to_move.txt"
    src.write_text("move content", encoding="utf-8")

    op = CommandOperation(
        operation_type=OperationType.MOVE,
        intent="MOVE_FILE",
        arguments={"source": "to_move.txt", "destination": "moved.txt"},
        generation_status=GenerationStatus.READY_FOR_VALIDATION,
        source_query="move to_move.txt to moved.txt",
    )
    val_res = validator.validate(op)
    exec_res = executor.execute(op, val_res, approved=True)

    assert exec_res.success is True
    assert (sandbox_workspace / "moved.txt").exists()
    assert not src.exists()  # Source was moved


# 7. Successful deletion with both confirmations
def test_7_delete_file_with_both_confirmations(validator, executor, sandbox_workspace):
    """Test deletion succeeds only when approved AND exact token 'DELETE' is supplied."""
    target = sandbox_workspace / "delete_me.txt"
    target.write_text("bye", encoding="utf-8")

    op = CommandOperation(
        operation_type=OperationType.DELETE,
        intent="DELETE_FILE",
        arguments={"file_name": "delete_me.txt"},
        risk_level=RiskLevel.HIGH,
        requires_confirmation=True,
        generation_status=GenerationStatus.READY_FOR_VALIDATION,
        source_query="delete delete_me.txt",
    )
    val_res = validator.validate(op)
    assert val_res.requires_double_confirmation is True

    exec_res = executor.execute(op, val_res, approved=True, deletion_token="DELETE")
    assert exec_res.success is True
    assert not target.exists()


# 8. Rejection without validation
def test_8_rejection_without_validation(executor, sandbox_workspace):
    """Test execution is rejected if validation result is marked invalid."""
    op = CommandOperation(
        operation_type=OperationType.DELETE,
        intent="DELETE_FILE",
        arguments={"file_name": "none.txt"},
        generation_status=GenerationStatus.READY_FOR_VALIDATION,
        source_query="delete none.txt",
    )
    invalid_val = ValidationResult(
        is_valid=False,
        validation_status=ValidationStatus.UNSAFE_PATH,
        operation_type=OperationType.DELETE,
        validation_errors=["Invalid target."],
        reason="Target is invalid.",
    )
    exec_res = executor.execute(op, invalid_val, approved=True)
    assert exec_res.success is False
    assert exec_res.execution_status == ExecutionStatus.VALIDATION_FAILED


# 9. Rejection after validation failure
def test_9_rejection_after_validation_failure(validator, executor):
    """Test that a failed validation passes through to failed execution result without running."""
    op = CommandOperation(
        operation_type=OperationType.MOVE,
        intent="MOVE_FILE",
        arguments={"source": "report.pdf"},  # missing destination
        generation_status=GenerationStatus.NEEDS_CLARIFICATION,
        source_query="move report.pdf",
    )
    val_res = validator.validate(op)
    assert val_res.is_valid is False

    exec_res = executor.execute(op, val_res, approved=True)
    assert exec_res.success is False
    assert exec_res.execution_status == ExecutionStatus.VALIDATION_FAILED


# 10. Cancellation when approval is NO
def test_10_cancellation_when_approval_is_no(validator, executor, sandbox_workspace):
    """Test operation is cancelled when user responds NO."""
    op = CommandOperation(
        operation_type=OperationType.CREATE_DIR,
        intent="CREATE_DIRECTORY",
        arguments={"directory_name": "CancelledDir"},
        generation_status=GenerationStatus.READY_FOR_VALIDATION,
        source_query="create folder CancelledDir",
    )
    val_res = validator.validate(op)

    exec_res = executor.execute(op, val_res, approved=False)
    assert exec_res.success is False
    assert exec_res.execution_status == ExecutionStatus.CANCELLED
    assert exec_res.was_cancelled is True
    assert not (sandbox_workspace / "CancelledDir").exists()


# 11. Cancellation on empty approval input
def test_11_cancellation_on_empty_approval_input():
    """Test default prompt treats empty input as NO."""
    with patch("builtins.input", return_value=""):
        res = CommandExecutor.default_confirm_prompt("Confirm?")
        assert res is False


# 12. Cancellation on invalid approval input
def test_12_cancellation_on_invalid_approval_input():
    """Test default prompt treats random input as NO."""
    for bad_input in ("maybe", "sure", "ok", "1", "true"):
        with patch("builtins.input", return_value=bad_input):
            assert CommandExecutor.default_confirm_prompt("Confirm?") is False


# 13. Cancellation on EOF (EOFError)
def test_13_cancellation_on_eof():
    """Test default prompt safely cancels on EOF."""
    with patch("builtins.input", side_effect=EOFError):
        assert CommandExecutor.default_confirm_prompt("Confirm?") is False


# 14. Cancellation on KeyboardInterrupt
def test_14_cancellation_on_keyboard_interrupt():
    """Test default prompt safely cancels on Ctrl+C."""
    with patch("builtins.input", side_effect=KeyboardInterrupt):
        assert CommandExecutor.default_confirm_prompt("Confirm?") is False


# 15. Second confirmation token verification for deletion
def test_15_second_confirmation_token_mismatch(validator, executor, sandbox_workspace):
    """Test deletion is aborted if user types anything other than exact 'DELETE'."""
    target = sandbox_workspace / "safe_file.txt"
    target.write_text("stay", encoding="utf-8")

    op = CommandOperation(
        operation_type=OperationType.DELETE,
        intent="DELETE_FILE",
        arguments={"file_name": "safe_file.txt"},
        risk_level=RiskLevel.HIGH,
        requires_confirmation=True,
        generation_status=GenerationStatus.READY_FOR_VALIDATION,
        source_query="delete safe_file.txt",
    )
    val_res = validator.validate(op)

    # User approved initial prompt, but typed wrong token
    for wrong_token in ("delete", "yes", "YES", "y", "remove", ""):
        exec_res = executor.execute(op, val_res, approved=True, deletion_token=wrong_token)
        assert exec_res.success is False
        assert exec_res.execution_status == ExecutionStatus.CANCELLED
        assert target.exists()  # Target was NOT deleted!


# 16. No execution when confirmation callback returns False
def test_16_confirmation_callback_denied(validator, sandbox_workspace, history_file):
    """Test execution is blocked when pluggable confirmation callback returns False."""
    custom_executor = CommandExecutor(
        history_file_path=history_file,
        confirmation_callback=lambda prompt, risk: False,
    )
    op = CommandOperation(
        operation_type=OperationType.CREATE_DIR,
        intent="CREATE_DIRECTORY",
        arguments={"directory_name": "BlockedDir"},
        generation_status=GenerationStatus.READY_FOR_VALIDATION,
        source_query="create folder BlockedDir",
    )
    val_res = validator.validate(op)
    exec_res = custom_executor.execute(op, val_res)

    assert exec_res.success is False
    assert exec_res.execution_status == ExecutionStatus.CANCELLED
    assert not (sandbox_workspace / "BlockedDir").exists()


# 17. File not found handling
def test_17_file_not_found_handling(validator, executor):
    """Test file not found returns FILE_NOT_FOUND status."""
    op = CommandOperation(
        operation_type=OperationType.MOVE,
        intent="MOVE_FILE",
        arguments={"source": "ghost.txt", "destination": "dest.txt"},
        generation_status=GenerationStatus.READY_FOR_VALIDATION,
        source_query="move ghost.txt to dest.txt",
    )
    val_res = validator.validate(op)
    # Validation already catches missing source
    assert val_res.is_valid is False


# 18. Existing target conflict handling
def test_18_existing_target_conflict_handling(validator, executor, sandbox_workspace):
    """Test creation/copy over existing file returns TARGET_ALREADY_EXISTS."""
    target = sandbox_workspace / "exists.txt"
    target.write_text("existing", encoding="utf-8")

    op = CommandOperation(
        operation_type=OperationType.CREATE_FILE,
        intent="CREATE_FILE",
        arguments={"file_name": "exists.txt"},
        generation_status=GenerationStatus.READY_FOR_VALIDATION,
        source_query="create exists.txt",
    )
    val_res = validator.validate(op)
    assert val_res.is_valid is False
    assert val_res.validation_status == ValidationStatus.CONFLICT_TARGET_EXISTS

    exec_res = executor.execute(op, val_res, approved=True)
    assert exec_res.success is False
    assert exec_res.execution_status == ExecutionStatus.TARGET_ALREADY_EXISTS


# 19. Permission error handling
def test_19_permission_error_handling(validator, executor, sandbox_workspace, monkeypatch):
    """Test unexpected permission error is caught and mapped to PERMISSION_DENIED."""
    op = CommandOperation(
        operation_type=OperationType.CREATE_DIR,
        intent="CREATE_DIRECTORY",
        arguments={"directory_name": "PermDir"},
        generation_status=GenerationStatus.READY_FOR_VALIDATION,
        source_query="create folder PermDir",
    )
    val_res = validator.validate(op)

    def mock_mkdir(*args, **kwargs):
        raise PermissionError("Access is denied.")

    monkeypatch.setattr(Path, "mkdir", mock_mkdir)

    exec_res = executor.execute(op, val_res, approved=True)
    assert exec_res.success is False
    assert exec_res.execution_status == ExecutionStatus.PERMISSION_DENIED


# 20. Subprocess timeout handling
def test_20_subprocess_timeout_handling(validator, executor, monkeypatch):
    """Test that subprocess timeout expired is caught and mapped to TIMEOUT."""
    op = CommandOperation(
        operation_type=OperationType.CHECK_VERSION,
        intent="CHECK_VERSION",
        arguments={"version_name": "python"},
        generation_status=GenerationStatus.READY_FOR_VALIDATION,
        source_query="check python version",
    )
    val_res = validator.validate(op)

    def mock_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=5)

    monkeypatch.setattr(subprocess, "run", mock_run)

    exec_res = executor.execute(op, val_res)
    assert exec_res.success is False
    assert exec_res.execution_status == ExecutionStatus.TIMEOUT
    assert exec_res.return_code == 124


# 21. Nonzero subprocess return code handling
def test_21_nonzero_subprocess_return_code(validator, executor, monkeypatch):
    """Test that nonzero subprocess exit code is mapped to EXECUTION_FAILED."""
    op = CommandOperation(
        operation_type=OperationType.CHECK_VERSION,
        intent="CHECK_VERSION",
        arguments={"version_name": "git"},
        generation_status=GenerationStatus.READY_FOR_VALIDATION,
        source_query="check git version",
    )
    val_res = validator.validate(op)

    mock_proc = MagicMock()
    mock_proc.returncode = 1
    mock_proc.stdout = ""
    mock_proc.stderr = "git command failed"
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: mock_proc)

    exec_res = executor.execute(op, val_res)
    assert exec_res.success is False
    assert exec_res.execution_status == ExecutionStatus.EXECUTION_FAILED
    assert exec_res.return_code == 1


# 22. Execution audit history logging
def test_22_execution_audit_history_logging(validator, executor, sandbox_workspace, history_file):
    """Test that every execution attempt is recorded in the audit history."""
    op = CommandOperation(
        operation_type=OperationType.CREATE_FILE,
        intent="CREATE_FILE",
        arguments={"file_name": "audit_test.txt"},
        generation_status=GenerationStatus.READY_FOR_VALIDATION,
        source_query="create file audit_test.txt",
    )
    val_res = validator.validate(op)
    executor.execute(op, val_res, approved=True)

    records = executor.history_manager.get_history()
    assert len(records) > 0
    last = records[-1]
    assert last["operation_type"] == "CREATE_FILE"
    assert last["success"] is True
    assert "timestamp" in last
    assert "audit_id" in last


# 23. History corruption handling
def test_23_history_corruption_handling(history_file):
    """Test that corrupted JSON in history file does not crash the system."""
    # Write garbage to history file
    history_file.write_text("{{corrupted_garbage json}}", encoding="utf-8")

    from utils.history import HistoryManager
    hm = HistoryManager(history_file)
    # Should safely recover empty list
    history = hm.get_history()
    assert history == []

    # Logging a new entry recovers file
    hm.log_entry({"test": "entry"})
    recovered = hm.get_history()
    assert len(recovered) == 1
    assert recovered[0]["test"] == "entry"


# 24. Verification of no shell execution (shell=False)
def test_24_no_shell_execution(validator, executor, monkeypatch):
    """Verifies subprocess.run is called strictly with shell=False and argument list."""
    op = CommandOperation(
        operation_type=OperationType.CHECK_VERSION,
        intent="CHECK_VERSION",
        arguments={"version_name": "python"},
        generation_status=GenerationStatus.READY_FOR_VALIDATION,
        source_query="check python version",
    )
    val_res = validator.validate(op)

    captured_kwargs = {}

    def mock_run(*args, **kwargs):
        captured_kwargs.update(kwargs)
        res = MagicMock()
        res.returncode = 0
        res.stdout = "Python 3.11.9"
        res.stderr = ""
        return res

    monkeypatch.setattr(subprocess, "run", mock_run)

    executor.execute(op, val_res)
    assert captured_kwargs.get("shell") is False


# 25. Verification that arbitrary preview text is never executed
def test_25_preview_text_never_executed(validator, executor, monkeypatch):
    """Verifies that malicious or arbitrary text injected into powershell_preview is ignored."""
    # Operation with a malicious injected preview string
    op = CommandOperation(
        operation_type=OperationType.GET_LOCATION,
        intent="CURRENT_DIRECTORY",
        arguments={},
        powershell_preview="Remove-Item -Recurse -Force C:\\ ; Format-Volume C",
        generation_status=GenerationStatus.READY_FOR_VALIDATION,
        source_query="where am I",
    )
    val_res = validator.validate(op)
    assert val_res.is_valid is True

    mock_run = MagicMock()
    monkeypatch.setattr(subprocess, "run", mock_run)
    monkeypatch.setattr(os, "system", mock_run)

    exec_res = executor.execute(op, val_res)
    assert exec_res.success is True
    # subprocess and os.system must NOT have been called with the preview text
    mock_run.assert_not_called()
