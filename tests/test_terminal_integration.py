"""Integration tests for Phase 8: Terminal UI and End-to-End Integration.

Tests cover:
1. Application initialization
2. NLP components initialized once and reused
3. Natural-language request reaches command generator
4. Generated operation reaches validator
5. Invalid operation never reaches execution
6. LOW-risk operations execute only after successful validation
7. MEDIUM-risk operations require explicit approval
8. Declining confirmation prevents execution
9. HIGH-risk deletion requires both approval and exact 'DELETE' token
10. Invalid deletion confirmation prevents execution
11. UNKNOWN intent does not execute
12. Missing parameters trigger clarification
13. History displays existing records
14. Empty or corrupted history does not crash application
15. $help, $history, $clear, and $exit work
16. EOF and KeyboardInterrupt are handled gracefully
17. Execution errors are displayed without crashing application
18. Application never executes powershell_preview directly
"""

from io import StringIO
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

import config
from app import AITerminalAssistant
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
from utils.terminal_ui import TerminalUI
from utils.history import HistoryManager


@pytest.fixture
def temp_env():
    """Provides an isolated workspace and history file in a temporary directory."""
    temp_dir = Path(tempfile.mkdtemp(prefix="assistant_test_"))
    ws_dir = temp_dir / "workspace"
    ws_dir.mkdir(parents=True, exist_ok=True)
    hist_file = temp_dir / "command_history.json"
    string_io = StringIO()
    console = Console(file=string_io, force_terminal=False, color_system=None)

    assistant = AITerminalAssistant(
        workspace_dir=ws_dir,
        safe_mode=True,
        console=console,
        history_file_path=hist_file,
    )

    yield {
        "temp_dir": temp_dir,
        "workspace_dir": ws_dir,
        "history_file": hist_file,
        "assistant": assistant,
        "console": console,
        "output_io": string_io,
    }

    # Teardown
    shutil.rmtree(temp_dir, ignore_errors=True)


# ------------------------------------------------------------------------------
# Test 1: Application Initialization
# ------------------------------------------------------------------------------
def test_01_application_initialization(temp_env):
    """Verifies that AITerminalAssistant initializes cleanly and creates the workspace."""
    assistant = temp_env["assistant"]
    assert assistant is not None
    assert assistant.workspace_dir.exists()
    assert assistant.workspace_dir.is_dir()
    assert isinstance(assistant.ui, TerminalUI)
    assert isinstance(assistant.validator, CommandValidator)
    assert isinstance(assistant.executor, CommandExecutor)
    assert isinstance(assistant.history_manager, HistoryManager)


# ------------------------------------------------------------------------------
# Test 2: NLP Components Initialized Once and Reused
# ------------------------------------------------------------------------------
def test_02_nlp_components_reused(temp_env):
    """Verifies that heavy NLP models are initialized once and reused across queries."""
    assistant = temp_env["assistant"]
    preprocessor_id = id(assistant.preprocessor)
    classifier_id = id(assistant.classifier)
    semantic_engine_id = id(assistant.semantic_engine)
    entity_extractor_id = id(assistant.entity_extractor)
    generator_id = id(assistant.generator)

    # Process first query
    assistant.process_query("show current directory")

    # Process second query
    assistant.process_query("list all files")

    # IDs must remain identical (no recreation)
    assert id(assistant.preprocessor) == preprocessor_id
    assert id(assistant.classifier) == classifier_id
    assert id(assistant.semantic_engine) == semantic_engine_id
    assert id(assistant.entity_extractor) == entity_extractor_id
    assert id(assistant.generator) == generator_id


# ------------------------------------------------------------------------------
# Test 3: Natural Language Request Reaches Command Generator
# ------------------------------------------------------------------------------
def test_03_request_reaches_generator(temp_env):
    """Verifies natural language queries produce a valid CommandOperation."""
    assistant = temp_env["assistant"]
    op, val, exc = assistant.process_query("show current directory")

    assert op is not None
    assert isinstance(op, CommandOperation)
    assert op.operation_type == OperationType.GET_LOCATION
    assert op.powershell_preview == "Get-Location"
    assert op.generation_status == GenerationStatus.READY_FOR_VALIDATION


# ------------------------------------------------------------------------------
# Test 4: Generated Operation Reaches Validator
# ------------------------------------------------------------------------------
def test_04_operation_reaches_validator(temp_env):
    """Verifies that generated operations undergo independent safety validation."""
    assistant = temp_env["assistant"]
    op, val, exc = assistant.process_query("show current directory")

    assert val is not None
    assert isinstance(val, ValidationResult)
    assert val.is_valid is True
    assert val.validation_status == ValidationStatus.VALID
    assert val.operation_type == OperationType.GET_LOCATION


# ------------------------------------------------------------------------------
# Test 5: Invalid Operation Never Reaches Execution
# ------------------------------------------------------------------------------
def test_05_invalid_operation_never_executes(temp_env):
    """Verifies that operations failing validation are blocked before execution."""
    assistant = temp_env["assistant"]

    # Traversal attempt escaping workspace
    op, val, exc = assistant.process_query("list files in ../../Windows")

    assert op is not None
    assert val is not None
    assert val.is_valid is False
    assert val.validation_status == ValidationStatus.UNSAFE_PATH
    # Execution must never have been called
    assert exc is None


# ------------------------------------------------------------------------------
# Test 6: LOW-Risk Operations Execute Only After Validation
# ------------------------------------------------------------------------------
def test_06_low_risk_executes_after_validation(temp_env):
    """Verifies that LOW-risk operations execute automatically only after valid validation."""
    assistant = temp_env["assistant"]
    op, val, exc = assistant.process_query("show current directory")

    assert op.risk_level == RiskLevel.LOW
    assert val.is_valid is True
    assert exc is not None
    assert exc.success is True
    assert exc.execution_status == ExecutionStatus.SUCCESS
    assert str(temp_env["workspace_dir"]) in exc.stdout


# ------------------------------------------------------------------------------
# Test 7: MEDIUM-Risk Operations Require Explicit Approval
# ------------------------------------------------------------------------------
def test_07_medium_risk_requires_approval(temp_env):
    """Verifies that MEDIUM-risk operations execute when explicitly approved."""
    assistant = temp_env["assistant"]
    ws = temp_env["workspace_dir"]
    target_file = ws / "test_notes.txt"

    op, val, exc = assistant.process_query(
        "create file test_notes.txt",
        approved=True,
    )

    assert op.risk_level == RiskLevel.MEDIUM
    assert val.requires_confirmation is True
    assert exc is not None
    assert exc.success is True
    assert exc.execution_status == ExecutionStatus.SUCCESS
    assert target_file.exists()


# ------------------------------------------------------------------------------
# Test 8: Declining Confirmation Prevents Execution
# ------------------------------------------------------------------------------
def test_08_declining_confirmation_cancels(temp_env):
    """Verifies that declining approval prevents operation execution."""
    assistant = temp_env["assistant"]
    ws = temp_env["workspace_dir"]
    target_dir = ws / "secret_folder"

    op, val, exc = assistant.process_query(
        "create directory secret_folder",
        approved=False,
    )

    assert op.risk_level == RiskLevel.MEDIUM
    assert exc is not None
    assert exc.success is False
    assert exc.execution_status == ExecutionStatus.CANCELLED
    assert exc.was_cancelled is True
    assert not target_dir.exists()


# ------------------------------------------------------------------------------
# Test 9: HIGH-Risk Deletion Requires Double Confirmation
# ------------------------------------------------------------------------------
def test_09_high_risk_deletion_double_confirmation(temp_env):
    """Verifies that file deletion requires both approval=True and deletion_token='DELETE'."""
    assistant = temp_env["assistant"]
    ws = temp_env["workspace_dir"]
    target_file = ws / "delete_me.txt"
    target_file.write_text("temporary data", encoding="utf-8")
    assert target_file.exists()

    op, val, exc = assistant.process_query(
        "delete file delete_me.txt",
        approved=True,
        deletion_token="DELETE",
    )

    assert op.risk_level == RiskLevel.HIGH
    assert val.requires_double_confirmation is True
    assert exc is not None
    assert exc.success is True
    assert exc.execution_status == ExecutionStatus.SUCCESS
    assert not target_file.exists()


# ------------------------------------------------------------------------------
# Test 10: Invalid Deletion Confirmation Prevents Execution
# ------------------------------------------------------------------------------
def test_10_invalid_deletion_token_cancels(temp_env):
    """Verifies that wrong deletion token cancels deletion even if approved=True."""
    assistant = temp_env["assistant"]
    ws = temp_env["workspace_dir"]
    target_file = ws / "precious.txt"
    target_file.write_text("precious content", encoding="utf-8")

    op, val, exc = assistant.process_query(
        "delete file precious.txt",
        approved=True,
        deletion_token="yes",  # Wrong token! Must be 'DELETE'
    )

    assert exc is not None
    assert exc.success is False
    assert exc.execution_status == ExecutionStatus.CANCELLED
    assert exc.was_cancelled is True
    assert target_file.exists()  # File must remain untouched


# ------------------------------------------------------------------------------
# Test 11: UNKNOWN Intent Does Not Execute
# ------------------------------------------------------------------------------
def test_11_unknown_intent_does_not_execute(temp_env):
    """Verifies that unsupported or out-of-scope requests do not execute."""
    assistant = temp_env["assistant"]
    op, val, exc = assistant.process_query("write me a poem about butterflies")

    assert op is not None
    assert op.operation_type == OperationType.UNKNOWN
    assert val is None
    assert exc is None


# ------------------------------------------------------------------------------
# Test 12: Missing Parameters Trigger Clarification
# ------------------------------------------------------------------------------
def test_12_missing_parameters_trigger_clarification(temp_env):
    """Verifies that commands with missing arguments trigger clarification without executing."""
    assistant = temp_env["assistant"]

    # "move file" without source or destination
    op, val, exc = assistant.process_query("move file")

    assert op is not None
    assert op.generation_status == GenerationStatus.NEEDS_CLARIFICATION
    assert len(op.missing_parameters) > 0
    assert val is None
    assert exc is None


# ------------------------------------------------------------------------------
# Test 13: History Displays Existing Records
# ------------------------------------------------------------------------------
def test_13_history_displays_records(temp_env):
    """Verifies that command executions are logged and rendered in history."""
    assistant = temp_env["assistant"]

    # Run two queries
    assistant.process_query("show current directory")
    records = assistant.history_manager.get_history()
    assert len(records) >= 1

    # Render history via UI
    assistant.ui.show_history(records)
    output = temp_env["output_io"].getvalue()
    assert "CURRENT_DIRECTORY" in output or "GET_LOCATION" in output


# ------------------------------------------------------------------------------
# Test 14: Empty or Corrupted History Does Not Crash Application
# ------------------------------------------------------------------------------
def test_14_corrupted_history_safe(temp_env):
    """Verifies that corrupted history files do not crash the application."""
    assistant = temp_env["assistant"]
    hist_file = temp_env["history_file"]

    # Write corrupt data
    hist_file.write_text("{invalid json", encoding="utf-8")

    # Must recover safely
    records = assistant.history_manager.get_history()
    assert records == []

    # Display empty history safely
    assistant.ui.show_history([])
    output = temp_env["output_io"].getvalue()
    assert "No execution history records found" in output


# ------------------------------------------------------------------------------
# Test 15: $help, $history, $clear, and $exit Work
# ------------------------------------------------------------------------------
def test_15_built_in_terminal_commands(temp_env):
    """Verifies that built-in commands ($help, $history, $clear, $exit) execute safely."""
    assistant = temp_env["assistant"]

    # Test show_help
    assistant.ui.show_help(assistant.workspace_dir)

    # Test show_history
    assistant.ui.show_history([])

    # Test clear_screen (mocked os.system)
    with patch("os.system") as mock_os:
        assistant.ui.clear_screen()
        mock_os.assert_called_once()

    # Test show_exit
    assistant.ui.show_exit()

    output = temp_env["output_io"].getvalue()
    assert "Supported Interactive Commands" in output
    assert "Exiting AI Terminal Assistant" in output


# ------------------------------------------------------------------------------
# Test 16: EOF and KeyboardInterrupt Handled Gracefully
# ------------------------------------------------------------------------------
def test_16_eof_and_interrupt_handling(temp_env):
    """Verifies that EOF and KeyboardInterrupt during prompts do not crash the app."""
    ui = temp_env["assistant"].ui

    # 1. EOF in prompt_user returns '$exit'
    with patch("builtins.input", side_effect=EOFError):
        res = ui.prompt_user()
        assert res == "$exit"

    # 2. KeyboardInterrupt in prompt_user returns '$exit'
    with patch("builtins.input", side_effect=KeyboardInterrupt):
        res = ui.prompt_user()
        assert res == "$exit"

    # 3. EOF in prompt_confirmation defaults to False
    with patch("builtins.input", side_effect=EOFError):
        approved = ui.prompt_confirmation("CREATE_DIR", RiskLevel.MEDIUM)
        assert approved is False

    # 4. KeyboardInterrupt in prompt_deletion_token returns empty string
    with patch("builtins.input", side_effect=KeyboardInterrupt):
        token = ui.prompt_deletion_token("secret.txt")
        assert token == ""


# ------------------------------------------------------------------------------
# Test 17: Execution Errors Displayed Without Crashing
# ------------------------------------------------------------------------------
def test_17_execution_errors_handled_gracefully(temp_env):
    """Verifies that runtime execution exceptions (e.g. PermissionError) are handled safely."""
    assistant = temp_env["assistant"]

    # Mock an execution failure
    with patch.object(
        assistant.executor,
        "_exec_get_location",
        side_effect=PermissionError("Simulated permission denied"),
    ):
        op, val, exc = assistant.process_query("show current directory")
        assert exc is not None
        assert exc.success is False
        assert exc.execution_status == ExecutionStatus.PERMISSION_DENIED
        assert "Simulated permission denied" in exc.stderr


# ------------------------------------------------------------------------------
# Test 18: Application Never Executes powershell_preview Directly
# ------------------------------------------------------------------------------
def test_18_never_executes_powershell_preview_directly(temp_env):
    """Verifies that execution never invokes shell=True or executes the powershell_preview string."""
    assistant = temp_env["assistant"]

    with patch("subprocess.run") as mock_subproc:
        # File operations must execute purely through Python standard library handlers
        op, val, exc = assistant.process_query(
            "create file secure_file.txt",
            approved=True,
        )

        assert exc is not None
        assert exc.success is True
        # subprocess.run must NOT have been called for file operations
        mock_subproc.assert_not_called()
        assert (temp_env["workspace_dir"] / "secure_file.txt").exists()
