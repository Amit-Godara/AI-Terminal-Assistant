"""Tests for project initialization, module imports, and configuration."""

import pytest
from pathlib import Path


def test_config_paths_exist() -> None:
    """Verify that configuration paths are valid and created."""
    import config
    assert config.BASE_DIR.exists()
    assert config.DATA_DIR.exists()
    assert config.MODELS_DIR.exists()
    assert len(config.SUPPORTED_INTENTS) == 12


def test_core_imports() -> None:
    """Verify that all core modules import without error."""
    from core.preprocessor import TextPreprocessor
    from core.intent_classifier import IntentClassifier
    from core.entity_extractor import EntityExtractor
    from core.semantic_engine import SemanticEngine
    from core.command_generator import CommandGenerator, GeneratedCommand
    from core.command_validator import CommandValidator
    from core.executor import CommandExecutor, ExecutionResult

    assert TextPreprocessor is not None
    assert IntentClassifier is not None
    assert EntityExtractor is not None
    assert SemanticEngine is not None
    assert CommandGenerator is not None
    assert CommandValidator is not None
    assert CommandExecutor is not None


def test_utils_imports() -> None:
    """Verify that utility modules import without error."""
    from utils.terminal_ui import TerminalUI
    from utils.history import HistoryManager
    import config

    assert TerminalUI is not None
    assert HistoryManager is not None

    hm = HistoryManager(config.COMMAND_HISTORY_PATH)
    assert isinstance(hm.get_history(), list)


def test_third_party_dependencies() -> None:
    """Verify that required third-party libraries can be imported."""
    import rich
    import pydantic
    import joblib
    import pandas
    import numpy
    import sklearn

    assert rich is not None
    assert pydantic is not None
    assert joblib is not None
    assert pandas is not None
    assert numpy is not None
    assert sklearn is not None
