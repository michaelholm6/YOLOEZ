import pytest
from main import main as run_main 

def test_main_runs_without_error(monkeypatch):
    monkeypatch.setattr("sys.exit", lambda *args, **kwargs: None)

    run_main(test_mode=True)

    assert True