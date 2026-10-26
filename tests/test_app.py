from pathlib import Path
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def test_app_reacts_to_inputs_without_retraining(monkeypatch):
    from sklearn.pipeline import Pipeline
    def forbidden_fit(*args, **kwargs):
        raise AssertionError("The website must not retrain models")
    monkeypatch.setattr(Pipeline, "fit", forbidden_fit)
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    assert not app.exception
    assert len(app.tabs) == 3
    initial = app.metric[0].value
    app.radio[0].set_value("pass").run()
    assert not app.exception
    assert app.metric[0].value == app.metric[2].value
    assert app.metric[0].value != initial
    app.slider(key="field_position").set_value(99).run()
    assert not app.exception
    assert app.number_input(key="distance").value == 1
    minutes = next(widget for widget in app.number_input if widget.label == "Minutes")
    minutes.set_value(15).run()
    assert not app.exception
    assert app.number_input(key="seconds").value == 0
    assert app.number_input(key="seconds").disabled
