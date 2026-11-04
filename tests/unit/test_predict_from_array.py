import numpy as np
from pytest import MonkeyPatch

import yolo_rgbd.prediction as prediction


def test_load_yolo_model_uses_model_path(monkeypatch: MonkeyPatch) -> None:
    class FakeYOLO:
        def __init__(self, path: str) -> None:
            self.path = path

    monkeypatch.setattr('ultralytics.YOLO', FakeYOLO)

    model = prediction.load_yolo_model('model.pt')

    assert isinstance(model, FakeYOLO)
    assert model.path == 'model.pt'


def test_predict_from_array_merges_multiple_models(monkeypatch: MonkeyPatch) -> None:
    image = np.zeros((20, 20, 4), dtype=np.uint8)
    calls = []

    def fake_predict_instances(
        model: object,
        source: np.ndarray,
        confidence: float,
    ) -> tuple[list[np.ndarray], list[float], list[int]]:
        calls.append((model, source.shape, confidence))
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[2:12, 2:12] = 1
        return [mask], [0.8], [1]

    monkeypatch.setattr(prediction, 'predict_instances', fake_predict_instances)

    instances, semantics, confidences = prediction.predict_from_array(
        [object(), object()],
        image,
    )

    assert len(calls) == 2
    assert all(call[1] == (20, 20, 4) for call in calls)
    assert instances.max() == 1
    assert semantics[5, 5] == 2
    assert np.isclose(confidences[5, 5], 0.8)


def test_predict_from_array_selects_rgb_channels(monkeypatch: MonkeyPatch) -> None:
    image = np.zeros((20, 20, 4), dtype=np.uint8)
    received = []

    def fake_predict_instances(
        model: object,
        source: np.ndarray,
        confidence: float,
    ) -> tuple[list[np.ndarray], list[float], list[int]]:
        received.append(source)
        return [np.zeros((20, 20), dtype=np.uint8)], [0.1], [1]

    monkeypatch.setattr(prediction, 'predict_instances', fake_predict_instances)

    prediction.predict_from_array(object(), image, pred_mode='rgb')

    assert received[0].shape == (20, 20, 3)


def test_predict_instances_injects_model() -> None:
    class FakeModel:
        def __init__(self) -> None:
            self.sources = []

        def predict(self, source: np.ndarray, conf: float) -> list:
            self.sources.append((source, conf))
            return [type('Result', (), {'masks': None})()]

    model = FakeModel()
    image = np.zeros((4, 5, 4), dtype=np.uint8)

    results = prediction.predict_instances(model, image)

    assert isinstance(results, tuple)
    assert len(model.sources) == 1
