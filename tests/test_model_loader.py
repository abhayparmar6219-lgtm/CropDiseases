from app import load_model_and_labels


def test_model_and_labels_load():
    model, labels = load_model_and_labels()

    assert model is not None
    assert len(labels) > 0
