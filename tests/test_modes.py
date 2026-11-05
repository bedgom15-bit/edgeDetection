import main


class FakeDetector:
    def apply(self, frame, params):
        return frame

    def close(self):
        pass


def test_mode_whose_model_cannot_load_is_skipped(monkeypatch):
    monkeypatch.setattr(main, "LOADERS", {"cerca": lambda: None, "manos": lambda: FakeDetector()})
    models = {}

    assert main.activate_mode("cerca", models) == "manos"
    assert list(models) == ["manos"]


def test_model_is_loaded_only_once(monkeypatch):
    loads = []
    monkeypatch.setattr(main, "LOADERS", {"cerca": lambda: loads.append(1) or FakeDetector()})
    models = {}

    main.activate_mode("cerca", models)
    main.activate_mode("cerca", models)

    assert len(loads) == 1


def test_next_mode_cycles_through_all_modes(monkeypatch):
    monkeypatch.setattr(main, "LOADERS", {"cerca": FakeDetector, "manos": FakeDetector})
    models = {}
    mode = "cerca"
    seen = []
    for _ in main.MODES:
        mode = main.next_mode(mode, models)
        seen.append(mode)

    assert seen == ["manos", "objetos", "bordes", "cerca"]
