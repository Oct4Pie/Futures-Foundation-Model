from pathlib import Path


def test_extended_pilot_imports_as_package_and_adapter_seeds_before_load(monkeypatch):
    import torch

    from scripts import pilot_extended_native_route as pilot
    from scripts import smoke_extended_native_route as smoke

    assert pilot.Adapter is smoke.Adapter
    route_key = "mantis_v1:R:official_crop_resize_contrastive"
    captured = []

    def fake_load_route(
        requested_route_key,
        *,
        model_snapshot,
        source_runtime,
        device,
        n_classes,
    ):
        assert requested_route_key == route_key
        captured.append(torch.rand(4))
        backbone = torch.nn.Linear(2, 2)
        return smoke.mantis_native.LoadedRoute(
            route_key=route_key,
            backbone=backbone,
            head=None,
            identity={"fixture": True},
        )

    monkeypatch.setattr(smoke.mantis_native, "load_route", fake_load_route)
    paths = {
        "model_snapshot": Path("model"),
        "source_runtime": Path("source"),
    }
    first = smoke.Adapter(route_key, paths, "cpu", 20260718)
    second = smoke.Adapter(route_key, paths, "cpu", 20260718)

    assert torch.equal(captured[0], captured[1])
    for name, value in first.initial_modules["backbone"].items():
        assert torch.equal(value, second.initial_modules["backbone"][name])
