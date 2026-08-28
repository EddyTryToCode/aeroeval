"""
Unit tests for ModelRegistry and ModelRunner.
"""

from aeroeval.models.registry import ModelRegistry


def test_model_registry_crud(tmp_path):
    registry = ModelRegistry()
    assert len(registry) == 0

    # Create dummy weight file
    weight_file = tmp_path / "dummy.pt"
    weight_file.write_text("dummy")

    # Register
    registry.register(
        name="test_model",
        path=weight_file,
        format="PyTorch",
        imgsz=640,
        description="A test model"
    )

    assert len(registry) == 1
    assert registry.get("test_model") is not None
    assert registry.get("test_model").name == "test_model"

    # Compare DataFrame
    df = registry.compare()
    assert len(df) == 1
    assert df.iloc[0]["Model Name"] == "test_model"

    # Remove
    assert registry.remove("test_model") is True
    assert len(registry) == 0
