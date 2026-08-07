from pathlib import Path
import torch


def export_torchscript_to_onnx(model_dir: str) -> Path:
    model_dir_path = Path(model_dir)

    model_path = model_dir_path / "model_torchscript.pt"
    onnx_path = model_dir_path / "model.onnx"

    if not model_path.exists():
        raise FileNotFoundError(f"Không tìm thấy model: {model_path}")

    model = torch.jit.load(str(model_path), map_location="cpu")
    model.eval()

    dummy = torch.randn(1, 3, 224, 224)

    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["input"],
        output_names=["logits"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False
    )

    return onnx_path


def main():
    model_dir = r"D:\Project Real\AZS\NAB.APP\NAB.APP\dll\model"
    onnx_path = export_torchscript_to_onnx(model_dir)
    print(f"Exported: {onnx_path}")


if __name__ == "__main__":
    main()