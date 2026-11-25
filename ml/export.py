"""Phase 5 — export a trained checkpoint for deployment.

- ONNX: reliable, PyTorch-only. Used by the backend and as the bridge to TFLite.
- TFLite (for the on-device app): convert the ONNX with `onnx2tf` (needs
  tensorflow). Command is printed at the end; kept as a separate step because
  tensorflow is a heavy dependency.

Usage:
  python export.py --tag mixstyle --dg mixstyle
"""
import argparse

import torch

import config
from dg import to_mixstyle
from models import build_model


def load_model(tag, model_name, dg):
    net = build_model(model_name, config.NUM_CLASSES, pretrained=False)
    if dg == "mixstyle":
        net = to_mixstyle(net)
    state = torch.load(config.OUTPUTS / f"{tag}_{model_name}.pt", map_location="cpu")
    net.load_state_dict(state)
    net.eval()
    return net


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="mixstyle")
    ap.add_argument("--model", default="mobilenet_v3_small")
    ap.add_argument("--dg", choices=["none", "mixstyle"], default="mixstyle")
    args = ap.parse_args()

    net = load_model(args.tag, args.model, args.dg)
    dummy = torch.randn(1, 3, config.IMG_SIZE, config.IMG_SIZE)

    onnx_path = config.OUTPUTS / f"{args.tag}_{args.model}.onnx"
    torch.onnx.export(
        net, dummy, str(onnx_path),
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17, dynamo=False, verbose=False,
    )
    print(f"Exported ONNX -> {onnx_path.relative_to(config.ROOT)}")

    # optional sanity check
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        out = sess.run(None, {"input": dummy.numpy()})[0]
        print(f"ONNX runtime check OK — output shape {out.shape}")
    except Exception as e:
        print(f"(onnxruntime check skipped: {e})")

    print("\nTo produce model.tflite for the on-device app (needs tensorflow):")
    print("  pip install onnx2tf tensorflow")
    print(f"  onnx2tf -i {onnx_path.name} -o tflite_out -oiqt   # -oiqt = INT8 quant")
    print("  # then copy tflite_out/*_float32.tflite (or the INT8 file) into")
    print("  #   rice_blast_detector/assets/model.tflite")


if __name__ == "__main__":
    main()
