"""Recompute Table IV per-class recall for the CANOPY cells on the DE-DUPLICATED
Philippines target (reviewer: Table IV must be regenerated after removing 32 images).
Single seed (s42) to match the 'illustrative single seed' framing. No-adapt -> AdaBN.
Prints the two canopy rows (MobileNetV3, EfficientNet); single-leaf rows are unaffected.
"""
import os
import config

CELLS = [("MobileNetV3", "mobilenet_v3_small"), ("EfficientNet", "efficientnet_b0")]
SEED = 42


def main():
    import torch
    from models import build_model
    from dg import adabn_adapt
    import train as T

    os.environ["RICE_SOURCE_DOMAIN"] = "paddy_doctor"
    os.environ["RICE_TARGET_DOMAIN"] = "philippines"
    os.environ["RICE_EXTERNAL_EXCLUDE"] = str(config.OUTPUTS / "canopy_leak_exclude.txt")
    import importlib; importlib.reload(config)
    import train as T2; importlib.reload(T2)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, _, _, ext = T2._loaders(64)

    def recall(net):
        m = T2._metrics(*T2._predict(net, ext, device))
        r = m["per_class_recall"]
        return {c: round(r[c]) for c in config.CLASSES}

    print("de-dup canopy per-class recall (no-adapt -> AdaBN), seed 42\n")
    for disp, model_name in CELLS:
        ck = config.OUTPUTS / f"baseline_{model_name}_paddy_doctor-philippines_s{SEED}_{model_name}.pt"
        net = build_model(model_name, config.NUM_CLASSES, pretrained=False)
        net.load_state_dict(torch.load(ck, map_location="cpu")); net = net.to(device).eval()
        r0 = recall(net)
        adabn_adapt(net, ext, device)
        r1 = recall(net)
        cells = " & ".join(f"${r0[c]}{{\\to}}{r1[c]}$" for c in config.CLASSES)
        print(f"{disp:12s} & Canopy & {cells} \\\\")
        print(f"   raw: Blast {r0['Rice Blast']}->{r1['Rice Blast']}  "
              f"BSpot {r0['Brown Spot']}->{r1['Brown Spot']}  "
              f"Healthy {r0['Healthy']}->{r1['Healthy']}")


if __name__ == "__main__":
    os.environ.setdefault("RICE_NUM_WORKERS", "0"); os.environ.setdefault("RICE_CACHE", "1")
    main()
