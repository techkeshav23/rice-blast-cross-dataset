# Rice Blast Detector (Flutter App)

Prototype Android app for the major project *"Improving Cross-Dataset Generalization
for Rice Blast Detection Under Real-Field Conditions."*

Currently uses a **mock predictor** so the full flow works before the real model is
trained. Works fully offline.

## Flow
Home → Take Photo / Choose from Gallery → Prediction + Confidence → Disease-management guidance.

## Install the APK on a phone
1. APK is at: `build/app/outputs/flutter-apk/app-release.apk`
2. Copy it to your Android phone (USB, WhatsApp-to-self, Google Drive, etc.).
3. On the phone, open the file → allow "Install from unknown sources" → Install.

## Run during development
```
flutter pub get
flutter run           # on a connected phone / emulator
flutter run -d chrome # quick test in browser
```

## Build the APK again
```
flutter build apk --release            # single APK (~43 MB)
flutter build apk --split-per-abi      # smaller per-architecture APKs
```

## Project structure
```
lib/
  main.dart          # app + Home screen + image pick + About
  predictor.dart     # Prediction model + MockPredictor  <-- swap point
  guidance.dart      # loads offline disease guidance
  result_screen.dart # result UI (prediction, confidence, guidance)
assets/
  guidance.json      # offline Blast/Brown Spot/Healthy management text
```

## Plug in the REAL model later (one place)
1. Train MobileNetV3/EfficientNet in Python, convert to `model.tflite` (+ INT8 quantize).
2. Put `model.tflite` and `labels.txt` in `assets/` and list them in `pubspec.yaml`.
3. Add `tflite_flutter` to `pubspec.yaml`.
4. Create `TFLitePredictor implements Predictor` (in `predictor.dart`): load the model,
   preprocess the image (resize 224×224, **same normalization as training**), run inference,
   return a `Prediction`.
5. In `main.dart`, change `final Predictor _predictor = MockPredictor();`
   to `TFLitePredictor()`. **Nothing else changes.**

⚠️ Preprocessing must match training exactly (resize, normalization, RGB order, orientation)
— this is the most common integration bug.

## Phase 2
- Grad-CAM heatmap (backend FastAPI or on-device).
- Verified IRRI/ICAR management content review.
