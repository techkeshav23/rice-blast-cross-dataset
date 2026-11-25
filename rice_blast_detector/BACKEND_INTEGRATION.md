# Backend Integration (Optional, Offline-First)

This app runs fully **offline-first**. The FastAPI backend is a *best-effort*
enhancement used only when the device is online. Every remote call fails
gracefully (returns `null`/`false`, never throws), so nothing here can break the
core offline experience.

## What the client does

Two new files were added, plus one dependency (`http: ^1.2.2`):

- **`lib/backend_config.dart`** — configuration constants:
  - `kBackendBaseUrl` — the deployed backend URL (placeholder until you deploy).
  - `kBackendEnabled` — master on/off switch for all remote calls.
  - `kBackendTimeout` — per-request timeout (20s).
- **`lib/backend_client.dart`** — `BackendClient`, a thin wrapper over
  `package:http` implementing this API contract:

| Method | Endpoint | Returns |
| ------ | -------- | ------- |
| `checkHealth()` | `GET /health` | `bool` (`true` if `{"status":"ok"}`) |
| `fetchModelInfo()` | `GET /model-info` | `Map<String,dynamic>?` |
| `fetchMetrics()` | `GET /metrics` | `Map<String,dynamic>?` |
| `predict(File)` | `POST /predict` (multipart `image`) | `RemotePrediction?` |
| `gradcam(File)` | `POST /gradcam` (multipart `image`) | `Uint8List?` (PNG bytes) |
| `sendFeedback(...)` | `POST /feedback` (multipart) | `bool` |

`RemotePrediction` fields: `label`, `confidence`, `scores` (`Map<String,double>`),
`lowConfidence`, `version`.

## How to set `kBackendBaseUrl` after deploy

1. Deploy the FastAPI backend to a Hugging Face Space.
2. Copy the Space's public URL, e.g. `https://your-name-rice-blast.hf.space`.
3. Open `lib/backend_config.dart` and replace the placeholder:

   ```dart
   const String kBackendBaseUrl = 'https://your-name-rice-blast.hf.space';
   ```

4. Leave `kBackendEnabled = true`. To disable all remote calls (e.g. for a
   fully offline build), set it to `false`.

No trailing slash is required — the client normalizes it.

## Example wiring snippets

> These are **documentation only** — they are NOT yet wired into any screen.
> Add them when you are ready to surface backend features in the UI.

### 1. "Expert Analysis" button on the result screen (predict + Grad-CAM)

```dart
import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import '../backend_client.dart';

// Inside your result screen widget, given `File imageFile`:
Future<void> _runExpertAnalysis(BuildContext context, File imageFile) async {
  final client = BackendClient();

  // Best-effort: both may return null if offline / backend disabled.
  final prediction = await client.predict(imageFile);
  final Uint8List? heatmap = await client.gradcam(imageFile);

  if (!context.mounted) return;
  if (prediction == null) {
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(content: Text('Expert analysis unavailable offline.')),
    );
    return;
  }

  showDialog(
    context: context,
    builder: (_) => AlertDialog(
      title: Text('Expert: ${prediction.label} '
          '(${(prediction.confidence * 100).toStringAsFixed(1)}%)'),
      content: heatmap != null
          ? Image.memory(heatmap)
          : const Text('Prediction only (no heatmap).'),
    ),
  );
}

// Button:
ElevatedButton(
  onPressed: () => _runExpertAnalysis(context, imageFile),
  child: const Text('Expert Analysis'),
);
```

### 2. "Report wrong result" button (feedback)

```dart
import 'dart:io';
import 'package:flutter/material.dart';
import '../backend_client.dart';

Future<void> _reportWrongResult(
  BuildContext context,
  File imageFile,
  String predictedLabel,
  String correctLabel,
) async {
  final ok = await BackendClient().sendFeedback(
    file: imageFile,
    predictedLabel: predictedLabel,
    correctLabel: correctLabel,
    notes: 'Reported from result screen',
  );

  if (!context.mounted) return;
  ScaffoldMessenger.of(context).showSnackBar(
    SnackBar(
      content: Text(ok
          ? 'Thanks! Feedback sent.'
          : 'Could not send feedback (saved for offline use).'),
    ),
  );
}

// Button:
TextButton(
  onPressed: () =>
      _reportWrongResult(context, imageFile, predictedLabel, correctLabel),
  child: const Text('Report wrong result'),
);
```

### 3. Results/Metrics screen calling `fetchMetrics` with offline fallback

```dart
import 'package:flutter/material.dart';
import '../backend_client.dart';

// Provide sensible bundled defaults so the UI always has something to show.
const Map<String, dynamic> _offlineMetrics = {
  'primary_metric': 'accuracy',
  'baseline': {'in_set': 0.0, 'unseen': 0.0, 'gap': 0.0},
  'mixstyle': {'in_set': 0.0, 'unseen': 0.0, 'gap': 0.0},
};

Future<Map<String, dynamic>> loadMetrics() async {
  final remote = await BackendClient().fetchMetrics();
  return remote ?? _offlineMetrics; // fall back when offline / disabled.
}

// In a widget:
FutureBuilder<Map<String, dynamic>>(
  future: loadMetrics(),
  builder: (context, snapshot) {
    if (!snapshot.hasData) {
      return const Center(child: CircularProgressIndicator());
    }
    final metrics = snapshot.data!;
    final baseline = metrics['baseline'] as Map? ?? const {};
    return Text('Baseline unseen: ${baseline['unseen']}');
  },
);
```
