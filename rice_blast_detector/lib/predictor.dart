import 'dart:io';
import 'dart:math';

/// Result of a single classification.
class Prediction {
  final String label;
  final double confidence; // 0..1
  final Map<String, double> allScores;

  const Prediction({
    required this.label,
    required this.confidence,
    required this.allScores,
  });
}

/// Common interface. Swap the implementation (mock -> real TFLite)
/// without changing any UI code.
abstract class Predictor {
  Future<Prediction> predict(File image);
}

/// TEMPORARY mock predictor.
///
/// Returns a random class + confidence among the project's 3 classes so the
/// full app flow works today, before the real model is trained.
///
/// >>> To use the real model later: create a TFLitePredictor that implements
/// >>> Predictor, load assets/model.tflite, run inference, and return a
/// >>> Prediction. Only main.dart's `predictor` variable needs to change.
class MockPredictor implements Predictor {
  static const List<String> labels = ['Rice Blast', 'Brown Spot', 'Healthy'];
  final Random _rng = Random();

  @override
  Future<Prediction> predict(File image) async {
    // Simulate a little processing time so the loading UI is visible.
    await Future.delayed(const Duration(milliseconds: 700));

    // Random raw scores, then softmax-like normalisation.
    final raw = List<double>.generate(labels.length, (_) => _rng.nextDouble());
    final sum = raw.fold<double>(0, (a, b) => a + b);
    final scores = raw.map((v) => v / sum).toList();

    int best = 0;
    for (int i = 1; i < scores.length; i++) {
      if (scores[i] > scores[best]) best = i;
    }

    final allScores = <String, double>{
      for (int i = 0; i < labels.length; i++) labels[i]: scores[i],
    };

    return Prediction(
      label: labels[best],
      confidence: scores[best],
      allScores: allScores,
    );
  }
}
