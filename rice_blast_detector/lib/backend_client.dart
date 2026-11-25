import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:http/http.dart' as http;

import 'backend_config.dart';

/// A single prediction returned by the backend `/predict` endpoint.
///
/// All numeric fields are coerced to [double] defensively so that integer JSON
/// values (e.g. `1`) do not cause a type error at runtime.
class RemotePrediction {
  const RemotePrediction({
    required this.label,
    required this.confidence,
    required this.scores,
    required this.lowConfidence,
    required this.version,
  });

  /// The predicted class label (e.g. `"blast"`).
  final String label;

  /// Confidence of the top prediction, in the range `[0.0, 1.0]`.
  final double confidence;

  /// Per-class confidence scores keyed by label.
  final Map<String, double> scores;

  /// Whether the backend flagged this result as low confidence.
  final bool lowConfidence;

  /// Model / API version string reported by the backend.
  final String version;

  /// Builds a [RemotePrediction] from a decoded JSON map.
  ///
  /// Numbers are coerced to [double] via [_toDouble] to tolerate ints.
  factory RemotePrediction.fromJson(Map<String, dynamic> json) {
    final rawScores = json['scores'];
    final scores = <String, double>{};
    if (rawScores is Map) {
      rawScores.forEach((key, value) {
        scores['$key'] = _toDouble(value);
      });
    }
    return RemotePrediction(
      label: '${json['label'] ?? ''}',
      confidence: _toDouble(json['confidence']),
      scores: scores,
      lowConfidence: json['lowConfidence'] == true ||
          json['low_confidence'] == true,
      version: '${json['version'] ?? ''}',
    );
  }

  /// Safely coerces a dynamic JSON value into a [double].
  static double _toDouble(dynamic value) {
    if (value is num) return value.toDouble();
    if (value is String) return double.tryParse(value) ?? 0.0;
    return 0.0;
  }
}

/// Best-effort HTTP client for the optional FastAPI backend.
///
/// Every method is defensive: it returns `null`/`false` on any error, timeout,
/// or non-200 response and NEVER throws to the caller. This keeps the
/// offline-first app resilient when the backend is disabled or unreachable.
///
/// Base URL and behaviour are controlled by `backend_config.dart`
/// ([kBackendBaseUrl], [kBackendEnabled], [kBackendTimeout]).
class BackendClient {
  /// Creates a client. Optionally override [baseUrl] (defaults to
  /// [kBackendBaseUrl]) — useful for tests or staging environments.
  BackendClient({String? baseUrl})
      : _baseUrl = (baseUrl ?? kBackendBaseUrl).replaceAll(RegExp(r'/+$'), '');

  final String _baseUrl;

  Uri _uri(String path) => Uri.parse('$_baseUrl$path');

  /// Pings `GET /health`.
  ///
  /// Returns `true` only when the backend is enabled and responds with
  /// `{"status":"ok"}`. Returns `false` on any error or if disabled.
  Future<bool> checkHealth() async {
    if (!kBackendEnabled) return false;
    try {
      final response =
          await http.get(_uri('/health')).timeout(kBackendTimeout);
      if (response.statusCode != 200) return false;
      final body = jsonDecode(response.body);
      return body is Map && body['status'] == 'ok';
    } catch (_) {
      return false;
    }
  }

  /// Fetches `GET /model-info`.
  ///
  /// Returns the decoded JSON map, or `null` on any error / non-200 / disabled.
  Future<Map<String, dynamic>?> fetchModelInfo() async {
    if (!kBackendEnabled) return null;
    try {
      final response =
          await http.get(_uri('/model-info')).timeout(kBackendTimeout);
      if (response.statusCode != 200) return null;
      final body = jsonDecode(response.body);
      return body is Map<String, dynamic> ? body : null;
    } catch (_) {
      return null;
    }
  }

  /// Fetches `GET /metrics`.
  ///
  /// Expected shape:
  /// ```json
  /// {
  ///   "primary_metric": "...",
  ///   "baseline":  {"in_set": ..., "unseen": ..., "gap": ...},
  ///   "mixstyle":  {"in_set": ..., "unseen": ..., "gap": ...}
  /// }
  /// ```
  /// Returns the decoded map, or `null` on any error / non-200 / disabled.
  Future<Map<String, dynamic>?> fetchMetrics() async {
    if (!kBackendEnabled) return null;
    try {
      final response =
          await http.get(_uri('/metrics')).timeout(kBackendTimeout);
      if (response.statusCode != 200) return null;
      final body = jsonDecode(response.body);
      return body is Map<String, dynamic> ? body : null;
    } catch (_) {
      return null;
    }
  }

  /// Uploads [file] to `POST /predict` (multipart, field name `image`).
  ///
  /// Returns a [RemotePrediction], or `null` on any error / non-200 / disabled.
  Future<RemotePrediction?> predict(File file) async {
    if (!kBackendEnabled) return null;
    try {
      final request = http.MultipartRequest('POST', _uri('/predict'))
        ..files.add(await http.MultipartFile.fromPath('image', file.path));
      final streamed = await request.send().timeout(kBackendTimeout);
      final response = await http.Response.fromStream(streamed);
      if (response.statusCode != 200) return null;
      final body = jsonDecode(response.body);
      return body is Map<String, dynamic>
          ? RemotePrediction.fromJson(body)
          : null;
    } catch (_) {
      return null;
    }
  }

  /// Uploads [file] to `POST /gradcam` (multipart, field name `image`).
  ///
  /// Returns the raw PNG bytes as a [Uint8List], or `null` on any error /
  /// non-200 / disabled.
  Future<Uint8List?> gradcam(File file) async {
    if (!kBackendEnabled) return null;
    try {
      final request = http.MultipartRequest('POST', _uri('/gradcam'))
        ..files.add(await http.MultipartFile.fromPath('image', file.path));
      final streamed = await request.send().timeout(kBackendTimeout);
      final response = await http.Response.fromStream(streamed);
      if (response.statusCode != 200) return null;
      return response.bodyBytes;
    } catch (_) {
      return null;
    }
  }

  /// Submits feedback to `POST /feedback` (multipart).
  ///
  /// Sends the [file] under field `image`, plus fields `predicted_label`,
  /// `correct_label`, and optional `notes`.
  ///
  /// Returns `true` on a 200 response, otherwise `false` (also on any error /
  /// disabled).
  Future<bool> sendFeedback({
    required File file,
    required String predictedLabel,
    required String correctLabel,
    String? notes,
  }) async {
    if (!kBackendEnabled) return false;
    try {
      final request = http.MultipartRequest('POST', _uri('/feedback'))
        ..files.add(await http.MultipartFile.fromPath('image', file.path))
        ..fields['predicted_label'] = predictedLabel
        ..fields['correct_label'] = correctLabel;
      if (notes != null && notes.isNotEmpty) {
        request.fields['notes'] = notes;
      }
      final streamed = await request.send().timeout(kBackendTimeout);
      return streamed.statusCode == 200;
    } catch (_) {
      return false;
    }
  }
}
