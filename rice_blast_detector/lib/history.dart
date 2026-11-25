import 'dart:convert';
import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:path_provider/path_provider.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'predictor.dart';

/// One saved scan (persisted).
class ScanRecord {
  final String imagePath;
  final String label;
  final double confidence;
  final DateTime time;

  ScanRecord({
    required this.imagePath,
    required this.label,
    required this.confidence,
    required this.time,
  });

  File get image => File(imagePath);
  Prediction get prediction =>
      Prediction(label: label, confidence: confidence, allScores: {label: confidence});

  Map<String, dynamic> toJson() => {
        'p': imagePath,
        'l': label,
        'c': confidence,
        't': time.millisecondsSinceEpoch,
      };

  factory ScanRecord.fromJson(Map<String, dynamic> j) => ScanRecord(
        imagePath: j['p'] as String,
        label: j['l'] as String,
        confidence: (j['c'] as num).toDouble(),
        time: DateTime.fromMillisecondsSinceEpoch(j['t'] as int),
      );
}

/// Persistent scan history (survives app restarts).
class History {
  static const _key = 'scan_history_v1';
  static final ValueNotifier<List<ScanRecord>> records =
      ValueNotifier<List<ScanRecord>>([]);

  /// Load saved records; drop any whose image file no longer exists.
  static Future<void> load() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      final raw = prefs.getStringList(_key) ?? [];
      final list = raw
          .map((s) => ScanRecord.fromJson(
              jsonDecode(s) as Map<String, dynamic>))
          .where((r) => r.image.existsSync())
          .toList();
      records.value = list;
    } catch (_) {
      records.value = [];
    }
  }

  /// Copy the picked image into app storage and save the record.
  static Future<void> add(File image, Prediction p) async {
    try {
      final dir = await getApplicationDocumentsDirectory();
      final stamp = DateTime.now().millisecondsSinceEpoch;
      final dest = File('${dir.path}/scan_$stamp.jpg');
      await image.copy(dest.path);
      final rec = ScanRecord(
        imagePath: dest.path,
        label: p.label,
        confidence: p.confidence,
        time: DateTime.now(),
      );
      records.value = [rec, ...records.value];
      await _persist();
    } catch (_) {
      // Non-fatal: still show the result even if saving failed.
    }
  }

  static Future<void> clear() async {
    for (final r in records.value) {
      try {
        if (r.image.existsSync()) r.image.deleteSync();
      } catch (_) {}
    }
    records.value = [];
    final prefs = await SharedPreferences.getInstance();
    await prefs.remove(_key);
  }

  static Future<void> _persist() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setStringList(
        _key, records.value.map((r) => jsonEncode(r.toJson())).toList());
  }
}
