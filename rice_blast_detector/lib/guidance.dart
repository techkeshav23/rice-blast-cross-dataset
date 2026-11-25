import 'dart:convert';
import 'package:flutter/services.dart' show rootBundle;

import 'i18n.dart';

/// Disease-management guidance for one class (already localized).
class Guidance {
  final String summary;
  final List<String> symptoms;
  final List<String> management;
  final String severity;

  const Guidance({
    required this.summary,
    required this.symptoms,
    required this.management,
    required this.severity,
  });
}

class _Entry {
  final String severity;
  final Map<String, String> name; // lang -> display name
  final Map<String, Guidance> byLang; // lang -> content
  _Entry(this.severity, this.name, this.byLang);
}

/// Loads localized guidance from the bundled offline JSON.
class GuidanceStore {
  static Map<String, _Entry>? _cache;
  static Map<String, String> _note = const {};

  static String get disclaimer =>
      _note[I18n.lang.value] ?? _note['en'] ?? '';

  static Future<void> load() async {
    if (_cache != null) return;
    final raw = await rootBundle.loadString('assets/guidance.json');
    final data = jsonDecode(raw) as Map<String, dynamic>;
    _note = _strMap(data['note']);
    final classes = (data['classes'] as Map).cast<String, dynamic>();
    _cache = classes.map((key, value) {
      final m = value as Map<String, dynamic>;
      final severity = m['severity'] as String? ?? 'unknown';
      final name = _strMap(m['name']);
      final byLang = <String, Guidance>{};
      for (final lang in ['en', 'hi']) {
        final c = m[lang];
        if (c is Map) {
          byLang[lang] = Guidance(
            summary: c['summary'] as String? ?? '',
            symptoms: _list(c['symptoms']),
            management: _list(c['management']),
            severity: severity,
          );
        }
      }
      return MapEntry(key, _Entry(severity, name, byLang));
    });
  }

  static Map<String, String> _strMap(dynamic v) =>
      (v is Map) ? v.map((k, val) => MapEntry(k.toString(), val.toString())) : {};

  static List<String> _list(dynamic v) =>
      (v is List) ? v.map((e) => e.toString()).toList() : [];

  static Guidance? forLabel(String label) {
    final e = _cache?[label];
    if (e == null) return null;
    return e.byLang[I18n.lang.value] ?? e.byLang['en'];
  }

  /// Never-null, localized lookup with a safe fallback for unknown labels.
  static Guidance forLabelSafe(String label) =>
      forLabel(label) ??
      Guidance(
        summary: I18n.lang.value == 'hi'
            ? 'यह स्थिति अभी ऑफ़लाइन गाइड में नहीं है। पुष्टि और सलाह के लिए स्थानीय कृषि विशेषज्ञ से संपर्क करें।'
            : 'This condition is not in the offline guide yet. Please consult a local agricultural expert for confirmation and advice.',
        symptoms: [
          I18n.lang.value == 'hi' ? 'ऑफ़लाइन उपलब्ध नहीं।' : 'Not available offline.'
        ],
        management: [
          I18n.lang.value == 'hi'
              ? 'कृषि विस्तार अधिकारी से सलाह लें।'
              : 'Consult an agricultural extension officer.'
        ],
        severity: 'unknown',
      );

  /// Localized display name for a class label (falls back to the label).
  static String displayName(String label) {
    final e = _cache?[label];
    if (e == null) return label;
    return e.name[I18n.lang.value] ?? e.name['en'] ?? label;
  }

  /// Internal class labels in a stable display order.
  static List<String> labels() {
    const order = ['Rice Blast', 'Brown Spot', 'Healthy'];
    final keys = _cache?.keys.toList() ?? [];
    keys.sort((a, b) {
      final ia = order.indexOf(a), ib = order.indexOf(b);
      return (ia == -1 ? 99 : ia).compareTo(ib == -1 ? 99 : ib);
    });
    return keys;
  }
}
