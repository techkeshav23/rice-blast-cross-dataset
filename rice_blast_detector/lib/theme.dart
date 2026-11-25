import 'package:flutter/material.dart';
import 'i18n.dart';

/// Design language: clean, modern, minimal. White canvas, generous space,
/// near-black ink, one green accent. Fonts (Sora + Manrope) are bundled as
/// assets, so the app is fully offline — no runtime font fetching.
const String _display = 'Sora';
const String _body = 'Manrope';

class AppColors {
  static const bg = Color(0xFFFFFFFF); // canvas
  static const surface = Color(0xFFF5F5F1); // subtle filled cards
  static const surfaceAlt = Color(0xFFFBFBF9);
  static const ink = Color(0xFF15160F); // primary text (warm near-black)
  static const inkSoft = Color(0xFF54564C); // secondary text (WCAG AA on white)
  static const line = Color(0xFFE7E7E1); // hairline borders
  static const accent = Color(0xFF1E7A4D); // single green accent
  static const accentSoft = Color(0xFFE7F1EB); // accent tint
  static const blast = Color(0xFFA5301B); // high severity (AA on white)
  static const spot = Color(0xFF8A5A12); // medium (AA on white)
  static const healthy = Color(0xFF1E7A4D); // none
}

Color severityColor(String severity) {
  switch (severity) {
    case 'high':
      return AppColors.blast;
    case 'medium':
      return AppColors.spot;
    case 'none':
      return AppColors.healthy;
    default:
      return AppColors.inkSoft; // unknown reads neutral, not "healthy" green
  }
}

String severityLabel(String severity) {
  switch (severity) {
    case 'high':
      return tr('sev_high');
    case 'medium':
      return tr('sev_medium');
    case 'none':
      return tr('sev_none');
    default:
      return tr('sev_unknown');
  }
}

/// Severity conveyed by SHAPE too, not colour alone (colour-blind / glare safe).
IconData severityIcon(String severity) {
  switch (severity) {
    case 'high':
      return Icons.warning_amber_rounded;
    case 'medium':
      return Icons.error_outline_rounded;
    case 'none':
      return Icons.check_circle_outline_rounded;
    default:
      return Icons.help_outline_rounded;
  }
}

class AppTheme {
  static ThemeData build() {
    final base = ThemeData(
      useMaterial3: true,
      fontFamily: _body,
      scaffoldBackgroundColor: AppColors.bg,
      colorScheme: ColorScheme.fromSeed(
        seedColor: AppColors.accent,
        primary: AppColors.accent,
        surface: AppColors.bg,
        brightness: Brightness.light,
      ),
    );

    TextStyle head(double s, FontWeight w, {double ls = -0.4}) => TextStyle(
        fontFamily: _display,
        fontSize: s,
        fontWeight: w,
        color: AppColors.ink,
        letterSpacing: ls,
        height: 1.08);
    TextStyle bodyf(double s,
            {Color c = AppColors.ink, FontWeight w = FontWeight.w400}) =>
        TextStyle(
            fontFamily: _body,
            fontSize: s,
            color: c,
            height: 1.45,
            fontWeight: w);

    return base.copyWith(
      textTheme: base.textTheme.copyWith(
        displayLarge: head(34, FontWeight.w700, ls: -0.8),
        displaySmall: head(26, FontWeight.w700, ls: -0.5),
        headlineSmall: head(21, FontWeight.w600),
        titleLarge: bodyf(16, w: FontWeight.w700),
        titleMedium: bodyf(14.5, w: FontWeight.w700),
        bodyLarge: bodyf(15.5),
        bodyMedium: bodyf(14, c: AppColors.inkSoft),
      ),
    );
  }

  static TextStyle overline(Color color) => const TextStyle(
        fontFamily: _body,
      ).copyWith(
        fontSize: 12,
        fontWeight: FontWeight.w800,
        letterSpacing: 1.6,
        color: color,
      );

  static TextStyle head(
          {double size = 26,
          FontWeight weight = FontWeight.w700,
          Color color = AppColors.ink,
          double ls = -0.5,
          double height = 1.08}) =>
      TextStyle(
          fontFamily: _display,
          fontSize: size,
          fontWeight: weight,
          color: color,
          letterSpacing: ls,
          height: height);
}
