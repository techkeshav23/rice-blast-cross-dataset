import 'dart:io';
import 'dart:math' as math;
import 'package:flutter/material.dart';
import 'i18n.dart';
import 'theme.dart';

/// Colour dot.
class Dot extends StatelessWidget {
  final Color color;
  final double size;
  const Dot(this.color, {super.key, this.size = 9});
  @override
  Widget build(BuildContext context) => Container(
        width: size,
        height: size,
        decoration: BoxDecoration(color: color, shape: BoxShape.circle),
      );
}

/// Severity pill: icon + label on a tinted background. Shape + colour (not
/// colour alone) so it stays legible for colour-blind users and in glare.
class SeverityPill extends StatelessWidget {
  final String severity;
  final double fontSize;
  const SeverityPill(this.severity, {super.key, this.fontSize = 12});
  @override
  Widget build(BuildContext context) {
    final color = severityColor(severity);
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 11, vertical: 6),
      decoration: BoxDecoration(
          color: color.withValues(alpha: 0.12),
          borderRadius: BorderRadius.circular(20)),
      child: Row(mainAxisSize: MainAxisSize.min, children: [
        Icon(severityIcon(severity), size: fontSize + 3, color: color),
        const SizedBox(width: 6),
        Text(severityLabel(severity),
            style: TextStyle(
                color: color, fontSize: fontSize, fontWeight: FontWeight.w700)),
      ]),
    );
  }
}

/// Minimal circular confidence gauge for a light background.
class ConfidenceRing extends StatelessWidget {
  final double value; // 0..1
  final Color color;
  final double size;
  final String? label;
  const ConfidenceRing(
      {super.key,
      required this.value,
      required this.color,
      this.size = 96,
      this.label});

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      width: size,
      height: size,
      child: CustomPaint(
        painter: _RingPainter(value, color),
        child: Center(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Text('${(value * 100).round()}',
                  style: AppTheme.head(
                      size: size * 0.30,
                      weight: FontWeight.w700,
                      color: AppColors.ink)),
              Text(label ?? tr('confidence'),
                  style: TextStyle(
                      fontFamily: 'Manrope',
                      fontSize: 9.5,
                      fontWeight: FontWeight.w800,
                      letterSpacing: 1.0,
                      color: color)),
            ],
          ),
        ),
      ),
    );
  }
}

class _RingPainter extends CustomPainter {
  final double value;
  final Color color;
  _RingPainter(this.value, this.color);

  @override
  void paint(Canvas canvas, Size size) {
    final center = size.center(Offset.zero);
    final radius = size.width / 2 - 5;
    const start = -math.pi / 2;

    final track = Paint()
      ..color = AppColors.line
      ..style = PaintingStyle.stroke
      ..strokeWidth = 7
      ..strokeCap = StrokeCap.round;
    canvas.drawCircle(center, radius, track);

    final arc = Paint()
      ..color = color
      ..style = PaintingStyle.stroke
      ..strokeWidth = 7
      ..strokeCap = StrokeCap.round;
    canvas.drawArc(Rect.fromCircle(center: center, radius: radius), start,
        math.pi * 2 * value.clamp(0.0, 1.0), false, arc);
  }

  @override
  bool shouldRepaint(covariant _RingPainter old) =>
      old.value != value || old.color != color;
}

/// Illustrative Grad-CAM style overlay over a leaf image. This is a DEMO
/// visualisation (not computed from a model) — clearly labelled as such.
/// When the real model lands, replace [_HeatPainter] with the model's CAM.
class GradcamView extends StatelessWidget {
  final File image;
  const GradcamView({super.key, required this.image});

  @override
  Widget build(BuildContext context) {
    return ClipRRect(
      borderRadius: BorderRadius.circular(18),
      child: Stack(
        children: [
          Image.file(image,
              height: 200,
              width: double.infinity,
              fit: BoxFit.cover,
              errorBuilder: (_, _, _) => Container(
                  height: 200,
                  color: AppColors.surface,
                  child: const Icon(Icons.image_not_supported_outlined,
                      color: AppColors.inkSoft))),
          Positioned.fill(
            child: CustomPaint(painter: _HeatPainter()),
          ),
          Positioned(
            left: 10,
            top: 10,
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 9, vertical: 5),
              decoration: BoxDecoration(
                  color: Colors.black.withValues(alpha: 0.62),
                  borderRadius: BorderRadius.circular(20)),
              child: Text(tr('gradcam_chip'),
                  style: const TextStyle(
                      color: Colors.white,
                      fontSize: 11,
                      fontWeight: FontWeight.w700)),
            ),
          ),
        ],
      ),
    );
  }
}

class _HeatPainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    void blob(Offset c, double r, Color col) {
      final paint = Paint()
        ..shader = RadialGradient(colors: [
          col.withValues(alpha: 0.55),
          col.withValues(alpha: 0.0),
        ]).createShader(Rect.fromCircle(center: c, radius: r))
        ..blendMode = BlendMode.srcOver;
      canvas.drawCircle(c, r, paint);
    }

    blob(Offset(size.width * 0.42, size.height * 0.5), size.width * 0.28,
        const Color(0xFFE53935));
    blob(Offset(size.width * 0.6, size.height * 0.4), size.width * 0.18,
        const Color(0xFFFFA000));
    blob(Offset(size.width * 0.3, size.height * 0.62), size.width * 0.12,
        const Color(0xFFFFC107));
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => false;
}

/// Reusable soft card with hairline border.
class SoftCard extends StatelessWidget {
  final Widget child;
  final EdgeInsets padding;
  final VoidCallback? onTap;
  final Color? color;
  final bool bordered;
  const SoftCard(
      {super.key,
      required this.child,
      this.padding = const EdgeInsets.all(18),
      this.onTap,
      this.color,
      this.bordered = true});

  @override
  Widget build(BuildContext context) {
    final content = Container(
      width: double.infinity,
      padding: padding,
      decoration: BoxDecoration(
        color: color ?? AppColors.bg,
        borderRadius: BorderRadius.circular(18),
        border: bordered ? Border.all(color: AppColors.line) : null,
      ),
      child: child,
    );
    if (onTap == null) return content;
    return Material(
      color: Colors.transparent,
      child: InkWell(
        borderRadius: BorderRadius.circular(18),
        onTap: onTap,
        child: content,
      ),
    );
  }
}

/// Staggered entrance: fades + lifts children in sequence on first build.
class Stagger extends StatefulWidget {
  final List<Widget> children;
  final CrossAxisAlignment crossAxisAlignment;
  const Stagger(
      {super.key,
      required this.children,
      this.crossAxisAlignment = CrossAxisAlignment.start});

  @override
  State<Stagger> createState() => _StaggerState();
}

class _StaggerState extends State<Stagger> with SingleTickerProviderStateMixin {
  late final AnimationController _c;

  @override
  void initState() {
    super.initState();
    _c = AnimationController(
        vsync: this, duration: const Duration(milliseconds: 850))
      ..forward();
  }

  @override
  void dispose() {
    _c.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final n = widget.children.length;
    return Column(
      crossAxisAlignment: widget.crossAxisAlignment,
      children: [
        for (int i = 0; i < n; i++)
          AnimatedBuilder(
            animation: _c,
            builder: (context, child) {
              final startT = (i / (n + 2)).clamp(0.0, 1.0);
              final t = Curves.easeOutCubic
                  .transform(((_c.value - startT) / (1 - startT)).clamp(0.0, 1.0));
              return Opacity(
                opacity: t,
                child: Transform.translate(
                    offset: Offset(0, 16 * (1 - t)), child: child),
              );
            },
            child: widget.children[i],
          ),
      ],
    );
  }
}
