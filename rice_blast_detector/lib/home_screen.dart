import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';

import 'guidance.dart';
import 'history.dart';
import 'i18n.dart';
import 'library_screen.dart';
import 'predictor.dart';
import 'result_screen.dart';
import 'theme.dart';
import 'widgets.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  // Single swap point: replace MockPredictor() with TFLitePredictor() later.
  final Predictor _predictor = MockPredictor();
  final ImagePicker _picker = ImagePicker();
  bool _busy = false;

  Future<void> _scan(ImageSource source) async {
    if (_busy) return;
    try {
      final picked = await _picker.pickImage(source: source, maxWidth: 1024);
      if (picked == null) return;
      final file = File(picked.path);

      setState(() => _busy = true);
      final prediction = await _predictor.predict(file);
      await History.add(file, prediction);
      if (!mounted) return;
      setState(() => _busy = false);

      Navigator.push(
        context,
        MaterialPageRoute(
            builder: (_) => ResultScreen(image: file, prediction: prediction)),
      );
    } catch (e) {
      setState(() => _busy = false);
      if (!mounted) return;
      ScaffoldMessenger.of(context)
          .showSnackBar(SnackBar(content: Text('Error: $e')));
    }
  }

  @override
  Widget build(BuildContext context) {
    if (_busy) return const _Analysing();

    final classes = GuidanceStore.labels();
    return ListView(
      padding: const EdgeInsets.fromLTRB(20, 4, 20, 28),
      children: [
        Stagger(
          children: [
            const SizedBox(height: 4),
            Text(tr('home_eyebrow'), style: AppTheme.overline(AppColors.accent)),
            const SizedBox(height: 12),
            Text(tr('home_title'),
                style: Theme.of(context).textTheme.displayLarge),
            const SizedBox(height: 14),
            Text(tr('home_sub'), style: Theme.of(context).textTheme.bodyLarge),
            const SizedBox(height: 20),
            _HeroScanCard(
              onCamera: () => _scan(ImageSource.camera),
              onGallery: () => _scan(ImageSource.gallery),
            ),
            const SizedBox(height: 26),
            _statStrip(),
            const SizedBox(height: 30),
            Row(
              children: [
                Text(tr('detects'), style: AppTheme.overline(AppColors.inkSoft)),
                const SizedBox(width: 12),
                const Expanded(child: Divider(color: AppColors.line)),
              ],
            ),
            const SizedBox(height: 16),
            for (final label in classes) _detectRow(label),
            const SizedBox(height: 14),
            _tip(),
          ],
        ),
      ],
    );
  }

  /// Trust strip: three quick facts that make the app feel substantial.
  Widget _statStrip() {
    Widget item(IconData ic, String big, String small) => Expanded(
          child: Column(
            children: [
              Icon(ic, size: 20, color: AppColors.accent),
              const SizedBox(height: 8),
              Text(big,
                  style: AppTheme.head(size: 17, weight: FontWeight.w700)),
              const SizedBox(height: 2),
              Text(small,
                  textAlign: TextAlign.center,
                  style: const TextStyle(
                      color: AppColors.inkSoft,
                      fontSize: 11,
                      height: 1.3,
                      fontWeight: FontWeight.w600)),
            ],
          ),
        );
    return SoftCard(
      padding: const EdgeInsets.symmetric(vertical: 18, horizontal: 12),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          item(Icons.eco_rounded, '3', tr('stat_diseases')),
          _vline(),
          item(Icons.wifi_off_rounded, '100%', tr('stat_offline')),
          _vline(),
          item(Icons.bolt_rounded, '<1s', tr('stat_fast')),
        ],
      ),
    );
  }

  Widget _vline() => Container(
      width: 1, height: 46, color: AppColors.line, margin: const EdgeInsets.symmetric(horizontal: 4));

  Widget _detectRow(String label) {
    final g = GuidanceStore.forLabelSafe(label);
    final color = severityColor(g.severity);
    return Padding(
      padding: const EdgeInsets.only(bottom: 10),
      child: SoftCard(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 15),
        onTap: () => Navigator.push(context,
            MaterialPageRoute(builder: (_) => DiseaseDetailScreen(label))),
        child: Row(
          children: [
            Container(
              width: 38,
              height: 38,
              decoration: BoxDecoration(
                  color: color.withValues(alpha: 0.12),
                  borderRadius: BorderRadius.circular(10)),
              child: Icon(severityIcon(g.severity), color: color, size: 20),
            ),
            const SizedBox(width: 14),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(GuidanceStore.displayName(label),
                      style: Theme.of(context).textTheme.titleLarge),
                  const SizedBox(height: 2),
                  Text(severityLabel(g.severity),
                      style: const TextStyle(
                          color: AppColors.inkSoft, fontSize: 12.5)),
                ],
              ),
            ),
            const Icon(Icons.chevron_right_rounded, color: AppColors.inkSoft),
          ],
        ),
      ),
    );
  }

  Widget _tip() {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: AppColors.surface,
        borderRadius: BorderRadius.circular(16),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Icon(Icons.wb_sunny_outlined, size: 19, color: AppColors.accent),
          const SizedBox(width: 12),
          Expanded(
            child: Text(tr('tip'),
                style: const TextStyle(
                    color: AppColors.inkSoft, fontSize: 13.5, height: 1.4)),
          ),
        ],
      ),
    );
  }
}

/// Signature hero: a gradient scan panel with a dashed camera frame and a leaf
/// motif, plus the two capture actions. This is the visual centrepiece.
class _HeroScanCard extends StatelessWidget {
  final VoidCallback onCamera;
  final VoidCallback onGallery;
  const _HeroScanCard({required this.onCamera, required this.onGallery});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(26),
        gradient: const LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [Color(0xFF1E7A4D), Color(0xFF115C3A)],
        ),
        boxShadow: [
          BoxShadow(
            color: AppColors.accent.withValues(alpha: 0.28),
            blurRadius: 28,
            offset: const Offset(0, 14),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              _chip(Icons.wifi_off_rounded, tr('offline')),
              const SizedBox(width: 8),
              _chip(Icons.verified_outlined, 'AI'),
            ],
          ),
          const SizedBox(height: 16),
          // Dashed scan frame with leaf motif.
          AspectRatio(
            aspectRatio: 1.7,
            child: CustomPaint(
              painter: _ScanFramePainter(),
              child: Center(
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Container(
                      width: 58,
                      height: 58,
                      decoration: BoxDecoration(
                        color: Colors.white.withValues(alpha: 0.14),
                        shape: BoxShape.circle,
                      ),
                      child: const Icon(Icons.eco_rounded,
                          color: Colors.white, size: 30),
                    ),
                    const SizedBox(height: 12),
                    Text(tr('hero_frame'),
                        style: TextStyle(
                            color: Colors.white.withValues(alpha: 0.92),
                            fontFamily: 'Manrope',
                            fontSize: 13.5,
                            fontWeight: FontWeight.w700)),
                  ],
                ),
              ),
            ),
          ),
          const SizedBox(height: 18),
          SizedBox(
            height: 56,
            child: FilledButton.icon(
              onPressed: onCamera,
              icon: const Icon(Icons.camera_alt_rounded, size: 21),
              label: Text(tr('btn_scan')),
              style: FilledButton.styleFrom(
                backgroundColor: Colors.white,
                foregroundColor: AppColors.accent,
                shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(15)),
                textStyle: const TextStyle(
                    fontSize: 16,
                    fontWeight: FontWeight.w800,
                    letterSpacing: 0.2),
              ),
            ),
          ),
          const SizedBox(height: 10),
          Center(
            child: TextButton.icon(
              onPressed: onGallery,
              icon: Icon(Icons.photo_library_outlined,
                  size: 18, color: Colors.white.withValues(alpha: 0.92)),
              label: Text(tr('btn_gallery'),
                  style: TextStyle(
                      color: Colors.white.withValues(alpha: 0.92),
                      fontSize: 14,
                      fontWeight: FontWeight.w700)),
            ),
          ),
        ],
      ),
    );
  }

  Widget _chip(IconData ic, String text) => Container(
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
        decoration: BoxDecoration(
            color: Colors.white.withValues(alpha: 0.16),
            borderRadius: BorderRadius.circular(20)),
        child: Row(mainAxisSize: MainAxisSize.min, children: [
          Icon(ic, size: 13, color: Colors.white),
          const SizedBox(width: 6),
          Text(text,
              style: const TextStyle(
                  color: Colors.white,
                  fontSize: 11.5,
                  fontWeight: FontWeight.w700)),
        ]),
      );
}

/// Dashed rounded-rectangle frame with corner ticks — a "viewfinder" motif.
class _ScanFramePainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    final rect = Offset.zero & size;
    final rrect = RRect.fromRectAndRadius(
        rect.deflate(2), const Radius.circular(18));

    // Dashed body.
    final dash = Paint()
      ..color = Colors.white.withValues(alpha: 0.45)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1.6;
    final path = Path()..addRRect(rrect);
    for (final metric in path.computeMetrics()) {
      double d = 0;
      const on = 9.0, off = 7.0;
      while (d < metric.length) {
        canvas.drawPath(metric.extractPath(d, d + on), dash);
        d += on + off;
      }
    }

    // Solid corner ticks for a camera-viewfinder feel.
    final tick = Paint()
      ..color = Colors.white
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.6
      ..strokeCap = StrokeCap.round;
    const len = 20.0;
    final r = rrect.outerRect.deflate(1);
    // top-left
    canvas.drawLine(Offset(r.left, r.top + len + 6), Offset(r.left, r.top + 18), tick);
    canvas.drawLine(Offset(r.left + 18, r.top), Offset(r.left + len + 6, r.top), tick);
    // top-right
    canvas.drawLine(Offset(r.right, r.top + 18), Offset(r.right, r.top + len + 6), tick);
    canvas.drawLine(Offset(r.right - len - 6, r.top), Offset(r.right - 18, r.top), tick);
    // bottom-left
    canvas.drawLine(Offset(r.left, r.bottom - len - 6), Offset(r.left, r.bottom - 18), tick);
    canvas.drawLine(Offset(r.left + 18, r.bottom), Offset(r.left + len + 6, r.bottom), tick);
    // bottom-right
    canvas.drawLine(Offset(r.right, r.bottom - 18), Offset(r.right, r.bottom - len - 6), tick);
    canvas.drawLine(Offset(r.right - len - 6, r.bottom), Offset(r.right - 18, r.bottom), tick);
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => false;
}

class _Analysing extends StatelessWidget {
  const _Analysing();
  @override
  Widget build(BuildContext context) {
    return Center(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          const SizedBox(
            width: 42,
            height: 42,
            child: CircularProgressIndicator(
                strokeWidth: 3, color: AppColors.accent),
          ),
          const SizedBox(height: 22),
          Text(tr('analysing_title'),
              style: AppTheme.head(size: 20, weight: FontWeight.w700)),
          const SizedBox(height: 6),
          Text(tr('analysing_sub'),
              style: const TextStyle(color: AppColors.inkSoft)),
        ],
      ),
    );
  }
}
