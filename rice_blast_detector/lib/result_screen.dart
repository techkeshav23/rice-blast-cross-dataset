import 'dart:io';
import 'package:flutter/material.dart';

import 'guidance.dart';
import 'i18n.dart';
import 'predictor.dart';
import 'theme.dart';
import 'widgets.dart';

class ResultScreen extends StatelessWidget {
  final File image;
  final Prediction prediction;

  const ResultScreen({
    super.key,
    required this.image,
    required this.prediction,
  });

  bool get _lowConfidence => prediction.confidence < 0.55;

  @override
  Widget build(BuildContext context) {
    final g = GuidanceStore.forLabelSafe(prediction.label);
    final color = severityColor(g.severity);

    return Scaffold(
      backgroundColor: AppColors.bg,
      appBar: AppBar(
        backgroundColor: AppColors.bg,
        surfaceTintColor: Colors.transparent,
        elevation: 0,
        foregroundColor: AppColors.ink,
        title: Text(tr('res_title'),
            style: AppTheme.head(size: 18, weight: FontWeight.w700)),
      ),
      body: ListView(
        padding: const EdgeInsets.fromLTRB(22, 4, 22, 32),
        children: [
          Stagger(
            children: [
              _resultHero(g, color),
              const SizedBox(height: 16),
              _mockBanner(),
              if (_lowConfidence) ...[
                const SizedBox(height: 16),
                _lowConfidenceNote(),
              ],
              const SizedBox(height: 26),
              _eyebrow(tr('res_gradcam_eyebrow')),
              const SizedBox(height: 10),
              GradcamView(image: image),
              const SizedBox(height: 6),
              Text(tr('res_gradcam_caption'),
                  style: const TextStyle(
                      color: AppColors.inkSoft, fontSize: 12.5)),
              const SizedBox(height: 26),
              _eyebrow(tr('symptoms')),
              const SizedBox(height: 10),
              ...g.symptoms.map((s) => _bullet(s, color)),
              const SizedBox(height: 26),
              _eyebrow(tr('management')),
              const SizedBox(height: 12),
              ..._numbered(g.management, color),
              const SizedBox(height: 18),
              _sources(),
              const SizedBox(height: 12),
              _disclaimer(),
            ],
          ),
        ],
      ),
    );
  }

  Widget _mockBanner() {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
      decoration: BoxDecoration(
        color: AppColors.spot.withValues(alpha: 0.12),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: AppColors.spot.withValues(alpha: 0.4)),
      ),
      child: Row(children: [
        Icon(Icons.science_outlined, size: 19, color: AppColors.spot),
        const SizedBox(width: 10),
        Expanded(
          child: Text(tr('res_mock_banner'),
              style: const TextStyle(
                  fontSize: 12.5,
                  color: AppColors.ink,
                  fontWeight: FontWeight.w600,
                  height: 1.35)),
        ),
      ]),
    );
  }

  /// Premium result hero: the photo with a dark gradient scrim, the diagnosis
  /// (severity pill + disease name) overlaid at the bottom, and a confidence
  /// chip top-right.
  Widget _resultHero(Guidance g, Color color) {
    return ClipRRect(
      borderRadius: BorderRadius.circular(22),
      child: Stack(
        children: [
          Image.file(image,
              height: 260,
              width: double.infinity,
              fit: BoxFit.cover,
              errorBuilder: (_, _, _) => Container(
                  height: 260,
                  color: AppColors.surface,
                  child: const Icon(Icons.image_not_supported_outlined,
                      color: AppColors.inkSoft, size: 32))),
          // bottom-up scrim for text legibility
          Positioned.fill(
            child: DecoratedBox(
              decoration: BoxDecoration(
                gradient: LinearGradient(
                  begin: Alignment.topCenter,
                  end: Alignment.bottomCenter,
                  colors: [
                    Colors.transparent,
                    Colors.black.withValues(alpha: 0.15),
                    Colors.black.withValues(alpha: 0.78),
                  ],
                  stops: const [0.35, 0.6, 1.0],
                ),
              ),
            ),
          ),
          // confidence chip (top-right)
          Positioned(
            top: 12,
            right: 12,
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 7),
              decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(30)),
              child: Row(mainAxisSize: MainAxisSize.min, children: [
                Icon(Icons.insights_rounded, size: 14, color: color),
                const SizedBox(width: 6),
                Text('${(prediction.confidence * 100).round()}% · ${tr('res_mock')}',
                    style: TextStyle(
                        color: AppColors.ink,
                        fontSize: 12,
                        fontWeight: FontWeight.w800)),
              ]),
            ),
          ),
          // diagnosis (bottom-left)
          Positioned(
            left: 16,
            right: 16,
            bottom: 16,
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Container(
                  padding:
                      const EdgeInsets.symmetric(horizontal: 11, vertical: 6),
                  decoration: BoxDecoration(
                      color: color, borderRadius: BorderRadius.circular(20)),
                  child: Row(mainAxisSize: MainAxisSize.min, children: [
                    Icon(severityIcon(g.severity),
                        size: 14, color: Colors.white),
                    const SizedBox(width: 6),
                    Text(severityLabel(g.severity),
                        style: const TextStyle(
                            color: Colors.white,
                            fontSize: 12,
                            fontWeight: FontWeight.w800)),
                  ]),
                ),
                const SizedBox(height: 10),
                Text(GuidanceStore.displayName(prediction.label),
                    style: AppTheme.head(
                        size: 30, weight: FontWeight.w700, color: Colors.white)),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _lowConfidenceNote() {
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: AppColors.blast.withValues(alpha: 0.08),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: AppColors.blast.withValues(alpha: 0.3)),
      ),
      child: Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Icon(Icons.help_outline_rounded, size: 19, color: AppColors.blast),
        const SizedBox(width: 10),
        Expanded(
          child: Text(tr('res_lowconf'),
              style: const TextStyle(
                  fontSize: 13, color: AppColors.ink, height: 1.4)),
        ),
      ]),
    );
  }

  Widget _eyebrow(String t) =>
      Text(t, style: AppTheme.overline(AppColors.inkSoft));

  Widget _bullet(String text, Color color) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 6),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Padding(
            padding: const EdgeInsets.only(top: 7, right: 12),
            child: Dot(color, size: 6),
          ),
          Expanded(
              child: Text(text,
                  style: const TextStyle(
                      color: AppColors.ink, fontSize: 15, height: 1.45))),
        ],
      ),
    );
  }

  List<Widget> _numbered(List<String> items, Color color) {
    return [
      for (int i = 0; i < items.length; i++)
        Padding(
          padding: const EdgeInsets.only(bottom: 10),
          child: Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Container(
                width: 26,
                height: 26,
                decoration: BoxDecoration(
                    color: color.withValues(alpha: 0.12),
                    shape: BoxShape.circle),
                child: Center(
                  child: Text('${i + 1}',
                      style: TextStyle(
                          color: color,
                          fontWeight: FontWeight.w800,
                          fontSize: 12.5)),
                ),
              ),
              const SizedBox(width: 14),
              Expanded(
                  child: Padding(
                padding: const EdgeInsets.only(top: 3),
                child: Text(items[i],
                    style: const TextStyle(
                        color: AppColors.ink, fontSize: 14.5, height: 1.45)),
              )),
            ],
          ),
        ),
    ];
  }

  Widget _sources() {
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Icon(Icons.menu_book_outlined, size: 15, color: AppColors.inkSoft),
        const SizedBox(width: 8),
        Expanded(
          child: Text(tr('sources'),
              style: const TextStyle(
                  color: AppColors.inkSoft,
                  fontSize: 11.5,
                  fontWeight: FontWeight.w600)),
        ),
      ],
    );
  }

  Widget _disclaimer() {
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: AppColors.surface,
        borderRadius: BorderRadius.circular(14),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Icon(Icons.info_outline_rounded,
              size: 17, color: AppColors.inkSoft),
          const SizedBox(width: 10),
          Expanded(
            child: Text(GuidanceStore.disclaimer,
                style: const TextStyle(
                    fontSize: 12, color: AppColors.inkSoft, height: 1.45)),
          ),
        ],
      ),
    );
  }
}
