import 'package:flutter/material.dart';

import 'guidance.dart';
import 'history.dart';
import 'i18n.dart';
import 'result_screen.dart';
import 'theme.dart';
import 'widgets.dart';

class HistoryScreen extends StatelessWidget {
  const HistoryScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return ValueListenableBuilder<List<ScanRecord>>(
      valueListenable: History.records,
      builder: (context, records, _) {
        return ListView(
          padding: const EdgeInsets.fromLTRB(22, 8, 22, 28),
          children: [
            const SizedBox(height: 6),
            Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(tr('hist_eyebrow'),
                          style: AppTheme.overline(AppColors.accent)),
                      const SizedBox(height: 12),
                      Text(tr('hist_title'),
                          style: Theme.of(context).textTheme.displaySmall),
                    ],
                  ),
                ),
                if (records.isNotEmpty)
                  TextButton(
                    onPressed: History.clear,
                    style:
                        TextButton.styleFrom(foregroundColor: AppColors.inkSoft),
                    child: Text(tr('hist_clear')),
                  ),
              ],
            ),
            const SizedBox(height: 20),
            if (records.isEmpty)
              _empty()
            else
              for (final r in records) _row(context, r),
          ],
        );
      },
    );
  }

  Widget _empty() {
    return Container(
      margin: const EdgeInsets.only(top: 70),
      child: Column(
        children: [
          Container(
            width: 74,
            height: 74,
            decoration: BoxDecoration(
                color: AppColors.surface, shape: BoxShape.circle),
            child: const Icon(Icons.eco_outlined,
                size: 34, color: AppColors.accent),
          ),
          const SizedBox(height: 18),
          Text(tr('hist_empty_title'),
              style: AppTheme.head(size: 20, weight: FontWeight.w700)),
          const SizedBox(height: 6),
          Text(tr('hist_empty_sub'),
              style: const TextStyle(color: AppColors.inkSoft)),
        ],
      ),
    );
  }

  Widget _row(BuildContext context, ScanRecord r) {
    final g = GuidanceStore.forLabelSafe(r.prediction.label);
    final color = severityColor(g.severity);
    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: SoftCard(
        padding: const EdgeInsets.all(12),
        onTap: () => Navigator.push(
            context,
            MaterialPageRoute(
                builder: (_) =>
                    ResultScreen(image: r.image, prediction: r.prediction))),
        child: Row(
          children: [
            ClipRRect(
              borderRadius: BorderRadius.circular(12),
              child: Image.file(r.image,
                  width: 54,
                  height: 54,
                  fit: BoxFit.cover,
                  cacheWidth: 120,
                  errorBuilder: (_, _, _) => Container(
                      width: 54,
                      height: 54,
                      color: AppColors.surface,
                      child: const Icon(Icons.image_not_supported_outlined,
                          size: 20, color: AppColors.inkSoft))),
            ),
            const SizedBox(width: 14),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(GuidanceStore.displayName(r.prediction.label),
                      style: Theme.of(context).textTheme.titleLarge),
                  const SizedBox(height: 3),
                  Row(children: [
                    Dot(color, size: 6),
                    const SizedBox(width: 6),
                    Text(
                        '${(r.prediction.confidence * 100).round()}%  ·  ${_time(r.time)}',
                        style: const TextStyle(
                            color: AppColors.inkSoft, fontSize: 12.5)),
                  ]),
                ],
              ),
            ),
            const Icon(Icons.chevron_right_rounded, color: AppColors.inkSoft),
          ],
        ),
      ),
    );
  }

  String _time(DateTime t) {
    final h = t.hour % 12 == 0 ? 12 : t.hour % 12;
    final m = t.minute.toString().padLeft(2, '0');
    final ap = t.hour < 12 ? 'AM' : 'PM';
    return '$h:$m $ap';
  }
}
