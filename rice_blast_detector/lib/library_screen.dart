import 'package:flutter/material.dart';

import 'guidance.dart';
import 'i18n.dart';
import 'theme.dart';
import 'widgets.dart';

/// "Field Guide" tab — an encyclopedia of the detectable conditions.
class LibraryScreen extends StatelessWidget {
  const LibraryScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final labels = GuidanceStore.labels();
    return ListView(
      padding: const EdgeInsets.fromLTRB(22, 8, 22, 28),
      children: [
        Stagger(
          children: [
            const SizedBox(height: 6),
            Text(tr('lib_eyebrow'),
                style: AppTheme.overline(AppColors.accent)),
            const SizedBox(height: 12),
            Text(tr('lib_title'),
                style: Theme.of(context).textTheme.displaySmall),
            const SizedBox(height: 24),
            for (final label in labels) _card(context, label),
          ],
        ),
      ],
    );
  }

  Widget _card(BuildContext context, String label) {
    final g = GuidanceStore.forLabelSafe(label);
    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: SoftCard(
        onTap: () => Navigator.push(context,
            MaterialPageRoute(builder: (_) => DiseaseDetailScreen(label))),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                SeverityPill(g.severity, fontSize: 11.5),
                const Spacer(),
                const Icon(Icons.arrow_outward_rounded,
                    color: AppColors.inkSoft, size: 19),
              ],
            ),
            const SizedBox(height: 14),
            Row(
              children: [
                Container(
                  width: 42,
                  height: 42,
                  decoration: BoxDecoration(
                      color: severityColor(g.severity).withValues(alpha: 0.12),
                      borderRadius: BorderRadius.circular(12)),
                  child: Icon(severityIcon(g.severity),
                      color: severityColor(g.severity), size: 22),
                ),
                const SizedBox(width: 14),
                Expanded(
                  child: Text(GuidanceStore.displayName(label),
                      style: AppTheme.head(size: 22, weight: FontWeight.w700)),
                ),
              ],
            ),
            const SizedBox(height: 10),
            Text(g.summary,
                maxLines: 2,
                overflow: TextOverflow.ellipsis,
                style: const TextStyle(
                    color: AppColors.inkSoft, fontSize: 13.5, height: 1.4)),
          ],
        ),
      ),
    );
  }
}

class DiseaseDetailScreen extends StatelessWidget {
  final String label;
  const DiseaseDetailScreen(this.label, {super.key});

  @override
  Widget build(BuildContext context) {
    final g = GuidanceStore.forLabelSafe(label);
    final color = severityColor(g.severity);
    return Scaffold(
      backgroundColor: AppColors.bg,
      appBar: AppBar(
        backgroundColor: AppColors.bg,
        surfaceTintColor: Colors.transparent,
        elevation: 0,
        foregroundColor: AppColors.ink,
      ),
      body: ListView(
        padding: const EdgeInsets.fromLTRB(22, 4, 22, 32),
        children: [
          Stagger(
            children: [
              SeverityPill(g.severity, fontSize: 12.5),
              const SizedBox(height: 16),
              Text(GuidanceStore.displayName(label),
                  style: AppTheme.head(size: 32, weight: FontWeight.w700)),
              const SizedBox(height: 14),
              Text(g.summary,
                  style: const TextStyle(
                      fontSize: 15.5, color: AppColors.ink, height: 1.55)),
              const SizedBox(height: 28),
              _block(tr('symptoms'), g.symptoms, color),
              const SizedBox(height: 26),
              _block(tr('management'), g.management, color),
              const SizedBox(height: 22),
              Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Icon(Icons.menu_book_outlined,
                      size: 15, color: AppColors.inkSoft),
                  const SizedBox(width: 8),
                  Expanded(
                    child: Text(tr('sources'),
                        style: const TextStyle(
                            color: AppColors.inkSoft,
                            fontSize: 11.5,
                            fontWeight: FontWeight.w600)),
                  ),
                ],
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _block(String eyebrow, List<String> items, Color color) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(eyebrow, style: AppTheme.overline(AppColors.inkSoft)),
        const SizedBox(height: 12),
        ...items.map((t) => Padding(
              padding: const EdgeInsets.symmetric(vertical: 6),
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Padding(
                    padding: const EdgeInsets.only(top: 7, right: 12),
                    child: Dot(color, size: 6),
                  ),
                  Expanded(
                      child: Text(t,
                          style: const TextStyle(
                              fontSize: 15,
                              color: AppColors.ink,
                              height: 1.45))),
                ],
              ),
            )),
      ],
    );
  }
}
