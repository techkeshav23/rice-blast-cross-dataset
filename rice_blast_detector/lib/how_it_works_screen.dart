import 'package:flutter/material.dart';
import 'i18n.dart';
import 'theme.dart';
import 'widgets.dart';

/// Tells the RESEARCH story: this is a cross-dataset generalization project,
/// not just a leaf classifier. Content mirrors PROJECT_MASTER_NOTES §2-4.
class HowItWorksScreen extends StatelessWidget {
  const HowItWorksScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.bg,
      appBar: AppBar(
        backgroundColor: AppColors.bg,
        surfaceTintColor: Colors.transparent,
        elevation: 0,
        foregroundColor: AppColors.ink,
        title: Text(tr('nav_how'),
            style: AppTheme.head(size: 18, weight: FontWeight.w700)),
      ),
      body: ListView(
        padding: const EdgeInsets.fromLTRB(22, 4, 22, 32),
        children: [
          Stagger(children: [
            Text(tr('how_eyebrow'), style: AppTheme.overline(AppColors.accent)),
            const SizedBox(height: 12),
            Text(tr('how_title'),
                style: AppTheme.head(size: 28, weight: FontWeight.w700)),
            const SizedBox(height: 20),
            _step('01', tr('how_s1_t'), tr('how_s1_b'), AppColors.blast),
            _step('02', tr('how_s2_t'), tr('how_s2_b'), AppColors.accent),
            _step('03', tr('how_s3_t'), tr('how_s3_b'), AppColors.spot),
            _step('04', tr('how_s4_t'), tr('how_s4_b'), AppColors.healthy),
            const SizedBox(height: 8),
            Container(
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                  color: AppColors.surface,
                  borderRadius: BorderRadius.circular(14)),
              child: Text(tr('how_note'),
                  style: const TextStyle(
                      fontSize: 13, color: AppColors.inkSoft, height: 1.45)),
            ),
          ]),
        ],
      ),
    );
  }

  Widget _step(String n, String title, String body, Color color) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 20),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(n,
              style: AppTheme.head(
                  size: 22, weight: FontWeight.w700, color: color)),
          const SizedBox(width: 16),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(title,
                    style: const TextStyle(
                        fontFamily: 'Manrope',
                        fontSize: 16,
                        fontWeight: FontWeight.w700,
                        color: AppColors.ink)),
                const SizedBox(height: 6),
                Text(body,
                    style: const TextStyle(
                        fontSize: 14.5, color: AppColors.ink, height: 1.5)),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
