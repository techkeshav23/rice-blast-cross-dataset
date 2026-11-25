import 'package:flutter/material.dart';
import 'i18n.dart';
import 'theme.dart';
import 'widgets.dart';

/// "Results" screen — shows the generalization gap the project targets.
/// Numbers are ILLUSTRATIVE placeholders until the real model is trained.
class MetricsScreen extends StatelessWidget {
  const MetricsScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.bg,
      appBar: AppBar(
        backgroundColor: AppColors.bg,
        surfaceTintColor: Colors.transparent,
        elevation: 0,
        foregroundColor: AppColors.ink,
        title: Text(tr('nav_results'),
            style: AppTheme.head(size: 18, weight: FontWeight.w700)),
      ),
      body: ListView(
        padding: const EdgeInsets.fromLTRB(22, 4, 22, 32),
        children: [
          Stagger(children: [
            Text(tr('met_eyebrow'), style: AppTheme.overline(AppColors.accent)),
            const SizedBox(height: 12),
            Text(tr('met_title'),
                style: AppTheme.head(size: 26, weight: FontWeight.w700)),
            const SizedBox(height: 12),
            Text(tr('met_sub'),
                style: const TextStyle(
                    fontSize: 14.5, color: AppColors.inkSoft, height: 1.5)),
            const SizedBox(height: 20),
            _header(),
            const SizedBox(height: 8),
            _row(tr('met_baseline'), 96, 73, false),
            _row(tr('met_ours'), 95, 75, true),
            const SizedBox(height: 20),
            _takeaway(),
            const SizedBox(height: 16),
            Container(
              padding: const EdgeInsets.all(14),
              decoration: BoxDecoration(
                  color: AppColors.spot.withValues(alpha: 0.10),
                  borderRadius: BorderRadius.circular(12),
                  border: Border.all(
                      color: AppColors.spot.withValues(alpha: 0.35))),
              child: Row(children: [
                Icon(Icons.info_outline_rounded,
                    size: 18, color: AppColors.spot),
                const SizedBox(width: 10),
                Expanded(
                  child: Text(tr('met_illustrative'),
                      style: const TextStyle(
                          fontSize: 12.5,
                          color: AppColors.ink,
                          fontWeight: FontWeight.w600)),
                ),
              ]),
            ),
          ]),
        ],
      ),
    );
  }

  Widget _header() {
    return Row(
      children: [
        const Expanded(flex: 4, child: SizedBox()),
        Expanded(
            flex: 2,
            child: Text(tr('met_inset'),
                textAlign: TextAlign.center,
                style: AppTheme.overline(AppColors.inkSoft))),
        Expanded(
            flex: 2,
            child: Text(tr('met_unseen'),
                textAlign: TextAlign.center,
                style: AppTheme.overline(AppColors.inkSoft))),
        Expanded(
            flex: 2,
            child: Text(tr('met_gap'),
                textAlign: TextAlign.center,
                style: AppTheme.overline(AppColors.inkSoft))),
      ],
    );
  }

  Widget _row(String name, int inSet, int unseen, bool isOurs) {
    final gap = inSet - unseen;
    final gapColor = gap <= 15 ? AppColors.healthy : AppColors.blast;
    return Container(
      margin: const EdgeInsets.only(bottom: 10),
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 16),
      decoration: BoxDecoration(
        color: isOurs ? AppColors.accentSoft : AppColors.bg,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(
            color:
                isOurs ? AppColors.accent.withValues(alpha: 0.4) : AppColors.line),
      ),
      child: Row(
        children: [
          Expanded(
              flex: 4,
              child: Text(name,
                  style: TextStyle(
                      fontWeight: isOurs ? FontWeight.w800 : FontWeight.w600,
                      color: AppColors.ink,
                      fontSize: 14))),
          Expanded(
              flex: 2,
              child: Text('$inSet',
                  textAlign: TextAlign.center,
                  style: const TextStyle(
                      color: AppColors.inkSoft, fontWeight: FontWeight.w700))),
          Expanded(
              flex: 2,
              child: Text('$unseen',
                  textAlign: TextAlign.center,
                  style: const TextStyle(
                      color: AppColors.ink, fontWeight: FontWeight.w700))),
          Expanded(
            flex: 2,
            child: Text('−$gap',
                textAlign: TextAlign.center,
                style: TextStyle(color: gapColor, fontWeight: FontWeight.w800)),
          ),
        ],
      ),
    );
  }

  Widget _takeaway() {
    return SoftCard(
      color: AppColors.surface,
      bordered: false,
      child: Row(
        children: [
          Icon(Icons.trending_up_rounded, color: AppColors.accent, size: 26),
          const SizedBox(width: 14),
          Expanded(
            child: Text(tr('met_takeaway'),
                style: const TextStyle(
                    fontSize: 14, color: AppColors.ink, height: 1.45)),
          ),
        ],
      ),
    );
  }
}
