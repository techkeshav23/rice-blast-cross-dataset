import 'package:flutter/material.dart';
import 'i18n.dart';
import 'theme.dart';
import 'widgets.dart';

class AboutScreen extends StatelessWidget {
  const AboutScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.bg,
      appBar: AppBar(
        backgroundColor: AppColors.bg,
        surfaceTintColor: Colors.transparent,
        elevation: 0,
        foregroundColor: AppColors.ink,
        title: Text(tr('nav_about'),
            style: AppTheme.head(size: 18, weight: FontWeight.w700)),
      ),
      body: ListView(
        padding: const EdgeInsets.fromLTRB(22, 4, 22, 32),
        children: [
          Stagger(
            children: [
              Text(tr('about_eyebrow'),
                  style: AppTheme.overline(AppColors.accent)),
              const SizedBox(height: 12),
              Text('Paddy Blast Lab',
                  style: AppTheme.head(size: 32, weight: FontWeight.w700)),
              const SizedBox(height: 14),
              Text(tr('about_body'),
                  style: const TextStyle(
                      fontSize: 15.5, color: AppColors.ink, height: 1.55)),
              const SizedBox(height: 26),
              _card(tr('about_team'), ['Harsh Kumar Sah', 'Keshav Upadhyay']),
              const SizedBox(height: 12),
              _card(tr('about_guide'), ['Mr. Prashant Kumar Singh']),
              const SizedBox(height: 12),
              _card(tr('about_institute'),
                  ['ABESEC Ghaziabad', tr('about_institute2')]),
              const SizedBox(height: 22),
              Container(
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: AppColors.surface,
                  borderRadius: BorderRadius.circular(14),
                ),
                child: Text(tr('about_disclaimer'),
                    style: const TextStyle(
                        fontSize: 13, color: AppColors.inkSoft, height: 1.45)),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _card(String eyebrow, List<String> lines) {
    return SoftCard(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(eyebrow, style: AppTheme.overline(AppColors.inkSoft)),
          const SizedBox(height: 10),
          ...lines.map((l) => Padding(
                padding: const EdgeInsets.symmetric(vertical: 2),
                child: Text(l,
                    style: const TextStyle(
                        color: AppColors.ink,
                        fontSize: 15.5,
                        fontWeight: FontWeight.w600)),
              )),
        ],
      ),
    );
  }
}
