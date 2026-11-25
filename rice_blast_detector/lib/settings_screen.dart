import 'package:flutter/material.dart';

import 'i18n.dart';
import 'theme.dart';
import 'widgets.dart';

class SettingsScreen extends StatelessWidget {
  const SettingsScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.bg,
      appBar: AppBar(
        backgroundColor: AppColors.bg,
        surfaceTintColor: Colors.transparent,
        elevation: 0,
        foregroundColor: AppColors.ink,
        title: Text(tr('nav_settings'),
            style: AppTheme.head(size: 18, weight: FontWeight.w700)),
      ),
      body: ValueListenableBuilder<String>(
        valueListenable: I18n.lang,
        builder: (context, lang, _) {
          return ListView(
            padding: const EdgeInsets.fromLTRB(22, 8, 22, 28),
            children: [
              Stagger(children: [
                Text(tr('settings_eyebrow'),
                    style: AppTheme.overline(AppColors.accent)),
                const SizedBox(height: 12),
                Text(tr('settings_title'),
                    style: AppTheme.head(size: 26, weight: FontWeight.w700)),
                const SizedBox(height: 20),
                _option('en', 'English', 'English', lang),
                const SizedBox(height: 12),
                _option('hi', 'हिन्दी', 'Hindi', lang),
                const SizedBox(height: 16),
                Text(tr('settings_hint'),
                    style: const TextStyle(
                        color: AppColors.inkSoft, fontSize: 13, height: 1.45)),
              ]),
            ],
          );
        },
      ),
    );
  }

  Widget _option(String code, String title, String subtitle, String current) {
    final selected = code == current;
    return SoftCard(
      color: selected ? AppColors.accentSoft : AppColors.bg,
      onTap: () => I18n.set(code),
      child: Row(
        children: [
          Container(
            width: 44,
            height: 44,
            decoration: BoxDecoration(
              color: selected ? AppColors.accent : AppColors.surface,
              borderRadius: BorderRadius.circular(12),
            ),
            child: Icon(Icons.translate_rounded,
                color: selected ? Colors.white : AppColors.inkSoft, size: 22),
          ),
          const SizedBox(width: 16),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(title,
                    style: TextStyle(
                        fontSize: 17,
                        fontWeight: FontWeight.w700,
                        color: selected ? AppColors.accent : AppColors.ink)),
                Text(subtitle,
                    style: const TextStyle(
                        color: AppColors.inkSoft, fontSize: 13)),
              ],
            ),
          ),
          Icon(
            selected
                ? Icons.check_circle_rounded
                : Icons.circle_outlined,
            color: selected ? AppColors.accent : AppColors.line,
          ),
        ],
      ),
    );
  }
}
