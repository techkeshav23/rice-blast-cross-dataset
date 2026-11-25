import 'package:flutter/material.dart';
import 'package:flutter_localizations/flutter_localizations.dart';

import 'about_screen.dart';
import 'guidance.dart';
import 'history.dart';
import 'history_screen.dart';
import 'home_screen.dart';
import 'how_it_works_screen.dart';
import 'i18n.dart';
import 'library_screen.dart';
import 'metrics_screen.dart';
import 'settings_screen.dart';
import 'theme.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await I18n.load();
  await GuidanceStore.load();
  await History.load();
  runApp(const RiceBlastApp());
}

class RiceBlastApp extends StatelessWidget {
  const RiceBlastApp({super.key});

  @override
  Widget build(BuildContext context) {
    // Rebuild the whole app when the language changes.
    return ValueListenableBuilder<String>(
      valueListenable: I18n.lang,
      builder: (context, lang, _) {
        return MaterialApp(
          title: 'Rice Blast Detector',
          debugShowCheckedModeBanner: false,
          theme: AppTheme.build(),
          locale: Locale(lang),
          supportedLocales: const [Locale('en'), Locale('hi')],
          localizationsDelegates: const [
            GlobalMaterialLocalizations.delegate,
            GlobalWidgetsLocalizations.delegate,
            GlobalCupertinoLocalizations.delegate,
          ],
          home: const MainShell(),
        );
      },
    );
  }
}

class MainShell extends StatefulWidget {
  const MainShell({super.key});

  @override
  State<MainShell> createState() => _MainShellState();
}

class _MainShellState extends State<MainShell> {
  int _index = 0;

  void _go(int i) => setState(() => _index = i);

  void _push(Widget screen) =>
      Navigator.push(context, MaterialPageRoute(builder: (_) => screen));

  @override
  Widget build(BuildContext context) {
    final titles = [tr('nav_scan'), tr('nav_guide_full'), tr('nav_history')];
    final pages = const [HomeScreen(), LibraryScreen(), HistoryScreen()];

    return Scaffold(
      backgroundColor: AppColors.bg,
      appBar: AppBar(
        backgroundColor: AppColors.bg,
        surfaceTintColor: Colors.transparent,
        elevation: 0,
        scrolledUnderElevation: 0.5,
        centerTitle: false,
        titleSpacing: 4,
        title: Text(titles[_index],
            style: AppTheme.head(size: 20, weight: FontWeight.w700)),
        iconTheme: const IconThemeData(color: AppColors.ink),
      ),
      drawer: _SideBar(
        index: _index,
        onSelect: (i) {
          Navigator.pop(context);
          _go(i);
        },
        onHow: () {
          Navigator.pop(context);
          _push(const HowItWorksScreen());
        },
        onResults: () {
          Navigator.pop(context);
          _push(const MetricsScreen());
        },
        onAbout: () {
          Navigator.pop(context);
          _push(const AboutScreen());
        },
        onSettings: () {
          Navigator.pop(context);
          _push(const SettingsScreen());
        },
      ),
      body: IndexedStack(index: _index, children: pages),
      bottomNavigationBar: _BottomNav(index: _index, onTap: _go),
    );
  }
}

// ---------------------------------------------------------------- Sidebar
class _SideBar extends StatelessWidget {
  final int index;
  final ValueChanged<int> onSelect;
  final VoidCallback onHow;
  final VoidCallback onResults;
  final VoidCallback onAbout;
  final VoidCallback onSettings;
  const _SideBar(
      {required this.index,
      required this.onSelect,
      required this.onHow,
      required this.onResults,
      required this.onAbout,
      required this.onSettings});

  @override
  Widget build(BuildContext context) {
    final primary = [
      (Icons.center_focus_strong_rounded, tr('nav_scan')),
      (Icons.menu_book_rounded, tr('nav_guide_full')),
      (Icons.history_rounded, tr('nav_history')),
    ];
    return Drawer(
      backgroundColor: AppColors.bg,
      shape: const RoundedRectangleBorder(
          borderRadius: BorderRadius.horizontal(right: Radius.circular(24))),
      child: SafeArea(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Padding(
              padding: const EdgeInsets.fromLTRB(22, 26, 22, 22),
              child: Row(
                children: [
                  Container(
                    width: 42,
                    height: 42,
                    decoration: BoxDecoration(
                        color: AppColors.accent,
                        borderRadius: BorderRadius.circular(12)),
                    child: const Icon(Icons.eco_rounded,
                        color: Colors.white, size: 23),
                  ),
                  const SizedBox(width: 12),
                  Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text('Paddy Blast Lab',
                          style:
                              AppTheme.head(size: 16, weight: FontWeight.w700)),
                      Text(tr('brand_tagline'),
                          style: const TextStyle(
                              color: AppColors.inkSoft, fontSize: 12.5)),
                    ],
                  ),
                ],
              ),
            ),
            const Divider(height: 1, color: AppColors.line),
            const SizedBox(height: 10),
            _eyebrow(tr('grp_scan')),
            for (int i = 0; i < primary.length; i++)
              _tile(primary[i].$1, primary[i].$2, i == index, () => onSelect(i)),
            const SizedBox(height: 8),
            const Divider(height: 1, color: AppColors.line),
            const SizedBox(height: 8),
            _eyebrow(tr('grp_project')),
            _tile(Icons.lightbulb_outline_rounded, tr('nav_how'), false, onHow),
            _tile(Icons.insights_rounded, tr('nav_results'), false, onResults),
            _tile(Icons.translate_rounded, tr('nav_settings'), false, onSettings),
            _tile(Icons.info_outline_rounded, tr('nav_about'), false, onAbout),
            const Spacer(),
            Padding(
              padding: const EdgeInsets.fromLTRB(24, 10, 22, 16),
              child: Row(children: [
                const Icon(Icons.wifi_off_rounded,
                    size: 14, color: AppColors.inkSoft),
                const SizedBox(width: 8),
                Text(tr('offline_footer'),
                    style: const TextStyle(
                        color: AppColors.inkSoft, fontSize: 12)),
              ]),
            ),
          ],
        ),
      ),
    );
  }

  Widget _eyebrow(String t) => Padding(
        padding: const EdgeInsets.fromLTRB(26, 6, 22, 6),
        child: Text(t, style: AppTheme.overline(AppColors.inkSoft)),
      );

  Widget _tile(IconData icon, String label, bool selected, VoidCallback onTap) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 2),
      child: Material(
        color: selected ? AppColors.accentSoft : Colors.transparent,
        borderRadius: BorderRadius.circular(12),
        child: InkWell(
          borderRadius: BorderRadius.circular(12),
          onTap: onTap,
          child: Container(
            constraints: const BoxConstraints(minHeight: 50),
            padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
            child: Row(
              children: [
                Icon(icon,
                    size: 21,
                    color: selected ? AppColors.accent : AppColors.inkSoft),
                const SizedBox(width: 14),
                Text(label,
                    style: TextStyle(
                        color: selected ? AppColors.accent : AppColors.ink,
                        fontWeight:
                            selected ? FontWeight.w700 : FontWeight.w600,
                        fontSize: 15)),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

// ------------------------------------------------------------- Bottom nav
class _BottomNav extends StatelessWidget {
  final int index;
  final ValueChanged<int> onTap;
  const _BottomNav({required this.index, required this.onTap});

  @override
  Widget build(BuildContext context) {
    final items = [
      (Icons.center_focus_strong_rounded, tr('nav_scan')),
      (Icons.menu_book_rounded, tr('nav_guide')),
      (Icons.history_rounded, tr('nav_history')),
    ];
    return Container(
      decoration: const BoxDecoration(
        color: AppColors.bg,
        border: Border(top: BorderSide(color: AppColors.line)),
      ),
      child: SafeArea(
        top: false,
        child: Padding(
          padding: const EdgeInsets.symmetric(vertical: 6, horizontal: 8),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.spaceAround,
            children: [
              for (int i = 0; i < items.length; i++)
                _item(i, items[i].$1, items[i].$2),
            ],
          ),
        ),
      ),
    );
  }

  Widget _item(int i, IconData icon, String label) {
    final sel = i == index;
    return Expanded(
      child: GestureDetector(
        behavior: HitTestBehavior.opaque,
        onTap: () => onTap(i),
        child: Container(
          height: 54,
          alignment: Alignment.center,
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Icon(icon,
                  size: 24, color: sel ? AppColors.accent : AppColors.inkSoft),
              const SizedBox(height: 4),
              Text(label,
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: TextStyle(
                      fontSize: 12.5,
                      fontWeight: sel ? FontWeight.w700 : FontWeight.w600,
                      color: sel ? AppColors.accent : AppColors.inkSoft)),
            ],
          ),
        ),
      ),
    );
  }
}
