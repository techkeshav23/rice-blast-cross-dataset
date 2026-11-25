import 'package:flutter_test/flutter_test.dart';

import 'package:rice_blast_detector/main.dart';

void main() {
  testWidgets('Home screen shows the headline and scan button', (tester) async {
    await tester.pumpWidget(const RiceBlastApp());

    expect(find.text('Scan a leaf'), findsOneWidget);
    expect(find.text('RICE LEAF DIAGNOSIS'), findsOneWidget);
  });
}
