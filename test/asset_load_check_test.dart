import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  test('Should load mobilefacenet.tflite asset via package path', () async {
    // This expects the future to complete without throwing an error
    await expectLater(
      rootBundle.load(
        'packages/face_auth_engine/assets/models/mobilefacenet.tflite',
      ),
      completes,
    );
  });
}
