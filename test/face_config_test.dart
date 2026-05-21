import 'package:face_auth_engine/face_auth_engine.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  group('FaceConfig', () {
    test('should use default values', () {
      const config = FaceConfig.defaultConfig;
      expect(config.recognitionThreshold, 1.0);
      expect(config.minFaceSize, 80);
      expect(config.maxRollAngle, 15.0);
    });

    test('should respect custom values', () {
      const config = FaceConfig(
        recognitionThreshold: 0.5,
        minFaceSize: 100,
        maxRollAngle: 10,
      );
      expect(config.recognitionThreshold, 0.5);
      expect(config.minFaceSize, 100);
      expect(config.maxRollAngle, 10.0);
    });

    test('should have correct check semantics', () {
      const config1 = FaceConfig.defaultConfig;
      const config2 = FaceConfig.defaultConfig;
      expect(config1, config2);
    });

    test('should provide string representation', () {
      const config = FaceConfig.defaultConfig;
      expect(config.toString(), contains('threshold: 1.0'));
      expect(config.toString(), contains('minSize: 80'));
    });

    test('should be const constructible', () {
      const config1 = FaceConfig.defaultConfig;
      const config2 = FaceConfig.defaultConfig;

      expect(identical(config1, config2), isTrue);
    });

    test('should have meaningful toString', () {
      const config = FaceConfig.defaultConfig;
      final str = config.toString();

      expect(str, contains('threshold'));
      expect(str, contains('threshold'));
      expect(str, contains('minSize'));
      expect(str, contains('maxRoll'));
    });
  });
}
