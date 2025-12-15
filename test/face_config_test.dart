import 'package:face_auth_engine/face_auth_engine.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  group('FaceConfig', () {
    test('should use default values when not specified', () {
      const config = FaceConfig();

      expect(config.recognitionThreshold, 1.0);
      expect(config.requiredEnrollmentSamples, 5);
      expect(config.minFaceSize, 80);
      expect(config.maxRollAngle, 15.0);
    });

    test('should allow custom values', () {
      const config = FaceConfig(
        recognitionThreshold: 0.8,
        requiredEnrollmentSamples: 3,
        minFaceSize: 100,
        maxRollAngle: 20.0,
      );

      expect(config.recognitionThreshold, 0.8);
      expect(config.requiredEnrollmentSamples, 3);
      expect(config.minFaceSize, 100);
      expect(config.maxRollAngle, 20.0);
    });

    test('should provide default config constant', () {
      const config = FaceConfig.defaultConfig;

      expect(config.recognitionThreshold, 1.0);
      expect(config.requiredEnrollmentSamples, 5);
      expect(config.minFaceSize, 80);
      expect(config.maxRollAngle, 15.0);
    });

    test('should be const constructible', () {
      const config1 = FaceConfig();
      const config2 = FaceConfig();

      expect(identical(config1, config2), isTrue);
    });

    test('should have meaningful toString', () {
      const config = FaceConfig();
      final str = config.toString();

      expect(str, contains('threshold'));
      expect(str, contains('samples'));
      expect(str, contains('minSize'));
      expect(str, contains('maxRoll'));
    });
  });
}
