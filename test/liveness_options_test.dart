import 'package:face_auth_engine/face_auth_engine.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  group('LivenessOptions', () {
    test('should use sensible default values', () {
      const options = LivenessOptions();

      expect(options.useGpu, false);
      expect(options.cpuThreads, 1);
      expect(options.threshold, 0.5);
      expect(options.applyLaplacianGate, false);
      expect(options.laplacianThreshold, 500);
      expect(options.normalizationType, NormalizationType.centered);
      expect(options.outputIsSpoofProbability, true);
      expect(options.warmUpIterations, 10);
      expect(options.applyLowLightGate, true);
      expect(options.lowLightThreshold, 40.0);
    });

    test('should respect custom values', () {
      const options = LivenessOptions(
        useGpu: true,
        cpuThreads: 4,
        threshold: 0.7,
        applyLaplacianGate: true,
        laplacianThreshold: 300,
        normalizationType: NormalizationType.scaled,
        outputIndex: 1,
        outputIsSpoofProbability: false,
        warmUpIterations: 5,
        applyLowLightGate: false,
        lowLightThreshold: 60,
      );

      expect(options.useGpu, true);
      expect(options.cpuThreads, 4);
      expect(options.threshold, 0.7);
      expect(options.applyLaplacianGate, true);
      expect(options.laplacianThreshold, 300);
      expect(options.normalizationType, NormalizationType.scaled);
      expect(options.outputIndex, 1);
      expect(options.outputIsSpoofProbability, false);
      expect(options.warmUpIterations, 5);
      expect(options.applyLowLightGate, false);
      expect(options.lowLightThreshold, 60.0);
    });

    test('should support adaptiveCentered normalization type', () {
      const options = LivenessOptions(
        normalizationType: NormalizationType.adaptiveCentered,
      );
      expect(options.normalizationType, NormalizationType.adaptiveCentered);
    });

    test('should support equality', () {
      const options1 = LivenessOptions(threshold: 0.6);
      const options2 = LivenessOptions(threshold: 0.6);
      const options3 = LivenessOptions(threshold: 0.7);

      expect(options1, equals(options2));
      expect(options1, isNot(equals(options3)));
    });

    test('should differentiate on lowLightThreshold', () {
      const options1 = LivenessOptions();
      const options2 = LivenessOptions(lowLightThreshold: 60);
      expect(options1, isNot(equals(options2)));
    });

    test('should have string representation including low-light fields', () {
      const options = LivenessOptions();
      final str = options.toString();

      expect(str, contains('LivenessOptions'));
      expect(str, contains('threshold'));
      expect(str, contains('normalizationType'));
      expect(str, contains('applyLowLightGate'));
      expect(str, contains('lowLightThreshold'));
    });

    test('should be const constructible', () {
      const options1 = LivenessOptions();
      const options2 = LivenessOptions();

      expect(identical(options1, options2), isTrue);
    });
  });

  group('LivenessResult', () {
    test('should store values correctly', () {
      const result = LivenessResult(
        isLive: true,
        score: 0.85,
        laplacian: 1500,
        duration: Duration(milliseconds: 100),
      );

      expect(result.isLive, true);
      expect(result.score, 0.85);
      expect(result.laplacian, 1500);
      expect(result.duration, const Duration(milliseconds: 100));
      expect(result.rejectionReason, LivenessRejectionReason.none);
      expect(result.brightness, isNull);
    });

    test('should store low-light rejection reason', () {
      const result = LivenessResult(
        isLive: false,
        score: 0,
        laplacian: 0,
        duration: Duration.zero,
        rejectionReason: LivenessRejectionReason.lowLight,
        brightness: 25,
      );

      expect(result.isLive, false);
      expect(result.rejectionReason, LivenessRejectionReason.lowLight);
      expect(result.brightness, 25.0);
    });

    test('should store blurry rejection reason', () {
      const result = LivenessResult(
        isLive: false,
        score: 0,
        laplacian: 100,
        duration: Duration.zero,
        rejectionReason: LivenessRejectionReason.blurry,
      );
      expect(result.rejectionReason, LivenessRejectionReason.blurry);
    });

    test('should store spoof rejection reason', () {
      const result = LivenessResult(
        isLive: false,
        score: 0.2,
        laplacian: 1000,
        duration: Duration.zero,
        rejectionReason: LivenessRejectionReason.spoof,
        brightness: 120,
      );
      expect(result.rejectionReason, LivenessRejectionReason.spoof);
    });

    test('should calculate spoofProb correctly', () {
      const result = LivenessResult(
        isLive: true,
        score: 0.8,
        laplacian: 1000,
        duration: Duration.zero,
      );

      expect(result.spoofProb, closeTo(0.2, 0.001));
    });

    test('should support equality', () {
      const r1 = LivenessResult(
        isLive: true,
        score: 0.85,
        laplacian: 1000,
        duration: Duration.zero,
      );
      const r2 = LivenessResult(
        isLive: true,
        score: 0.85,
        laplacian: 1000,
        duration: Duration.zero,
      );
      expect(r1, equals(r2));
    });

    test('should have readable string representation', () {
      const result = LivenessResult(
        isLive: true,
        score: 0.85,
        laplacian: 1500,
        duration: Duration(milliseconds: 100),
        brightness: 120,
      );

      final str = result.toString();
      expect(str, contains('isLive: true'));
      expect(str, contains('score: 0.850'));
      expect(str, contains('laplacian: 1500'));
      expect(str, contains('rejectionReason'));
      expect(str, contains('brightness'));
    });
  });

  group('NormalizationType', () {
    test('should have all three normalization types', () {
      expect(NormalizationType.values, contains(NormalizationType.scaled));
      expect(NormalizationType.values, contains(NormalizationType.centered));
      expect(
        NormalizationType.values,
        contains(NormalizationType.adaptiveCentered),
      );
    });
  });

  group('LivenessRejectionReason', () {
    test('should have all four reasons', () {
      expect(
        LivenessRejectionReason.values,
        containsAll([
          LivenessRejectionReason.none,
          LivenessRejectionReason.lowLight,
          LivenessRejectionReason.blurry,
          LivenessRejectionReason.spoof,
        ]),
      );
    });
  });
}
