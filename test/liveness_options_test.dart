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
    });

    test('should support equality', () {
      const options1 = LivenessOptions(threshold: 0.6);
      const options2 = LivenessOptions(threshold: 0.6);
      const options3 = LivenessOptions(threshold: 0.7);

      expect(options1, equals(options2));
      expect(options1, isNot(equals(options3)));
    });

    test('should have string representation', () {
      const options = LivenessOptions();
      final str = options.toString();

      expect(str, contains('LivenessOptions'));
      expect(str, contains('threshold'));
      expect(str, contains('normalizationType'));
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

    test('should have readable string representation', () {
      const result = LivenessResult(
        isLive: true,
        score: 0.85,
        laplacian: 1500,
        duration: Duration(milliseconds: 100),
      );

      final str = result.toString();
      expect(str, contains('isLive: true'));
      expect(str, contains('score: 0.850'));
      expect(str, contains('laplacian: 1500'));
    });
  });

  group('NormalizationType', () {
    test('should have both normalization types', () {
      expect(NormalizationType.values, contains(NormalizationType.scaled));
      expect(NormalizationType.values, contains(NormalizationType.centered));
    });
  });
}
