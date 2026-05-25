import 'dart:developer' as developer;

import 'package:face_auth_engine/src/image/face_image_provider.dart';
import 'package:face_auth_engine/src/liveness/image_to_nhwc.dart';
import 'package:face_auth_engine/src/liveness/laplacian.dart';
import 'package:face_auth_engine/src/liveness/liveness_options.dart';
import 'package:face_auth_engine/src/liveness/liveness_result.dart';
import 'package:face_auth_engine/src/liveness/tflite_runner.dart';

class LivenessEngine {
  const LivenessEngine(this._runner, this._options, this._imageProvider);
  final TFLiteRunner _runner;
  final LivenessOptions _options;
  final FaceImageProvider _imageProvider;

  Future<LivenessResult> analyze(FaceImageBuffer face) async {
    final sw = Stopwatch()..start();

    // Resize with explicit interpolation
    final FaceImageBuffer resized;
    if (face.width == 224 && face.height == 224) {
      resized = face;
    } else {
      resized = await _imageProvider.resize(face, 224, 224);
    }

    // Calculate brightness only if needed for gating or reporting
    final needsBrightness =
        _options.applyLowLightGate || _options.applyOverExposureGate;
    final brightness = needsBrightness ? brightnessScore(resized) : 127.5;

    // ── Step 1: Low-light gate (runs BEFORE blur check)
    // Dark images have low contrast → Laplacian would give a misleading
    // "too blurry" rejection. By checking brightness first we surface the
    // true reason to the caller.
    if (_options.applyLowLightGate) {
      developer.log(
        'Brightness score: ${brightness.toStringAsFixed(1)} '
        '(threshold: ${_options.lowLightThreshold})',
      );

      if (brightness < _options.lowLightThreshold) {
        developer.log(
          'Image rejected: too dark '
          '(brightness=${brightness.toStringAsFixed(1)})',
        );
        sw.stop();
        return LivenessResult(
          isLive: false,
          score: 0,
          laplacian: 0,
          duration: sw.elapsed,
          rejectionReason: LivenessRejectionReason.lowLight,
          brightness: brightness,
        );
      }
    }

    // ── Step 1.1: Over-exposure gate
    if (_options.applyOverExposureGate) {
      if (brightness > _options.overExposureThreshold) {
        developer.log(
          'Image rejected: too bright '
          '(brightness=${brightness.toStringAsFixed(1)})',
        );
        sw.stop();
        return LivenessResult(
          isLive: false,
          score: 0,
          laplacian: 0,
          duration: sw.elapsed,
          rejectionReason: .overExposed,
          brightness: brightness,
        );
      }
    }

    // ── Step 2: Laplacian blur gate
    final int? laplacian;
    if (_options.applyLaplacianGate) {
      laplacian = laplacianScore(
        resized,
        laplacePixelThreshold: _options.laplacePixelThreshold,
      );
      developer.log(
        'Laplacian score: $laplacian '
        '(threshold: ${_options.laplacianThreshold})',
      );
    } else {
      laplacian = null; // Bypass
    }

    // Early reject if image is too blurry
    if (_options.applyLaplacianGate &&
        laplacian != null &&
        laplacian < _options.laplacianThreshold) {
      developer.log('Image rejected: too blurry (laplacian=$laplacian)');
      sw.stop();
      return LivenessResult(
        isLive: false,
        score: 0,
        laplacian: laplacian,
        duration: sw.elapsed,
        rejectionReason: .blurry,
        brightness: brightness,
      );
    }

    // ── Step 3: Convert image to tensor with configured normalization
    final input = toNHWC(resized, normalization: _options.normalizationType);

    // ── Step 4: Run inference
    final rawProb = await _runner.inferProb(
      input,
      outputIndex: _options.outputIndex,
    );

    // ── Step 5: Convert to liveness probability
    final double liveProb;
    if (_options.outputIsSpoofProbability) {
      liveProb = 1.0 - rawProb;
    } else {
      liveProb = rawProb;
    }

    final isLive = liveProb >= _options.threshold;

    sw.stop();
    final duration = sw.elapsed;

    developer.log(
      'Liveness: isLive=$isLive, '
      'liveProb=${liveProb.toStringAsFixed(3)}, '
      'threshold=${_options.threshold}, '
      'brightness=${brightness.toStringAsFixed(1)}, '
      'duration=${duration.inMilliseconds}ms',
    );

    return LivenessResult(
      isLive: isLive,
      score: liveProb,
      laplacian: laplacian,
      duration: duration,
      rejectionReason: isLive ? .none : .spoof,
      brightness: brightness,
    );
  }
}
