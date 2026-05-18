import 'dart:developer' as developer;

import '../image/face_image_provider.dart';
import 'image_to_nhwc.dart';
import 'laplacian.dart';
import 'liveness_options.dart';
import 'liveness_result.dart';
import 'tflite_runner.dart';

class LivenessEngine {
  final TFLiteRunner _runner;
  final LivenessOptions _options;
  final FaceImageProvider _imageProvider;

  const LivenessEngine(this._runner, this._options, this._imageProvider);

  Future<LivenessResult> analyze(FaceImageBuffer face) async {
    final sw = Stopwatch()..start();

    // Resize with explicit interpolation
    final FaceImageBuffer resized;
    if (face.width == 224 && face.height == 224) {
      resized = face;
    } else {
      resized = await _imageProvider.resize(face, 224, 224);
    }

    // Calculate Laplacian score for blur detection
    final int laplacian;
    if (_options.applyLaplacianGate) {
      laplacian = laplacianScore(
        resized,
        laplacePixelThreshold: _options.laplacePixelThreshold,
      );
      developer.log(
        'Laplacian score: $laplacian (threshold: ${_options.laplacianThreshold})',
      );
    } else {
      laplacian = 999999; // Bypass
    }

    // Early reject if image is too blurry
    if (_options.applyLaplacianGate &&
        laplacian < _options.laplacianThreshold) {
      developer.log('Image rejected: too blurry (laplacian=$laplacian)');
      return LivenessResult(
        isLive: false,
        score: 0.0, // Definitely spoof if too blurry
        laplacian: laplacian,
        duration: Duration.zero,
      );
    }

    // Convert image to tensor with configured normalization
    final input = toNHWC(resized, normalization: _options.normalizationType);

    // Run inference
    final rawProb = await _runner.inferProb(
      input,
      outputIndex: _options.outputIndex,
    );

    // Convert to liveness probability
    final double liveProb;
    if (_options.outputIsSpoofProbability) {
      // Model outputs spoof probability, invert it
      liveProb = 1.0 - rawProb;
    } else {
      // Model outputs live probability directly
      liveProb = rawProb;
    }

    final isLive = liveProb >= _options.threshold;

    sw.stop();
    final duration = sw.elapsed;

    developer.log(
      'Liveness: isLive=$isLive, liveProb=${liveProb.toStringAsFixed(3)}, '
      'threshold=${_options.threshold}, duration=${duration.inMilliseconds}ms',
    );

    return LivenessResult(
      isLive: isLive,
      score: liveProb, // Now correctly represents liveness probability
      laplacian: laplacian,
      duration: duration,
    );
  }
}
