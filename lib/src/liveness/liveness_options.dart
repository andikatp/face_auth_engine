import 'package:meta/meta.dart';

/// Normalization type for image preprocessing.
enum NormalizationType {
  /// x / 255.0 (scaled to 0–1).
  scaled,

  /// (x - 127.5) / 127.5 (scaled to -1 to 1, ImageNet-style).
  centered,

  /// Adaptive: scales relative to the image's actual min/max luma range.
  /// More robust to lighting variation for moderate-light images.
  /// Equivalent to: (x - imgMin) / (imgMax - imgMin) → [-1, 1].
  adaptiveCentered,
}

/// Configuration options for liveness detection.
@immutable
class LivenessOptions {
  const LivenessOptions({
    this.useGpu = false,
    this.cpuThreads = 1,
    this.threshold = 0.5,
    this.applyLaplacianGate = false,
    this.laplacianThreshold = 500,
    this.laplacePixelThreshold = 50,
    this.normalizationType = NormalizationType.centered,
    this.outputIndex = 0,
    this.outputIsSpoofProbability = true,
    this.warmUpIterations = 10,
    this.applyLowLightGate = true,
    this.lowLightThreshold = 40.0,
  });

  /// Use GPU for processing.
  final bool useGpu;

  /// Number of threads for CPU inference.
  final int cpuThreads;

  /// Liveness threshold (probability must be >= this to be considered live).
  final double threshold;

  /// Whether to apply laplacian gate (blur detection).
  final bool applyLaplacianGate;

  /// Laplacian score threshold (lower = more blur allowed).
  /// Typical mobile camera selfies score 300–1500.
  final int laplacianThreshold;

  /// Number of pixels that must exceed the laplacianThreshold.
  final int laplacePixelThreshold;

  /// Normalization type for image preprocessing.
  final NormalizationType normalizationType;

  /// Which output index to read from the model.
  /// Set to 0 if model outputs single spoof probability.
  /// Set to 1 if model outputs [live_prob, spoof_prob].
  final int outputIndex;

  /// Whether model output is spoof probability (true)
  /// or live probability (false).
  final bool outputIsSpoofProbability;

  /// Number of warm-up inference runs.
  final int warmUpIterations;

  /// Whether to apply the low-light gate before inference.
  /// When enabled, images with mean luma below [lowLightThreshold] are
  /// rejected with `LivenessRejectionReason.lowLight` instead of being
  /// misclassified as spoofs. Enabled by default.
  final bool applyLowLightGate;

  /// Mean luma threshold (0–255) below which an image is considered too dark.
  /// Default 40.0 maps to roughly the bottom 16% of luma range.
  /// Typical well-lit selfies score 80–160.
  final double lowLightThreshold;

  @override
  String toString() {
    return 'LivenessOptions('
        'useGpu: $useGpu, '
        'cpuThreads: $cpuThreads, '
        'threshold: $threshold, '
        'applyLaplacianGate: $applyLaplacianGate, '
        'laplacianThreshold: $laplacianThreshold, '
        'normalizationType: $normalizationType, '
        'applyLowLightGate: $applyLowLightGate, '
        'lowLightThreshold: $lowLightThreshold'
        ')';
  }

  @override
  bool operator ==(Object other) {
    if (identical(this, other)) return true;
    return other is LivenessOptions &&
        other.useGpu == useGpu &&
        other.cpuThreads == cpuThreads &&
        other.threshold == threshold &&
        other.applyLaplacianGate == applyLaplacianGate &&
        other.laplacianThreshold == laplacianThreshold &&
        other.laplacePixelThreshold == laplacePixelThreshold &&
        other.normalizationType == normalizationType &&
        other.outputIndex == outputIndex &&
        other.outputIsSpoofProbability == outputIsSpoofProbability &&
        other.warmUpIterations == warmUpIterations &&
        other.applyLowLightGate == applyLowLightGate &&
        other.lowLightThreshold == lowLightThreshold;
  }

  @override
  int get hashCode => Object.hash(
    useGpu,
    cpuThreads,
    threshold,
    applyLaplacianGate,
    laplacianThreshold,
    laplacePixelThreshold,
    normalizationType,
    outputIndex,
    outputIsSpoofProbability,
    warmUpIterations,
    applyLowLightGate,
    lowLightThreshold,
  );
}
