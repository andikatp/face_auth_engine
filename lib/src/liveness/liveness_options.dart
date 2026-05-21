/// Normalization type for image preprocessing
enum NormalizationType {
  /// x / 255.0 (scaled to 0-1)
  scaled,

  /// (x - 127.5) / 127.5 (scaled to -1 to 1, ImageNet-style)
  centered,
}

/// Configuration options for liveness detection
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
  });

  /// Use GPU for processing
  final bool useGpu;

  /// Number of threads for CPU inference
  final int cpuThreads;

  /// Liveness threshold (probability must be >= this to be considered live)
  final double threshold;

  /// Whether to apply laplacian gate (blur detection)
  final bool applyLaplacianGate;

  /// Laplacian score threshold (lower = more blur allowed)
  /// Typical mobile camera selfies score 300-1500
  final int laplacianThreshold;

  /// Number of pixels that must exceed the laplacianThreshold
  final int laplacePixelThreshold;

  /// Normalization type for image preprocessing
  final NormalizationType normalizationType;

  /// Which output index to read from the model
  /// Set to 0 if model outputs single spoof probability
  /// Set to 1 if model outputs [live_prob, spoof_prob]
  final int outputIndex;

  /// Whether model output is spoof probability (true)
  /// or live probability (false)
  final bool outputIsSpoofProbability;

  /// Number of warm-up inference runs
  final int warmUpIterations;

  @override
  String toString() {
    return 'LivenessOptions('
        'useGpu: $useGpu, '
        'cpuThreads: $cpuThreads, '
        'threshold: $threshold, '
        'applyLaplacianGate: $applyLaplacianGate, '
        'laplacianThreshold: $laplacianThreshold, '
        'normalizationType: $normalizationType'
        ')';
  }


}
