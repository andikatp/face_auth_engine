import 'package:face_auth_engine/src/image/face_image_provider.dart';
import 'package:face_auth_engine/src/liveness/liveness_options.dart';

/// Convert image to NHWC tensor format with configurable normalization.
/// Returns a Float32List wrapped in NHWC shape for TFLite inference.
List<List<List<List<double>>>> toNHWC(
  FaceImageBuffer resized, {
  NormalizationType normalization = .centered,
}) {
  final bytes = resized.pixels;
  var index = 0;

  // Pre-compute adaptive range if needed
  var adaptiveMin = 255;
  var adaptiveMax = 0;
  if (normalization == .adaptiveCentered) {
    for (var p = 0; p < bytes.length; p++) {
      final v = bytes[p];
      if (v < adaptiveMin) adaptiveMin = v;
      if (v > adaptiveMax) adaptiveMax = v;
    }
    // Avoid division by zero on a fully uniform image
    if (adaptiveMax == adaptiveMin) adaptiveMax = adaptiveMin + 1;
  }

  final range = (adaptiveMax - adaptiveMin).toDouble();

  return [
    List.generate(
      224,
      (_) => List.generate(224, (_) {
        final r = bytes[index++].toDouble();
        final g = bytes[index++].toDouble();
        final b = bytes[index++].toDouble();

        switch (normalization) {
          case .scaled:
            // x / 255.0 -> [0, 1]
            return [r / 255.0, g / 255.0, b / 255.0];
          case .centered:
            // (x - 127.5) / 127.5 -> [-1, 1]
            return [
              (r - 127.5) / 127.5,
              (g - 127.5) / 127.5,
              (b - 127.5) / 127.5,
            ];
          case .adaptiveCentered:
            // Scale to [0, 1] relative to image's actual range, then to [-1, 1]
            final rn = (r - adaptiveMin) / range * 2.0 - 1.0;
            final gn = (g - adaptiveMin) / range * 2.0 - 1.0;
            final bn = (b - adaptiveMin) / range * 2.0 - 1.0;
            return [rn, gn, bn];
        }
      }),
    ),
  ];
}
