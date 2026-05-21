import 'package:face_auth_engine/src/image/face_image_provider.dart';
import 'package:face_auth_engine/src/liveness/liveness_options.dart';

/// Convert image to NHWC tensor format with configurable normalization
/// Returns a Float32List wrapped in NHWC shape for TFLite inference
List<List<List<List<double>>>> toNHWC(
  FaceImageBuffer resized, {
  NormalizationType normalization = NormalizationType.centered,
}) {
  final bytes = resized.pixels;
  var index = 0;

  return [
    List.generate(
      224,
      (_) => List.generate(224, (_) {
        final r = bytes[index++].toDouble();
        final g = bytes[index++].toDouble();
        final b = bytes[index++].toDouble();

        switch (normalization) {
          case NormalizationType.scaled:
            // x / 255.0 -> [0, 1]
            return [r / 255.0, g / 255.0, b / 255.0];
          case NormalizationType.centered:
            // (x - 127.5) / 127.5 -> [-1, 1]
            return [
              (r - 127.5) / 127.5,
              (g - 127.5) / 127.5,
              (b - 127.5) / 127.5,
            ];
        }
      }),
    ),
  ];
}
