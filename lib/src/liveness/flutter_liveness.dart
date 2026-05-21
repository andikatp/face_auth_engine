import 'package:face_auth_engine/face_auth_engine.dart';
import 'package:face_auth_engine/src/liveness/liveness_engine.dart';
import 'package:face_auth_engine/src/liveness/tflite_runner.dart';

class FlutterLiveness {
  const FlutterLiveness._(this._engine, this._runner, this.imageProvider);
  final LivenessEngine _engine;
  final TFLiteRunner _runner;

  final FaceImageProvider imageProvider;

  static Future<FlutterLiveness> create({
    required FaceImageProvider imageProvider,
    LivenessOptions options = const LivenessOptions(),
  }) async {
    final runner = await TFLiteRunner.create(
      useGpu: options.useGpu,
      threads: options.cpuThreads,
    );
    final engine = LivenessEngine(runner, options, imageProvider);
    return FlutterLiveness._(engine, runner, imageProvider);
  }

  Future<LivenessResult> analyze(FaceImageBuffer face) => _engine.analyze(face);

  Future<void> dispose() => _runner.dispose();
}
