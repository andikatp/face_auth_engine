import 'dart:developer' as developer;
import 'dart:io';

import 'package:tflite_flutter/tflite_flutter.dart';

class TFLiteRunner {
  final Interpreter _interp;
  final IsolateInterpreter _iso;

  const TFLiteRunner._(this._interp, this._iso);

  static Future<TFLiteRunner> create({
    required bool useGpu,
    required int threads,
  }) async {
    const assetPath = 'packages/face_auth_engine/assets/models/model.tflite';
    final options = await _createOptions(useGpu: useGpu, threads: threads);
    final i = await Interpreter.fromAsset(assetPath, options: options);
    final iso = await IsolateInterpreter.create(address: i.address);

    // Log model info for debugging
    final inputShape = i.getInputTensor(0).shape;
    final outputShape = i.getOutputTensor(0).shape;
    developer.log(
      'Liveness model loaded: input=$inputShape, output=$outputShape',
    );

    final runner = TFLiteRunner._(i, iso);
    return runner;
  }

  Future<void> dispose() async {
    await _iso.close();
    _interp.close();
  }

  /// Warm up the model with dummy inferences
  Future<void> warmUp({int times = 10}) async {
    final input = [
      List.generate(224, (_) => List.generate(224, (_) => [0.0, 0.0, 0.0])),
    ];
    for (var i = 0; i < times; i++) {
      await inferProb(input);
    }
  }

  /// Run inference and return raw output
  /// Returns the probability value at [outputIndex]
  Future<double> inferProb(
    List<List<List<List<double>>>> nhwc, {
    int outputIndex = 0,
  }) async {
    // Check output shape to determine buffer size
    final outputShape = _interp.getOutputTensor(0).shape;
    final outputSize = outputShape.length > 1 ? outputShape[1] : 1;

    final List<List<double>> output = [List.filled(outputSize, 0.0)];
    final Map<int, List<List<double>>> outputs = {0: output};

    await _iso.runForMultipleInputs([nhwc], outputs);

    // Log raw output for debugging
    developer.log('Liveness raw output: ${output[0]}');

    // Return probability at specified index
    if (outputIndex >= outputSize) {
      developer.log(
        'Warning: outputIndex $outputIndex >= outputSize $outputSize, using 0',
      );
      return output[0][0];
    }

    return output[0][outputIndex];
  }
}

Future<InterpreterOptions> _createOptions({
  required bool useGpu,
  required int threads,
}) async {
  final options = InterpreterOptions();

  if (useGpu) {
    try {
      if (Platform.isAndroid) {
        var gpuDelegate = GpuDelegateV2(
          options: GpuDelegateOptionsV2(
            isPrecisionLossAllowed: true,
            inferencePriority1: 2,
          ),
        );
        options.addDelegate(gpuDelegate);
      } else if (Platform.isIOS) {
        var gpuDelegate = GpuDelegate(
          options: GpuDelegateOptions(allowPrecisionLoss: true),
        );
        options.addDelegate(gpuDelegate);
      }
    } catch (e) {
      developer.log('GPU delegate failed, falling back to CPU: $e');
      options.threads = threads;
    }
  } else {
    options.threads = threads;
  }

  return options;
}
