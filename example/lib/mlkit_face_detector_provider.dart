import 'dart:io';
import 'dart:math';

import 'package:face_auth_engine/face_auth_engine.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart' as mlkit;

/// Google ML Kit implementation for FaceDetectorProvider.
/// Copy this file into your main application and pass an instance of this
/// class to FaceAuthEngine and LivenessDetector.
class MLKitFaceDetectorProvider implements FaceDetectorProvider {
  late mlkit.FaceDetector _faceDetector;

  MLKitFaceDetectorProvider() {
    final options = mlkit.FaceDetectorOptions(
      enableLandmarks: true,
      enableClassification: false,
      enableTracking: false,
      performanceMode: mlkit.FaceDetectorMode.accurate,
    );
    _faceDetector = mlkit.FaceDetector(options: options);
  }

  @override
  Future<FaceDetectionResult> detectFace(File imageFile) async {
    final inputImage = mlkit.InputImage.fromFile(imageFile);
    List<mlkit.Face> faces = await _faceDetector.processImage(inputImage);

    if (faces.isEmpty) {
      throw Exception(
        'No face detected. Please ensure:\n'
        '• Your face is clearly visible\n'
        '• Good lighting conditions\n'
        '• Face is frontal (not side profile)',
      );
    }
    if (faces.length > 1) {
      throw Exception(
        'Multiple faces detected (${faces.length} faces).\n'
        'Please ensure only one person is in the image.',
      );
    }

    final face = faces.first;
    final leftEye = face.landmarks[mlkit.FaceLandmarkType.leftEye];
    final rightEye = face.landmarks[mlkit.FaceLandmarkType.rightEye];
    final noseTip = face.landmarks[mlkit.FaceLandmarkType.noseBase];
    final leftMouth = face.landmarks[mlkit.FaceLandmarkType.leftMouth];
    final rightMouth = face.landmarks[mlkit.FaceLandmarkType.rightMouth];

    if (leftEye == null || rightEye == null || noseTip == null || leftMouth == null || rightMouth == null) {
      throw Exception('Required face landmarks missing. Please use a clear frontal face photo.');
    }

    return FaceDetectionResult(
      landmarks: [
        FaceLandmark(Point(leftEye.position.x, leftEye.position.y)),
        FaceLandmark(Point(rightEye.position.x, rightEye.position.y)),
        FaceLandmark(Point(noseTip.position.x, noseTip.position.y)),
        FaceLandmark(Point(leftMouth.position.x, leftMouth.position.y)),
        FaceLandmark(Point(rightMouth.position.x, rightMouth.position.y)),
      ],
      boundingBox: face.boundingBox,
    );
  }

  @override
  void dispose() {
    _faceDetector.close();
  }
}
