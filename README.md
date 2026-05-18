# face_auth_engine

Model: MobileFaceNet (Recognition) / MobileNetV2 (Liveness) | License: MIT

A stateless face authentication engine for Flutter, featuring 1:1 face verification and passive liveness detection.

✨ **Features**

- **Stateless Architecture** — pure functional API, no internal state
- **Face Recognition** — MobileFaceNet embeddings (192-dim)
- **Liveness Detection** — On-device anti-spoofing (MobileNetV2)
- **Agnostic Face Detection** — Bring your own face detector (e.g. `google_mlkit_face_detection`), so this package doesn't bloat your app size with duplicate ML models.
- **Precise Alignment** — Landmark-based similarity transformation
- **Optimized Performance** — TFLite GPU delegation support

🧠 **Model Information**

**Face Recognition:**

- **Model**: MobileFaceNet
- **Output**: 192-dimensional float vector (L2 normalized)
- **Metric**: Euclidean distance / Cosine similarity

**Liveness Detection:**

- **Model**: MobileNetV2 (based on open liveness research)
- **Input**: 224x224 RGB
- **Classes**:
  - 0 = Real
  - 1 = Spoof

🚀 **Usage**

### 1. Face Recognition

**Initialize Engine**

First, you must create a `FaceDetectorProvider` and `FaceImageProvider` (see the `example` folder for implementations). Then:

```dart
import 'package:face_auth_engine/face_auth_engine.dart';

final engine = FaceAuthEngine(
  faceDetector: MLKitFaceDetectorProvider(), // Your implementation
  imageProvider: ImageProviderImpl(), // Your implementation
  config: FaceConfig(
    recognitionThreshold: 1.0, // Adjust stricter/looser
  ),
);
```

**Extract Embedding**

```dart
try {
  // Convert image file path to embedding
  final List<double> embedding = await engine.convertToEmbedded('/path/to/image.jpg');

  // Store 'embedding' in your database
} catch (e) {
  print('Error: $e'); // e.g. No face detected, or multiple faces
}
```

**Verify Person (1:1)**

```dart
try {
  // Check if new image matches a known embedding
  bool isSame = await engine.isThePersonTheSame(
    '/path/to/check.jpg',
    knownEmbedding, // List<double> from DB
  );

  print(isSame ? 'Verified!' : 'Not the same person');
} catch (e) {
  print('Error: $e');
}
```

**Compare Two Images**

```dart
bool match = await engine.compareFaces(path1, path2);
```

### 2. Liveness Detection

**Initialize Detector**

```dart
import 'package:face_auth_engine/face_auth_engine.dart';
import 'package:image/image.dart' as imglib;

late final LivenessDetector liveness;

Future<void> initLiveness() async {
  liveness = await LivenessDetector.create(
    faceDetector: MLKitFaceDetectorProvider(), // Your implementation
    options: LivenessOptions(
      useGpu: true,
      options: 0.5, // Threshold
    ),
  );
}
```

**Analyze Face**

```dart
/// Option A: Pass a full image file (auto-detects & crops face)
Future<void> checkLiveness(File imageFile) async {
  try {
    final result = await liveness.detectLiveness(imageFile);

    print("Live: ${result.isLive}");
    print("Score: ${result.score}");
    print("Laplacian: ${result.laplacian}");
  } catch (e) {
    print("Error: $e"); // e.g. "No face detected"
  }
}

/// Option B: Pass a pre-cropped face image
Future<void> checkLivenessCropped(imglib.Image faceCrop) async {
  final result = await liveness.analyze(faceCrop);
  // ...
}
```

**Cleanup**

```dart
@override
void dispose() {
  engine.dispose();
  liveness.dispose();
  super.dispose();
}
```

📊 **Output Example (Liveness)**

```text
Live: true
Score: 0.98
Laplacian: 8540.2
Time: 45 ms
```

⚙️ **Configuration**

**FaceConfig**

```dart
FaceConfig(
  recognitionThreshold: 1.0,
  minFaceSize: 80,
  maxRollAngle: 15.0,
)
```

**LivenessOptions**

```dart
LivenessOptions(
  threshold: 0.5,       // Score required to be considered 'live'
  laplacianThreshold: 5000, // Min clarity score
  useGpu: true,
)
```

❗ **Usage Notes**

- **Liveness Input**: use `liveness.detectLiveness(file)` for full images (auto-crop), or `liveness.analyze(image)` if you already have a cropped face.
- **Threading**: Liveness analysis runs on an Isolate to prevent UI jank.
- **Security**: This package provides probabilistic estimations. Use as part of a broader security pipeline (e.g. combine with backend verification).

🔐 **Disclaimer**

This library provides estimation tools for utility and basic security. It should not be the sole line of defense for high-value financial applications. Evaluate accordance with your security requirements.

🤝 **Acknowledgements**

- TensorFlow Lite team
- MobileFaceNet authors
- Google ML Kit

## 📄 License

This project is licensed under the MIT License.

### Third-Party Models

This package includes pre-trained TFLite models sourced from public repositories.

The authors of this repository do not claim ownership of these models.  
All rights and licenses belong to their respective original authors.

Users are responsible for verifying and complying with the licenses of the original model sources when using this package, especially for commercial use.
