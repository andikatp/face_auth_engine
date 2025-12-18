# Face Auth Engine

A stateless face recognition engine for Flutter with ML Kit detection, landmark alignment, and MobileFaceNet embeddings.

## Features

- 🔍 **Face Detection**: Google ML Kit for robust face detection
- 📐 **Face Alignment**: Landmark-based similarity transform alignment
- 🧠 **Embedding Generation**: MobileFaceNet 192-dimensional embeddings
- ✅ **Verification**: 1:1 Face comparison
- 🎯 **Stateless Design**: Purely functional API, no internal state management

## Important Design Principle

This package **does not store any face data**. It provides tools to extract embeddings and compare them.

## Installation

Add to your `pubspec.yaml`:

```yaml
dependencies:
  face_auth_engine:
    git:
      url: https://github.com/<your-username>/face_auth_engine.git
```

## Requirements

### 1. Add MobileFaceNet Model

Place the `mobilefacenet.tflite` model file in your app's `assets/models/` directory:

```
your_app/
├── assets/
│   └── models/
│       └── mobilefacenet.tflite
└── pubspec.yaml
```

Update your `pubspec.yaml`:

```yaml
flutter:
  assets:
    - assets/models/mobilefacenet.tflite
```

### 2. Configure ML Kit

Follow [Google ML Kit Face Detection setup](https://pub.dev/packages/google_mlkit_face_detection) for your platform.

## Quick Start

### 1. Initialize Engine

```dart
import 'package:face_auth_engine/face_auth_engine.dart';

final engine = FaceAuthEngine(
  config: FaceConfig(
    recognitionThreshold: 1.0,
  ),
);
```

### 2. Extract Embedding

```dart
try {
  // Convert image file path to embedding
  final List<double> embedding = await engine.convertToEmbedded('/path/to/image.jpg');
  
  // Persist "embedding" to your database (e.g. as JSON or blob)
} catch (e) {
  print('Error: $e'); // e.g. No face detected
}
```

### 3. Verify Person (1:1 Verification)

```dart
try {
  // Check if the person in the new image matches a known embedding
  bool isSamePerson = await engine.isThePersonTheSame(
    '/path/to/new_image.jpg',
    knownEmbedding, // List<double> you stored directly
  );

  if (isSamePerson) {
    print('Verified!');
  } else {
    print('Not the same person.');
  }
} catch (e) {
  print('Error: $e');
}
```

### 4. Bulk Extraction

```dart
final paths = ['/path/1.jpg', '/path/2.jpg', '/path/3.jpg'];
// Returns List<List<double>>
final allEmbeddings = await engine.convertFromListToEmbedded(paths);
```

## API Reference

### FaceAuthEngine

| Method                                                  | Description                                                |
| ------------------------------------------------------- | ---------------------------------------------------------- |
| `convertToEmbedded(String path)`                        | Extract embedding from image file path                     |
| `convertFromListToEmbedded(List<String> paths)`         | Extract embeddings from multiple image paths               |
| `isThePersonTheSame(String path, List<double> known)`   | Verify if image matches known embedding                    |
| `matchFaceAgainstList(String path, List<List<double>>)` | Verify if image matches any known embedding in a list      |
| `compareFaces(String path1, String path2)`              | Verify if two images belong to the same person             |
| `dispose()`                                             | Release resources                                          |

### FaceConfig

```dart
const FaceConfig({
  double recognitionThreshold = 1.0,  // Lower = stricter
  int requiredEnrollmentSamples = 5,  // (Unused in stateless mode)
  int minFaceSize = 80,
  double maxRollAngle = 15.0,
});
```

## Error Handling

The package throws exceptions for:

- No face detected
- Multiple faces detected
- Missing facial landmarks
- Face quality too low (size, angle)

Always wrap calls in try-catch blocks.

## License

MIT License
