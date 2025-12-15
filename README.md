# Face Auth Engine

A stateless face recognition engine for Flutter with ML Kit detection, landmark alignment, and MobileFaceNet embeddings.

## Features

- 🔍 **Face Detection**: Google ML Kit for robust face detection
- 📐 **Face Alignment**: Landmark-based similarity transform alignment
- 🧠 **Embedding Generation**: MobileFaceNet 192-dimensional embeddings
- 📊 **Enrollment**: Multi-sample enrollment with averaging
- ✅ **Recognition**: L2 distance-based matching
- 🎯 **Stateless Design**: No internal persistence, app controls storage

## Important Design Principle

This package **does not store any face data**. All persistence decisions are delegated to your application. You must:

1. Receive `FaceEmbedding` objects from enrollment
2. Persist them using your preferred method (SQLite, SharedPreferences, Firebase, etc.)
3. Load and import embeddings before recognition

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
    requiredEnrollmentSamples: 5,
  ),
);
```

### 2. Enroll a Person

```dart
try {
  // Pick 5 images of the same person
  for (int i = 0; i < 5; i++) {
    final imageFile = await pickImage(); // Your image picker
    final embedding = await engine.extractEmbedding(imageFile);
    engine.enrollSample('person_id', embedding);

    print('Sample ${i + 1}/5 enrolled');
  }

  // Build final averaged embedding
  final finalEmbedding = engine.buildFinalEmbedding('person_id');

  if (finalEmbedding != null) {
    // Create FaceEmbedding and persist it
    final faceEmbedding = FaceEmbedding(
      id: 'person_id',
      embedding: finalEmbedding,
    );

    // Save to your database
    await yourDatabase.saveFaceEmbedding(faceEmbedding.toJson());
  }
} catch (e) {
  print('Enrollment error: $e');
}
```

### 3. Recognize a Face

```dart
try {
  // Load embeddings from your database
  final storedEmbeddings = await yourDatabase.loadAllFaceEmbeddings();
  final faceEmbeddings = storedEmbeddings
      .map((json) => FaceEmbedding.fromJson(json))
      .toList();

  // Import embeddings into engine
  engine.importEmbeddings(faceEmbeddings);

  // Extract embedding from query image
  final queryFile = await pickImage();
  final queryEmbedding = await engine.extractEmbedding(queryFile);

  // Recognize
  final personId = engine.recognize(queryEmbedding);

  if (personId != null) {
    print('Recognized: $personId');
  } else {
    print('Unknown person');
  }
} catch (e) {
  print('Recognition error: $e');
}
```

## API Reference

### FaceAuthEngine

| Method                                  | Description                            |
| --------------------------------------- | -------------------------------------- |
| `extractEmbedding(File)`                | Extract face embedding from image file |
| `enrollSample(String, Float32List)`     | Add enrollment sample for a person     |
| `isEnrollmentComplete(String)`          | Check if enrollment is complete        |
| `buildFinalEmbedding(String)`           | Get averaged embedding for person      |
| `exportAveragedEmbeddings()`            | Export all enrolled embeddings         |
| `importEmbeddings(List<FaceEmbedding>)` | Import embeddings for recognition      |
| `recognize(Float32List)`                | Recognize face from embedding          |
| `clear()`                               | Clear all enrollment data              |
| `dispose()`                             | Release resources                      |

### FaceConfig

```dart
const FaceConfig({
  double recognitionThreshold = 1.0,  // Lower = stricter
  int requiredEnrollmentSamples = 5,
  int minFaceSize = 80,
  double maxRollAngle = 15.0,
});
```

### FaceEmbedding

```dart
class FaceEmbedding {
  final String id;
  final Float32List embedding;

  Map<String, dynamic> toJson();
  factory FaceEmbedding.fromJson(Map<String, dynamic>);
}
```

## Error Handling

The package throws exceptions for:

- No face detected
- Multiple faces detected
- Missing facial landmarks
- Face quality too low (size, angle)
- Inconsistent enrollment samples

Always wrap calls in try-catch blocks.

## License

MIT License
