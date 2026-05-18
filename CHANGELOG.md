## 0.0.3

* Completely removed `package:image` dependency, enabling true dependency inversion with `FaceImageProvider`.

## 0.0.2

* Abstracted face detection so the library no longer forcefully bundles `google_mlkit_face_detection`, allowing host apps to inject their own provider to save app bundle size.

## 0.0.1

* Initial release.
