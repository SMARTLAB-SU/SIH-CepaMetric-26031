# Manual Review Register

Complete these items before submission or public release:

1. Replace `TeamName` and `SIHID`; add the institution, team members, maintainer, and contact information.
2. Confirm the Kaggle dataset URL, version, license, class counts, split sizes, augmentation statistics, and redistribution conditions.
3. Reconcile notebook labels (`molded`, `normal`, `rotten`, `sprouted`) with application labels (`Moldy`, `Normal`, `Rotten`, `Sprouted`).
4. Independently evaluate every retained checkpoint. No complete test metrics or evaluation report was present in the source project.
5. Confirm the architecture and preprocessing compatibility of all five base-model weight files and the stacking meta-classifier.
6. Only the Xception training notebook was supplied. Add reproducible training code/configuration for ConvNeXt Tiny, EfficientNetV2-B0, DenseNet121, ResNet50V2, and the stacking model.
7. The source contained two byte-identical Xception weight files. One duplicate was omitted from the new repository.
8. `XAI Gradcam/xai_manifest.json` was actually a NumPy binary, not JSON. It is retained as `xai_manifest_mislabeled.npy` for investigation.
9. Validate the three supplied Grad-CAM examples and document the method, model checkpoint, normalization, and interpretation limits.
10. Confirm Intel RealSense D455f availability, calibration procedure, camera permissions, and metric measurement accuracy on the target conveyor.
11. Review hard-coded hardware defaults: Arduino `COM3`, 9600 baud, camera index `0`, 4-second capture interval, and 850 mm conveyor depth.
12. Validate the classical box/onion detection and tracking subsystem against the intended physical onion geometry. Its supplied README primarily describes box measurement.
13. Add an installer if required. No application installer, executable package, or setup definition was found.
14. Add formal project documentation, SIH presentation, application screenshots, and a demo video if required. Only Grad-CAM result images and a measurement CSV were available.
15. Confirm ownership and redistribution rights for model weights, the SMART Lab logo, source code, dataset-derived outputs, and third-party dependencies before applying the MIT license publicly.
16. Install Git LFS before committing. Several model files exceed GitHub's normal file-size limit.

