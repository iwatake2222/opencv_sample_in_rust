# YOLOX Object Detection with OpenCV in Rust

![00_doc/yolox.jpg](00_doc/yolox.jpg)

Sample project to run YOLOX Object Detection with OpenCV in Rust.

## How to Run
1. Install Rust and OpenCV
    - https://www.rust-lang.org/tools/install
    - https://lib.rs/crates/opencv
2. Download the model
    - https://github.com/PINTO0309/PINTO_model_zoo/blob/main/132_YOLOX/download_nano.sh
    - copy `saved_model_yolox_nano_320x320/yolox_nano_320x320.onnx` to `resource/model/yolox_nano_320x320.onnx`
3. `cargo run`


## Acknowledgements
- https://github.com/PINTO0309/PINTO_model_zoo
- https://github.com/Megvii-BaseDetection/YOLOX

