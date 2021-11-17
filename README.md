# OpenCV Sample Projects in Rust 

# Resource
- https://docs.rs/opencv/0.60.0/opencv/
- https://lib.rs/crates/opencv
- https://github.com/twistedfall/opencv-rust/tree/master/examples

# Setup
## Windows
- Install LLVM
    - https://github.com/llvm/llvm-project/releases/download/llvmorg-13.0.0/LLVM-13.0.0-win64.exe
        - https://github.com/llvm/llvm-project
    - Add LLVM to the system Path while installing
- Download OpenCV
    - https://github.com/opencv/opencv/releases/download/4.5.4/opencv-4.5.4-vc14_vc15.exe
        - https://github.com/opencv/opencv
    - Extract to `C:/opencv` (If you use a different directory, change the commands in the next step)
- Set environment variables (Power Shell)
    ```ps
    $env:Path+=";C:/opencv/build/x64/vc15/bin"
    $env:OPENCV_LINK_LIBS="opencv_world454"
    $env:OPENCV_LINK_PATHS="C:/opencv/build/x64/vc15/lib"
    $env:OPENCV_INCLUDE_PATHS="C:/opencv/build/include"
    ```

## Linux
todo

## Add crate
```toml
[dependencies]
opencv = "0.60"
```

## Acknowledgements
- OpenCV
    - https://github.com/opencv/opencv
    - Licensed under the Apache License, Version 2.0
    - Some images are copied from this project
