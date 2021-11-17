/* Copyright 2021 iwatake2222
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

use opencv::{prelude::*, imgcodecs::*, highgui::*};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mat = imread("../resource/fruits.jpg", IMREAD_COLOR)?;
    println!("{:?}", mat);
    imshow("test", &mat)?;
    wait_key(-1)?;
    Ok(())
}

// fn main() {
//     let mat = opencv::imgcodecs::imread("../resource/fruits.jpg", opencv::imgcodecs::IMREAD_COLOR).unwrap();
//     println!("{:?}", mat);
//     opencv::highgui::imshow("test", &mat).unwrap();
//     opencv::highgui::wait_key(-1).unwrap();
// }
