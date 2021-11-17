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

use opencv::{prelude::*, core, highgui, imgcodecs, imgproc, videoio};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // let mut cam = videoio::VideoCapture::new(0, videoio::CAP_ANY)?;
    let mut cam = videoio::VideoCapture::from_file("../resource/Megamind.avi", videoio::CAP_ANY)?;
    if !videoio::VideoCapture::is_opened(&cam).unwrap() {
        panic!("Unable to open capture");
    }
    loop {
        let mut mat = Mat::default();
        cam.read(&mut mat)?;
        if mat.size()?.width <= 0 {
            break;
        }
        highgui::imshow("test", &mat)?;
        let key = highgui::wait_key(1)? as u8 as char;
        if key == 'q' || key as u8 == 27 {
            break;
        }
    }

    Ok(())
}
