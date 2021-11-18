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
#[allow(unused_imports)]
use opencv::{prelude::*, core, highgui, imgcodecs, imgproc, videoio};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    /* Read Video */
    // let mut cap = videoio::VideoCapture::new(0, videoio::CAP_ANY)?;
    let mut cap = videoio::VideoCapture::from_file("../resource/Megamind.avi", videoio::CAP_ANY)?;
    if !videoio::VideoCapture::is_opened(&cap).unwrap() {
        panic!("Unable to open capture");
    }
    
    let mut fps = cap.get(videoio::CAP_PROP_FPS)?;
    let width = cap.get(videoio::CAP_PROP_FRAME_WIDTH)? as i32;
    let height = cap.get(videoio::CAP_PROP_FRAME_HEIGHT)? as i32;
    println!("fps = {:}, width = {:}, height = {:}", fps, width, height);
    fps = fps.min(120.0);
    if fps == 0.0 {
        fps = 30.0;
    }

    /* Write Video */
    let mut writer = videoio::VideoWriter::new("test.mp4", videoio::VideoWriter::fourcc('M' as i8,'P' as i8,'4' as i8,'V' as i8)?
        , fps, core::Size { width: width, height: height }, true)?;

    loop {
        /* Read Video */
        let mut mat = Mat::default();
        cap.read(&mut mat)?;
        if mat.empty() {
            break;
        }
        highgui::imshow("test", &mat)?;
        let key = highgui::wait_key(1)? as u8 as char;
        if key == 'q' || key as u8 == 27 {
            break;
        }


        /* Write Video */
        if writer.is_opened()? {
            writer.write(&mat)?;
        }
    }

    /* Write Video */
    if writer.is_opened()? {
        writer.release()?;
    }

    Ok(())
}
