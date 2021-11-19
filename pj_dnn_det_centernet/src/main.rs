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

/////////////////////////////////////////////////////////////////
#[allow(unused_imports)]
use opencv::{prelude::*, core, highgui, imgcodecs, imgproc};

mod detection_engine;
use detection_engine::detection_engine::DetectionEngine;

/////////////////////////////////////////////////////////////////
const TESTIMAGE_FILENAME: &str = "../resource/dog.jpg";

/////////////////////////////////////////////////////////////////
fn main() {
    test_single_image(TESTIMAGE_FILENAME);
}

fn test_single_image(image_filename: &str) {
    /* Create detection engine */
    let mut engine = DetectionEngine::new();

    /* Read input image */
    let mut mat_org = imgcodecs::imread(image_filename, imgcodecs::IMREAD_COLOR).unwrap();

    /* Run detection */
    let bbox_list = engine.process(&mat_org);

    /* Draw bounding box */
    for bbox in bbox_list {
        imgproc::rectangle(&mut mat_org, core::Rect::new(bbox.x, bbox.y, bbox.w, bbox.h), core::Scalar::new(255., 0., 0., 0.), 2, imgproc::LINE_8, 0).unwrap();
        imgproc::put_text(&mut mat_org, &bbox.label, core::Point::new(bbox.x, bbox.y), highgui::QT_STYLE_NORMAL, 0.8, core::Scalar::new(255., 0., 0., 0.), 2, imgproc::LINE_8, false).unwrap();
    }

    highgui::imshow("result", &mat_org).unwrap();

    highgui::wait_key(-1).unwrap();
}
