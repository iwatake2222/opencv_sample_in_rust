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
use opencv::{prelude::*, core, highgui, imgcodecs, imgproc, videoio};

mod detection_engine;
use detection_engine::detection_engine::DetectionEngine;
mod helper_cv;
use helper_cv::*;

/////////////////////////////////////////////////////////////////
#[allow(dead_code)]
enum CaptureSource<'a> {
    Camera(i32),
    Video(&'a str),
}

/////////////////////////////////////////////////////////////////
fn main() {
    test_with_single_image("../resource/dashcam_01.jpg");
    // test_with_cap(CaptureSource::Camera(0));
    // test_with_cap(CaptureSource::Video("D:/devel/video/4K Drive Shin Yokohama to Minato Mirai 21 and Yokohama Bay Bridge.mp4"));
}

#[allow(dead_code)]
fn test_with_single_image(image_filename: &str) {

    let color_generator = ColorGenerator::new(20, 30);

    /* Create detection engine */
    let mut engine = DetectionEngine::new();

    /* Read input image */
    let mut mat_org = imgcodecs::imread(image_filename, imgcodecs::IMREAD_COLOR).unwrap();

    /* Run detection */
    let bbox_list = engine.process(&mat_org);

    /* Draw bounding box */
    for (i, bbox) in bbox_list.iter().enumerate() {
        let color = color_generator.get(bbox.class_id);
        // let color = color_generator.get(i as i32);
        imgproc::rectangle(&mut mat_org, core::Rect::new(bbox.x, bbox.y, bbox.w, bbox.h), color, 2, imgproc::LINE_8, 0).unwrap();
        // draw_text(&mut mat_org, &bbox.label, core::Point::new(bbox.x, bbox.y - 20), 0.6, 1, core::Scalar::new(255., 255., 255., 255.), color, true); 
    }

    highgui::imshow("result", &mat_org).unwrap();
    highgui::wait_key(-1).unwrap();
}

#[allow(dead_code)]
fn test_with_cap(capture_source: CaptureSource) {
    
    let color_generator = ColorGenerator::new(20, 30);

    /* Open capture */
    let mut cap = match capture_source {
        CaptureSource::Camera(id) => videoio::VideoCapture::new(id, videoio::CAP_ANY).unwrap(),
        CaptureSource::Video(filename) => videoio::VideoCapture::from_file(filename, videoio::CAP_ANY).unwrap(),
    };
    if !videoio::VideoCapture::is_opened(&cap).unwrap() {
        panic!("Unable to open capture");
    }

    /* Create detection engine */
    let mut engine = DetectionEngine::new();

    let mut t_all_previous = std::time::Instant::now();
    loop {
        /* Read image */
        let mut mat_org = Mat::default();
        cap.read(&mut mat_org).unwrap();
        if mat_org.empty() {
            break;
        }

        /* Run detection */
        let t_detection_start = std::time::Instant::now();
        let bbox_list = engine.process(&mat_org);
        let t_detection = t_detection_start.elapsed();

        /* Draw bounding box */
        for (i, bbox) in bbox_list.iter().enumerate() {
            let color = color_generator.get(bbox.class_id);
            // let color = color_generator.get(i as i32);
            imgproc::rectangle(&mut mat_org, core::Rect::new(bbox.x, bbox.y, bbox.w, bbox.h), color, 2, imgproc::LINE_8, 0).unwrap();
            // draw_text(&mut mat_org, &bbox.label, core::Point::new(bbox.x, bbox.y - 20), 0.6, 2, core::Scalar::new(255., 255., 255., 255.), color, true); 
        }

        /* Calculate processing time */
        let t_all = t_all_previous.elapsed();
        t_all_previous = std::time::Instant::now();
        let text = format!("FPS = {:5.1}, Process = {:.1} ms", 1.0 / t_all.as_secs_f32(), t_detection.as_secs_f32() * 1000.0);
        draw_text(&mut mat_org, &text, core::Point::new(0, 0), 0.8, 2, core::Scalar::new(255., 0., 0., 255.), core::Scalar::new(200., 200., 200., 255.), true); 

        /* Display the result image */
        highgui::imshow("result", &mat_org).unwrap();
        let key = highgui::wait_key(1).unwrap() as u8 as char;
        if key == 'q' || key as u8 == 27 {
            break;
        }
    }
}
