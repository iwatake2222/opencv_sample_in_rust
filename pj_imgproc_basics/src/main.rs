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

use opencv::{prelude::*, core, highgui, imgcodecs, imgproc};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mat_src = imgcodecs::imread("../resource/fruits.jpg", imgcodecs::IMREAD_COLOR)?;
    imgcodecs::imwrite("test.jpg", &mat_src, &core::Vector::default())?;

    let mut mat = Mat::default();
    imgproc::resize(&mat_src, &mut mat, core::Size { width: 100, height: 100, }, 0.0, 0.0, imgproc::INTER_LINEAR)?;
    highgui::imshow("test0", &mat)?;

    let mut mat = Mat::default();
    imgproc::cvt_color(&mat_src, &mut mat, imgproc::COLOR_BGR2GRAY, 0)?;
    highgui::imshow("test1", &mat)?;

    // let mat = Mat::new_rows_cols_with_default(64, 128, core::CV_8UC3, core::Scalar::all(128.))?;
    let mat = Mat::new_rows_cols_with_default(64, 128, core::CV_8UC3, core::Scalar::new(255., 0., 0., 0.))?;
    highgui::imshow("test2", &mat)?;

    let mut mat = Mat::new_rows_cols_with_default(200, 200, core::CV_8UC3, core::Scalar::default())?;
    imgproc::rectangle(&mut mat, core::Rect::new(10, 10, 50, 50), core::Scalar::new(255., 0., 0., 0.), 2, imgproc::LINE_8, 0)?;
    imgproc::line(&mut mat, core::Point::new(50, 50), core::Point::new(50, 100), core::Scalar::new(255., 0., 0., 0.), 2, imgproc::LINE_8, 0)?;
    imgproc::circle(&mut mat, core::Point::new(100, 100), 10, core::Scalar::new(255., 0., 0., 0.), 2, imgproc::LINE_8, 0)?;
    imgproc::put_text(&mut mat, "TEST", core::Point::new(100, 100), highgui::QT_STYLE_NORMAL, 0.8, core::Scalar::new(255., 0., 0., 0.), 1, imgproc::LINE_8, false)?;
    let text = String::from("ABC");
    imgproc::put_text(&mut mat, &text, core::Point::new(100, 150), highgui::QT_STYLE_NORMAL, 0.8, core::Scalar::new(255., 0., 0., 0.), 1, imgproc::LINE_8, false)?;
    highgui::imshow("test3", &mat)?;

    highgui::wait_key(-1)?;
    Ok(())
}

