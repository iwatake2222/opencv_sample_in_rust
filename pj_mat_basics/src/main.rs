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
use opencv::{prelude::*, core, highgui, imgcodecs, imgproc, types::{VectorOfPoint2f, VectorOfString, VectorOfVectorOfPoint},};

fn print_mat_info(mat: &mut core::Mat) {
    println!("dims = {:}, rows = {:}, cols = {:}, depth = {:}, channels = {:}, mat_size = {:?}, mat_step = {:?}, elem_size = {:?}, elem_size1 = {:?}, size = {:?}, total = {:?}"
        , mat.dims(), mat.rows(), mat.cols(), mat.depth(), mat.channels(), mat.mat_size(), mat.mat_step(), mat.elem_size(), mat.elem_size1(), mat.size(), mat.total());
    println!("{:?}, {:?}", &mat as *const _, mat.as_raw_mut_Mat());
    println!("{:?}, {:?}, {:?}", mat.data_mut(), mat.ptr_2d(0, 0).unwrap(), mat.at_2d_mut::<core::Vec3b>(0, 0).unwrap() as *const _);   // address of pixel data
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    const WIDTH: i32 = 200;
    const HEIGHT: i32 = 100;

    /* Create Mat */
    let mut mat = Mat::default();
    let mut mat = Mat::zeros(HEIGHT, WIDTH, core::CV_8UC3)?.to_mat()?;
    let mut mat = Mat::new_rows_cols_with_default(HEIGHT, WIDTH, core::CV_8UC3, core::Scalar::default())?;

    println!("\n=== mat original ===");
    print_mat_info(&mut mat);

    /* copy (we usually don't use this) */
    let mut mat2 = Mat::copy(&mut mat)?;    // pixel data is not copied (the address of pixel data is the same)
    println!("\n=== mat2 ===");
    print_mat_info(&mut mat2);

    /* copy_to */
    let mut mat3 = Mat::default();
    mat.copy_to(&mut mat3)?;
    println!("\n=== mat3 ===");
    print_mat_info(&mut mat3);

    /* clone */
    let mut mat4 = mat.clone();
    println!("\n=== mat4 ===");
    print_mat_info(&mut mat4);

    imgproc::put_text(&mut mat, "mat0", core::Point::new(10, 10), highgui::QT_STYLE_NORMAL, 0.8, core::Scalar::new(255., 0., 0., 0.), 1, imgproc::LINE_8, false)?;
    imgproc::put_text(&mut mat2, "mat2", core::Point::new(10, 10), highgui::QT_STYLE_NORMAL, 0.8, core::Scalar::new(255., 0., 0., 0.), 1, imgproc::LINE_8, false)?;
    imgproc::put_text(&mut mat3, "mat3", core::Point::new(10, 10), highgui::QT_STYLE_NORMAL, 0.8, core::Scalar::new(255., 0., 0., 0.), 1, imgproc::LINE_8, false)?;
    imgproc::put_text(&mut mat4, "mat4", core::Point::new(10, 10), highgui::QT_STYLE_NORMAL, 0.8, core::Scalar::new(255., 0., 0., 0.), 1, imgproc::LINE_8, false)?;
    highgui::imshow("mat", &mat)?;
    highgui::imshow("mat2", &mat2)?;
    highgui::imshow("mat3", &mat3)?;
    highgui::imshow("mat4", &mat4)?;
    


    /* Overlay using ROI (directly modify ROI area) */
    let mut mat = Mat::new_rows_cols_with_default(HEIGHT, WIDTH, core::CV_8UC3, core::Scalar::new(255., 0., 0., 0.))?;
    let mut mat_roi = Mat::roi(&mat, core::Rect::new(10, 10, 50, 50))?;     // the addres of pixel data is inside that of mat
    imgproc::rectangle(&mut mat_roi, core::Rect::new(0, 0, 50, 50), core::Scalar::new(0., 255., 0., 0.), 2, imgproc::LINE_8, 0)?;
    imgproc::circle(&mut mat, core::Point::new(30, 30), 10, core::Scalar::new(0., 255., 0., 0.), 2, imgproc::LINE_8, 0)?;
    highgui::imshow("overlay0_mat", &mat)?;
    highgui::imshow("overlay0_mat_roi", &mat_roi)?;

    /* Overlay using ROI (copy mask image to ROI area) */
    let mut mat = Mat::new_rows_cols_with_default(HEIGHT, WIDTH, core::CV_8UC3, core::Scalar::new(255., 0., 0., 0.))?;
    let mut mat_roi = Mat::roi(&mat, core::Rect::new(10, 10, 50, 50))?;     // the addres of pixel data is inside that of mat
    let mut mat_mask = Mat::new_rows_cols_with_default(50, 50, core::CV_8UC3, core::Scalar::new(0., 0., 255., 0.))?;
    // mat_roi = mat_I_want_to_write.clone();   // can't use clone because clone create completely new mat with newly allocated data
    mat_mask.copy_to(&mut mat_roi)?;
    highgui::imshow("overlay1_mat", &mat)?;
    highgui::imshow("overlay1_mat_roi", &mat_roi)?;
    highgui::imshow("overlay1_mat_mask", &mat_mask)?;


    highgui::wait_key(-1)?;
    Ok(())
}
