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

use opencv::{prelude::*, core, highgui, imgcodecs, imgproc, types::{VectorOfPoint2f, VectorOfString, VectorOfVectorOfPoint},};


fn main() -> Result<(), Box<dyn std::error::Error>> {
    let width = 200;
    let height = 100;

    // let mut mat = Mat::zeros(height, width, core::CV_8UC3)?;     // todo: how to convert MatExpr to Mat
    let mut mat = Mat::new_rows_cols_with_default(height, width, core::CV_8UC3, core::Scalar::new(255., 0., 0., 0.))?;
    println!("rows = {:}, cols = {:}, depth = {:}, channels = {:}, mat_step = {:?}, elem_size = {:?}, size = {:?}",
        mat.rows(), mat.cols(), mat.depth(), mat.channels(), mat.mat_step(), mat.elem_size(), mat.size());

    /* Read color of a specific point using at_2d */
    let color = mat.at_2d::<core::Vec3b>(10, 10)?;
    println!("Color at (10, 10) = B:{:}, G:{:}, R:{:}", color[0], color[1], color[2]);

    /* Write color of a specific point using at_2d_mut */
    let new_color = core::Vec3b::from([0, 255, 0]); // [B, G, R]
    for y in 0..10 {
        for x in 0..20 {
            let color_org = mat.at_2d_mut::<core::Vec3b>(y, x)?;
            *color_org = new_color;
        }
    }

    /* Get the pointer of data using ptr_2d */
    let ptr_org: *mut u8 = mat.ptr_2d(0, 0)? as _;
    println!("{:?}", ptr_org);
    unsafe {
        for y in 20..30 {
            for x in 0..20 {
                let ptr = ptr_org.add(x * mat.elem_size()? + y * mat.mat_step()[0]);
                *ptr.add(0) = new_color[0]; // B
                *ptr.add(1) = new_color[1]; // G
                *ptr.add(2) = new_color[2]; // R

            }
        }
    }

    /* Create Array from Mat (the address of pixel date is the same, so I can overwrite pixel data using array) */
    let ptr_org: *mut u8 = mat.ptr_2d(0, 0)? as _;
    let mut arr = unsafe {std::slice::from_raw_parts_mut(ptr_org, mat.total() * mat.elem_size()?)};
    println!("mat_ptr = {:?}, arr = {:?}", ptr_org, arr.as_ptr());
    println!("Color at (10, 10) = B:{:}, G:{:}, R:{:}", arr[0], arr[1], arr[2]);
    for y in 30..40 {
        for x in 0..20 {
            arr[(y * width * 3 + x * 3 + 0) as usize] = 0 ;
            arr[(y * width * 3 + x * 3 + 1) as usize] = 0 ;
            arr[(y * width * 3 + x * 3 + 2) as usize] = 255 ;
        }
    }
    highgui::imshow("test0", &mat)?;

    /* Create Vec from Mat (Pixel data is copied to the new heap memory) */
    let ptr_org: *mut u8 = mat.ptr_2d(0, 0)? as _;
    let mut v = unsafe {std::slice::from_raw_parts(ptr_org, mat.total() * mat.elem_size()?).to_vec()};
    println!("mat_ptr = {:?}, v = {:?}", ptr_org, v.as_ptr());
    println!("Color at (10, 10) = B:{:}, G:{:}, R:{:}", v[0], v[1], v[2]);

    /* Create Mat from Vec */
    let mut val: Vec<u8> = Vec::new();
    for y in 0..height {
        for x in 0..width {
            val.push(0);    // B
            val.push(0);    // G
            val.push(255);  // R
        }
    }
    let mut mat = unsafe {
        Mat::new_rows_cols_with_data(
            height, width, core::CV_8UC3,
            val.as_mut_ptr() as *mut std::os::raw::c_void,
            core::Mat_AUTO_STEP
        )?
    };
    highgui::imshow("test2", &mat)?;

    /* Overlay using ROI */
    let mut mat = Mat::new_rows_cols_with_default(height, width, core::CV_8UC3, core::Scalar::new(255., 0., 0., 0.))?;
    let mut mat_roi = Mat::roi(&mat, core::Rect::new(10, 10, 50, 50))?;
    imgproc::rectangle(&mut mat_roi, core::Rect::new(0, 0, 50, 50), core::Scalar::new(0., 255., 0., 0.), 2, imgproc::LINE_8, 0)?;
    imgproc::circle(&mut mat, core::Point::new(30, 30), 10, core::Scalar::new(0., 255., 0., 0.), 2, imgproc::LINE_8, 0)?;
    highgui::imshow("test3_roi", &mat_roi)?;
    highgui::imshow("test3", &mat)?;
    
    
    highgui::wait_key(-1)?;
    Ok(())
}

