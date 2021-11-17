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
use opencv::{prelude::*, core, highgui, imgcodecs, imgproc, types::{VectorOfPoint2f, VectorOfString, VectorOfVectorOfPoint}, dnn};
use std::fs::File;
use std::io::{BufRead, BufReader};

fn main() -> Result<(), Box<dyn std::error::Error>> {

    /* Read label */
    let mut label_list = Vec::<String>::new();
    let label_file = File::open("../resource/imagenet_labels.txt")?;
    let reader = BufReader::new(label_file);
    for (_, line) in reader.lines().enumerate() {
        label_list.push(line?);
    }
    // println!("{:?}", label_list);

    /* Read input image */
    let mat = imgcodecs::imread("../resource/parrot.jpg", imgcodecs::IMREAD_COLOR)?;

    /* Pre Process */
    let mut mat_resized = Mat::default();
    imgproc::resize(&mat, &mut mat_resized, core::Size { width: 224, height: 224, }, 0.0, 0.0, imgproc::INTER_LINEAR)?;
    let mut mat_normalized = Mat::default();
    mat_resized.convert_to(&mut mat_normalized, core::CV_32FC3, 1.0 / 255.0, 0.0)?;
    // cv::subtract(img_src, cv::Scalar(cv::Vec<float, 3>(input_tensor_info.normalize.mean)), img_src);
    // cv::multiply(img_src, cv::Scalar(cv::Vec<float, 3>(input_tensor_info.normalize.norm)), img_src);
    let mat_blob = dnn::blob_from_image(&mat_normalized, 1.0f64, core::Size::default(), core::Scalar::default(), true, false, core::CV_32F)?;
    
    /* Load model */
    let mut net = dnn::read_net_from_onnx("../resource/mobilenetv2-1.0.onnx")?;

    /* Feed input data */
    net.set_input(&mat_blob, "data", 1.0, core::Scalar::default())?;

    /* Run inference */
    let mut output_blobs = core::Vector::<core::Mat>::new();
    let mut out_blob_names = core::Vector::<String>::new();
    out_blob_names.push("mobilenetv20_output_flatten0_reshape0");
    net.forward(&mut output_blobs, &out_blob_names)?;

    /* Retrieve output */
    let output_blob = &output_blobs.to_vec()[0];
    println!("{:?}", output_blob);


    /* Convert the output to vector */
    let v = unsafe {std::slice::from_raw_parts(output_blob.ptr(0)? as *mut f32, output_blob.total()).to_vec()};
    // println!("{:?}", v);

    /* Find the max score */
    let max_value = v.iter().fold(0.0f32, |max, &val| if val > max{ val } else{ max });
    let index = v.iter().position(|&r| r == max_value).unwrap();
    println!("{:} ({:}) : {:}", label_list[index], index, max_value);

    highgui::wait_key(-1)?;
    Ok(())
}

