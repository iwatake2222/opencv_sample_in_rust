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


/* Model Parameters */
const MODEL_FILENAME: &str = "../resource/model/mobilenetv2-1.0.onnx";
const LABEL_FILENAME: &str = "../resource/model/imagenet_labels.txt";
const MODEL_NORMALIZE_MEAN: (f64, f64, f64) = (0.485, 0.456, 0.406);
const MODEL_NORMALIZE_NORM: (f64, f64, f64) = (0.229, 0.224, 0.225);
const MODEL_WIDTH: i32 = 224;
const MODEL_HEIGHT: i32 = 224;
const MODEL_NAME_INPUT_0: &str = "data";
const MODEL_NAME_OUTPUT_0: &str = "mobilenetv20_output_flatten0_reshape0";

const TESTIMAGE_FILENAME: &str = "../resource/parrot.jpg";

fn normalize(mat: &mut core::Mat, mean: (f64, f64, f64), norm: (f64, f64, f64)) -> core::Mat {
    let mean = core::Scalar::from((mean.0, mean.1, mean.2));
    let norm = core::Scalar::from((norm.0, norm.1, norm.2));

    let mut mat_normalized = Mat::default();
    let mut mat_normalized_sub = Mat::default();
    let mut mat_normalized_div = Mat::default();
    mat.convert_to(&mut mat_normalized, core::CV_32FC3, 1.0 / 255.0, 0.0).unwrap();
    core::subtract(&mat_normalized, &mean, &mut mat_normalized_sub, &core::no_array(), -1).unwrap();
    core::divide2(&mat_normalized_sub, &norm, &mut mat_normalized_div, 1.0, -1).unwrap();

    // /* just to check calculation */
    // println!("{:?}", mat.at_2d::<core::Vec3b>(10, 10).unwrap());
    // println!("{:?}", mat_normalized.at_2d::<core::Vec3f>(10, 10).unwrap());
    // println!("{:?}", mat_normalized_sub.at_2d::<core::Vec3f>(10, 10).unwrap());
    // println!("{:?}", mat_normalized_div.at_2d::<core::Vec3f>(10, 10).unwrap());

    mat_normalized_div
}

fn main() -> Result<(), Box<dyn std::error::Error>> {

    /* Read label */
    let mut label_list = Vec::<String>::new();
    let label_file = File::open(LABEL_FILENAME)?;
    let reader = BufReader::new(label_file);
    for (_, line) in reader.lines().enumerate() {
        label_list.push(line?);
    }
    // println!("{:?}", label_list);

    /* Read input image */
    let mat = imgcodecs::imread(TESTIMAGE_FILENAME, imgcodecs::IMREAD_COLOR)?;

    /* Pre Process */
    let mut mat_resized = Mat::default();
    imgproc::resize(&mat, &mut mat_resized, core::Size { width: MODEL_WIDTH, height: MODEL_HEIGHT }, 0.0, 0.0, imgproc::INTER_LINEAR)?;
    let mat_normalized = normalize(&mut mat_resized, MODEL_NORMALIZE_MEAN, MODEL_NORMALIZE_NORM);
    let mat_blob = dnn::blob_from_image(&mat_normalized, 1.0, core::Size::default(), core::Scalar::default(), true, false, core::CV_32F)?;
    
    /* Load model */
    let mut net = dnn::read_net_from_onnx(MODEL_FILENAME)?;

    /* Feed input data */
    net.set_input(&mat_blob, MODEL_NAME_INPUT_0, 1.0, core::Scalar::default())?;

    /* Run inference */
    let mut output_blobs = core::Vector::<core::Mat>::new();
    let mut out_blob_names = core::Vector::<String>::new();
    out_blob_names.push(MODEL_NAME_OUTPUT_0);
    net.forward(&mut output_blobs, &out_blob_names)?;

    /* Retrieve output */
    let output_0 = &output_blobs.to_vec()[0];
    println!("{:?}", output_0);

    /* Convert the output to vector */
    let v = unsafe {std::slice::from_raw_parts(output_0.ptr(0)? as *mut f32, output_0.total()).to_vec()};
    // println!("{:?}", v);

    /* Find the max score */
    let max_value = v.iter().fold(0.0f32, |max, &val| if val > max{ val } else{ max });
    let index = v.iter().position(|&r| r == max_value).unwrap();
    println!("{:} ({:}) : {:}", label_list[index], index, max_value);

    highgui::wait_key(-1)?;
    Ok(())
}

