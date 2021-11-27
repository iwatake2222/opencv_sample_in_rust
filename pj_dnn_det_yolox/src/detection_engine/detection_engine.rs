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
use opencv::{prelude::*, core, imgproc, dnn};
use std::fs::File;
use std::io::{BufRead, BufReader};

// use crate::detection_engine::boundnig_box::BoundingBox;
use super::boundnig_box::{BoundingBox, nms};


/////////////////////////////////////////////////////////////////
pub struct DetectionEngine {
    net: dnn::Net,
    out_blob_names: core::Vector::<String>,
    normalize_mean: core::Scalar,
    normalize_norm: core::Scalar,
    label_list: Vec::<String>,
}

impl DetectionEngine {
    /* Model Parameters */
    const MODEL_FILENAME: &'static str = "../resource/model/yolox_nano_320x320.onnx";
    const LABEL_FILENAME: &'static str = "../resource/model/label_coco_80.txt";
    const MODEL_NORMALIZE_MEAN: (f64, f64, f64) = (0.485, 0.456, 0.406);
    const MODEL_NORMALIZE_NORM: (f64, f64, f64) = (0.229, 0.224, 0.225);
    const MODEL_WIDTH: i32 = 320;
    const MODEL_HEIGHT: i32 = 320;
    const MODEL_NAME_INPUT_0: &'static str = "images";
    const MODEL_NAME_OUTPUT_0: &'static str = "output";
    const MODEL_GRID_SCALE_LIST: [i32; 3] = [ 8, 16, 32 ];
    const MODEL_GRID_CHANNEL: i32 = 1;
    const MODEL_NUMBER_OF_CLASS: i32 = 80;
    const MODEL_ELEMENT_NUM_OF_ANCHOR: i32 = Self::MODEL_NUMBER_OF_CLASS + 5;    // x, y, w, h, bbox confidence, [class confidence]

    /* Other Parameters */
    const SCORE_BOX_THRESHOLD: f32 = 0.4;
    const SCORE_CLASS_THRESHOLD: f32 = 0.2;
    const NMS_IOU_THRESHOLD: f32 = 0.6;
    
    pub fn new() -> Self {
        /* Load model */
        let mut net = dnn::read_net_from_onnx(Self::MODEL_FILENAME).unwrap();
        net.set_preferable_backend(dnn::DNN_BACKEND_OPENCV).unwrap();
        net.set_preferable_target(dnn::DNN_TARGET_CPU).unwrap();
        // net.set_preferable_target(dnn::DNN_TARGET_OPENCL).unwrap();

        /* Set output names */
        let mut out_blob_names = core::Vector::<String>::new();
        out_blob_names.push(Self::MODEL_NAME_OUTPUT_0);

        /* Initialize normalize param */
        let normalize_mean = core::Scalar::from(Self::MODEL_NORMALIZE_MEAN);
        let normalize_norm = core::Scalar::from(Self::MODEL_NORMALIZE_NORM);

        /* Read label */
        let mut label_list = Vec::<String>::new();
        let label_file = File::open(Self::LABEL_FILENAME).unwrap();
        let reader = BufReader::new(label_file);
        for (_, line) in reader.lines().enumerate() {
            label_list.push(line.unwrap());
        }
        // println!("{:?}", label_list);
        
        DetectionEngine {
            net: net,
            out_blob_names: out_blob_names,
            normalize_mean: normalize_mean,
            normalize_norm: normalize_norm,
            label_list: label_list
        }
    }

    fn normalize(&self, mat: &mut core::Mat) -> core::Mat {
        let mut mat_normalized = Mat::default();
        let mut mat_normalized_sub = Mat::default();
        let mut mat_normalized_div = Mat::default();
        mat.convert_to(&mut mat_normalized, core::CV_32FC3, 1.0 / 255.0, 0.0).unwrap();
        core::subtract(&mat_normalized, &self.normalize_mean, &mut mat_normalized_sub, &core::no_array(), -1).unwrap();
        core::divide2(&mat_normalized_sub, &self.normalize_norm, &mut mat_normalized_div, 1.0, -1).unwrap();
        mat_normalized_div
    }

    pub fn process(&mut self, mat: &core::Mat) -> Vec::<BoundingBox> {
        /* Pre Process */
        let mut mat_resized = Mat::default();
        imgproc::resize(mat, &mut mat_resized, core::Size { width: Self::MODEL_WIDTH, height: Self::MODEL_HEIGHT }, 0.0, 0.0, imgproc::INTER_LINEAR).unwrap();
        let mat_normalized = self.normalize(&mut mat_resized);
        let mat_blob = dnn::blob_from_image(&mat_normalized, 1.0, core::Size::default(), core::Scalar::default(), true, false, core::CV_32F).unwrap();

        /* Feed input data */
        self.net.set_input(&mat_blob, Self::MODEL_NAME_INPUT_0, 1.0, core::Scalar::default()).unwrap();

        /* Run inference */
        let mut output_blobs = core::Vector::<core::Mat>::new();
        self.net.forward(&mut output_blobs, &self.out_blob_names).unwrap();

        /* Retrieve output */
        let output_data = &output_blobs.to_vec()[0];

        // /* Convert the output to vector */
        let output_data = unsafe {std::slice::from_raw_parts(output_data.ptr(0).unwrap() as *mut f32, output_data.total()).to_vec()};
        //  println!("{}, {}", output_data[10], output_data.len());
        
        /* Decode bbox */
        let mut bbox_list = Vec::<BoundingBox>::new();

        let mut index: usize = 0;
        for grid_scale in Self::MODEL_GRID_SCALE_LIST.iter() {
            let grid_w = Self::MODEL_WIDTH / grid_scale;
            let grid_h = Self::MODEL_HEIGHT / grid_scale;

            let scale_x = *grid_scale as f32 * mat.cols() as f32 / Self::MODEL_WIDTH as f32;      /* scale to original image */
            let scale_y = *grid_scale as f32 * mat.rows() as f32 / Self::MODEL_HEIGHT as f32;

            self.get_bounding_box(&output_data, index, scale_x, scale_y, grid_w, grid_h, &mut bbox_list);
            index += (grid_w * grid_h * Self::MODEL_GRID_CHANNEL * Self::MODEL_ELEMENT_NUM_OF_ANCHOR) as usize;
        }

        /* NMS */
        let bbox_nms_list = nms(&mut bbox_list, Self::NMS_IOU_THRESHOLD);

        bbox_nms_list
    }

    fn get_bounding_box(&mut self, data: &Vec::<f32>, mut index: usize, scale_x: f32, scale_y: f32, grid_w: i32, grid_h: i32, bbox_list: &mut Vec::<BoundingBox>) {
        for grid_y in 0 .. grid_h {
            for grid_x in 0 .. grid_w {
                for _grid_c in 0 .. Self::MODEL_GRID_CHANNEL {
                    let box_confidence = data[index + 4];

                    if box_confidence >= Self::SCORE_BOX_THRESHOLD {
                        let mut class_id = 0;
                        let mut confidence: f32 = 0.0;
                        for class_index in 0 .. Self::MODEL_NUMBER_OF_CLASS {
                            let confidence_of_class = data[index + 5 + class_index as usize];
                            if confidence_of_class > confidence {
                                confidence = confidence_of_class;
                                class_id = class_index;
                            }
                        }

                        if confidence >= Self::SCORE_CLASS_THRESHOLD {
                            let cx = ((data[index + 0] + grid_x as f32) * scale_x) as i32;
                            let cy = ((data[index + 1] + grid_y as f32) * scale_y) as i32;
                            let w  = (data[index + 2].exp() * scale_x) as i32;
                            let h  = (data[index + 3].exp() * scale_y) as i32;

                            let bbox = BoundingBox {
                                class_id: class_id,
                                label: self.label_list[class_id as usize].clone(),
                                score: confidence,
                                x: cx - w / 2,
                                y: cy - h / 2,
                                w: w,
                                h: h,
                            };
                            bbox_list.push(bbox);
                        }
                    }
                    index += Self::MODEL_ELEMENT_NUM_OF_ANCHOR as usize;
                }
            }
        }
    }
}

