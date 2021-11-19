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
    const MODEL_FILENAME: &'static str = "../resource/ctdet_coco_dlav0_384.onnx";
    const LABEL_FILENAME: &'static str = "../resource/label_coco_80.txt";
    const MODEL_NORMALIZE_MEAN: (f64, f64, f64) = (0.485, 0.456, 0.406);
    const MODEL_NORMALIZE_NORM: (f64, f64, f64) = (0.229, 0.224, 0.225);
    const MODEL_WIDTH: i32 = 384;
    const MODEL_HEIGHT: i32 = 384;
    const MODEL_NAME_INPUT_0: &'static str = "input.1";
    const MODEL_NAME_OUTPUT_0: &'static str = "508";
    const MODEL_NAME_OUTPUT_1: &'static str = "511";
    const MODEL_NAME_OUTPUT_2: &'static str = "514";
    const MODEL_HM_SCALE: i32 = 4;
    const MODEL_HM_HEIGHT: i32  = Self::MODEL_WIDTH / Self::MODEL_HM_SCALE;
    const MODEL_HM_WIDTH: i32   = Self::MODEL_HEIGHT / Self::MODEL_HM_SCALE;
    const MODEL_HM_CHANNEL: i32 = 80;

    /* Other Parameters */
    const SCORE_THRESHOLD: f32 = 0.4;
    const NMS_IOU_THRESHOLD: f32 = 0.6;
    
    pub fn new() -> Self {
        /* Load model */
        let net = dnn::read_net_from_onnx(Self::MODEL_FILENAME).unwrap();

        /* Set output names */
        let mut out_blob_names = core::Vector::<String>::new();
        out_blob_names.push(Self::MODEL_NAME_OUTPUT_0);
        out_blob_names.push(Self::MODEL_NAME_OUTPUT_1);
        out_blob_names.push(Self::MODEL_NAME_OUTPUT_2);

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
        let hm_list = &output_blobs.to_vec()[0];
        let reg_xy_list = &output_blobs.to_vec()[1];
        let reg_wh_list = &output_blobs.to_vec()[2];

        /* Convert the output to vector */
        let hm_list = unsafe {std::slice::from_raw_parts(hm_list.ptr(0).unwrap() as *mut f32, hm_list.total()).to_vec()};
        let reg_xy_list = unsafe {std::slice::from_raw_parts(reg_xy_list.ptr(0).unwrap() as *mut f32, reg_xy_list.total()).to_vec()};
        let reg_wh_list = unsafe {std::slice::from_raw_parts(reg_wh_list.ptr(0).unwrap() as *mut f32, reg_wh_list.total()).to_vec()};
        //  println!("{}, {}", hm_list[10], hm_list.len());
        
        /* Decode bbox */
        let mut bbox_list = Vec::<BoundingBox>::new();
        let threshold_score_logit: f32 = Self::logit(Self::SCORE_THRESHOLD);
        let scale_w: f32 = mat.cols() as f32 / Self::MODEL_WIDTH as f32;
        let scale_h: f32 = mat.rows() as f32 / Self::MODEL_HEIGHT as f32;
        for hm_y in 0 .. Self::MODEL_HM_HEIGHT {
            for hm_x in 0 .. Self::MODEL_HM_WIDTH {
                let mut max_class_id = -1;
                let mut max_score_logit: f32 = std::f32::MIN;
                for class_id in 0 .. Self::MODEL_HM_CHANNEL {
                    let index_score: usize = (hm_x + Self::MODEL_HM_WIDTH * hm_y + Self::MODEL_HM_WIDTH * Self::MODEL_HM_HEIGHT * class_id) as usize;
                    let score_logit = hm_list[index_score];
                    // max_score_logit = max_score_logit.max(score_logit);
                    if score_logit > max_score_logit {
                        max_score_logit = score_logit;
                        max_class_id = class_id;
                    }
                }
                if max_score_logit > threshold_score_logit && max_class_id >= 0 {
                    let index_x: usize = (Self::MODEL_HM_WIDTH * hm_y + hm_x) as usize;
                    let index_y: usize = index_x + (Self::MODEL_HM_HEIGHT * Self::MODEL_HM_WIDTH) as usize;
                    let width = reg_wh_list[index_x];
                    let height = reg_wh_list[index_y];
                    let cx = hm_x as f32 + reg_xy_list[index_x];  /* no need to add +0.5f according to sample code */
                    let cy = hm_y as f32 + reg_xy_list[index_y];
                    let x0 = cx - width / 2.0;
                    let y0 = cy - height / 2.0;
                    // println!("{}, {}, {}, {}", x0, y0, width, height);
                    let bbox = BoundingBox{
                        class_id: max_class_id,
                        label: self.label_list[max_class_id as usize].clone(),
                        score: Self::sigmoid(max_score_logit),
                        x: (x0 * Self::MODEL_HM_SCALE as f32 * scale_w) as i32,
                        y: (y0 * Self::MODEL_HM_SCALE as f32 * scale_h) as i32,
                        w: (width * Self::MODEL_HM_SCALE as f32 * scale_w) as i32,
                        h: (height * Self::MODEL_HM_SCALE as f32 * scale_h) as i32,
                    };
                    bbox_list.push(bbox);
                }
            }
        }

        /* NMS */
        let bbox_nms_list = nms(&mut bbox_list, Self::NMS_IOU_THRESHOLD);

        bbox_nms_list
    }

    fn sigmoid(x: f32) -> f32 {
        if x >= 0.0 {
            return 1.0 / (1.0 + (-x).exp());
        } else {
            return x.exp() / (1.0 + x.exp());   /* to aovid overflow */
        }
    }

    fn logit(x: f32) -> f32 {
        if x <= 0.0 {
            return std::f32::MIN;
        } else if x >= 1.0 {
            return std::f32::MAX;
        } else {
            return (x / (1.0 - x)).ln();
        }
    }
}

