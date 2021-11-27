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
#[derive(Debug, Clone)]
pub struct BoundingBox {
    pub class_id: i32,
    pub label: String,
    pub score: f32,
    pub x: i32,
    pub y: i32,
    pub w: i32,
    pub h: i32,
}

pub fn calculate_iou(bbox0: &BoundingBox, bbox1: &BoundingBox) -> f32 {
    let interx0 = std::cmp::max(bbox0.x, bbox1.x);
    let intery0 = std::cmp::max(bbox0.y, bbox1.y);
    let interx1 = std::cmp::min(bbox0.x + bbox0.w, bbox1.x + bbox1.w);
    let intery1 = std::cmp::min(bbox0.y + bbox0.h, bbox1.y + bbox1.h);
    if interx1 < interx0 || intery1 < intery0 {
        return 0.0;
    }

    let area0 = bbox0.w * bbox0.h;
    let area1 = bbox1.w * bbox1.h;
    let area_inter = (interx1 - interx0) * (intery1 - intery0);
    let area_sum = area0 + area1 - area_inter;
    return area_inter as f32 / area_sum as f32;
}


pub fn nms(bbox_list: &mut Vec::<BoundingBox>, iou_threshold: f32) -> Vec::<BoundingBox> {
    let mut bbox_list_new = Vec::<BoundingBox>::new();
    // /* Note: Since BoundingBox contains String type, I can't use copy */
    // bbox_list_new.push(bbox_list[0].clone());
    
    /* Descending order of score: High score -> Low score */
    bbox_list.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

    let mut is_merged_list = Vec::<bool>::new();
    is_merged_list.resize(bbox_list.len(), false);

    for index_high_score in 0 .. bbox_list.len() {
        if is_merged_list[index_high_score] {
            continue;
        }
        bbox_list_new.push(bbox_list[index_high_score].clone());
        for index_low_score in index_high_score + 1 .. bbox_list.len() {
            if is_merged_list[index_low_score] {
                continue;
            }
            if calculate_iou(&bbox_list[index_high_score], &bbox_list[index_low_score]) > iou_threshold {
                is_merged_list[index_low_score] = true;
            }
        }
    }
    bbox_list_new
}

#[allow(dead_code)]
pub fn fit_in_screen(bbox: &mut BoundingBox, width: i32, height: i32) {
    bbox.x = std::cmp::max(0, bbox.x);
    bbox.y = std::cmp::max(0, bbox.y);
    bbox.w = std::cmp::min(width - bbox.x, bbox.w);
    bbox.h = std::cmp::min(height - bbox.y, bbox.h);
}