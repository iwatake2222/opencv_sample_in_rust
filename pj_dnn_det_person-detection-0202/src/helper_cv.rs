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
use opencv::{core, highgui, imgproc};
use rand::prelude::*;


/////////////////////////////////////////////////////////////////
/* pos = left top */
pub fn draw_text(mat: &mut core::Mat, text: &str, pos: core::Point, font_scale: f64, thickness: i32, color_front: core::Scalar, color_back: core::Scalar, is_text_on_rect: bool) {
    let mut pos = pos;
    let mut base_line: i32 = 0;
    let text_size: core::Size = imgproc::get_text_size(&text, highgui::QT_STYLE_NORMAL, font_scale, thickness, &mut base_line).unwrap();
    base_line += thickness;
    pos.y += text_size.height;

    if is_text_on_rect {
        imgproc::rectangle(mat, core::Rect::new(pos.x, pos.y - text_size.height, text_size.width, base_line + text_size.height), color_back, -1, imgproc::LINE_8, 0).unwrap();
        imgproc::put_text(mat, &text, pos, highgui::QT_STYLE_NORMAL, font_scale, color_front, thickness, imgproc::LINE_8, false).unwrap();
    } else {
        imgproc::put_text(mat, &text, pos, highgui::QT_STYLE_NORMAL, font_scale, color_back, thickness * 3, imgproc::LINE_8, false).unwrap();
        imgproc::put_text(mat, &text, pos, highgui::QT_STYLE_NORMAL, font_scale, color_front, thickness, imgproc::LINE_8, false).unwrap();
    }

}


pub struct ColorGenerator {
    color_list: Vec::<core::Scalar>,
}

impl ColorGenerator {
    pub fn conv_hsl2rgb(h: u32, s: u32, l: u32) -> (u8, u8, u8) {
        /* https://www.peko-step.com/tool/hslrgb.html */
        
        let h = h.min(360).max(0) as f32;   /* [0, 360] */
        let s = s.min(100).max(0) as f32;   /* [0, 100] */
        let l = l.min(100).max(0) as f32;   /* [0, 100] */
        let max;
        let min;
        if l < 50.0 {
            max = 2.55 * (l + l * (s / 100.0));
            min = 2.55 * (l - l * (s / 100.0));
        } else {
            max = 2.55 * (l + (100.0 - l) * (s / 100.0));
            min = 2.55 * (l - (100.0 - l) * (s / 100.0));
        }
        let red;
        let green;
        let blue;
        if h < 60.0 {
            red = max;
            green = (h / 60.0) * (max - min) + min;
            blue = min;
        } else if h < 120.0 {
            red = ((120.0 - h) / 60.0) * (max - min) + min;
            green = max;
            blue = min;
        } else if h < 180.0 {
            red = min;
            green = max;
            blue = ((h - 120.0)/ 60.0) * (max - min) + min;
        } else if h < 240.0 {
            red = min;
            green = ((240.0 - h) / 60.0) * (max - min) + min;
            blue = max;
        } else if h < 300.0 {
            red = ((h - 240.0) / 60.0) * (max - min) + min;
            green = min;
            blue = max;
        } else {
            red = max;
            green = min;
            blue = ((360.0- h) / 60.0) * (max - min) + min;
        }
        (red as u8, green as u8, blue as u8)
    }

    pub fn new(num: u32, lightness: u32) -> ColorGenerator {
        let mut color_list = Vec::<core::Scalar>::new();
        let mut rng = rand::thread_rng();
        for i in 0 .. num {
            /* Use HSL to get nice color */
            // let h: u32 = (rng.gen_range(0..num) * 360) / num;
            let h: u32 = 360 * (i + 1) / num;
            let rgb = Self::conv_hsl2rgb(h, 100, lightness);
            color_list.push(core::Scalar::new(rgb.2 as f64, rgb.1 as f64, rgb.0 as f64, 255.0));
        }
        color_list.shuffle(&mut rng);
        
        ColorGenerator{color_list: color_list}
    }

    pub fn get(&self, mut id: i32) -> core::Scalar {
        id %= self.color_list.len() as i32;
        self.color_list[id as usize]
    }
}
