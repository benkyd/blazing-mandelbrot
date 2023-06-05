use rayon::prelude::*;
use std::arch::x86_64::*;
use std::time::Duration;

use minifb::{Key, Window, WindowOptions};
use num::Complex;

struct Timer {
    pub fps: f64,
    pub ms: f64,
    start: std::time::Instant,
    last_10_times: Vec<Duration>,
}

impl Timer {
    fn new() -> Timer {
        Timer {
            fps: 0.0,
            ms: 0.0,
            start: std::time::Instant::now(),
            last_10_times: Vec::new(),
        }
    }

    fn start(&mut self) {
        self.start = std::time::Instant::now();
    }

    fn end(&mut self) -> f64 {
        let end = std::time::Instant::now();
        let duration = end - self.start;

        if self.last_10_times.len() > 10 {
            self.last_10_times.remove(0);
        }
        self.last_10_times.push(duration);
        // calculate average duration
        let mut total_duration = Duration::new(0, 0);
        for time in self.last_10_times.iter() {
            total_duration += *time;
        }
        let average_duration = total_duration / self.last_10_times.len() as u32;

        self.fps = 1.0 / average_duration.as_secs_f64();
        self.ms = average_duration.as_secs_f64() * 1000.0;
        self.ms
    }
}

struct Mandlebrot {
    pub view_width: usize,
    pub view_height: usize,
    pub view_x: f64,
    pub view_y: f64,
    pub view_zoom: f64,
    pub max_iterations: u32,
}

impl Mandlebrot {
    fn new() -> Mandlebrot {
        let height: usize = 600;
        let width: usize = 800;
        Mandlebrot {
            view_width: width,
            view_height: height,
            view_x: 0.0,
            view_y: 0.0,
            view_zoom: 1.0,
            max_iterations: 512,
        }
    }

    fn zoom_relative_translate(&mut self, x: f64, y: f64, zoom: f64) {
        self.view_x += x * self.view_zoom;
        self.view_y += y * self.view_zoom;
        // zoom should never be, and the zoom value should be scaled based on the actual zoom
        self.view_zoom *= 1.0 + zoom;
        println!(
            "x: {}, y: {}, zoom: {}",
            self.view_x, self.view_y, self.view_zoom
        );
    }

    fn get_pixel(&self, x: u32, y: u32) -> Complex<f64> {
        let cx = self.view_zoom * (x as f64 / self.view_width as f64 - 0.5) + self.view_x;
        let cy = self.view_zoom * (y as f64 / self.view_height as f64 - 0.5) + self.view_y;
        Complex { re: cx, im: cy }
    }

    fn mandlebrot(&self, c: Complex<f64>, limit: u32) -> Option<u32> {
        let mut z = Complex { re: 0.0, im: 0.0 };
        for i in 0..limit {
            z = z * z + c;
            if z.norm() > 2.0 {
                return Some(i);
            }
        }
        None
    }

    fn get_pixel_color(&self, x: u32, y: u32) -> u32 {
        let c = self.get_pixel(x, y);
        match self.mandlebrot(c, self.max_iterations) {
            None => 0,
            Some(count) => 255 << 16 | count << 8 | 255,
        }
    }

    fn get_buffer(&self) -> Vec<u32> {
        let mut buffer: Vec<u32> = vec![0; self.view_width as usize * self.view_height as usize];

        buffer.par_chunks_exact_mut(100).enumerate().for_each(|(i, pixel)| {
            for (j, p) in pixel.iter_mut().enumerate() {
                let x = (i * 100 + j) % self.view_width as usize;
                let y = (i * 100 + j) / self.view_width as usize;
                *p = self.get_pixel_color(x as u32, y as u32);
            }
        });

        buffer
    }
}

fn main() {
    let mut mandlebrot = Mandlebrot::new();

    let mut buffer: Vec<u32> = vec![0; mandlebrot.view_width * mandlebrot.view_height];
    let mut window = Window::new(
        "Mandlebrot",
        mandlebrot.view_width,
        mandlebrot.view_height,
        WindowOptions::default(),
    )
    .unwrap_or_else(|e| {
        panic!("{}", e);
    });

    let mut timer = Timer::new();
    while window.is_open() && !window.is_key_down(Key::Escape) {
        timer.start();

        if window.is_key_down(Key::W) {
            mandlebrot.zoom_relative_translate(0.0, -0.1, 0.0);
        }
        if window.is_key_down(Key::S) {
            mandlebrot.zoom_relative_translate(0.0, 0.1, 0.0);
        }
        if window.is_key_down(Key::A) {
            mandlebrot.zoom_relative_translate(-0.1, 0.0, 0.0);
        }
        if window.is_key_down(Key::D) {
            mandlebrot.zoom_relative_translate(0.1, 0.0, 0.0);
        }

        if window.is_key_down(Key::Q) {
            mandlebrot.zoom_relative_translate(0.0, 0.0, -0.1);
        }
        if window.is_key_down(Key::E) {
            mandlebrot.zoom_relative_translate(0.0, 0.0, 0.1);
        }

        if window.is_key_down(Key::Left) {
            mandlebrot.max_iterations -= 1;
            println!("Iterations: {}", mandlebrot.max_iterations);
        }
        if window.is_key_down(Key::Right) {
            mandlebrot.max_iterations += 1;
            println!("Iterations: {}", mandlebrot.max_iterations);
        }

        buffer = mandlebrot.get_buffer();

        timer.end();

        println!("Time: {:.3}ms at {:.3}FPS", timer.ms, timer.fps);
        window
            .update_with_buffer(&buffer, mandlebrot.view_width, mandlebrot.view_height)
            .unwrap();
    }

}
