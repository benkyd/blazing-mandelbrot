#![feature(core_intrinsics)]
use rand::Rng;
use std::f64::{consts::PI, INFINITY};

use std::time::Duration;

use minifb::{Key, Window, WindowOptions};
use num::{Complex, Float};
use rayon::prelude::*;

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
    pub supersampling_range: f64,
    pub supersampling_min_dist: f64,
    pub max_iterations: u32,
}

impl Mandlebrot {
    fn new() -> Mandlebrot {
        let height: usize = 1800;
        let width: usize = 1800;
        Mandlebrot {
            view_width: width,
            view_height: height,
            view_x: -0.6,
            view_y: 0.0,
            view_zoom: 2.2,
            supersampling_range: 4.0,
            supersampling_min_dist: 0.5,
            max_iterations: 1000,
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

    fn mandlebrot_sochastic_supersample(&self, x: u32, y: u32, limit: u32) -> Option<u32> {
        // first we need to make the pisson disc of out sampling point and generate c for them
        let cx = self.view_zoom * (x as f64 / self.view_width as f64 - 0.5) + self.view_x;
        let cy = self.view_zoom * (y as f64 / self.view_height as f64 - 0.5) + self.view_y;

        // this is the fast bridson algorithm first outlined here:
        // https://www.cs.ubc.ca/~rbridson/docs/bridson-siggraph07-poissondisk.pdf
        let pisson_disc_sample = |cx: f64, cy: f64| -> Vec<Complex<f64>> {
            let mut ret = Vec::<Complex<f64>>::new();
            let mut rng = rand::thread_rng();

            let cell_size = self.supersampling_min_dist / 1.41421356237; // sqrt of 2
            let grid_size = (self.supersampling_range / cell_size).ceil();

            let mut grid = Vec::<(f64, f64)>::with_capacity(
                (self.supersampling_range * self.supersampling_range) as usize,
            );
            let mut proc = Vec::<(f64, f64)>::new(); // treat as a stack

            let in_area = |p: (f64, f64)| -> bool {
                p.0 > 0.0
                    && p.0 < self.supersampling_range
                    && p.1 > 0.0
                    && p.1 < self.supersampling_range
            };

            let squared_distance = |a: (f64, f64), b: (f64, f64)| -> f64 {
                let delta_x = a.0 - b.0;
                let delta_y = a.1 - b.1;
                delta_x * delta_x * delta_y * delta_y
            };

            let point_around = |p: (f64, f64)| -> (f64, f64) {
                let radius =
                    self.supersampling_min_dist * (rng.gen_range(0.0..3.0) + 1.0f64).sqrt();
                let angle = rng.gen_range(0.0..PI * 2.0);
                (p.0 + angle.cos() * radius, p.1 + angle.sin() * radius)
            };

            let mut set = |p: (f64, f64)| {
                let cell = ((p.0 / cell_size) as usize, (p.1 / cell_size) as usize);
                grid[cell.1 * self.supersampling_range as usize + cell.0] = p;
            };

            let mut add = |p: (f64, f64)| {
                let c = Complex { re: p.0, im: p.1 };
                ret.push(c);
                proc.push(p);
                set(p);
            };

            let mut p_too_close = |p: (f64, f64)| -> bool {
                let xi = (p.0 / cell_size).floor() as usize;
                let yi = (p.1 / cell_size).floor() as usize;

                if grid[yi * grid_size as usize + xi].0 != INFINITY {
                    return true;
                }

                let min_dist_sq = self.supersampling_min_dist * self.supersampling_min_dist;
                let minx = (xi - 2).max(0);
                let miny = (yi - 2).max(0);
                let maxx = (xi + 2).min(grid_size as usize - 1);
                let maxy = (yi + 2).min(grid_size as usize - 1);

                for y in miny..=maxy {
                    for x in minx..=maxx {
                        let point = grid[y * grid_size as usize + x];
                        let exists = point.0 != INFINITY;
                        if exists && squared_distance(p, point) < min_dist_sq {
                            return true;
                        }
                    }
                }

                false
            };

            let start_p = (
                rng.gen_range(0.0..self.supersampling_range),
                rng.gen_range(0.0..self.supersampling_range),
            );
            add(start_p);

            while !proc.is_empty() {
                let point = proc.pop().unwrap(); // we can safely unwrap here
                for i in 0..30 {
                    let p = point_around(point);
                    if in_area(p) && !p_too_close(p) {
                        add(p);
                    }
                }
            }

            ret
        };

        let samples = pisson_disc_sample(cx, cy);
        let samples: Vec<Option<u32>> = samples
            .iter()
            .map(|c| -> Option<u32> {
                let mut z = Complex { re: 0.0, im: 0.0 };

                for i in 0..limit {
                    z = z * z + c;
                    if z.norm() > 2.0 {
                        return Some(i);
                    }
                }
                None
            })
            .collect();
        let summed_iterations: u32 = samples
            .iter()
            .map(|x| if x.is_some() { x.unwrap() } else { 0 })
            .sum();

        Some(summed_iterations / samples.len() as u32)
    }

    fn mandlebrot(&self, x: u32, y: u32, limit: u32) -> Option<u32> {
        let cx = self.view_zoom * (x as f64 / self.view_width as f64 - 0.5) + self.view_x;
        let cy = self.view_zoom * (y as f64 / self.view_height as f64 - 0.5) + self.view_y;

        let c = Complex { re: cx, im: cy };
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
        let cyclic_shading = |value: u32, upper: u32| -> (u8, u8, u8) {
            let scaled_v: u8 = ((value * upper) / 255) as u8;
            (scaled_v, scaled_v, scaled_v)
        };

        match self.mandlebrot(x, y, self.max_iterations) {
            None => 0,
            Some(count) => {
                let col = cyclic_shading(count, self.max_iterations);
                (col.0 as u32) << 16 | (col.1 as u32) << 8 | col.2 as u32
            }
        }
    }

    fn get_buffer(&self) -> Vec<u32> {
        let mut buffer: Vec<u32> = vec![0; self.view_width as usize * self.view_height as usize];

        buffer
            .par_chunks_exact_mut(100)
            .enumerate()
            .for_each(|(i, pixel)| {
                for (j, p) in pixel.iter_mut().enumerate() {
                    let x = (i * 100 + j) % self.view_width as usize;
                    let y = (i * 100 + j) / self.view_width as usize;
                    *p = self.get_pixel_color(x as u32, y as u32);
                }
            });

        buffer
    }
}

struct FixedPoint512([u64; 8]);

impl FixedPoint512 {
    fn new() -> Self {
        let mut fp = FixedPoint512([0; 8]);
        fp.0[7] = 18_446_744_073_709_551_612;
        fp
    }
}

impl std::fmt::Display for FixedPoint512 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut s = String::new();
        for i in 0..8 {
            s.push_str(&format!("{:0>16}", self.0[i].to_string()));
        }
        write!(f, "{}", s)
    }
}

impl std::ops::Add for FixedPoint512 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let mut res = FixedPoint512::new();
        let mut carry = 0;
        for i in 0..8 {
            let (val, carry1) = self.0[i].overflowing_add(rhs.0[i]);
            let (val, carry2) = val.overflowing_add(carry);
            res.0[i] = val;
            carry = if carry1 || carry2 { 1 } else { 0 };
        }
        res
    }
}

fn main() {
    let mut mandlebrot = Mandlebrot::new();

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

        // we want mouse pan and scroll as opposed to keyboard pan and scroll

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

        let buffer = mandlebrot.get_buffer();

        timer.end();

        println!("Time: {:.3}ms at {:.3}FPS", timer.ms, timer.fps);
        window
            .update_with_buffer(&buffer, mandlebrot.view_width, mandlebrot.view_height)
            .unwrap();
    }
}
