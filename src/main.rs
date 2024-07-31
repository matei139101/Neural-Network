mod engine;
mod utils;

const INPUT_DATA: [[i8; 5]; 5]= [
    [1, 1, 1, 0, 1],
    [1, 0, 0, 1, 1],
    [0, 1, 1, 1, 0],
    [0, 1, 0, 0, 1],
    [1, 1, 1, 1, 0]
];

const CORRECTION_DATA: [bool; 5] = [
    true, true, false, false, false
];

fn main() {
    //println!("{}", rand::thread_rng().gen_range(-1.0..1.0));
    let mut weights: [[f32; 5]; 1] = [
        [0.0; 5]
    ];

    for layer in weights.iter_mut() {
        for weight in layer.iter_mut() {
            *weight = utils::random_number(-1f32, 1f32);
        }
    }

    engine::run_engine(&INPUT_DATA, &CORRECTION_DATA, &mut weights, 1);
}