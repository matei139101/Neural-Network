mod engine;
mod utils;

const INPUT_DATA: [[f32; 5]; 5]= [
    [1f32, 1f32, 1f32, 0f32, 1f32],
    [1f32, 0f32, 0f32, 1f32, 1f32],
    [0f32, 1f32, 1f32, 1f32, 0f32],
    [0f32, 1f32, 0f32, 0f32, 1f32],
    [1f32, 1f32, 1f32, 1f32, 0f32]
];

const CORRECTION_DATA: [f32; 5] = [
    1f32, 1f32, 0f32, 0f32, 0f32
];

///NOTES:
///https://developers-dot-devsite-v2-prod.appspot.com/machine-learning/crash-course/backprop-scroll
/// 
fn main() {
    let mut weights: [[f32; 5]; 1] = [
        [0.0; 5]
    ];

    for layer in weights.iter_mut() {
        for weight in layer.iter_mut() {
            *weight = utils::random_number(-1f32, 1f32);
        }
    }

    engine::run_engine(&INPUT_DATA, &CORRECTION_DATA, &mut weights, 100);
}