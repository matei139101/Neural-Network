mod neuralengine;
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

const learning_state: bool = true;
///NOTES:
///https://developers-dot-devsite-v2-prod.appspot.com/machine-learning/crash-course/backprop-scroll
/// 
fn main() {
    let mut weights: [[f32; 5]; 1] = [
        [0.0; 5]
    ];

    utils::randomize_weights(&mut weights);

    //Setup engine values and variables
    neuralengine::setup_engine(&INPUT_DATA, &CORRECTION_DATA, &mut weights, learning_state, 100);

    //Run engine calculus

    //Run engine learning
    //  Set up generations
    //  Loop generations
    //  Learn

    //Exit
}