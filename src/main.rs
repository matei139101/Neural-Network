mod neuralengine;
mod utils;
mod logger;

const INPUT_DATA: [[f32; 5]; 1] = [
    [1f32, 1f32, 1f32, 0f32, 1f32]
];

const INPUT_TRAINING_DATA: [[f32; 5]; 5] = [
    [1f32, 1f32, 1f32, 0f32, 1f32],
    [1f32, 0f32, 0f32, 1f32, 1f32],
    [0f32, 1f32, 1f32, 1f32, 0f32],
    [0f32, 1f32, 0f32, 0f32, 1f32],
    [1f32, 1f32, 1f32, 1f32, 0f32]
];

const CORRECTION_DATA: [f32; 5] = [
    1f32, 1f32, 0f32, 0f32, 0f32
];

const LEARNING_STATE: bool = true;
///NOTES:
///https://developers-dot-devsite-v2-prod.appspot.com/machine-learning/crash-course/backprop-scroll
/// 
fn main() {
    let mut weights: [[f32; 5]; 1] = [
        [0.0; 5]
    ];

    utils::randomize_weights(&mut weights);

    let input: &[[f32; 5]] = if LEARNING_STATE { &INPUT_TRAINING_DATA } else { &INPUT_DATA };

    //Setup engine values and variables
    neuralengine::setup_engine(input, &CORRECTION_DATA, &mut weights, LEARNING_STATE, 100);

    //Run engine calculus

    //Run engine learning
    //  Set up generations
    //  Loop generations
    //  Learn

    //Exit
}