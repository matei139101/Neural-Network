use std::result;

use crate::utils::math;

use super::lossfunction::LossFunction;

pub struct MeanSquaredError {}

impl LossFunction for MeanSquaredError {
    fn multi_calculate(&self, inputs: Vec<f32>, correct_values: Vec<f32>) -> Vec<f32> {
        if inputs.len() != correct_values.len() {
            panic!("MSE failed to process. Inputs and correct values are not of the same length!")
        }

        let mut error: Vec<f32> = vec![];

        for (index, input) in inputs.iter().enumerate() {
            error.push(math::meansquarederror(&input, &correct_values[index]));
        }

        return error;
    }

    fn single_calculate(&self, inputs: Vec<f32>, correct_values: Vec<f32>) -> f32 {
        if inputs.len() != correct_values.len() {
            panic!("MSE failed to process. Inputs and correct values are not of the same length!")
        }
        
        let mut error: f32 = 0.0;

        for (index, input) in inputs.iter().enumerate() {
            error += math::meansquarederror(&input, &correct_values[index]);
        }

        return error / inputs.len() as f32;
    }
}