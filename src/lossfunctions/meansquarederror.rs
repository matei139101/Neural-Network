use std::result;

use crate::utils::math;

use super::lossfunction::LossFunction;

pub struct MeanSquaredError {}

impl LossFunction for MeanSquaredError {
    fn calculate(&self, input: &f32, target: &f32) -> f32 {
        return math::meansquarederror(&input, &target);
    }

    fn derivative(&self, output: &f32, target: &f32) -> f32 {
        return 2f32 * (output - target);
    }
}