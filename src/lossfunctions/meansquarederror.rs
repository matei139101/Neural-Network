use crate::utils::math;
use super::lossfunction::LossFunction;

pub struct MeanSquaredError {}

impl LossFunction for MeanSquaredError {
    fn calculate(&self, input_set: &Vec<f32>, target_set: &Vec<f32>) -> f32 {
        let mut loss: f32 = 0f32;

        let zipped_input_target_set = input_set.iter().zip(target_set);
        let length: f32 = zipped_input_target_set.len() as f32;
        for (input, target) in zipped_input_target_set {
            loss += math::meansquarederror(&input, &target);
        }

        return loss / length;
    }

    fn derivative(&self, output: &f32, target: &f32, mean_size: usize) -> f32 {
        return (2f32 * (output - target)) / mean_size as f32;
    }
}