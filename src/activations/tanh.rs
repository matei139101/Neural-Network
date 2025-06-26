use crate::utils::math;

use super::activation::Activation;

pub struct Tanh {

}

impl Activation for Tanh {
    fn calculate(&self, inputs: &Vec<f32>) -> Vec<f32> {
        let mut results: Vec<f32> = vec![];

        for input in inputs {
            results.push(2f32 * math::sigmoid(input) - 1f32);
        }

        return results;
    }

    fn derivative(&self, _net_input: &f32) -> f32 {
        todo!();
    }
}