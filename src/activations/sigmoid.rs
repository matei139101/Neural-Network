use crate::utils::math::{self, sigmoid};

use super::activation::Activation;

pub struct Sigmoid {

}

impl Activation for Sigmoid {
    fn calculate(&self, inputs: &Vec<f32>) -> Vec<f32> {
        let mut results: Vec<f32> = vec![];

        for input in inputs {
            results.push(math::sigmoid(input));
        }

        return results;
    }

    fn derivative(&self, net_input: &f32) -> f32 {
        return sigmoid(net_input) * (1f32 - sigmoid(net_input));
    }
}