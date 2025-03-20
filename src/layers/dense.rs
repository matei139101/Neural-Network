use crate::{utils::logger::{self, DebugTier}, utils::math};
use super::layer::Layer;


pub struct Dense {
    inputs: usize,
    neurons: usize
}

impl Dense {
    pub fn new(inputs: usize, neurons: usize) -> Self {
        Dense { inputs: (inputs), neurons: (neurons) }
    }
}

impl Layer for Dense {
    fn process(&self, input: &Vec<f32>, weights: &Vec<Vec<f32>>) -> Vec<f32> {
        let output: Vec<f32> = math::dot_product(&input, &weights);

        logger::log(DebugTier::MEDIUM, format!("Inputs: {}, Neurons: {}", self.inputs, self.neurons));

        return output;
    }

    fn make_weights (&self) -> Vec<Vec<f32>> {
        let mut layer_weights: Vec<Vec<f32>> = vec![];

        for _input in 0..self.inputs {
            let mut input_weights: Vec<f32> = vec![];

            for _weight in 0..self.neurons {
                input_weights.push(math::random_number(0.0, 1.0));
            }

            layer_weights.push(input_weights);
        }
        
        return layer_weights;
    }

    fn allign (&self, aligner: &usize) -> bool {
        return aligner == &self.inputs;
    }

    fn get_inputs(&self) -> usize {
        return self.inputs;
    }

    fn get_neurons(&self) -> usize {
        return self.neurons;
    }
}