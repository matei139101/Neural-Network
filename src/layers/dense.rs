use crate::{activations::activation::Activation, utils::{logger::{self, DebugTier}, math}};
use super::layer::Layer;


pub struct Dense {
    inputs: usize,
    neurons: usize,
    activation: Box<dyn Activation>
}

impl Dense {
    pub fn new(inputs: usize, neurons: usize, activation: Box<dyn Activation>) -> Self {
        logger::log(DebugTier::MEDIUM, format!("Layer created with {} inputs and {} neurons", inputs, neurons));

        Dense { 
            inputs: inputs,
            neurons: neurons,
            activation: activation
        }
    }
}

impl Layer for Dense {
    fn process(&self, input: &Vec<f32>, weights: &Vec<Vec<f32>>) -> Vec<f32> {
        logger::log(DebugTier::LOW, format!("Processing layer... "));

        let output: Vec<f32> = self.activation.calculate(math::dot_product(&input, &weights));

        logger::logln(DebugTier::LOW, format!("Done!"));
        return output;
    }

    fn make_weights (&self) -> Vec<Vec<f32>> {
        let mut layer_weights: Vec<Vec<f32>> = vec![];

        for _input in 0..self.inputs {
            let mut input_weights: Vec<f32> = vec![];

            for _weight in 0..self.neurons {
                input_weights.push(math::random_number(-1.0, 1.0));
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