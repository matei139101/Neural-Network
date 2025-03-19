use std::vec;
use crate::{logger::{self, DebugTier}, math};

pub trait Layer {
    fn process (&self, input: &Vec<f32>, weights: &Vec<Vec<f32>>) -> Vec<f32>;
    fn make_weights (&self) -> Vec<Vec<f32>>;
}

pub struct Model {
    input: Vec<f32>,
    weights: Vec<Vec<Vec<f32>>>,
    layers: Vec<Box<dyn Layer>>
}

impl Model {
    pub fn new(input: Vec<f32>) -> Self {
        Model {
            input: input,
            weights: vec![],
            layers: vec![],
        }
    }

    pub fn add_layer(&mut self, layer: Box<dyn Layer>) {
        self.layers.push(layer);
    }

    pub fn predict(&self) {
        let mut output: Vec<f32> = self.input.clone();

        for (index, layer) in self.layers.iter().enumerate() {
            output = layer.process(&output, &self.weights[index]);
        }
    }

    pub fn fit(&mut self) {
        for layer in &self.layers {
            self.weights.push(layer.make_weights());
        }

        logger::log(DebugTier::MEDIUM, format!("Weights: {:?}", self.weights));
    }
}

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
        let mut output: Vec<f32> = vec![];

        for neuron in 0..self.neurons {
            output.push(math::dot_product(&input, &weights[neuron]));
        }

        logger::log(DebugTier::HIGH, format!("Inputs: {}, Neurons: {}, Output: {:?}", self.inputs, self.neurons, output));

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
}