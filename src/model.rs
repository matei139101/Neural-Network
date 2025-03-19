use std::{vec};
use crate::{logger::{self, DebugTier}, math};

pub trait Layer {
    fn run (&self, input: &Vec<f32>) -> Vec<f32>;
}

pub struct Model {
    input: Vec<f32>,
    layers: Vec<Box<dyn Layer>>
}

impl Model {
    pub fn new(input: Vec<f32>) -> Self {
        Model {
            input: input,
            layers: vec![],
        }
    }

    pub fn add_layer(&mut self, layer: Box<dyn Layer>) {
        self.layers.push(layer);
    }

    pub fn run(&self) {
        let mut output: Vec<f32> = self.input.clone();

        for layer in &self.layers {
            output = layer.run(&output);
        }
    }
}

pub struct DenseLayer {
    inputs: i8,
    neurons: i8
}

impl DenseLayer {
    pub fn new(inputs: i8, neurons: i8) -> Self {
        DenseLayer { inputs: (inputs), neurons: (neurons) }
    }
}

impl Layer for DenseLayer {
    fn run(&self, input: &Vec<f32>) -> Vec<f32> {
        let mut output: Vec<f32> = vec![];

        for _neuron in 1..self.neurons {
            output.push(math::dot_product(&input, &input));
        }

        logger::log(DebugTier::HIGH, format!("Inputs: {}, Neurons: {}, Output: {:?}", self.inputs, self.neurons, output));

        return output;
    }
}