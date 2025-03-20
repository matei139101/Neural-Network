use std::{usize, vec};
use crate::{logger::{self, DebugTier}, math};

pub trait Layer {
    fn process (&self, input: &Vec<f32>, weights: &Vec<Vec<f32>>) -> Vec<f32>;
    fn make_weights (&self) -> Vec<Vec<f32>>;
    fn allign(&self, alligner: &usize) -> bool;
    fn get_inputs(&self) -> usize;
    fn get_neurons(&self) -> usize;
}

pub struct Model {
    input: Vec<f32>,
    weights: Vec<Vec<Vec<f32>>>,
    layers: Vec<Box<dyn Layer>>,
    ok: bool
}

impl Model {
    pub fn new(input: Vec<f32>) -> Self {
        Model {
            input: input,
            weights: vec![],
            layers: vec![],
            ok: true
        }
    }

    pub fn add_layer(&mut self, layer: Box<dyn Layer>) {
        self.layers.push(layer);
    }

    pub fn predict(&self) {
        if !self.ok { return };

        let mut output: Vec<f32> = self.input.clone();
        for (index, layer) in self.layers.iter().enumerate() {
            output = layer.process(&output, &self.weights[index]);
        }
    }

    pub fn fit(&mut self) {
        let mut size: usize = self.input.len();

        for (index, layer) in self.layers.iter().enumerate() {
            if !layer.allign(&size) {
                logger::log(DebugTier::HIGH, format!("Missalignment on layer {}. Got {} but have {} inputs", index, &size, layer.get_inputs()));
                self.ok = false;
                break;
            }

            size = layer.get_neurons();
            self.weights.push(layer.make_weights());
        }

        logger::log(DebugTier::LOW, format!("Weights: {:?}", self.weights));
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

        for neuron in 1..self.neurons {
            output.push(math::dot_product(&input, &weights[neuron - 1]));
        }

        logger::log(DebugTier::MEDIUM, format!("Inputs: {}, Neurons: {}, Output: {:?}", self.inputs, self.neurons, output));

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