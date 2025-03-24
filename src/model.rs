use std::{usize, vec};
use crate::{layers::layer::Layer, lossfunctions::lossfunction::{self, LossFunction}, utils::logger::{self, DebugTier}};

pub struct Model {
    input: Vec<f32>,
    weights: Vec<Vec<Vec<f32>>>,
    layers: Vec<Box<dyn Layer>>,
    outputs: Vec<Vec<f32>>,
    lossfunction: Box<dyn LossFunction>
}

impl Model {
    pub fn new(input: Vec<f32>, lossfunction: Box<dyn LossFunction>) -> Self {
        Model {
            input,
            weights: vec![],
            layers: vec![],
            outputs: vec![],
            lossfunction
        }
    }

    pub fn add_layer(&mut self, layer: Box<dyn Layer>) {
        self.layers.push(layer);
    }

    pub fn predict(&mut self) {
        let mut output: Vec<f32> = self.input.clone();
        for (index, layer) in self.layers.iter().enumerate() {
            output = layer.process(&output, &self.weights[index]);
        }

        self.outputs.push(output);
    }

    pub fn fit(&mut self) {
        let mut size: usize = self.input.len();

        for (index, layer) in self.layers.iter().enumerate() {
            if !layer.allign(&size) {
                panic!("Missalignment on layer {}. Got {} but have {} inputs", index, &size, layer.get_inputs());
            }

            size = layer.get_neurons();
            self.weights.push(layer.make_weights());
        }

        logger::log(DebugTier::LOW, format!("Weights: {:?}", self.weights));
    }

    pub fn train(&self, correct_values: Vec<f32>, epochs: usize) {
        logger::log(DebugTier::HIGH, format!("Error: {:?}", self.lossfunction.single_calculate(self.outputs[self.outputs.len() - 1].clone(), correct_values)));

        for epoch in 1..epochs {
            
        }
    }
}