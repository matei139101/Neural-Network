use std::{usize, vec};
use crate::{layers::layer::Layer, utils::logger::{self, DebugTier}};

pub struct Model {
    input: Vec<f32>,
    weights: Vec<Vec<Vec<f32>>>,
    layers: Vec<Box<dyn Layer>>,
    outputs: Vec<Vec<f32>>
}

impl Model {
    pub fn new(input: Vec<f32>) -> Self {
        Model {
            input: input,
            weights: vec![],
            layers: vec![],
            outputs: vec![]
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
}