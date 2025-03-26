use std::{usize, vec};
use crate::{layers::layer::Layer, lossfunctions::lossfunction::LossFunction, utils::logger::{self, DebugTier}};

pub struct Model {
    input: Vec<f32>,
    output: Vec<f32>,
    weights: Vec<Vec<Vec<f32>>>,
    layers: Vec<Box<dyn Layer>>,
    lossfunction: Box<dyn LossFunction>
}

impl Model {
    pub fn new(input: Vec<f32>, lossfunction: Box<dyn LossFunction>) -> Self {
        Model {
            output: vec![],
            input,
            weights: vec![],
            layers: vec![],
            lossfunction
        }
    }

    pub fn add_layer(&mut self, layer: Box<dyn Layer>) {
        self.layers.push(layer);
    }

    pub fn predict(&mut self) {
        let mut output: Vec<f32> = self.input.clone();

        for (index, layer) in self.layers.iter_mut().enumerate() {
            output = layer.process(&output, &self.weights[index]).to_vec();
        }

        self.output = output;
    }

    pub fn fit(&mut self) {
        let mut size: usize = self.input.len();

        for (index, layer) in self.layers.iter_mut().enumerate() {
            if !layer.allign(&size) {
                panic!("Missalignment on layer {}. Got {} but have {} inputs", index, &size, layer.get_inputs());
            }

            size = layer.get_neurons();
            layer.make_weights();
            self.weights.push(layer.get_weights().clone());
        }

        logger::log(DebugTier::LOW, format!("Weights: {:?}", self.weights));
    }

    pub fn calculate_loss(&self, output: &Vec<f32>, targets: &Vec<f32>) -> Vec<f32> {
        let mut losses: Vec<f32> = vec![];

        //Naming is hard
        //TO-DO: Make a proper name insteado of x
        for x in output.iter().zip(targets) {
            losses.push(self.lossfunction.calculate(&x.0, &x.1));
        }

        return losses;
    }

    pub fn train(&self) {

        todo!()
    }
}