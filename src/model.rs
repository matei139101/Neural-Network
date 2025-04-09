use std::{usize, vec};
use crate::{layers::layer::Layer, lossfunctions::lossfunction::LossFunction, utils::logger::{self, DebugTier}};

pub struct Model {
    weights: Vec<Vec<Vec<f32>>>,
    layers: Vec<Box<dyn Layer>>,
    lossfunction: Box<dyn LossFunction>
}

impl Model {
    pub fn new(lossfunction: Box<dyn LossFunction>) -> Self {
        Model {
            weights: vec![],
            layers: vec![],
            lossfunction
        }
    }

    pub fn add_layer(&mut self, layer: Box<dyn Layer>) {
        self.layers.push(layer);
    }

    pub fn predict(&mut self, input: &Vec<f32>) -> Vec<f32> {
        logger::log(DebugTier::HIGH, format!("Starting model..."));
        
        //TO-DO: I don't like this clone...
        let mut output: Vec<f32> = input.clone();
        for layer in &mut self.layers {
            output = layer.process(&output).to_vec();
        }

        logger::log(DebugTier::HIGH, format!("Ended model"));
        return output;
    }

    pub fn prepare(&mut self) {
        let mut size: usize = self.layers[0].get_inputs();

        for (index, layer) in self.layers.iter_mut().enumerate() {
            if !layer.allign(&size) { panic!("Missalignment on layer {}. Got {} but have {} inputs", index, &size, layer.get_inputs()); }

            size = layer.get_neurons();
            layer.make_weights();
            //TO-DO: I don't like this clone
            self.weights.push(layer.get_weights().clone());
        }
    }

    pub fn train(&mut self, input: &Vec<Vec<f32>>, targets: &Vec<Vec<f32>>, learning_rate: f32, epochs: usize) {
        for epoch in 0..epochs {
            let zipped_input_targets = input.iter().zip(targets);
            let mut epoch_output: Vec<Vec<f32>> = vec![];
            for (zipped_input, zipped_targets) in zipped_input_targets {
                let output: Vec<f32> = self.predict(zipped_input);
                epoch_output.push(output.clone());

                self.back_propagate(&output, zipped_targets);
            }

            for layer in &mut self.layers {
                layer.train(learning_rate);
            }

            logger::log(DebugTier::IMPORTANT, format!("Epoch: {}, Loss: {}", epoch+1, self.loss(&epoch_output, targets)));
        }
    }

    fn loss(&self, output: &Vec<Vec<f32>>, targets: &Vec<Vec<f32>>) -> f32 {
        let mut loss: f32 = 0f32;
        let zipped_output_targets = output.iter().zip(targets);
        let count = zipped_output_targets.len() as f32;

        for (zipped_output, zipped_target) in zipped_output_targets {
            loss += self.lossfunction.calculate(zipped_output, zipped_target)
        }

        return loss / count;
    }

    fn back_propagate(&mut self, output: &Vec<f32>, targets: &Vec<f32>) {
        let mut loss_derivatives: Vec<f32> = vec![];
        for (zipped_output, zipped_target) in output.iter().zip(targets) {
            loss_derivatives.push(self.lossfunction.derivative(zipped_output, zipped_target));
        }

        let mut weight_derivatives: Vec<Vec<f32>> = vec![];
        weight_derivatives.push(loss_derivatives);
        for layer in (0..self.layers.len()).rev() {
            weight_derivatives.push(self.layers[layer].back_propagate(&weight_derivatives[&weight_derivatives.len()-1]));
        }
    }
}