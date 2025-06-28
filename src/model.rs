use std::{usize, vec};

use crate::{layers::layer::Layer, lossfunctions::lossfunction::LossFunction, utils::{logger::{self, DebugTier}, outputwrappers::{LayerOutput, ModelOutput}}};

pub struct Model {
    pub layers: Vec<Box<dyn Layer + Sync>>,

    weights: Vec<Vec<Vec<f32>>>,
    lossfunction: Box<dyn LossFunction + Sync>,
}

impl Model {
    pub fn new(lossfunction: Box<dyn LossFunction + Sync>) -> Self {
        Model {
            weights: vec![],
            layers: vec![],
            lossfunction
        }
    }

    pub fn add_layer(&mut self, layer: Box<dyn Layer + Sync>) {
        self.layers.push(layer);
    }

    pub fn predict(&self, input: &Vec<f32>) -> ModelOutput {
        logger::log(DebugTier::HIGH, format!("Starting model..."));
        
        let mut output: ModelOutput = ModelOutput::new();
        for (index, layer) in self.layers.iter().enumerate() {
            if index == 0 {
                output.push(layer.process(&LayerOutput::new(vec![], input.clone())));
            } else {
                output.push(layer.process(&output.last()));
            }
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

    pub fn loss(&self, output: &Vec<ModelOutput>, targets: &Vec<Vec<f32>>) -> f32 {
        let mut loss: f32 = 0f32;
        let activated_outputs: Vec<&Vec<f32>> = output
            .iter()
            .flat_map(|layer_outputs| {
                layer_outputs.output.iter().map(|lo| &lo.activated_output)
            })
            .collect();

        let zipped_output_targets = activated_outputs.iter().zip(targets);
        let count = zipped_output_targets.len() as f32;
    
        for (zipped_output, zipped_target) in zipped_output_targets {
            loss += self.lossfunction.calculate(zipped_output, zipped_target)
        }

        return loss / count;
    }

    pub fn back_propagate(&self, input: &Vec<f32>, output: &ModelOutput, targets: &Vec<f32>) -> Vec<Vec<Vec<f32>>> {
        let mut deltas: Vec<Vec<Vec<f32>>> = vec![];

        let mut loss_derivatives: Vec<Vec<f32>> = vec![];
        for (index, _) in output.last().net_output.iter().enumerate() {
            loss_derivatives.push(vec![self.lossfunction.derivative(&output.last().net_output[index], &targets[index], output.last().net_output.len())]);
        }
        deltas.push(loss_derivatives);


        for layer_index in (0..self.layers.len()).rev() {
            if layer_index == 0 {
                deltas.push(self.layers[layer_index].back_propagate(&deltas.last().unwrap(), &input, &output.get_by_layer(layer_index)));
            } else {
                deltas.push(self.layers[layer_index].back_propagate(&deltas.last().unwrap(), &output.get_by_layer(layer_index - 1).activated_output, &output.get_by_layer(layer_index)));
            }
        }

        return deltas;
    }
}