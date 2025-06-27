use std::{usize, vec};
use crate::{layers::layer::Layer, lossfunctions::lossfunction::LossFunction, utils::{layeroutput::LayerOutput, logger::{self, DebugTier}}};

pub struct Model {
    pub layers: Vec<Box<dyn Layer>>,

    weights: Vec<Vec<Vec<f32>>>,
    lossfunction: Box<dyn LossFunction>,
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

    pub fn predict(&mut self, input: &Vec<f32>) -> Vec<LayerOutput> {
        logger::log(DebugTier::HIGH, format!("Starting model..."));
        
        let mut output: Vec<LayerOutput> = vec![LayerOutput::new(vec![], input.clone())];
        for layer in &mut self.layers {
            output.push(layer.process(&output[output.len() - 1].activated_output));
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

    pub fn loss(&self, output: &Vec<Vec<LayerOutput>>, targets: &Vec<Vec<f32>>) -> f32 {
        let mut loss: f32 = 0f32;
        let activated_outputs: Vec<Vec<f32>> = output
            .iter()
            .map(|layer| layer.activated_output.clone())
            .collect();
        let zipped_output_targets = output.iter().zip(targets);
        let count = zipped_output_targets.len() as f32;
    
        for (zipped_output, zipped_target) in zipped_output_targets {
            loss += self.lossfunction.calculate(zipped_output, zipped_target)
        }

        return loss / count;
    }

    pub fn back_propagate(&mut self, input: &Vec<f32>, output: &Vec<LayerOutput>, targets: &Vec<f32>) -> Vec<Vec<Vec<f32>>> {
        // To change
        let mut loss_derivatives: Vec<Vec<f32>> = vec![];
        for (index, _) in output[output.len() - 1].net_output.iter().enumerate() {
            loss_derivatives.push(vec![self.lossfunction.derivative(&output[output.len() - 1].net_output[index], &targets[index], output[output.len() - 1].net_output.len())]);
        }

        let mut deltas: Vec<Vec<Vec<f32>>> = vec![loss_derivatives];
        for layer in (0..self.layers.len()).rev() {
            deltas.push(self.layers[layer].back_propagate(&deltas[&deltas.len()-1], input, &output[layer]));
        }

        return deltas;
    }
}