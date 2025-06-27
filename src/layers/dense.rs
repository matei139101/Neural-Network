use std::vec;

use crate::{activations::activation::Activation, utils::{logger::{self, DebugTier}, math}};
use super::layer::Layer;


pub struct Dense {
    inputs: usize,
    neurons: usize,
    activation: Box<dyn Activation>,
    input_values: Vec<f32>,
    weights: Vec<Vec<f32>>,
    net_output: Vec<f32>,
    activated_output: Vec<f32>,
}

impl Dense {
    pub fn new(inputs: usize, neurons: usize, activation: Box<dyn Activation>) -> Self {
        logger::log(DebugTier::MEDIUM, format!("Layer created with {} inputs and {} neurons", inputs, neurons));

        Dense { 
            inputs: inputs,
            neurons: neurons,
            activation: activation,
            input_values: vec![],
            weights: vec![],
            net_output: vec![],
            activated_output: vec![]
        }
    }
}

impl Layer for Dense {
    fn process(&mut self, input: &Vec<f32>) -> &Vec<f32> {
        logger::log(DebugTier::MEDIUM, format!("Processing layer... "));
        //logger::log(DebugTier::LOW, format!("Layer weights: {:?}", weights));

        self.input_values = input.clone();
        self.net_output = math::dot_product(&input, &self.weights);
        self.activated_output = self.activation.calculate(&self.net_output);

        logger::logln(DebugTier::MEDIUM, format!("Done!"));
        return &self.activated_output;
    }

    fn make_weights (&mut self) {
        let mut layer_weights: Vec<Vec<f32>> = vec![];

        for _input in 0..self.inputs {
            let mut input_weights: Vec<f32> = vec![];

            for _weight in 0..self.neurons {
                input_weights.push(math::random_number(-1.0, 1.0));
            }

            layer_weights.push(input_weights);
        }
        
        self.weights = layer_weights;
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

    fn get_weights(&self) -> &Vec<Vec<f32>> {
        return &self.weights;
    }

    fn back_propagate(&mut self, backwards_derivatives: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        let backward_deltas_sums: Vec<f32> = backwards_derivatives
            .iter()
            .map(|v| v.iter().sum())
            .collect();

        let mut layer_deltas: Vec<Vec<f32>> = vec![];
        for input in &self.input_values {
            let mut derivative_set: Vec<f32> = vec![];
            for derivative in backward_deltas_sums.iter().enumerate() {
                let derivative_value: f32 = derivative.1 * self.activation.derivative(&self.net_output[derivative.0]) * input;
                derivative_set.push(derivative_value);
            }
            layer_deltas.push(derivative_set);
        }

        return layer_deltas;
    }

    fn train(&mut self, learning_rate: f32, deltas: Vec<Vec<Vec<f32>>>) {
        logger::log(DebugTier::LOW, format!("Old layer weights: {:?}", self.weights));
        for (input, weight_derivative_set) in math::average_third_dimension(deltas).iter().enumerate() {
            for (index, derivative) in weight_derivative_set.iter().enumerate() {
                self.weights[input][index] -= derivative * learning_rate;
            }
        }
        logger::log(DebugTier::LOW, format!("New layer weights: {:?}", self.weights));
    }
}