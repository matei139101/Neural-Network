use crate::{activations::activation::Activation, utils::{logger::{self, DebugTier}, math}};
use super::layer::Layer;


pub struct Dense {
    inputs: usize,
    neurons: usize,
    activation: Box<dyn Activation>,
    input_values: Vec<f32>,
    weights: Vec<Vec<f32>>,
    net_output: Vec<f32>,
    activated_output: Vec<f32>
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
    fn process(&mut self, input: &Vec<f32>, weights: &Vec<Vec<f32>>) -> &Vec<f32> {
        logger::log(DebugTier::LOW, format!("Processing layer... "));

        self.input_values = input.clone();
        self.net_output = math::dot_product(&input, &weights);
        self.activated_output = self.activation.calculate(&self.net_output);

        logger::logln(DebugTier::LOW, format!("Done!"));
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

    fn back_propagate(&mut self, backwards_derivatives: &Vec<f32>, learning_rate: f32) -> Vec<f32> {       
        let mut net_derivatives: Vec<f32> = vec![];
        for net_output_value in &self.net_output {
            net_derivatives.push(self.activation.derivative(&net_output_value));
        }

        let mut new_backwards_derivatives: Vec<f32> = vec![];
        for  {
            for weight in 0..self.weights[weight_set].len()-1 {

            }
        }
        //println!("{:?}", input_weight_derivatives );
        return net_derivatives;
    }
}