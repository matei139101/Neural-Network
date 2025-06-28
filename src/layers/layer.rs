use crate::utils::outputwrappers::LayerOutput;

pub trait Layer {
    fn get_inputs(&self) -> usize;
    fn get_neurons(&self) -> usize;
    fn get_weights(&self) -> &Vec<Vec<f32>>;

    fn process (&self, input: &LayerOutput) -> LayerOutput;
    fn make_weights (&mut self);
    fn allign(&self, alligner: &usize) -> bool;
    fn back_propagate(&self, backwards_derivative: &Vec<Vec<f32>>, input: &Vec<f32>, layer_output: &LayerOutput) -> Vec<Vec<f32>>;
    fn train(&mut self, learning_rate: f32, deltas: Vec<Vec<Vec<f32>>>);
}