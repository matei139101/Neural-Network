pub trait Layer {
    fn process (&self, input: &Vec<f32>, weights: &Vec<Vec<f32>>) -> Vec<f32>;
    fn make_weights (&self) -> Vec<Vec<f32>>;
    fn allign(&self, alligner: &usize) -> bool;
    fn get_inputs(&self) -> usize;
    fn get_neurons(&self) -> usize;
}