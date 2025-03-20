use layers::dense::Dense;
use model::Model;

mod layers;
mod activations;
mod utils;
mod model;

fn main() {
    let input: Vec<f32> = vec![2f32, 2f32];

    let mut model: Model = Model::new(input);
    model.add_layer(Box::new(Dense::new(2, 5)));
    model.add_layer(Box::new(Dense::new(5, 2)));
    model.fit();
    model.predict();
}