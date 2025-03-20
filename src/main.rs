use layers::dense::Dense;
use model::Model;

mod layers;
mod utils;
mod model;

fn main() {
    let input: Vec<f32> = vec![2f32, 2f32];

    let mut model: Model = Model::new(input);
    model.add_layer(Box::new(Dense::new(2, 20)));
    model.add_layer(Box::new(Dense::new(20, 100)));
    model.add_layer(Box::new(Dense::new(100, 100)));
    model.add_layer(Box::new(Dense::new(100, 10)));
    model.fit();
    model.predict();
}