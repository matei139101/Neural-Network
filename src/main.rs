use model::{Dense, Model};

mod math;
mod logger;
mod model;

fn main() {
    let input: Vec<f32> = vec![2f32];

    let mut model: Model = Model::new(input);
    model.add_layer(Box::new(Dense::new(1, 2)));
    model.add_layer(Box::new(Dense::new(2, 2)));
    model.fit();
    model.predict();
}