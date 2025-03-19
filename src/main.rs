use model::{DenseLayer, Model};

mod math;
mod logger;
mod model;

fn main() {
    let input: Vec<f32> = vec![2f32, 2f32, 2f32, 2f32, 2f32];

    let weights: Vec<Vec<f32>> = vec![
        vec![2f32, 2f32, 2f32, 2f32, 2f32],
        vec![4f32, 4f32, 4f32, 4f32, 4f32]
    ];

    let mut model: Model = Model::new(input, weights);
    model.add_layer(Box::new(DenseLayer::new(5, 5)));
    model.add_layer(Box::new(DenseLayer::new(5, 5)));
    model.run();
}