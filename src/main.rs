use model::{DenseLayer, Model};

mod math;
mod logger;
mod model;

fn main() {
    let mut model: Model = Model::new(vec![2f32, 2f32, 2f32, 2f32, 2f32]);
    model.add_layer(Box::new(DenseLayer::new(4, 8)));
    model.add_layer(Box::new(DenseLayer::new(4, 8)));
    model.add_layer(Box::new(DenseLayer::new(4, 8)));
    model.run();
}