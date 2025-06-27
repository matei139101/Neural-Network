use crate::{model::Model, utils::logger::{self, DebugTier}};

pub struct Trainer {
    model: Model,
    input: Vec<Vec<f32>>,
    targets: Vec<Vec<f32>>
}

impl Trainer {
    pub fn new(model: Model, input: Vec<Vec<f32>>, targets: Vec<Vec<f32>>) -> Self {
        Trainer {
            model,
            input,
            targets
        }
    }

    pub fn train(&mut self, epochs: usize, learning_rate: f32) {
        for epoch in 0..epochs {
            // Run model epochs
            let mut epoch_output: Vec<Vec<f32>> = vec![];
            let mut epoch_deltas: Vec<Vec<Vec<Vec<f32>>>> = vec![];
            for (index, _) in self.input.iter().enumerate() {
                let output = self.model.predict(&self.input[index]);
                epoch_output.push(output.clone());
                epoch_deltas.push(self.model.back_propagate(&output, &self.targets[index]));
            }
            
            // Train model layers
            for layer in &mut self.model.layers {
                layer.train(learning_rate, Trainer::average_deltas(&epoch_deltas));
            }

            logger::log(DebugTier::IMPORTANT, format!("Epoch: {}, Loss: {}", epoch+1, self.model.loss(&epoch_output, &self.targets)));
        }
    }

    fn average_deltas(inputs: &Vec<Vec<Vec<Vec<f32>>>>) -> Vec<Vec<Vec<f32>>> {
        let n = inputs.len() as f32;
        let mut avg = inputs[0].clone();
    
        for input in inputs.iter().skip(1) {
            for (l, layer) in input.iter().enumerate() {
                for (o, output) in layer.iter().enumerate() {
                    for (w, &weight) in output.iter().enumerate() {
                        avg[l][o][w] += weight;
                    }
                }
            }
        }
    
        for layer in &mut avg {
            for output in layer {
                for weight in output {
                    *weight /= n;
                }
            }
        }
    
        return avg;
    }
}