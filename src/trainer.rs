use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};

use crate::{model::Model, utils::{logger::{self, DebugTier}, outputwrappers::{ModelOutput}}};

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
            let results: Vec<(ModelOutput, Vec<Vec<Vec<f32>>>)> = self.input
                .par_iter()
                .zip(self.targets.par_iter())
                .map(|(input, target)| {
                    let output = self.model.predict(input); // returns ModelOutput
                    let deltas = self.model.back_propagate(input, &output, target);
                    (output, deltas)
                })
                .collect();

            let mut epoch_output = Vec::new();
            let mut epoch_deltas = Vec::new();
                
            for (output, deltas) in results {
                epoch_output.push(output);
                epoch_deltas.push(deltas);
            }
            // Run model epochs
            //Needs to get parallelized
            // let mut epoch_output: Vec<ModelOutput> = vec![];
            // let mut epoch_deltas: Vec<Vec<Vec<Vec<f32>>>> = vec![];
            // for (index, _) in self.input.iter().enumerate() {
            //     let output = self.model.predict(&self.input[index]);
            //     epoch_deltas.push(self.model.back_propagate(&self.input[index], &output, &self.targets[index]));
            //     epoch_output.push(output);
            // }
            
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