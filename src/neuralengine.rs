use crate::utils;
use crate::logger;

pub fn setup_engine(input_index_data: &[[f32; 5]], correction_data: &[f32], weights: &mut[[f32; 5]], learning_state: bool, generations: u8) {
    if learning_state {
        train_network(generations, input_index_data, weights, correction_data);
    } else {
        execute_engine(input_index_data, weights);
    }
}

pub fn execute_engine(inputs: &[[f32; 5]], weights: &[[f32; 5]]) {
    for input_index in 0..inputs.len() {
        let output: f32 = calculate_layer(&inputs[input_index], &weights[0]);

        logger::log(logger::debug_tier::HIGH, output.to_string().as_str());
    }
}

fn train_network(generations: u8, inputs: &[[f32; 5]], weights: &mut [[f32; 5]], correction_data: &[f32]) {
    let mut outputs: Vec<f32> = vec![];
    let mut corrected_outputs: Vec<f32> = vec![];
    let mut best_weights: [f32; 5] = [0f32; 5];

    for generation in 0..generations {
        for input_index in 0..inputs.len() {   
            let output: [f32; 2] = network(inputs, weights, correction_data, input_index);

            outputs.push(output[0]);
            corrected_outputs.push(output[1]);
        }

        learn(outputs.clone(), weights, &mut best_weights);

        println!("\nGeneration {generation} end.")
    }
}

fn network(input_index_data: &[[f32; 5]], weights: &[[f32; 5]], correction_data: &[f32], input_index: usize) -> [f32; 2] {
        let output: f32 = calculate_layer(&input_index_data[input_index], &weights[0]);

        let correctness: f32 = utils::compare(output, correction_data[input_index] as i32 as f32);

        print!("O: {output} C: {correctness} ");

        return [output, correctness];
}

//Only calculates for one node as of now
fn calculate_layer(vector: &[f32; 5], weights: &[f32; 5]) -> f32 {
    let mut output: f32 = utils::dot_product(vector, weights);
    output = utils::sygmoid(output);

    return output;
}

//This needs to be redone somehow to be more readable and less repeating
fn learn(outputs: Vec<f32>, weights: &mut[[f32; 5]], best_weights: &mut [f32; 5]) {
    let mut newest_output: f32 = 0f32;
    let mut previous_output: f32 = 0f32;

    if outputs.len() >= 5 {
        for i in 1..5 {
            newest_output += outputs[outputs.len() - i];
        }
        newest_output = newest_output / 5f32;

        if outputs.len() >= 10 {
            for i in 1..5 {
                previous_output += outputs[outputs.len() - i - 5];
            }
            previous_output = previous_output / 5f32;
        }

        if newest_output < previous_output {
            *best_weights = weights[0];

            for weight in weights[0].iter_mut() {
                *weight += utils::random_number(-newest_output, newest_output)
            }

            println!("\nFound new best weights");
        } else {
            weights[0] = *best_weights;

            for weight in weights[0].iter_mut() {
                *weight += utils::random_number(-newest_output, newest_output)
            }

            println!("\nReverted to old weights");
        }
    }
}