use crate::utils;

pub fn run_engine(input_index_data: &[[i8; 5]], correction_data: &[bool], weights: &mut[[f32; 5]], generations: i32) -> Vec<f32> {
    let mut outputs: Vec<f32> = vec![];
    let mut corrected_outputs: Vec<f32> = vec![];
    let mut best_weights: [f32; 5] = [0f32; 5];

    for generation in 0..generations {        
        for input_index in 0..input_index_data.len() {
            let mut output: f32 = utils::dot_product(&input_index_data[input_index], &weights[0]);
            output = utils::sygmoid(output);

            let absolute_answer: f32 = utils::compare(output,correction_data[input_index]);
            let correction: f32 = correction_data[input_index] as i32 as f32;
            print!("O: {output} D: {absolute_answer} ");

            outputs.push(output);
            corrected_outputs.push(absolute_answer);
        }

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
                best_weights = weights[0];
    
                for weight in weights[0].iter_mut() {
                    *weight += utils::random_number(-newest_output, newest_output)
                }
    
                println!("\nFound new best weights");
            } else {
                weights[0] = best_weights;
    
                for weight in weights[0].iter_mut() {
                    *weight += utils::random_number(-newest_output, newest_output)
                }
    
                println!("\nReverted to old weights");
            }
        }

        println!("\nGeneration {generation} end.")
    }
    return outputs;
}