use crate::utils;

pub fn run_engine(input_data: &[[i8; 5]], _correction_data: &[bool], weights: &[[f32; 5]], generations: i32) -> Vec<f32> {
    //println!("Hello world!");
    //println!("{}", crate::utils::sygmoid(1.0));
    //println!("{}", crate::utils::dot_product(vec![1.0, 2.0, 3.0], vec![1.0, 2.0, 3.0]));
    let mut outputs: Vec<f32> = vec![];

    for generation in 1..generations + 1 {
        println!("Generation {generation} outputs:");
        
        for input in 0..input_data.len() {
            let weighted_output: f32 = utils::dot_product(&input_data[input], &weights[0]);
            let normalized_output: f32 = utils::sygmoid(weighted_output);

            print!("{normalized_output} ");

            outputs.push(normalized_output);
        }

        println!("\nGeneration {generation} end")
    }
    return outputs;
}