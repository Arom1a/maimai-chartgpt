use data_preprocess::parser::parse_entire_file;
use std::fs;

fn main() {
    // for file in "./dataset/*/*/maidata.txt".chars() {
    //     let output = "processed.txt";
    //     println!("Pre-processing {}", file);
    // }

    let file = fs::read_to_string("tests/test-files/cryptarithm.txt").unwrap();
    let parsed = parse_entire_file(&file).unwrap();
}
