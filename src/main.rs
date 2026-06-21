use data_preprocess::parser::{ProcessError, parse_entire_file};
use std::fs;
use std::path::Path;

fn main() {
    let dataset_dir = Path::new("./dataset");
    let mut processed_cnt = 0;

    for version in fs::read_dir(dataset_dir).unwrap().flatten() {
        let version_path = version.path();
        if !version_path.is_dir() {
            continue;
        }
        for id in fs::read_dir(&version_path).unwrap().flatten() {
            let id_path = id.path();
            if !id_path.is_dir() {
                continue;
            }
            let file = id_path.join("maidata.txt");
            println!("Processing {}", file.to_string_lossy());
            let content = fs::read_to_string(&file).unwrap();
            let processed = match parse_entire_file(&content) {
                Ok(ok) => ok,
                Err(ProcessError::Utage) => continue,
                Err(ProcessError::Nom(e)) => {
                    eprint!("Parse error: {}", e);
                    panic!();
                }
            };
            processed_cnt += 1;
            let output = file.with_file_name("processed.json");
            let processed_json = serde_json::to_string_pretty(&processed).unwrap();
            fs::write(output, processed_json).unwrap();
        }
    }

    println!("Successfully processed {} files", processed_cnt);
}

fn remove_all_processed_json(dir: &Path) {
    for entry in fs::read_dir(dir).unwrap().flatten() {
        let path = entry.path();
        let ft = entry.file_type().unwrap();
        if ft.is_dir() {
            remove_all_processed_json(&path);
        } else if ft.is_file() && path.file_name().unwrap_or_default() == "processed.json" {
            let _ = fs::remove_file(&path);
        }
    }
}
