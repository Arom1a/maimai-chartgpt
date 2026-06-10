use crate::schema::*;
use nom::{
    IResult, Parser,
    branch::alt,
    bytes::complete::{tag, take_until},
    character::complete::{char, line_ending, multispace0, not_line_ending},
    multi::many0,
    sequence::preceded,
};
use std::collections::HashMap;

#[derive(Debug)]
pub enum FileItem<'a> {
    Metadata(&'a str, &'a str),
    ChartSection { level: u8, raw: &'a str },
}

fn parse_metadata_line(input: &str) -> IResult<&str, FileItem<'_>> {
    let start = char('&');
    let key = take_until("=");
    let equal_sign = char('=');
    let value = not_line_ending;
    let new_line = line_ending;

    let (input, (_, key, _, value, _)) = (start, key, equal_sign, value, new_line).parse(input)?;
    Ok((input, FileItem::Metadata(key, value)))
}

fn parse_inote_raw(input: &str) -> IResult<&str, FileItem<'_>> {
    let inote_tag = tag("&inote_");
    let level = take_until("=");
    let equal_sign = char('=');
    let raw_chart = take_until("\nE");
    let end = tag("\nE");

    let (input, (_, level, _, raw, _)) =
        (inote_tag, level, equal_sign, raw_chart, end).parse(input)?;
    let level = level.parse().unwrap();
    let raw = raw.trim();
    Ok((input, FileItem::ChartSection { level, raw }))
}

pub fn parse_file_items(input: &str) -> IResult<&str, Vec<FileItem<'_>>> {
    many0(preceded(
        multispace0,
        alt((parse_inote_raw, parse_metadata_line)),
    ))
    .parse(input)
}

fn parse_constant(s: &str) -> (u8, u8) {
    let mut parts = s.split('.');
    let major = parts.next().unwrap().parse().unwrap();
    let minor = parts.next().unwrap().parse().unwrap();
    (major, minor)
}

struct TimingState {
    bpm: u32,
    divider: f64,
    curr_time_ms: f64,
    bpm_list: Vec<BpmRecord>,
}
impl TimingState {
    fn new() -> Self {
        Self {
            bpm: 0,
            divider: 1.0,
            curr_time_ms: 0.0,
            bpm_list: Vec::new(),
        }
    }

    fn beat_duration_ms(&self) -> f64 {
        // let the BPM value is B and the length divider is T,
        // per-comma length = 240 / B / T (seconds)
        240.0 / self.bpm as f64 / self.divider * 1000.0
    }

    fn advance(&mut self) {
        self.curr_time_ms += self.beat_duration_ms();
    }
}

fn parse_simai_tokens(input: &str) -> IResult<&str, Vec<SimaiToken>> {
    todo!()
}

//                                                 we use a vector here in case the string represent an each
fn parse_note_string(input: &str) -> IResult<&str, Vec<Note>> {
    todo!()
}

fn parse_raw_chart(input: &str) -> IResult<&str, (Vec<BpmRecord>, Vec<Note>)> {
    let mut state = TimingState::new();
    let mut notes = Vec::new();

    let (rest, tokens) = parse_simai_tokens(input)?;

    for token in tokens {
        match token {
            SimaiToken::BpmChange(bpm) => {
                todo!();
                // continue here since bpmchange does not update the time state
                continue;
            }
            SimaiToken::DividerChange(divider) => {
                todo!()
            }
            SimaiToken::Empty => {}
            SimaiToken::Note(note_string) => {
                let (rest, mut note) = parse_note_string(note_string)?;
                assert!(rest == "");
                notes.append(&mut note);
            }
            SimaiToken::End => {
                break;
            }
        }
        state.advance();
    }

    Ok((todo!(), (state.bpm_list, notes)))
}

pub fn parse_entire_file(input: &str) -> Result<ProcessedFile, nom::Err<nom::error::Error<&str>>> {
    let (_, items) = parse_file_items(input)?;
    let mut headers = HashMap::new();
    let mut chart_sections = Vec::new();
    for item in items {
        match item {
            FileItem::Metadata(key, value) => {
                headers.insert(key, value);
            }
            FileItem::ChartSection { level, raw } => chart_sections.push((level, raw)),
        }
    }

    let title = headers.get("title").unwrap().to_string();
    let cabinet = match *headers.get("cabinet").unwrap() {
        "SD" => Cabinet::SD,
        "DX" => Cabinet::DX,
        _ => panic!(),
    };
    let version = headers.get("version").unwrap().to_string();
    assert!(headers.get("first").is_none()); // not present in my dataset, so assert here for future reference

    let mut all_charts = Vec::new();

    for (level, raw) in chart_sections {
        let lv_key = format!("lv_{}", level);
        let constant = parse_constant(&headers.get(lv_key.as_str()).unwrap());
        let des_key = format!("des_{}", level);
        let designer = headers.get(des_key.as_str()).unwrap().to_string();

        let (rest, (bpm_list, notes)) = parse_raw_chart(raw)?;

        all_charts.push(Chart {
            constant,
            designer,
            bpm_list,
            notes,
        });
    }

    Ok(ProcessedFile {
        title,
        cabinet,
        version,
        charts: all_charts,
    })
}
