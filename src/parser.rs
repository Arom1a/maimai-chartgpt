use crate::schema::*;
use nom::{
    IResult, Parser,
    branch::alt,
    bytes::complete::{tag, take_until, take_while1},
    character::complete::{char, line_ending, multispace0, not_line_ending, one_of},
    combinator::{all_consuming, map, map_res, opt, recognize},
    multi::many0,
    sequence::{delimited, preceded, terminated},
};
use std::collections::{BTreeSet, HashMap};

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

    let (input, (_, level, _, raw)) = (
        inote_tag,
        level,
        equal_sign,
        recognize(terminated(raw_chart, end)),
    )
        .parse(input)?;
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

fn parse_segment(input: &str) -> IResult<&str, Vec<SimaiToken<'_>>> {
    fn parse_bpm(input: &str) -> IResult<&str, SimaiToken<'_>> {
        map_res(
            delimited(char('('), take_while1(|c: char| c.is_digit(10)), char(')')),
            |s: &str| s.parse().map(SimaiToken::BpmChange),
        )
        .parse(input)
    }
    fn parse_divider(input: &str) -> IResult<&str, SimaiToken<'_>> {
        map_res(
            delimited(char('{'), take_while1(|c: char| c.is_digit(10)), char('}')),
            |s: &str| s.parse().map(SimaiToken::DividerChange),
        )
        .parse(input)
    }
    fn parse_note_or_empty(input: &str) -> IResult<&str, SimaiToken<'_>> {
        if input.is_empty() {
            Ok(("", SimaiToken::Empty))
        } else if input == "E" {
            Ok(("", SimaiToken::End))
        } else {
            Ok(("", SimaiToken::Note(input)))
        }
    }

    let mut rtn = Vec::new();
    let mut rest = input;
    loop {
        rest = rest.trim();
        if let Ok((remain, bpm)) = parse_bpm(rest) {
            rtn.push(bpm);
            rest = remain;
        } else if let Ok((remain, div)) = parse_divider(rest) {
            rtn.push(div);
            rest = remain;
        } else {
            let (_, token) = parse_note_or_empty(rest)?;
            rtn.push(token);
            break;
        }
    }

    Ok((rest, rtn))
}

fn parse_simai_tokens(input: &str) -> IResult<&str, Vec<SimaiToken<'_>>> {
    let mut tokens = Vec::new();
    let mut rest = input;

    loop {
        let seg = match rest.find(',') {
            Some(comma_pos) => {
                let rtn = rest[..comma_pos].trim();
                rest = &rest[comma_pos + 1..];
                rtn
            }
            None => {
                let rtn = rest.trim();
                rest = "";
                rtn
            }
        };

        let (_, mut notes) = parse_segment(seg)?;
        tokens.append(&mut notes);

        if rest.is_empty() {
            break;
        }
    }

    println!("{}", input);
    println!("{:?}", tokens);
    Ok((rest, tokens))
}

fn parse_starting_pos(input: &str) -> IResult<&str, Pos> {
    alt((
        map_res(one_of("12345678"), |btn| (None, btn).try_into()),
        map_res((one_of("ABDE"), one_of("12345678")), |(s, n)| {
            (Some(s), n).try_into()
        }),
        map(preceded(char('C'), opt(one_of("12"))), |_| Pos::C),
    ))
    .parse(input)
}

fn parse_decoration_and_hold(input: &str) -> IResult<&str, (BTreeSet<NoteDecoration>, bool)> {
    todo!()
}

fn parse_slide_segments(input: &str) -> IResult<&str, Vec<SlideSegment>> {
    todo!()
}

fn parse_slide_decoration(input: &str) -> IResult<&str, BTreeSet<SlideDeco>> {
    todo!()
}

fn parse_duration_expression(input: &str) -> IResult<&str, DurationExpr> {
    todo!()
}

fn parse_single_note(input: &str, starting_pos: Pos) -> IResult<&str, UnresolvedNote> {
    let (rest, (deco, is_hold)) = parse_decoration_and_hold(input)?;

    if is_hold {
        let (rest, dur_expr) = parse_duration_expression(rest)?;
        let kind = if starting_pos.is_button() {
            NoteKind::Hold
        } else {
            NoteKind::TouchHold
        };
        return Ok((
            rest,
            UnresolvedNote {
                kind,
                pos: starting_pos,
                deco,
                duration_expr: Some(dur_expr),
                slide_segments: vec![],
                slide_deco: BTreeSet::new(),
            },
        ));
    }

    if let Ok((rest, segments)) = parse_slide_segments(rest) {
        let (rest, slide_deco) = parse_slide_decoration(rest)?;
        let (rest, duration_expr) = opt(parse_duration_expression).parse(rest)?;
        assert!(duration_expr.is_some());
        return Ok((
            rest,
            UnresolvedNote {
                kind: NoteKind::Slide,
                pos: starting_pos,
                deco,
                duration_expr,
                slide_segments: segments,
                slide_deco,
            },
        ));
    }

    let kind = if starting_pos.is_button() {
        NoteKind::Tap
    } else {
        NoteKind::Touch
    };
    Ok((
        rest,
        UnresolvedNote {
            kind,
            pos: starting_pos,
            deco,
            duration_expr: None,
            slide_segments: vec![],
            slide_deco: BTreeSet::new(),
        },
    ))
}

//                                                 we use a vector here in case the string represent an each
//                                                 or multiple slides
fn parse_note_string(input: &str) -> IResult<&str, Vec<UnresolvedNote>> {
    println!("{}", input);
    let mut rtn = Vec::new();

    // first split by '/' and then by '*'
    for sub_str in input.split('/') {
        let (rest, starting_pos) = parse_starting_pos(sub_str)?;

        for sub_note in rest.split('*') {
            // then parse each part
            let (_, note) =
                all_consuming(|s| parse_single_note(s, starting_pos)).parse(sub_note)?;
            rtn.push(note);
        }
    }

    Ok(("", rtn))
}

fn parse_raw_chart(input: &str) -> IResult<&str, (Vec<BpmRecord>, Vec<Note>)> {
    let mut state = TimingState::new();
    let mut notes = Vec::new();

    let (rest, tokens) = parse_simai_tokens(input)?;

    for token in tokens {
        match token {
            SimaiToken::BpmChange(bpm) => {
                if state.bpm != bpm {
                    state.bpm_list.push(BpmRecord {
                        bpm,
                        timestamp: state.curr_time_ms as _,
                    });
                    state.bpm = bpm;
                }
                // continue here since bpmchange does not update the time state
                continue;
            }
            SimaiToken::DividerChange(divider) => {
                state.divider = divider;
                // continue for the same reason
                continue;
            }
            SimaiToken::Empty => {}
            SimaiToken::Note(note_string) => {
                let (_, parsed_notes) = parse_note_string(note_string)?;
                let mut resolved_notes = parsed_notes
                    .into_iter()
                    .map(|note| Note {
                        timestamp: state.curr_time_ms as _,
                        kind: note.kind,
                        pos: note.pos,
                        deco: note.deco,
                        duration: note.duration_expr.map(|expr| expr.resolve_ms(state.bpm)),
                        slide_segments: note.slide_segments,
                        slide_deco: note.slide_deco,
                    })
                    .collect();
                notes.append(&mut resolved_notes);
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn note_string_tap1() {
        let input = "1";
        let (rest, output) = parse_note_string(input).unwrap();
        let note = Note {
            timestamp: 0,
            kind: NoteKind::Tap,
            pos: Pos::Btn1,
            deco: BTreeSet::new(),
            duration: None,
            slide_segments: Vec::new(),
            slide_deco: BTreeSet::new(),
        };
        assert!(rest.is_empty());
        // assert_eq!(output, vec![note]);
    }
    #[test]
    fn note_string_tap2() {
        let input = "8";
    }

    #[test]
    fn note_string_each1() {
        let input = "1/8";
    }
    #[test]
    fn note_string_each2() {
        let input = "7/3";
    }

    #[test]
    fn note_string_hold1() {
        let input = "2h[8:5]";
    }
    #[test]
    fn note_string_hold2() {
        let input = "4h[2:7]";
    }
    fn note_string_hold3() {
        let input = "7h[4:0]";
    }
    fn note_string_hold4() {
        let input = "5h";
    }

    #[test]
    fn note_string_tap_hold_each1() {
        let input = "6/7h[4:3]";
    }
    #[test]
    fn note_string_tap_hold_each2() {
        let input = "8h[8:2]/3";
    }

    // #[test]
    // fn note_string_slide_festival1() {
    // let input = "7-4-1-6-3-8[4:4]"
    // }
}
