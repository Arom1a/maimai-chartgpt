use crate::schema::*;
use nom::{
    IResult, Parser,
    branch::alt,
    bytes::complete::{tag, take_until, take_while1},
    character::complete::{char, line_ending, multispace0, not_line_ending, one_of},
    combinator::{all_consuming, map, map_res, opt, recognize},
    multi::{many0, many1},
    sequence::{delimited, preceded, terminated},
};
use std::collections::{BTreeSet, HashMap};

#[derive(Debug)]
enum FileItem<'a> {
    Metadata(&'a str, &'a str),
    ChartSection { level: u8, raw: &'a str },
}

fn parse_float1(input: &str) -> IResult<&str, &str> {
    take_while1(|c: char| c.is_digit(10) || c == '.').parse(input)
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

fn parse_file_items(input: &str) -> IResult<&str, Vec<FileItem<'_>>> {
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
    bpm10: u32,
    divider: u32,
    curr_time_ms: f64,
    bpm10_list: Vec<BpmRecord>,
}
impl TimingState {
    fn new() -> Self {
        Self {
            bpm10: 0,
            divider: 0,
            curr_time_ms: 0.0,
            bpm10_list: Vec::new(),
        }
    }

    fn beat_duration_ms(&self) -> f64 {
        // let the BPM value is B and the length divider is T,
        // per-comma length = 240 / B / T (seconds)
        // 240.0 * 10.0 * 1000.0 = 2400000.0, as it is bpm10 and ms
        2400000.0 / self.bpm10 as f64 / self.divider as f64
    }

    fn advance(&mut self) {
        self.curr_time_ms += self.beat_duration_ms();
    }
}

fn parse_segment(input: &str) -> IResult<&str, Vec<SimaiToken<'_>>> {
    fn parse_bpm10(input: &str) -> IResult<&str, SimaiToken<'_>> {
        map_res(delimited(char('('), parse_float1, char(')')), |s: &str| {
            s.parse()
                .map(|bpm: f32| SimaiToken::Bpm10Change((bpm * 10.0) as u32))
        })
        .parse(input)
    }
    fn parse_divider(input: &str) -> IResult<&str, SimaiToken<'_>> {
        map_res(delimited(char('{'), parse_float1, char('}')), |s: &str| {
            s.parse().map(SimaiToken::DividerChange)
        })
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
        if let Ok((remain, bpm10)) = parse_bpm10(rest) {
            rtn.push(bpm10);
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
    let (rest, decos) = many0(alt((char('b'), char('x'), char('f'), char('h')))).parse(input)?;
    let mut is_hold = false;
    Ok((
        rest,
        (
            decos
                .into_iter()
                .filter_map(|c| {
                    if c == 'h' {
                        is_hold = true;
                        None
                    } else {
                        Some(match c {
                            'b' => NoteDecoration::Break,
                            'x' => NoteDecoration::Ex,
                            'f' => NoteDecoration::Firework,
                            _ => unreachable!(),
                        })
                    }
                })
                .collect(),
            is_hold,
        ),
    ))
}

fn parse_slide_chain(input: &str) -> IResult<&str, Vec<SlideSegment>> {
    fn parse_shape(input: &str) -> IResult<&str, SlideShape> {
        alt((
            map(char('-'), |_| SlideShape::Straight),
            map(char('<'), |_| SlideShape::ArcLeft),
            map(char('>'), |_| SlideShape::ArcRight),
            map(char('v'), |_| SlideShape::Center),
            map(char('s'), |_| SlideShape::ZigzagS),
            map(char('z'), |_| SlideShape::ZigzagZ),
            map(tag("pp"), |_| SlideShape::PP),
            map(tag("qq"), |_| SlideShape::QQ),
            map_res(preceded(char('V'), one_of("12345678")), |btn| {
                (None, btn).try_into().map(SlideShape::Reflect)
            }),
            map(char('p'), |_| SlideShape::P),
            map(char('q'), |_| SlideShape::Q),
            map(char('w'), |_| SlideShape::Wifi),
        ))
        .parse(input)
    }
    fn parse_segment(input: &str) -> IResult<&str, SlideSegment> {
        let (rest, shape) = parse_shape(input)?;
        let (rest, btn) = one_of("12345678").parse(rest)?;
        let end = (None, btn).try_into()?;
        Ok((rest, SlideSegment { shape, end }))
    }

    many1(parse_segment).parse(input)
}

fn parse_slide_decoration(input: &str) -> IResult<&str, BTreeSet<SlideDeco>> {
    let (rest, decos) = many0(alt((map(char('b'), |_| SlideDeco::Break),))).parse(input)?;
    Ok((rest, decos.into_iter().collect()))
}

fn parse_duration_expression(
    input: &str,
) -> IResult<&str, (Option<DurationExpr>, Option<DurationExpr>)> {
    fn parse_inner_string(
        input: &str,
    ) -> IResult<&str, (Option<DurationExpr>, Option<DurationExpr>)> {
        let opt_wait_override = opt(terminated(parse_float1, tag("##")));
        let opt_bpm_override = opt(terminated(parse_float1, char('#')));
        let div_mul = map(
            (parse_float1, char(':'), parse_float1),
            |(div_s, _, mul_s)| {
                let div: f64 = div_s.parse().unwrap();
                debug_assert!(format!("{:?}", div).ends_with('0'));
                let div: u32 = div as u32;
                let mul: u32 = mul_s.parse().unwrap();
                DurationExpr::DividerMultiplier(div, mul)
            },
        );
        let abs = map(parse_float1, |s: &str| {
            DurationExpr::AbsoluteMs(s.parse::<f64>().unwrap() * 1000.0)
        });
        let div_mul_or_abs = alt((div_mul, abs));

        let (rest, (wait, bpm, dur_expr)) =
            (opt_wait_override, opt_bpm_override, div_mul_or_abs).parse(input)?;

        let wait = wait.map(|s| DurationExpr::AbsoluteMs(s.parse::<f64>().unwrap() * 1000.0));

        if let Some(bpm_s) = bpm
            && let DurationExpr::DividerMultiplier(div, mul) = dur_expr
        {
            let bpm: f64 = bpm_s.parse().unwrap();
            let ms = 240.0 / bpm / div as f64 * mul as f64 * 1000.0;
            Ok((rest, (wait, Some(DurationExpr::AbsoluteMs(ms)))))
        } else {
            Ok((rest, (wait, Some(dur_expr))))
        }
    }

    let (rest, rtn) = delimited(char('['), parse_inner_string, char(']')).parse(input)?;

    Ok((rest, rtn))
}

fn parse_single_note(input: &str, starting_pos: Pos) -> IResult<&str, Note> {
    let (rest, (deco, is_hold)) = parse_decoration_and_hold(input)?;

    // found h, so this note is a hold
    if is_hold {
        let (rest, (wait, dur_expr)) = parse_duration_expression(rest)?;
        debug_assert!(wait.is_none());
        let kind = if starting_pos.is_button() {
            NoteKind::Hold
        } else {
            NoteKind::TouchHold
        };
        return Ok((
            rest,
            Note {
                timestamp_ms: 0,
                kind,
                pos: starting_pos,
                deco,
                wait,
                duration: dur_expr,
                slide_segments: vec![],
                slide_deco: BTreeSet::new(),
            },
        ));
    }

    // found slide shape chars, so this note is a slide
    if let Ok((rest, slide_segments)) = parse_slide_chain(rest) {
        let (rest, mut slide_deco) = parse_slide_decoration(rest)?;
        let (rest, (wait, duration)) = parse_duration_expression(rest)?;
        let (rest, slide_deco_back) = parse_slide_decoration(rest)?;
        slide_deco.extend(slide_deco_back.into_iter());
        return Ok((
            rest,
            Note {
                timestamp_ms: 0,
                kind: NoteKind::Slide,
                pos: starting_pos,
                deco,
                wait,
                duration,
                slide_segments,
                slide_deco,
            },
        ));
    }

    // not found anything special, so this note is either a tap or touch
    let kind = if starting_pos.is_button() {
        NoteKind::Tap
    } else {
        NoteKind::Touch
    };
    Ok((
        rest,
        Note {
            timestamp_ms: 0,
            kind,
            pos: starting_pos,
            deco,
            wait: None,
            duration: None,
            slide_segments: vec![],
            slide_deco: BTreeSet::new(),
        },
    ))
}

//                                                 we use a vector here in case the string represent an each
//                                                 or multiple slides
fn parse_note_string(input: &str) -> IResult<&str, Vec<Note>> {
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
            SimaiToken::Bpm10Change(bpm10) => {
                if state.bpm10 != bpm10 {
                    state.bpm10_list.push(BpmRecord {
                        bpm10,
                        change_timestamp_ms: state.curr_time_ms as _,
                    });
                    state.bpm10 = bpm10;
                }
                // continue here since bpmchange does not update the time state
                continue;
            }
            SimaiToken::DividerChange(divider) => {
                debug_assert!(format!("{:?}", divider).ends_with('0'), "{:2?}", divider);
                state.divider = divider as u32;
                // continue for the same reason
                continue;
            }
            SimaiToken::Empty => {}
            SimaiToken::Note(note_string) => {
                let (_, mut parsed_notes) = parse_note_string(note_string)?;
                for note in &mut parsed_notes {
                    note.timestamp_ms = state.curr_time_ms as _;
                }
                notes.append(&mut parsed_notes);
            }
            SimaiToken::End => {
                break;
            }
        }
        state.advance();
    }

    Ok((rest, (state.bpm10_list, notes)))
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
    debug_assert!(headers.get("first").is_none()); // not present in my dataset, so assert here for future reference

    let mut all_charts = Vec::new();

    for (level, raw) in chart_sections {
        let lv_key = format!("lv_{}", level);
        let constant = parse_constant(&headers.get(lv_key.as_str()).unwrap());
        let des_key = format!("des_{}", level);
        let designer = headers.get(des_key.as_str()).unwrap().to_string();

        let (_rest, (bpm10_list, notes)) = parse_raw_chart(raw)?;

        all_charts.push(Chart {
            constant,
            designer,
            bpm10_list,
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
            timestamp_ms: 0,
            kind: NoteKind::Tap,
            pos: Pos::Btn1,
            deco: BTreeSet::new(),
            wait: None,
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
