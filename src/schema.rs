use serde::Serialize;
use std::collections::BTreeSet;

#[derive(Debug, PartialEq, Serialize)]
pub enum NoteKind {
    Tap,
    Hold,
    Slide,
    Touch,
    TouchHold,
}

#[rustfmt::skip]
#[derive(Debug, PartialEq, Clone, Copy, Serialize)]
pub enum Pos {
    // button 1-8
    Btn1, Btn2, Btn3, Btn4, Btn5, Btn6, Btn7, Btn8,
    // censor A 1-8
    A1, A2, A3, A4, A5, A6, A7, A8,
    // censor B 1-8
    B1, B2, B3, B4, B5, B6, B7, B8,
    // cencor C, map C1 and C2 to C
    C,
    // censor D
    D1, D2, D3, D4, D5, D6, D7, D8,
    // censor E
    E1, E2, E3, E4, E5, E6, E7, E8,
}

impl Pos {
    #[rustfmt::skip]
    pub fn is_button(&self) -> bool {
        matches!(
            self,
              Pos::Btn1 | Pos::Btn2 | Pos::Btn3 | Pos::Btn4
            | Pos::Btn5 | Pos::Btn6 | Pos::Btn7 | Pos::Btn8
        )
    }

    pub fn is_touch(&self) -> bool {
        !self.is_button()
    }
}

impl TryFrom<(Option<char>, char)> for Pos {
    type Error = nom::Err<nom::error::Error<&'static str>>;

    #[rustfmt::skip]
    fn try_from((sensor, btn): (Option<char>, char)) -> Result<Self, Self::Error> {
        let err = Err(nom::Err::Error(nom::error::Error::new(
            "invalid position",
            nom::error::ErrorKind::Char,
        )));
        match sensor {
            None => match btn {
                '1' => Ok(Pos::Btn1), '2' => Ok(Pos::Btn2), '3' => Ok(Pos::Btn3), '4' => Ok(Pos::Btn4),
                '5' => Ok(Pos::Btn5), '6' => Ok(Pos::Btn6), '7' => Ok(Pos::Btn7), '8' => Ok(Pos::Btn8),
                _ => err,
            },
            Some('A') => match btn {
                '1' => Ok(Pos::A1), '2' => Ok(Pos::A2), '3' => Ok(Pos::A3), '4' => Ok(Pos::A4),
                '5' => Ok(Pos::A5), '6' => Ok(Pos::A6), '7' => Ok(Pos::A7), '8' => Ok(Pos::A8),
                _ => err,
            },
            Some('B') => match btn {
                '1' => Ok(Pos::B1), '2' => Ok(Pos::B2), '3' => Ok(Pos::B3), '4' => Ok(Pos::B4),
                '5' => Ok(Pos::B5), '6' => Ok(Pos::B6), '7' => Ok(Pos::B7), '8' => Ok(Pos::B8),
                _ => err,
            },
            Some('C') => match btn {
                '1' => Ok(Pos::C), '2' => Ok(Pos::C), _ => err,
            },
            Some('D') => match btn {
                '1' => Ok(Pos::D1), '2' => Ok(Pos::D2), '3' => Ok(Pos::D3), '4' => Ok(Pos::D4),
                '5' => Ok(Pos::D5), '6' => Ok(Pos::D6), '7' => Ok(Pos::D7), '8' => Ok(Pos::D8),
                _ => err,
            },
            Some('E') => match btn {
                '1' => Ok(Pos::E1), '2' => Ok(Pos::E2), '3' => Ok(Pos::E3), '4' => Ok(Pos::E4),
                '5' => Ok(Pos::E5), '6' => Ok(Pos::E6), '7' => Ok(Pos::E7), '8' => Ok(Pos::E8),
                _ => err,
            },
            _ => err,
        }
    }
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub enum NoteDecoration {
    Break,
    Ex,
    // omitted as not present in my dataset
    // PseudoEach,
    Firework,
}

#[derive(Debug, PartialEq, Serialize)]
pub enum SlideShape {
    Straight,     // -
    ArcLeft,      // <, do not use ^, auto convert to left/right
    Center,       // v, furthermore, ^ is not present in my dataset
    ArcRight,     // >
    ZigzagS,      // s
    ZigzagZ,      // z
    P,            // TODO
    Q,            // TODO
    PP,           // TODO
    QQ,           // TODO
    Reflect(Pos), // V
    Wifi,         // w
}

#[derive(Debug, PartialEq, Serialize)]
pub struct SlideSegment {
    pub shape: SlideShape,
    pub end: Pos,
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub enum SlideDeco {
    Break,
    // omitted as not present in normal charts
    // Fade,
    // StarVisible
}

#[derive(Debug, PartialEq, Serialize)]
pub enum DurationExpr {
    DividerMultiplier(u32, u32),
    AbsoluteMs(f64),
    // we will process this to AbsoluteMs automatically
    // Bpm10OverideDividerMultiplier {
    //     bpm10: u32,
    //     divider: f64,
    //     multiplier: u32,
    // },
}

#[derive(Debug, PartialEq, Serialize)]
pub struct Note {
    pub timestamp_ms: u64,
    pub kind: NoteKind,
    pub pos: Pos,
    pub deco: BTreeSet<NoteDecoration>,
    pub wait: Option<DurationExpr>,
    pub duration: Option<DurationExpr>,
    pub slide_segments: Vec<SlideSegment>,
    pub slide_deco: BTreeSet<SlideDeco>,
}

#[derive(Debug, Serialize)]
pub struct Chart {
    pub constant: (u8, u8), // major, minor
    pub designer: String,
    pub bpm10_list: Vec<BpmRecord>,
    pub notes: Vec<Note>,
}

#[derive(Debug, Serialize)]
pub struct BpmRecord {
    pub bpm10: u32,
    pub change_timestamp_ms: u64,
}

#[derive(Debug, Serialize)]
pub enum Cabinet {
    SD,
    DX,
}

#[derive(Debug, Serialize)]
pub struct ProcessedFile {
    pub title: String,
    pub cabinet: Cabinet,
    pub version: String,
    pub charts: Vec<Chart>,
}

#[derive(Debug)]
pub enum SimaiToken<'a> {
    Bpm10Change(u32),
    DividerChange(f64),
    Empty,
    Note(&'a str),
    End,
}

// TODO: invariants for files and each note kind
pub fn check_invariant() -> bool {
    todo!()
}
