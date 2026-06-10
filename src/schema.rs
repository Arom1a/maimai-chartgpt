use std::collections::BTreeSet;

pub enum NoteKind {
    Tap,
    Hold,
    Slide,
    Touch,
    TouchHold,
}

#[rustfmt::skip]
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

pub enum NoteDecoration {
    Break,
    Ex,
    PseudoEach,
    Firework,
}

pub enum SlideShape {
    Straight,     // -
    ArcLeft,      // <, do not use ^, auto convert to left/right
    ArcRight,     // >
    Center,       // v
    ZigzagS,      // s
    ZigzagZ,      // z
    P,            // TODO
    Q,            // TODO
    PP,           // TODO
    QQ,           // TODO
    Reflect(Pos), // V
    Wifi,         // w
}

pub struct SlideSegment {
    pub shape: SlideShape,
    pub end: Pos,
}

pub enum SlideDeco {
    Break,
    // omitted as not present in normal charts
    // Fade,
    // StarVisible
}

pub struct Note {
    pub timestamp: u64,
    pub kind: NoteKind,
    pub pos: Pos,
    pub deco: BTreeSet<NoteDecoration>,
    pub duration: Option<u64>,
    pub slide_segments: Vec<SlideSegment>,
    pub slide_deco: BTreeSet<SlideDeco>,
}

pub struct Chart {
    pub constant: (u8, u8), // major, minor
    pub designer: String,
    pub bpm_list: Vec<BpmRecord>,
    pub notes: Vec<Note>,
}

pub struct BpmRecord {
    pub bpm: u32,
    pub timestamp: u64,
}

pub enum Cabinet {
    SD,
    DX,
}

pub struct ProcessedFile {
    pub title: String,
    pub cabinet: Cabinet,
    pub version: String,
    pub charts: Vec<Chart>,
}

pub enum SimaiToken<'a> {
    BpmChange(u32),
    DividerChange(f64),
    Empty,
    Note(&'a str),
    End,
}

// TODO: invariants for files and each note kind
pub fn check_invariant() -> bool {
    todo!()
}
