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
    Fan,          // w
}

pub struct SlideSegment {
    shape: SlideShape,
    end: Pos,
}

pub enum SlideDeco {
    Break,
    // omitted as not present in normal charts
    // Fade,
    // StarVisible
}

pub struct Note {
    timestamp: u64,
    kind: NoteKind,
    pos: Pos,
    deco: BTreeSet<NoteDecoration>,
    duration: Option<u64>,
    slide_segments: Vec<SlideSegment>,
    slide_deco: BTreeSet<SlideDeco>,
}

pub struct Chart {
    constant: (u8, u8), // major, minor
    designer: String,
    notes: Vec<Note>,
}

pub struct BpmRecord {
    bpm: u32,
    timestamp: u64,
}

enum Cabinet {
    SD,
    DX,
}

pub struct ProcessedFile {
    title: String,
    cabinet: Cabinet,
    version: String,
    bpm_list: Vec<BpmRecord>,
    charts: Vec<Chart>,
}

// TODO: invariants for files and each note kind
pub fn check_invariant() -> bool {
    todo!()
}
