use nom::{
    IResult, Parser,
    bytes::{complete::take_until, tag},
    character::complete::{char, not_line_ending},
};

pub fn parse_metadata_line(input: &str) -> IResult<&str, (&str, &str)> {
    let start = char('&');
    let key = take_until("=");
    let equal_sign = char('=');
    let value = not_line_ending;

    let (input, (_, key, _, value)) = (start, key, equal_sign, value).parse(input)?;
    Ok((input, (key, value)))
}

pub fn parse_inote(input: &str) -> IResult<&str, (&str, &str)> {
    let inote_tag = tag("&inote_");
    let level = take_until("=");
    let equal_sign = char('=');
    let chart = take_until("E");

    let (input, (_, level, _, chart)) = (inote_tag, level, equal_sign, chart).parse(input)?;
    Ok((input, (level, chart)))
}
