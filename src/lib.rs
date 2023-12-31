use std::{
  collections::{hash_map::Entry::*, HashMap, HashSet},
  fmt::Display,
};

use thiserror::Error;

#[derive(Clone, Debug)]
pub enum RispExpr {
  Boolean(bool),
  Float(f64),
  List(Vec<RispExpr>),
  Symbol(String),
  Function(fn(&[RispExpr]) -> Result<RispExpr, RispError>),
}

impl Display for RispExpr {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(
      f,
      "{}",
      match self {
        RispExpr::Boolean(b) => b.to_string(),
        RispExpr::Float(n) => n.to_string(),
        RispExpr::List(l) => format!("{:?}", dbg!(l)),
        RispExpr::Symbol(s) => s.to_string(),
        RispExpr::Function(f) => format!("{:?}", dbg!(f)),
      }
    )
  }
}

impl TryFrom<RispExpr> for f64 {
  type Error = RispError;

  fn try_from(value: RispExpr) -> Result<Self, Self::Error> {
    match value {
      RispExpr::Float(n) => Ok(n),
      _ => Err(RispError::ExpectedNumber),
    }
  }
}

macro_rules! risp_elementary_arithmetic {
  ($operator:tt) => {
    RispExpr::Function(|argv| {
      Ok(RispExpr::Float(
        parse_float_list(argv)?
          .iter()
          .copied()
          .reduce(|acc, x| acc $operator x)
          .ok_or(RispError::ExpectedNumber)?,
      ))
    })
  };
}

fn modulo(a: f64, b: f64) -> f64 {
  ((a % b) + b) % b
}

fn remainder(a: f64, b: f64) -> f64 {
  a % b
}

macro_rules! risp_modulo_remainder {
  ($operator:ident) => {
    RispExpr::Function(|argv| {
      if argv.len() > 2 {
        return Err(RispError::ExpectedTwoForms(
          stringify!($operator)[..3].to_string(),
        ));
      }

      let argv = argv
        .iter()
        .cloned()
        .map(TryInto::<f64>::try_into)
        .collect::<Vec<Result<f64, RispError>>>();
      let (a, b) = (
        argv.get(0).ok_or(RispError::ExpectedNumber)?.clone()?,
        argv.get(1).ok_or(RispError::ExpectedNumber)?.clone()?,
      );

      Ok(RispExpr::Float($operator(a, b)))
    })
  };
}

macro_rules! risp_extremum {
  ($operator:ident) => {
    RispExpr::Function(|argv| {
      Ok(RispExpr::Float(
        parse_float_list(argv)?
          .iter()
          .$operator(|a, b| a.total_cmp(b))
          .copied()
          .ok_or(RispError::ExpectedNumber)?,
      ))
    })
  };
}

macro_rules! risp_logical_condition {
  ($operator:tt) => {
    RispExpr::Function(|argv| {
      fn compare(curr: &f64, nums: &[f64]) -> bool {
        match nums.first() {
          Some(next) => (curr $operator next) && compare(next, &nums[1..]),
          None => true,
        }
      }

      Ok(RispExpr::Boolean(compare(
        &parse_float(&argv.first().ok_or(RispError::ExpectedNumber)?.clone())?,
        &parse_float_list(argv)?[1..],
      )))
    })
  };
  (!=:tt) => {
    RispExpr::Function(|argv| {
      if argv.is_empty() {
        return Err(RispError::ExpectedNumber);
      }

      Ok(RispExpr::Boolean((1..argv.len()).any(|i| argv[i..].contains(&argv[i - 1]))))
    })
  };
}

pub struct RispEnv {
  reserved_keywords: HashSet<String>,
  var: HashMap<String, RispExpr>,
}

impl RispEnv {
  pub fn new_default_env() -> Self {
    Self::default()
  }
}

impl Default for RispEnv {
  fn default() -> Self {
    let reserved_keywords: HashSet<String> = HashSet::from_iter(
      [
        "defvar", "setq", "if", "max", "min", "mod", "rem", "+", "-", "*", "/", "=", "!=", ">",
        "<", ">=", "<=",
      ]
      .map(|s| s.to_string()),
    );

    let mut var: HashMap<String, RispExpr> = HashMap::new();

    var.insert("+".to_string(), risp_elementary_arithmetic!(+));
    var.insert("-".to_string(), risp_elementary_arithmetic!(-));
    var.insert("*".to_string(), risp_elementary_arithmetic!(*));
    var.insert("/".to_string(), risp_elementary_arithmetic!(/));

    var.insert("mod".to_string(), risp_modulo_remainder!(modulo));
    var.insert("rem".to_string(), risp_modulo_remainder!(remainder));

    var.insert("max".to_string(), risp_extremum!(max_by));
    var.insert("min".to_string(), risp_extremum!(min_by));

    var.insert("=".to_string(), risp_logical_condition!(==));
    var.insert("!=".to_string(), risp_logical_condition!(!=));
    var.insert(">".to_string(), risp_logical_condition!(>));
    var.insert("<".to_string(), risp_logical_condition!(<));
    var.insert(">=".to_string(), risp_logical_condition!(>=));
    var.insert("<=".to_string(), risp_logical_condition!(<=));

    RispEnv {
      reserved_keywords,
      var,
    }
  }
}

#[derive(Clone, Debug, Error)]
pub enum RispError {
  #[error("Could not read token.")]
  CouldNotReadToken,

  #[error("Expected `(`.")]
  ExpectedLeftBracket,

  #[error("Unexpected `)`.")]
  UnexpectedRightBracket,

  #[error("Missing `)`.")]
  MissingRightBracket,

  #[error("Expected a number.")]
  ExpectedNumber,

  #[error("Unexpected symbol '{0}'.")]
  UnexpectedSymbol(String),

  #[error("Expected a nonempty list.")]
  ExpectedNonEmptyList,

  #[error("Invalid form.")]
  InvalidForm,

  #[error("First form must be a function.")]
  SymbolNotFunction,

  #[error("Failed to read line from stdin.")]
  StdinError,

  #[error("Expected a condition in `if`.")]
  ExpectedIfCondition,

  #[error("Invalid condition '{0}' in `if`.")]
  InvalidIfCondition(String),

  #[error("Invalid `if` form.")]
  InvalidIfForm,

  #[error("Expected symbol in `{0}`.")]
  ExpectedSymbol(String),

  #[error("Expected value in `{0}`.")]
  ExpectedValue(String),

  #[error("Invalid a symbol '{0}' in `{1}`.")]
  InvalidSymbol(String, String),

  #[error("Invalid a value '{0}' in `{1}`.")]
  InvalidValue(String, String),

  #[error("Variable '{0}' is already defined.")]
  VariableAlreadyDefined(String),

  #[error("Variable '{0}' is not yet defined.")]
  VariableNotYetDefined(String),

  #[error("Expected only two forms in `{0}`.")]
  ExpectedTwoForms(String),
}

pub fn interpret(exprs: &str, env: &mut RispEnv) -> Result<RispExpr, RispError> {
  let exprs = exprs.trim();

  if exprs.chars().next().ok_or(RispError::CouldNotReadToken)? != '(' {
    return Err(RispError::ExpectedLeftBracket);
  }

  evaluate(&parse(&tokenise(exprs))?.0, env)
}

fn tokenise(expr: &str) -> Vec<String> {
  expr
    .replace('(', " ( ")
    .replace(')', " ) ")
    .split_whitespace()
    .map(|x| x.to_string())
    .collect()
}

fn parse(tokens: &[String]) -> Result<(RispExpr, &[String]), RispError> {
  let (token, unparsed_tokens) = tokens.split_first().ok_or(RispError::CouldNotReadToken)?;

  match token.as_str() {
    "(" => parse_sequence(unparsed_tokens),
    ")" => Err(RispError::UnexpectedRightBracket),
    _ => Ok((parse_atom(token), unparsed_tokens)),
  }
}

fn parse_sequence(tokens: &[String]) -> Result<(RispExpr, &[String]), RispError> {
  let mut parsed_exprs: Vec<RispExpr> = Vec::new();
  let mut unparsed_tokens = tokens;
  let mut parsed_token;

  loop {
    let (current_token, remaining_tokens) = unparsed_tokens
      .split_first()
      .ok_or(RispError::MissingRightBracket)?;

    if current_token == ")" {
      return Ok((RispExpr::List(parsed_exprs), remaining_tokens));
    }

    (parsed_token, unparsed_tokens) = parse(unparsed_tokens)?;
    parsed_exprs.push(parsed_token.clone());
  }
}

fn parse_atom(token: &str) -> RispExpr {
  if let Ok(num) = token.parse() {
    return RispExpr::Float(num);
  }

  match token {
    "true" => RispExpr::Boolean(true),
    "false" => RispExpr::Boolean(false),
    s => RispExpr::Symbol(s.to_string()),
  }
}

fn parse_float_list(numbers: &[RispExpr]) -> Result<Vec<f64>, RispError> {
  numbers.iter().map(parse_float).collect()
}

fn parse_float(expr: &RispExpr) -> Result<f64, RispError> {
  match expr {
    RispExpr::Float(n) => Ok(*n),
    _ => Err(RispError::ExpectedNumber),
  }
}

fn evaluate(expr: &RispExpr, env: &mut RispEnv) -> Result<RispExpr, RispError> {
  match expr {
    RispExpr::List(list) => {
      let (first_form, argv) = list.split_first().ok_or(RispError::ExpectedNonEmptyList)?;
      match evaluate_built_in_form(first_form, argv, env) {
        Some(expr) => expr,
        None => match evaluate(first_form, env)? {
          RispExpr::Function(func) => func(
            &argv
              .iter()
              .map(|arg| evaluate(arg, env))
              .collect::<Result<Vec<RispExpr>, RispError>>()?,
          ),
          _ => Err(RispError::SymbolNotFunction),
        },
      }
    }
    RispExpr::Symbol(symb) => env
      .var
      .get(symb)
      .ok_or(RispError::UnexpectedSymbol(symb.to_string()))
      .cloned(),
    RispExpr::Boolean(_) => Ok(expr.clone()),
    RispExpr::Float(_) => Ok(expr.clone()),
    RispExpr::Function(_) => Err(RispError::InvalidForm),
  }
}

fn evaluate_built_in_form(
  first_form: &RispExpr,
  argv: &[RispExpr],
  env: &mut RispEnv,
) -> Option<Result<RispExpr, RispError>> {
  match first_form {
    RispExpr::Symbol(symb) => match symb.as_str() {
      "defvar" => Some(evaluate_defvar(argv, env)),
      "setq" => Some(evaluate_setq(argv, env)),
      "if" => Some(evaluate_if(argv, env)),
      _ => None,
    },
    _ => None,
  }
}

fn evaluate_defvar(argv: &[RispExpr], env: &mut RispEnv) -> Result<RispExpr, RispError> {
  if argv.len() > 2 {
    return Err(RispError::ExpectedTwoForms("defvar".to_string()));
  }

  let first_form = argv
    .first()
    .ok_or(RispError::ExpectedSymbol("defvar".to_string()))?;
  let symbol = match first_form {
    RispExpr::Symbol(symb) => Ok(symb.to_string()),
    expr => Err(RispError::InvalidSymbol(
      expr.to_string(),
      "defvar".to_string(),
    )),
  }?;

  if env.reserved_keywords.contains(symbol.as_str()) {
    return Err(RispError::InvalidSymbol(symbol, "defvar".to_string()));
  }

  let value = if argv.len() == 2 {
    evaluate(&argv[1], env)?
  } else {
    RispExpr::Float(0_f64)
  };

  if let Vacant(e) = env.var.entry(symbol.clone()) {
    e.insert(value);
  } else {
    return Err(RispError::VariableAlreadyDefined(symbol));
  }

  Ok(first_form.clone())
}

fn evaluate_setq(argv: &[RispExpr], env: &mut RispEnv) -> Result<RispExpr, RispError> {
  if argv.len() > 2 {
    return Err(RispError::ExpectedTwoForms("setq".to_string()));
  }

  let first_form = argv
    .first()
    .ok_or(RispError::ExpectedSymbol("setq".to_string()))?;
  let symbol = match first_form {
    RispExpr::Symbol(symb) => Ok(symb.to_string()),
    expr => Err(RispError::InvalidSymbol(
      expr.to_string(),
      "setq".to_string(),
    )),
  }?;

  let value = if argv.len() == 2 {
    evaluate(&argv[1], env)?
  } else {
    return Err(RispError::ExpectedValue("setq".to_string()));
  };

  if let Occupied(mut e) = env.var.entry(symbol.clone()) {
    e.insert(value);
  } else {
    return Err(RispError::VariableNotYetDefined(symbol));
  }

  Ok(first_form.clone())
}

fn evaluate_if(argv: &[RispExpr], env: &mut RispEnv) -> Result<RispExpr, RispError> {
  match evaluate(argv.first().ok_or(RispError::ExpectedIfCondition)?, env)? {
    RispExpr::Boolean(b) => evaluate(
      argv
        .get(if b { 1 } else { 2 })
        .ok_or(RispError::InvalidIfForm)?,
      env,
    ),
    _ => Err(RispError::InvalidIfCondition(argv[0].to_string())),
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_empty_input() {
    let mut env = RispEnv::new_default_env();

    assert!(matches!(
      interpret("", &mut env),
      Err(RispError::CouldNotReadToken)
    ));
    assert!(matches!(
      interpret(" ", &mut env),
      Err(RispError::CouldNotReadToken)
    ));
  }

  #[test]
  fn test_addition() {
    let mut env = RispEnv::new_default_env();

    let two_numbers = "(+ 3 2)";
    let many_numbers = "(+ 5.4 4.3 3.2 2.1 1.0)";

    assert!(matches!(
      interpret(two_numbers, &mut env),
      Ok(RispExpr::Float(n)) if n == (3_f64 + 2_f64)
    ));
    assert!(matches!(
      interpret(many_numbers, &mut env),
      Ok(RispExpr::Float(n)) if n == (5.4 + 4.3 + 3.2 + 2.1 + 1.0)
    ));

    assert!(matches!(
      interpret("(+)", &mut env),
      Err(RispError::ExpectedNumber)
    ));
  }

  #[test]
  fn test_subtraction() {
    let mut env = RispEnv::new_default_env();

    let two_numbers = "(- 3 2)";
    let many_numbers = "(- 5.4 4.3 3.2 2.1 1.0)";

    assert!(matches!(
      interpret(two_numbers, &mut env),
      Ok(RispExpr::Float(n)) if n == (3_f64 - 2_f64)
    ));
    assert!(matches!(
      interpret(many_numbers, &mut env),
      Ok(RispExpr::Float(n)) if n == (5.4 - 4.3 - 3.2 - 2.1 - 1.0)
    ));

    assert!(matches!(
      interpret("(-)", &mut env),
      Err(RispError::ExpectedNumber)
    ));
  }

  #[test]
  fn test_multiplication() {
    let mut env = RispEnv::new_default_env();

    let two_numbers = "(* 3 2)";
    let many_numbers = "(* 5.4 4.3 3.2 2.1 1.0)";

    assert!(matches!(
      interpret(two_numbers, &mut env),
      Ok(RispExpr::Float(n)) if n == (3_f64 * 2_f64)
    ));
    assert!(matches!(
      interpret(many_numbers, &mut env),
      Ok(RispExpr::Float(n)) if n == (5.4 * 4.3 * 3.2 * 2.1 * 1.0)
    ));

    assert!(matches!(
      interpret("(*)", &mut env),
      Err(RispError::ExpectedNumber)
    ));
  }

  #[test]
  fn test_division() {
    let mut env = RispEnv::new_default_env();

    let two_numbers = "(/ 3 2)";
    let many_numbers = "(/ 5.4 4.3 3.2 2.1 1.0)";

    assert!(matches!(
      interpret(two_numbers, &mut env),
      Ok(RispExpr::Float(n)) if n == (3_f64 / 2_f64)
    ));
    assert!(matches!(
      interpret(many_numbers, &mut env),
      Ok(RispExpr::Float(n)) if n == (5.4 / 4.3 / 3.2 / 2.1 / 1.0)
    ));

    assert!(matches!(
      interpret("(/)", &mut env),
      Err(RispError::ExpectedNumber)
    ));
  }

  #[test]
  fn test_elementary_arithmetic() {
    let mut env = RispEnv::new_default_env();

    let add_and_subtract = "(- 5.4 4.3 3.2 2.1 1.0 (+ 3 2))";
    let multiply_and_divide = "(/ 5.4 4.3 3.2 2.1 1.0 (* 3 2))";
    let elementary_arithmetic = "(/ 5.4 4.3 3.2 2.1 1.0 (* 3 2) (- 5.4 4.3 3.2 2.1 1.0 (+ 3 2)))";

    assert!(matches!(
      interpret(add_and_subtract, &mut env),
      Ok(RispExpr::Float(n)) if n == (5.4 - 4.3 - 3.2 - 2.1 - 1.0 - (3_f64 + 2_f64))
    ));
    assert!(matches!(
      interpret(multiply_and_divide, &mut env),
      Ok(RispExpr::Float(n)) if n == (5.4 / 4.3 / 3.2 / 2.1 / 1.0 / (3_f64 * 2_f64))
    ));
    assert!(matches!(
      interpret(elementary_arithmetic, &mut env),
      Ok(RispExpr::Float(n))
        if (n - (5.4 / 4.3 / 3.2 / 2.1 / 1.0 / (3_f64 * 2_f64) / (5.4 - 4.3 - 3.2 - 2.1 - 1.0 
        - (3_f64 + 2_f64)))).abs() < 1e-9));
  }

  #[test]
  fn test_modulo() {
    let mut env = RispEnv::new_default_env();

    let two_numbers = "(mod 3 2)";
    let two_numbers_with_negative = "(mod -3 2)";
    let many_numbers = "(mod 3 3 3)";

    assert!(
      matches!(interpret(two_numbers, &mut env), Ok(RispExpr::Float(n)) if n == (3_f64 % 2_f64))
    );
    assert!(matches!(interpret(two_numbers_with_negative, &mut env),
      Ok(RispExpr::Float(n)) if n == (((-3_f64 % 2_f64) + 2_f64) % 2_f64)));
    assert!(matches!(interpret(many_numbers, &mut env),
      Err(RispError::ExpectedTwoForms(s)) if s == "mod"));

    assert!(matches!(
      interpret("(mod)", &mut env),
      Err(RispError::ExpectedNumber)
    ));
  }

  #[test]
  fn test_remainder() {
    let mut env = RispEnv::new_default_env();

    let two_numbers = "(rem 3 2)";
    let two_numbers_with_negative = "(rem -3 2)";
    let many_numbers = "(rem 3 3 3)";

    assert!(
      matches!(interpret(two_numbers, &mut env), Ok(RispExpr::Float(n)) if n == (3_f64 % 2_f64))
    );
    assert!(matches!(interpret(two_numbers_with_negative, &mut env),
      Ok(RispExpr::Float(n)) if n == (-3_f64 % 2_f64)));
    assert!(matches!(interpret(many_numbers, &mut env),
      Err(RispError::ExpectedTwoForms(s)) if s == "rem"));

    assert!(matches!(
      interpret("(rem)", &mut env),
      Err(RispError::ExpectedNumber)
    ));
  }

  #[test]
  fn test_max() {
    let mut env = RispEnv::new_default_env();

    let two_numbers = "(max 3.3 2.2)";
    let many_numbers = "(max (* 3.3 3.3) (+ 2.2 2.2) (- 1.1 1.1))";

    assert!(matches!(
      interpret(two_numbers, &mut env),
      Ok(RispExpr::Float(n)) if n == 3.3
    ));
    assert!(matches!(
      interpret(many_numbers, &mut env),
      Ok(RispExpr::Float(n)) if n == (3.3 * 3.3)
    ));

    assert!(matches!(
      interpret("(max)", &mut env),
      Err(RispError::ExpectedNumber)
    ));
  }

  #[test]
  fn test_min() {
    let mut env = RispEnv::new_default_env();

    let two_numbers = "(min 3.3 2.2)";
    let many_numbers = "(min (* 3.3 3.3) (+ 2.2 2.2) (- 1.1 1.1))";

    assert!(matches!(
      interpret(two_numbers, &mut env),
      Ok(RispExpr::Float(n)) if n == 2.2
    ));
    assert!(matches!(
      interpret(many_numbers, &mut env),
      Ok(RispExpr::Float(n)) if n == (1.1 - 1.1)
    ));

    assert!(matches!(
      interpret("(min)", &mut env),
      Err(RispError::ExpectedNumber)
    ));
  }

  #[test]
  fn test_equal_condition() {
    let mut env = RispEnv::new_default_env();

    let two_equal_numbers = "(= 3 3)";
    let many_equal_numbers = "(= (+ 3 3) (+ 3 3) (+ 3 3))";
    let two_not_equal_numbers = "(= 3.3 1.1)";
    let many_not_equal_numbers = "(= (+ 3.3 3.3) (+ 3.3 3.3) (+ 1.1 1.1))";

    assert!(matches!(
      interpret(two_equal_numbers, &mut env),
      Ok(RispExpr::Boolean(true))
    ));
    assert!(matches!(
      interpret(many_equal_numbers, &mut env),
      Ok(RispExpr::Boolean(true))
    ));
    assert!(matches!(
      interpret(two_not_equal_numbers, &mut env),
      Ok(RispExpr::Boolean(false))
    ));
    assert!(matches!(
      interpret(many_not_equal_numbers, &mut env),
      Ok(RispExpr::Boolean(false))
    ));

    assert!(matches!(
      interpret("(=)", &mut env),
      Err(RispError::ExpectedNumber)
    ));
  }

  #[test]
  fn test_not_equal_condition() {
    let mut env = RispEnv::new_default_env();

    let two_not_equal_numbers = "(!= 3.3 1.1)";
    let many_not_equal_numbers = "(!= (+ 3.3 3.3) (+ 2.2 2.2) (+ 1.1 1.1))";
    let two_equal_numbers = "(!= 3 3)";
    let many_equal_numbers = "(!= (+ 3 3) (+ 3 3) (+ 3 3))";

    assert!(matches!(
      interpret(two_not_equal_numbers, &mut env),
      Ok(RispExpr::Boolean(true))
    ));
    assert!(matches!(
      interpret(many_not_equal_numbers, &mut env),
      Ok(RispExpr::Boolean(true))
    ));
    assert!(matches!(
      interpret(two_equal_numbers, &mut env),
      Ok(RispExpr::Boolean(false))
    ));
    assert!(matches!(
      interpret(many_equal_numbers, &mut env),
      Ok(RispExpr::Boolean(false))
    ));

    assert!(matches!(
      interpret("(!=)", &mut env),
      Err(RispError::ExpectedNumber)
    ));
  }

  #[test]
  fn test_greater_condition() {
    let mut env = RispEnv::new_default_env();

    let two_sorted_numbers = "(> 3 2)";
    let many_sorted_numbers = "(> (+ 3 3) (+ 2 2) (+ 1 1))";
    let two_unsorted_numbers = "(> 3 4)";
    let many_unsorted_numbers = "(> (+ 3 3) (+ 3 3) (+ 4 4))";

    assert!(matches!(
      interpret(two_sorted_numbers, &mut env),
      Ok(RispExpr::Boolean(true))
    ));
    assert!(matches!(
      interpret(many_sorted_numbers, &mut env),
      Ok(RispExpr::Boolean(true))
    ));
    assert!(matches!(
      interpret(two_unsorted_numbers, &mut env),
      Ok(RispExpr::Boolean(false))
    ));
    assert!(matches!(
      interpret(many_unsorted_numbers, &mut env),
      Ok(RispExpr::Boolean(false))
    ));

    assert!(matches!(
      interpret("(>)", &mut env),
      Err(RispError::ExpectedNumber)
    ));
  }

  #[test]
  fn test_smaller_condition() {
    let mut env = RispEnv::new_default_env();

    let two_sorted_numbers = "(< 2 3)";
    let many_sorted_numbers = "(< (+ 1 1) (+ 2 2) (+ 3 3))";
    let two_unsorted_numbers = "(< 4 3)";
    let many_unsorted_numbers = "(< (+ 4 4) (+ 3 3) (+ 3 3))";

    assert!(matches!(
      interpret(two_sorted_numbers, &mut env),
      Ok(RispExpr::Boolean(true))
    ));
    assert!(matches!(
      interpret(many_sorted_numbers, &mut env),
      Ok(RispExpr::Boolean(true))
    ));
    assert!(matches!(
      interpret(two_unsorted_numbers, &mut env),
      Ok(RispExpr::Boolean(false))
    ));
    assert!(matches!(
      interpret(many_unsorted_numbers, &mut env),
      Ok(RispExpr::Boolean(false))
    ));

    assert!(matches!(
      interpret("(<)", &mut env),
      Err(RispError::ExpectedNumber)
    ));
  }

  #[test]
  fn test_greater_or_equal_condition() {
    let mut env = RispEnv::new_default_env();

    let two_sorted_numbers = "(>= 3 3)";
    let many_sorted_numbers = "(>= (+ 3 3) (+ 2 2) (+ 2 2))";
    let two_unsorted_numbers = "(>= 3 4)";
    let many_unsorted_numbers = "(>= (+ 3 3) (+ 3 3) (+ 4 4))";

    assert!(matches!(
      interpret(two_sorted_numbers, &mut env),
      Ok(RispExpr::Boolean(true))
    ));
    assert!(matches!(
      interpret(many_sorted_numbers, &mut env),
      Ok(RispExpr::Boolean(true))
    ));
    assert!(matches!(
      interpret(two_unsorted_numbers, &mut env),
      Ok(RispExpr::Boolean(false))
    ));
    assert!(matches!(
      interpret(many_unsorted_numbers, &mut env),
      Ok(RispExpr::Boolean(false))
    ));

    assert!(matches!(
      interpret("(>=)", &mut env),
      Err(RispError::ExpectedNumber)
    ));
  }

  #[test]
  fn test_smaller_or_equal_condition() {
    let mut env = RispEnv::new_default_env();

    let two_sorted_numbers = "(<= 3 3)";
    let many_sorted_numbers = "(<= (+ 2 2) (+ 2 2) (+ 3 3))";
    let two_unsorted_numbers = "(<= 4 3)";
    let many_unsorted_numbers = "(<= (+ 4 4) (+ 3 3) (+ 3 3))";

    assert!(matches!(
      interpret(two_sorted_numbers, &mut env),
      Ok(RispExpr::Boolean(true))
    ));
    assert!(matches!(
      interpret(many_sorted_numbers, &mut env),
      Ok(RispExpr::Boolean(true))
    ));
    assert!(matches!(
      interpret(two_unsorted_numbers, &mut env),
      Ok(RispExpr::Boolean(false))
    ));
    assert!(matches!(
      interpret(many_unsorted_numbers, &mut env),
      Ok(RispExpr::Boolean(false))
    ));

    assert!(matches!(
      interpret("(<=)", &mut env),
      Err(RispError::ExpectedNumber)
    ));
  }

  #[test]
  fn test_defvar() {
    let mut env = RispEnv::new_default_env();

    let defvar_with_valid_symb = "(defvar a 0)";
    let defvar_with_invalid_symb = "(defvar 0 0)";
    let defvar_with_reserved_keyword = "(defvar defvar 0)";
    let defvar_with_number = "(defvar b (* 0 0))";
    let defvar_with_boolean_expression = "(defvar c (= 0 0))";
    let defvar_with_boolean_value = "(defvar d true)";
    let defvar_without_value = "(defvar e)";

    assert!(matches!(interpret(defvar_with_valid_symb, &mut env),
      Ok(RispExpr::Symbol(s)) if s == "a"));
    assert!(matches!(
      interpret(defvar_with_invalid_symb, &mut env),
      Err(RispError::InvalidSymbol(symb, func)) if symb == "0" && func == "defvar"
    ));
    assert!(matches!(
      interpret(defvar_with_reserved_keyword, &mut env),
      Err(RispError::InvalidSymbol(symb, func)) if symb == "defvar" && func == "defvar"
    ));
    assert!(
      matches!(interpret(defvar_with_number, &mut env),
      Ok(RispExpr::Symbol(s)) if s == "b")
        && matches!(env.var.get("b"), Some(RispExpr::Float(n)) if *n == (0_f64 * 0_f64))
    );
    assert!(
      matches!(interpret(defvar_with_boolean_expression, &mut env),
      Ok(RispExpr::Symbol(s)) if s == "c")
        && matches!(env.var.get("c"), Some(RispExpr::Boolean(b)) if *b)
    );
    assert!(
      matches!(interpret(defvar_with_boolean_value, &mut env),
      Ok(RispExpr::Symbol(s)) if s == "d")
        && matches!(env.var.get("d"), Some(RispExpr::Boolean(b)) if *b)
    );
    assert!(
      matches!(interpret(defvar_without_value, &mut env),
      Ok(RispExpr::Symbol(s)) if s == "e")
        && matches!(env.var.get("e"), Some(RispExpr::Float(n)) if *n == 0_f64)
    );
    assert!(matches!(interpret(defvar_without_value, &mut env),
      Err(RispError::VariableAlreadyDefined(s)) if s == "e"));

    assert!(matches!(
      interpret("(defvar)", &mut env),
      Err(RispError::ExpectedSymbol(func)) if func == "defvar"
    ));
  }

  #[test]
  fn test_setq() {
    let mut env = RispEnv::new_default_env();
    env.var.insert("a".to_string(), RispExpr::Float(3_f64));
    env.var.insert("b".to_string(), RispExpr::Boolean(false));

    let setq_with_number = "(setq a 0)";
    let setq_with_boolean = "(setq b true)";

    assert!(
      matches!(interpret(setq_with_number, &mut env),
      Ok(RispExpr::Symbol(s)) if s == "a")
        && matches!(env.var.get("a"), Some(RispExpr::Float(n)) if *n == 0_f64)
    );
    assert!(
      matches!(interpret(setq_with_boolean, &mut env),
      Ok(RispExpr::Symbol(s)) if s == "b")
        && matches!(env.var.get("b"), Some(RispExpr::Boolean(b)) if *b)
    );

    assert!(matches!(
      interpret("(setq)", &mut env),
      Err(RispError::ExpectedSymbol(func)) if func == "setq"
    ));
  }

  #[test]
  fn test_if() {
    let mut env = RispEnv::new_default_env();

    let true_condition = "(if (< 1 2) true false)";
    let false_condition = "(if (> 1 2) true false)";

    assert!(matches!(interpret(true_condition, &mut env), Ok(RispExpr::Boolean(b)) if b));
    assert!(matches!(interpret(false_condition, &mut env), Ok(RispExpr::Boolean(b)) if !b));

    assert!(matches!(
      interpret("(if)", &mut env),
      Err(RispError::ExpectedIfCondition)
    ));
  }
}
