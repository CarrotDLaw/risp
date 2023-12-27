use std::io::{self, Write};

use risp::*;

fn main() {
  let mut global_env = RispEnv::new_default_env();

  loop {
    print!("risp > ");
    io::stdout().flush().ok();

    let mut input = String::new();
    io::stdin()
      .read_line(&mut input)
      .map_err(|_| eprintln!("{}", RispError::StdinError))
      .ok();

    match interpret(input.trim(), &mut global_env) {
      Ok(result) => println!("{result}"),
      Err(e) => eprintln!("error: {e}"),
    }
  }
}
