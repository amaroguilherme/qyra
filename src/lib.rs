use pyo3::prelude::*;
use rand::Rng;

struct Problem {
    matrix: Vec<Vec<f64>>,
    num_vars: usize,
}

impl Problem {
    fn new(matrix: Vec<Vec<f64>>) -> Self {
        let num_vars = matrix.len();
        Problem { matrix, num_vars }
    }

    fn calculate_full_energy(&self, solution: &Vec<u8>) -> f64 {
        let mut energy = 0.0;
        for i in 0..self.num_vars {
            if solution[i] == 1 {
                for j in 0..self.num_vars {
                    if solution[j] == 1 {
                        energy += self.matrix[i][j];
                    }
                }
            }
        }
        energy
    }

    fn calculate_delta(&self, solution: &Vec<u8>, var_idx: usize) -> f64 {
        let is_turning_on = solution[var_idx] == 0;
        let self_term = self.matrix[var_idx][var_idx];

        let mut interaction_term = 0.0;
        for i in 0..self.num_vars {
            if i != var_idx && solution[i] == 1 {
                interaction_term += self.matrix[var_idx][i] + self.matrix[i][var_idx];
            }
        }

        if is_turning_on {
            self_term + interaction_term
        } else {
            -(self_term + interaction_term)
        }
    }
}

fn run_optimizer(qubo: &Problem, max_iterations: usize, priors: &Vec<f64>) -> (Vec<u8>, f64) {
    let mut rng = rand::thread_rng();

    let mut current_solution: Vec<u8> = priors.iter().map(|&p| {
        if rng.r#gen::<f64>() < p { 1 } else { 0 }
    }).collect();

    let mut current_energy = qubo.calculate_full_energy(&current_solution);
    
    let mut best_solution = current_solution.clone();
    let mut best_energy = current_energy;

    let mut temperature = 100.0;
    let cooling_rate = 0.9995;

    for _ in 0..max_iterations {
        let var_to_flip = rng.gen_range(0..qubo.num_vars);
        
        let delta = qubo.calculate_delta(&current_solution, var_to_flip);
        let new_energy = current_energy + delta;

        let accept = if delta < 0.0 {
            true
        } else {
            let probability = (-delta / temperature).exp();
            rng.r#gen::<f64>() < probability
        };

        if accept {
            current_solution[var_to_flip] = 1 - current_solution[var_to_flip];
            current_energy = new_energy;

            if current_energy < best_energy {
                best_energy = current_energy;
                best_solution = current_solution.clone();
            }
        }
        temperature *= cooling_rate;
    }

    (best_solution, best_energy)
}

// --- Model interface ---

/// # Arguments
/// - matrix: Q matrix (list of lists of floats)
/// - priors: Probabilities calculated by the model (list of floats)
/// - iterations: Iterations number of the optimizer
#[pyfunction]
fn solve(matrix: Vec<Vec<f64>>, priors: Vec<f64>, iterations: usize) -> PyResult<(Vec<u8>, f64)> {
    let problem = Problem::new(matrix);
    
    if priors.len() != problem.num_vars {
        return Err(pyo3::exceptions::PyValueError::new_err("The size of the priors vector must be equal to the number of variables."));
    }

    let (solution, energy) = run_optimizer(&problem, iterations, &priors);

    Ok((solution, energy))
}

#[pymodule]
fn qyra(m: &Bound<'_, PyModule>) -> PyResult<()> {
    eprintln!("Echo received by the solver.");
    m.add_function(wrap_pyfunction!(solve, m)?)?;
    Ok(())
}