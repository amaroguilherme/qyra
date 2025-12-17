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

    fn calculate_energy(&self, solution: &Vec<u8>) -> f64 {
        let mut energy = 0.0;
        for i in 0..self.num_vars {
            for j in 0..self.num_vars {
                if solution[i] == 1 && solution[j] == 1 {
                    energy += self.matrix[i][j];
                }
            }
        }
        energy
    }
}

// --- Hybrid Simulated Annealing ---
fn run_optimizer(qubo: &Problem, max_iterations: usize, priors: &Vec<f64>) -> (Vec<u8>, f64) {
    eprintln!("Optimizing...");
    let mut rng = rand::thread_rng();

    // Warm Start using model's "priors"
    let mut current_solution: Vec<u8> = priors.iter().map(|&p| {
        if rng.r#gen::<f64>() < p { 1 } else { 0 }
    }).collect();

    let mut current_energy = qubo.calculate_energy(&current_solution);
    let mut best_solution = current_solution.clone();
    let mut best_energy = current_energy;

    let mut temperature = 10.0;
    let cooling_rate = 0.99;

    for _ in 0..max_iterations {
        let var_to_flip = rng.gen_range(0..qubo.num_vars);
        
        let mut new_solution = current_solution.clone();
        new_solution[var_to_flip] = 1 - new_solution[var_to_flip];
        
        let new_energy = qubo.calculate_energy(&new_solution);
        let energy_diff = new_energy - current_energy;

        let accept = if energy_diff < 0.0 {
            true
        } else {
            let probability = (-energy_diff / temperature).exp();
            rng.r#gen::<f64>() < probability
        };

        if accept {
            current_solution = new_solution;
            current_energy = new_energy;
            if current_energy < best_energy {
                best_energy = current_energy;
                best_solution = current_solution.clone();
            }
        }
        temperature *= cooling_rate;
    }
    eprintln!("Optimization completed.");
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
fn core_qyra_solver(m: &Bound<'_, PyModule>) -> PyResult<()> {
    eprintln!("Echo received by the solver.");
    m.add_function(wrap_pyfunction!(solve, m)?)?;
    Ok(())
}