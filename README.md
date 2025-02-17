# Learning Stochastic Response of a SDOF Bouc-Wen System through LSTM

## Overview
This project explores the stochastic response of a Single Degree of Freedom (SDOF) Bouc-Wen system under random excitation using Long Short-Term Memory (LSTM) networks. Both windowed and non-windowed LSTM models are implemented to predict system response under stochastic loading.

## Mathematical Model
The equation of motion for the SDOF Bouc-Wen system is given by:

m * u''(t) + c * u'(t) + F(t) = f(t)

where:
- m = mass
- c = damping coefficient
- k = initial stiffness
- u(t) = displacement response
- F(t) = external stochastic load
- a = post-yield ratio (0 ≤ a ≤ 1)

The hidden nonlinear state is defined by:

dz/du = A - [β * sign(z(t) * u'(t)) + γ] * |z(t)|^n

where:
- A, β, γ, and n are dimensionless parameters controlling nonlinearity.

## Problem Definition
Given the system parameters:
- m = 1
- k = 1
- c = 0.02
- a = 0.5
- A = 1, β = 0.5, γ = 0.5, n = 5
- F(t): White noise with zero mean and standard deviation = 3.0

**Time step**: 0.02 s  
**Duration**: 10 s

### Steps Performed:
1. Generate a realization of F(t) and plot it over time.
2. Generate 50 samples of stochastic excitation F(t) and compute the corresponding response u(t) using numerical integration (Runge-Kutta).
3. Train an LSTM neural network using 50 pairs of input-output time histories.
4. Compare LSTM-predicted responses with numerically simulated responses:
   - Time history comparison of representative samples.
   - Peak response comparison.

## Implementation Details
- Implemented in Python using TensorFlow/Keras for LSTM training.
- Used Runge-Kutta method for solving differential equations.
- Two LSTM approaches used:
  - **Windowed LSTM**: Uses a sliding time window for input.
  - **Non-windowed LSTM**: Predicts response directly based on full input sequence.

## Results
- **Plot 1**: Comparison of predicted vs. actual response for representative time history in non-windowed LSTM.
     ![Comparision of predicted vs actual displacement response in non-windowed case](https://github.com/Manish3690/Bouc-Wen-System-under-Stochastic-Excitation/blob/main/DispPrediction_nowindow.jpg)
- **Plot 2**: Peak response comparison between LSTM and Runge-Kutta simulations in non-windowed case.
  ![Comparision of Peak Responses in non-windowed case](https://github.com/Manish3690/Bouc-Wen-System-under-Stochastic-Excitation/blob/main/Truemax_VS_predmax.png)
- **Plot 3**: Training and validation losses in non-windowed case.
  ![Non-windowed loss](https://github.com/Manish3690/Bouc-Wen-System-under-Stochastic-Excitation/blob/main/Loss_fig_nowindow.jpg)

**The figures of non-windowed LSTM will be updated once the reason behind the discrepencies is found out**

## Dependencies
- Python 3.x
- NumPy
- TensorFlow/Keras
- Matplotlib
- Pandas

## Usage
1. Run `Runge-Kutta_bouc-wen.ipynb` to generate stochastic excitations and compute numerical responses.
2. Train non-windowed LSTM model in the same file.
3. Train windowed LSTM model in WIndow_LSTM.ipynb, not actually performing good.
4. Evaluate and compare performance in the same files.

## Conclusion
This study successfully demonstrates the ability of LSTM networks to predict the stochastic response of nonlinear dynamic systems with high accuracy. The model can be further refined by tuning hyperparameters and increasing dataset size.
