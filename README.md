# wec_design_optimization
Open source ocean wave energy converter design automation and optimization, using Capytaine for hydrodynamics.

Steps to achieve design optimization success:
1. Create parameterized model in Capytaine
2. Implement fitness measure to judge each design
3. Package the analysis into one function that takes design parameters as an input and outputs the fitness measure
4. Add a relevant optimization algorithm based on your design problem
5. Run algorithm until converged

Examples here include:
- Optimizing a submerged flexible tube that generates power by damping structural deformations
- Analyzing the structural requirements of a bottom mounted oscillating surge wave energy converter
