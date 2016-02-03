# RustModels.jl

[WORK IN PROGRESS]

This package provides tools to work with the dynamic discrete choice models of Economics. Dynamic discrete choice models are closely related to Markov decision processes. They describe how an individual makes repeated decisions as a state variable evolves and his decisions impact the evolution of the state. Some fundamental parameters describe the individual's preferences but are unknown to the analyst. She tries to infer them from observing states and decisions.

This package enforces a set of assumptions that provide a good trade-off between model flexibility and tractability of the analysis. Models following these assumptions are often referred to as **Rust models**, after a beautiful article by John Rust [John Rust 1987 Econometrica: Optimal replacement of GMC bus engines: An empirical model of Harold Zurcher]. This package covers two types of models:

- **Classical Rust models**: States and decisions are fully observed. Probabilistically speaking, the data follows a Markov chain whose transition matrix is constrained by an economic model of decision-making. As used in [Rust 1987] and elsewhere.

- **Hidden Rust models**: The state has an observed component and an unobserved component. Probabilistically speaking, the data follows something similar to a Hidden Markov model, with transition matrices constrained by the same economic model of decision-making. Hidden Rust models allow for much richer dynamics (such as serially correlated unobserved state variables) without loosing too much of the tractability of classical Rust models.

I (Ben Connault, package author, hi!) studied hidden Rust models in my thesis, feel free to check it out on my [webpage](http://economics.sas.upenn.edu/~connault/).


## Installation

~~~julia
julia> Pkg.clone("git://github.com/BenConnault/RustModels.jl.git")
~~~

## Overview

Rust models have two layers:

- An **economic layer**: going back and forth between the deep parameterization in terms of the individual's preferences and the dynamics (the transition matrices for the random variables). Known as "solving the dynamic program" (deep parameters -> transition matrices) and "2-step projecting" or "minimum distance projecting" (transition matrices -> deep parameters). This part is common for both classical and hidden Rust models and handled by [RustModels.jl](https://github.com/BenConnault/RustModels.jl).

- A **statistical layer**: going back and forth between the dynamics (the transition matrices) and the data. This part is different for classical and hidden Rust models. Hidden Rust models have hidden-Markov-type dynamics which are handled externally by [DynamicDiscreteModels.jl](https://github.com/BenConnault/DynamicDiscreteModels.jl). Classical Rust models have much simpler plain Markov dynamics which are handled internally.
