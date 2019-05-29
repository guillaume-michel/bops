(in-package :bops)

(defgeneric select (population count fitnesses selection-strategy)
  (:documentation "Select count individuals in the population according to selection strategy.
returns a list of count pairs (fitness individual)"))
