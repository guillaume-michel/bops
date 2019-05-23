(in-package :bops)

(defgeneric run-inference (operator inputs outputs)
  (:documentation "perform forward inference for the given operator & inputs and returns its results in outputs"))
