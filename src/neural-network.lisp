(in-package :bops)

(defgeneric operator-output-shape (operator input-shape)
  (:documentation "returns the operator's output tensor shape for the given input"))

(defgeneric make-operator-output (operator input-shape)
  (:documentation "ask an operator to construct its output tensor"))

(defgeneric run-inference (operator inputs outputs)
  (:documentation "perform forward inference for the given operator & inputs and returns its results in outputs"))
