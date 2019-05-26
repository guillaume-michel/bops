(in-package :bops)

(defgeneric operator-output-shape (operator input-shape)
  (:documentation "returns the operator's output tensor shape for the given input"))

(defgeneric make-operator-output (operator input-shape)
  (:documentation "ask an operator to construct its output tensor"))

(defgeneric run-inference (operator inputs outputs)
  (:documentation "perform forward inference for the given operator & inputs and returns its results in outputs"))

(defgeneric operator-parameters (operator)
  (:documentation "return the parameters for the given operator. If multiple types of parameters it is returned as a list.")
  (:method (operator)))

;;; API for genetic algorithm

(defgeneric mutate (operator strategy)
  (:documentation "Mutate the given operator with eht egicven mutation strategy")
  (:method (operator strategy)
    ;; by default no mutation
    operator))
