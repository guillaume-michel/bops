(in-package #:bops)

(declaim (inline flatten))
(defun flatten (arr)
  (make-array (array-total-size arr)
              :element-type (array-element-type arr)
              :displaced-to arr))

(defun make-random-bit-vector (dims)
  "Creates a new bit array of the given dimensions.
Elements are initialized randomly."
  (let* ((data (make-array dims :element-type 'bit))
         (displaced-data (flatten data)))
    (map-into displaced-data
              (lambda (b) (declare (ignore b)) (random 2))
              displaced-data)
    data))
