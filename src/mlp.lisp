(in-package :bops)

(defun check-all-multiple-of (m l)
  (reduce (lambda (acc val)
            (and acc
                 (= (mod val m)
                    0)))
          l))

(defun mlp-check-dims (dims)
  (assert (>= (length dims)
              2))
  (check-all-multiple-of 64 (cdr (butlast dims))))

(defun make-mlp-operators (dims B)
  (iter (for i below (- (length dims) 1))
        (collect (make-instance 'binary-fully-connected
                                :input-neurones (nth i dims)
                                :output-neurones (nth (+ i 1) dims)
                                :bitplanes B) into operators)
          (finally (return (append operators (list (make-instance 'softmax :theta 255)))))))

(defun make-mlp (dims &key (B 8))
  (mlp-check-dims dims)
  (make-instance 'sequential :operators (make-mlp-operators dims B)))
