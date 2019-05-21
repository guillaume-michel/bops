(in-package :bops)

#|

(destructuring-bind (x-train y-train x-test y-test) (prepare-mnist (load-mnist))
    (print (array-dimensions x-train))
    (print (array-dimensions y-train))
    (print (array-dimensions x-test))
    (print (array-dimensions y-test)))

(defparameter datas (prepare-mnist (load-mnist)))
(defparameter arr-x (first datas))
(defparameter arr-w (make-random-bit-vector '(8 10 1024)))
(defparameter arr-b (make-random-bias-vector '(8 10) 125))
(defparameter arr-y (make-array '(60000 8 10) :element-type 'bit))

(dense-v1 arr-y arr-w arr-x arr-b)

(defparameter res (fuse-bitplane-uint8 (aops:permute '(0 2 1) arr-y)))


(defparameter train-labels (second datas))

|#
