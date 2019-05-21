(in-package :bops)

#|

(multiple-value-bind (x-train y-train x-test y-test) (prepare-mnist (load-mnist))
    (print (array-dimensions x-train))
    (print (array-dimensions y-train))
    (print (array-dimensions x-test))
    (print (array-dimensions y-test)))

(defparameter data (idx:read-from-file "/home/gmichel/data/mnist/train-images-idx3-ubyte"))
(defparameter x-train (prepare-mnist-x data '((0 0) (2 2) (2 2)) 0))
(defparameter w (make-random-bit-vector '(8 10 1024)))
(defparameter bias (make-random-bias-vector '(8 10) 125))
(defparameter arr-y (make-array '(60000 8 10) :element-type 'bit))

(dense-v1 arr-y w x-train bias)

(defparameter res (fuse-bitplane-uint8 (aops:permute '(0 2 1) arr-y)))


(defparameter train-labels (idx:read-from-file "/home/gmichel/data/mnist/train-labels-idx1-ubyte"))
|#
