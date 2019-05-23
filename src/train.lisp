(in-package :bops)

(defun uniform-bit-mutation (w pmut)
  (let ((result (make-random-bit-vector (array-dimensions w) :probability-one pmut)))
    (bit-xor w result result)
    result))

(defun uniform-bit-crossover (w1 w2 pcross)
  (let* ((mask (make-random-bit-vector (array-dimensions w1) :probability-one pcross))
         (bitdiff (bit-and (bit-xor w1 w2)
                           mask)))
    (list
     (bit-xor w1 bitdiff)
     (bit-xor w2 bitdiff))))

(defun get-batch-slice (data batch-size index)
  (let ((N (array-dimension data 0)))
    (cons (* index batch-size)
          (if (>= (* (+ index 1) batch-size)
                  N)
              nil
              (* (+ index 1) batch-size)))))

(defun get-x-batch (data batch-size index)
  (cl-slice:slice data
                  (get-batch-slice data batch-size index)
                  t
                  t))

(defun get-y-batch (data batch-size index)
  (cl-slice:slice data
                  (get-batch-slice data batch-size index)))

(defun train (datas &key
                      (dims '(1024 128 10))
                      (batch-size 32)
                      (B 8))
  (destructuring-bind (x-train y-train x-test y-test) datas
    (let* ((mlp (make-mlp dims batch-size :B B))
           (weights (make-mlp-weights dims :B B))
           (biases (make-mlp-biases dims :B B))
           (train-predictions nil)
           (train-predicted-labels nil))
      (iter (for i below (/ (array-dimension x-train 0) batch-size))
            (setf train-predictions (mlp-run-inference mlp (get-x-batch x-train batch-size i) weights biases))
            (setf train-predicted-labels (argmax train-predictions))
            (format t "TRAIN for batch: ~D Accuracy=~A - Loss=~A~%"
                    i
                    (accuracy train-predicted-labels (get-y-batch y-train batch-size i))
                    (loss train-predictions (get-y-batch y-train batch-size i)))))))

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
