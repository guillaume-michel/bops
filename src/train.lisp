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
    (let ((mlp (make-mlp dims :B B)))
      (let ((train-predictions (make-operator-output mlp (array-dimensions (get-x-batch x-train batch-size 0))))
            (train-predicted-labels nil))
        (iter (for i below (/ (array-dimension x-train 0) batch-size))
              (run-inference mlp (get-x-batch x-train batch-size i) train-predictions)
              (setf train-predicted-labels (argmax train-predictions))
              (format t "TRAIN for batch: ~D Accuracy=~A - Loss=~A~%"
                      i
                      (accuracy train-predicted-labels (get-y-batch y-train batch-size i))
                      (loss train-predictions (get-y-batch y-train batch-size i)))))
      (let ((test-predictions (make-operator-output mlp (array-dimensions x-test)))
            (test-predicted-labels nil))
        (run-inference mlp x-test test-predictions)
        (setf test-predicted-labels (argmax test-predictions))
        (format t "TEST Accuracy=~A - Loss=~A~%"
                (accuracy test-predicted-labels y-test)
                (loss test-predictions y-test))))))

#|

(defparameter datas (prepare-mnist (load-mnist)))

(mapcar #'type-of datas)

(destructuring-bind (x-train y-train x-test y-test) datas
    (print (array-dimensions x-train))
    (print (array-dimensions y-train))
    (print (array-dimensions x-test))
    (print (array-dimensions y-test)))

(defparameter mlp (make-mlp '(1024 256 10) :B 8))

(defparameter input (car datas))
(defparameter output (make-operator-output mlp (array-dimensions input)))

(time (run-inference mlp input output))

(time (train datas :batch-size 32))
|#
