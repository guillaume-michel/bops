(in-package :bops)

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

(defun make-population (dims B P)
    "create population of P individuals (each is an mlp)"
  (iter (repeat P)
        (collect (make-mlp dims :B B))))

(defun group-pairwise (list)
  (assert (= (mod (length list) 2) 0))
  (iter (for i below (length list) by 2)
        (collect (list (nth i list)
                       (nth (+ i 1) list)))))

(defun minimum-k (k list &key (accessor #'identity))
  "return the k minimum elements in list"
  (subseq (sort list (lambda (x y)
                       (< (funcall accessor x)
                          (funcall accessor y))))
          0 k))

(defun keep-best-individuals (k parents children)
  (minimum-k k (append parents children) :accessor #'first))

(defun train (datas &key
                      (dims '(1024 128 10))
                      (batch-size 600)
                      (B 8)
                      (generation 10)
                      (pcross 0.9)
                      (pmut 0.3)
                      (tournament-size 4)
                      (population-size 100)
                      (num-threads 8))
  (destructuring-bind (x-train y-train x-test y-test) datas
    (let* ((lparallel:*kernel* (lparallel:make-kernel num-threads))
           (population (make-population dims B population-size))
           (fitnesses nil)
           (mut-strategy (make-instance 'uniform-mutation :prob pmut))
           (mate-strategy (make-instance 'uniform-crossover :prob pcross))
           (selection-strategy (make-instance 'tournament-selection :tournament-size tournament-size))
           (train-predictions-outputs (mapcar (lambda (mlp)
                                                (make-operator-output mlp
                                                                      (array-dimensions (get-x-batch x-train
                                                                                                     batch-size
                                                                                                     0))))
                                              population))
           (new-train-predictions-outputs (mapcar (lambda (mlp)
                                                    (make-operator-output mlp
                                                                          (array-dimensions (get-x-batch x-train
                                                                                                         batch-size
                                                                                                         0))))
                                                  population)))

      (lparallel:pmapcar (lambda (mlp train-predictions)
                (run-inference mlp (get-x-batch x-train batch-size 0) train-predictions))
              population train-predictions-outputs)

      (setf fitnesses (mapcar (lambda (train-predictions)
                                (loss train-predictions (get-y-batch y-train batch-size 0)))
                              train-predictions-outputs))

      (iter (for gen below generation)
            (format t "============= Generation: ~d ============~%" gen)
            (iter (for i below (/ (array-dimension x-train 0) batch-size))
                  (let ((candidates (mapcar (lambda (ind)
                                              (mutate ind mut-strategy))
                                            (alexandria:flatten (mapcar (lambda (pair)
                                                                          (crossover (first pair) (second pair) mate-strategy))
                                                                        (group-pairwise (mapcar #'cdr
                                                                                                (select population
                                                                                                        (length population)
                                                                                                        fitnesses
                                                                                                        selection-strategy))))))))
                    (lparallel:pmapcar (lambda (mlp train-predictions)
                              (run-inference mlp (get-x-batch x-train batch-size i) train-predictions))
                            population train-predictions-outputs)

                    (lparallel:pmapcar (lambda (mlp train-predictions)
                              (run-inference mlp (get-x-batch x-train batch-size i) train-predictions))
                            candidates new-train-predictions-outputs)

                    (let* ((train-losses (mapcar (lambda (train-predictions)
                                                   (loss train-predictions (get-y-batch y-train batch-size i)))
                                                 train-predictions-outputs))
                           (new-train-losses (mapcar (lambda (train-predictions)
                                                       (loss train-predictions (get-y-batch y-train batch-size i)))
                                                     new-train-predictions-outputs))
                           (parents (mapcar #'list train-losses population))
                           (children (mapcar #'list new-train-losses candidates))
                           (best-individuals (keep-best-individuals (length parents) parents children)))

                      (setf population (mapcar #'second best-individuals))
                      (setf fitnesses (mapcar #'first best-individuals))

                      (format t "TRAIN for batch: ~D Loss=~A~%"
                              i
                              (caar best-individuals)))))

            (let ((test-predictions (make-operator-output (car population) (array-dimensions x-test)))
                    (test-predicted-labels nil))
                (run-inference (car population) x-test test-predictions)

                (setf test-predicted-labels (argmax test-predictions))
                (let ((test-accuracy (accuracy test-predicted-labels y-test))
                      (test-loss (loss test-predictions y-test)))
                  (format t "TEST Accuracy=~A - Loss=~A~%"
                          test-accuracy
                          test-loss))))
      population)))

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
