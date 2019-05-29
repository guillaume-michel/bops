(in-package :bops)

(defclass tournament-selection ()
  ((tournament-size :initarg :tournament-size
                    :type (integer 1 most-positive-fixnum)
                    :initform (error "You should provide tournament-size")
                    :accessor tournament-selection-size
                    :documentation "tournament selection size"))
  (:documentation "Select the best individual among tournament-size randomly chosen individuals."))

(defmethod print-object ((object tournament-selection) stream)
  (print-unreadable-object (object stream :type t :identity t)
    (with-slots (tournament-size) object
      (format stream ":tournament-size ~D" tournament-size))))

(defun select-random-index (range count)
  "create a list of count integer (indexes) in the specified range [0; range)"
  (iter (repeat count)
        (collect (random range))))

(defun minimum (list &key (accessor #'identity))
 "recursive function to return the minimum value of a list of numbers"
 (cond ((null list) nil)
       ((null (rest list)) (first list))
       ((< (funcall accessor (first list))
           (funcall accessor (second list)))
        (minimum (cons (first list)
                       (rest (rest list))) :accessor accessor))
       (t (minimum (rest list) :accessor accessor))))

(defmethod select ((population list)
                   count
                   (fitnesses list)
                   (selection-strategy tournament-selection))
  (assert (= (length population)
             (length fitnesses)))

  (assert (> count 0))

  (assert (<= count (length population)))

  (assert (<= (tournament-selection-size selection-strategy)
              (length population)))
  (iter (repeat count)
        (collect (minimum (mapcar (lambda (i)
                                    (cons (nth i fitnesses)
                                          (nth i population)))
                                  (select-random-index (length population)
                                                       (tournament-selection-size selection-strategy)))
                          :accessor #'first))))
