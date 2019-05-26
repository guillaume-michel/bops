(in-package :bops)

(defclass sequential ()
  ((operators :initarg :operators
              :type list
              :initform '()
              :reader sequential-operators
              :documentation "list of operators in order of execution")
   (scratchs :type list
             :initform '()
             :reader sequential-scratchs
             :documentation "list of scratch tensors for intermediate values"))
  (:documentation "Linear sequence of operators lined to each other"))

(defmethod print-object ((object sequential) stream)
  (print-unreadable-object (object stream :type t :identity t)
    (with-slots (operators) object
      (format stream ":operators ~A" operators))))

(defmethod operator-output-shape ((operator sequential) input-shape)
    (with-slots (operators) operator
      (let ((shapes (run-shape-inference operators input-shape)))
        (cadr (car (last shapes))))))

(defun run-shape-inference (operators input-shape)
  (when (null operators)
    (error "Trying to run-shape-inference with empty sequential operator"))

  (let ((current-input-shape input-shape))
    (iter (for operator in operators)
          (let ((output-shape (operator-output-shape operator current-input-shape)))
            (collect (cons current-input-shape output-shape) into shapes)
            (setf current-input-shape output-shape)
            (finally (return shapes))))))

(defmethod make-operator-output ((operator sequential) input-shape)
    (with-slots (operators) operator
      (if (null operators)
          (error "Trying to call make-operator-output for empty sequential operator")
          (let* ((shapes (run-shape-inference operators input-shape))
                 (last-operator (car (last operators)))
                 (last-operator-shapes (car (last shapes))))
            (make-operator-output last-operator
                                  (car last-operator-shapes))))))

(defun make-sequential-scratchs (operators input-shape)
  (when (null operators)
      (error "Trying to call make-sequential-scratchs with empty sequential operator"))

  (let ((shapes (run-shape-inference operators input-shape)))
    (iter (for i below (- (length operators) 1))
          (collect (make-operator-output (nth i operators)
                                         (car (nth i shapes)))))))

(defun reshape-sequential-scratchs (operator input-shape)
  (with-slots (operators scratchs) operator
    (cond ((<= (length operators) 1) (setf scratchs '()))
          (t (if (or (null scratchs)
                     (not (= (array-dimension (car scratchs) 0)
                             (car input-shape))))
                 (setf scratchs (make-sequential-scratchs operators input-shape)))))))

(defun run-sequential-inference (operators scratchs input)
  (when (null operators)
    (error "Trying to run-inference for empty sequential operator"))

  (run-inference (car operators)
                 input
                 (car scratchs))

  (when (not (null (cdr operators)))
    ;; there are a following operators
    (run-sequential-inference (cdr operators) (cdr scratchs) (car scratchs))))

(defmethod run-inference ((operator sequential) (inputs list) (outputs list))
  (assert (and (not (null inputs))
               (not (null outputs))))
  (let ((input (car inputs))
        (output (car outputs)))
    (run-inference operator input output)))

(defmethod run-inference ((operator sequential) input output)
  (reshape-sequential-scratchs operator (array-dimensions input))

  (with-slots (operators scratchs) operator
    (if (null operators)
        (error "Trying to run-inference for empty sequential operator")
        (run-sequential-inference operators (append scratchs (list output)) input))))

(defmethod operator-parameters ((operator sequential))
  (with-slots (operators scratchs) operator
    (alexandria:flatten (mapcar #'operator-parameters operators))))

(defmethod mutate ((operator sequential) strategy)
  (with-slots (operators scratchs) operator
    (let ((new-operators (mapcar (lambda (o)
                                   (mutate o strategy))
                                 operators)))
      (make-instance 'sequential
                     :operators new-operators))))
