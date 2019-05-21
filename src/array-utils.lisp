(in-package #:bops)

(defun simple-array-vector (array)
  (declare (simple-array array))
  (if (sb-kernel:array-header-p array)
      (sb-kernel:%array-data-vector array)
      array))

(defun make-random-bit-vector (dims)
  "Creates a new bit array of the given dimensions.
Elements are initialized randomly."
  (let* ((data (make-array dims :element-type 'bit))
         (displaced-data (aops:flatten data)))
    (map-into displaced-data
              (lambda (b) (declare (ignore b)) (random 2))
              displaced-data)
    data))

(defun make-random-bias-vector (dims extent)
  (aops:generate* 'fixnum (lambda () (- (random extent))) dims))

(defun bit-vector->integer (bit-vector)
  "Create a positive integer from a bit-vector."
  (reduce #'(lambda (first-bit second-bit)
              (+ (* first-bit 2) second-bit))
          bit-vector))

(defun integer->bit-vector (integer)
  "Create a bit-vector from a positive integer."
  (labels ((integer->bit-list (int &optional accum)
             (cond ((> int 0)
                    (multiple-value-bind (i r) (truncate int 2)
                      (integer->bit-list i (push r accum))))
                   ((null accum) (push 0 accum))
                   (t accum))))
    (coerce (integer->bit-list integer) 'bit-vector)))

(defun split-bitplane (arr)
  (let* ((res (make-array (append (array-dimensions arr) '(8)) :element-type 'bit :initial-element 0))
         (iflat (simple-array-vector arr))
         (oflat (simple-array-vector res)))
    (sb-sys:with-pinned-objects (arr res iflat oflat)
      (sb-kernel:system-area-ub8-copy (sb-sys:vector-sap iflat) 0
                                      (sb-sys:vector-sap oflat) 0
                                      (array-total-size arr)))
    res))

(defun fuse-bitplane-uint8 (arr)
  "arr must have the bitplanes as the last dimension"
  (assert (>= (array-rank arr)
              2))
  (assert (= (the fixnum (car (last (array-dimensions arr))))
             8))
  (assert (typep arr '(simple-array bit *)))

  (let* ((res (make-array (butlast (array-dimensions arr)) :element-type '(unsigned-byte 8)))
         (iflat (simple-array-vector arr))
         (oflat (simple-array-vector res)))

    (sb-sys:with-pinned-objects (arr res iflat oflat)
      (sb-kernel:system-area-ub8-copy (sb-sys:vector-sap iflat) 0
                                      (sb-sys:vector-sap oflat) 0
                                      (array-total-size res)))
    res))

(defun padded-dimensions (dims paddings)
  (mapcar (lambda (dim padding)
            (destructuring-bind (before after) padding
              (+ before dim after)))
          dims
          paddings))

(defun padded-slice (dims paddings)
  (mapcar (lambda (dim padding)
            (destructuring-bind (before after) padding
              (if (= after 0)
                  (cons before nil)
                  (cons before (- dim after)))))
            dims
          paddings))

(defun array-pad (arr &key
                        (paddings (make-list (array-rank arr) :initial-element '(0 0)))
                        (pad-value 0))
  (assert (= (array-rank arr)
             (length paddings)))
  (let* ((padded-dimensions (padded-dimensions (array-dimensions arr) paddings))
         (result (make-array padded-dimensions
                             :element-type (array-element-type arr)
                             :initial-element (coerce pad-value (array-element-type arr)))))
    (setf (apply #'cl-slice:slice result (padded-slice padded-dimensions paddings))
          arr)
    result))
