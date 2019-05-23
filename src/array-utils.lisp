(in-package #:bops)

(defun simple-array-vector (array)
  (declare (simple-array array))
  (if (sb-kernel:array-header-p array)
      (sb-kernel:%array-data-vector array)
      array))

(defun make-random-bit-vector (dims &key (probability-one 0.5f0))
  "Creates a new bit array of the given dimensions.
Elements are initialized randomly."
  (let* ((data (make-array dims :element-type 'bit))
         (displaced-data (aops:flatten data)))
    (map-into displaced-data
              (lambda (b)
                (declare (ignore b))
                (if (<= (random 1.0f0)
                        probability-one)
                    1
                    0))
              displaced-data)
    data))

(defun make-random-bias-vector (dims extent)
  (aops:generate* 'fixnum (lambda () (- (random extent))) dims))

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
