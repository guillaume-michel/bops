(in-package :bops)

(defun vec-xor (N pr px py)
  (declare (type (unsigned-byte 64) N)
           (type system-area-pointer pr px py)
           (optimize (speed 3) (debug 0) (safety 0)))
  (loop :for i of-type fixnum :below (/ N 64) :by 1 :do
       (%vec-xor (* 8 i) pr px py)))

(defun test-xor (&key (N 256))
  (let* ((repeat-count 1000)
         (x1 (make-array N :element-type 'bit :initial-element 1))
         (x2 (make-array N :element-type 'bit :initial-element 0))
         (r (make-array N :element-type 'bit :initial-element 0)))
    (sb-sys:with-pinned-objects (r x1 x2)
      (let ((pr (sb-sys:vector-sap r))
            (px1 (sb-sys:vector-sap x1))
            (px2 (sb-sys:vector-sap x2)))
        (time (dotimes (count repeat-count) (vec-xor N pr px1 px2)))))
    (equalp r
            (make-array N :element-type 'bit :initial-element 1))))
