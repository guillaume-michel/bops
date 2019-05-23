(in-package :bops)

(defun is-multiple-of-64 (x)
  (declare (type fixnum x)
           (optimize (speed 3) (debug 0) (safety 0)))
  (= (mod x 64) 0))
