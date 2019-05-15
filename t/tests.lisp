;;;; (asdf:test-system 'bops)

(in-package :bops-tests)

(fiveam:def-suite all-tests
    :description "bops test suite.")

(fiveam:in-suite all-tests)

;; some tests datas
(defparameter *a1*
  ;;|low         64-bits value=1                               high||low         64-bits value=3                               high|
  #*10000000000000000000000000000000000000000000000000000000000000001100000000000000000000000000000000000000000000000000000000000000)

(defparameter *a1-v-u64* #(1 3))

(defparameter *a2*
  ;;|low         64-bits value=9223372036854775809             high||low         64-bits value=9223372036854775811             high|
  #*10000000000000000000000000000000000000000000000000000000000000011100000000000000000000000000000000000000000000000000000000000001)

(defparameter *a2-v-u64* #(9223372036854775809 9223372036854775811))

(fiveam:test test-bit-and-vec
  (dotimes (i 100)
    (let ((N (* (+ i 1) 1024)))
      (dotimes (j 10)
        (let ((a1 (bops:make-random-bit-vector N))
              (a2 (bops:make-random-bit-vector N))
              (res (make-array N :element-type 'bit))
              (ref (make-array N :element-type 'bit)))
          (bops:bit-and-vec a1 a2 res (/ N 8))
          (bit-and a1 a2 ref)
          (fiveam:is (equal res ref)))))))

(fiveam:test test-sap-ref-u64
  (sb-sys:with-pinned-objects (*a1*)
    (let* ((p (sb-sys:vector-sap *a1*))
           (v0 (bops::%sap-ref-u64 p 0))
           (v1 (bops::%sap-ref-u64 p 8)))
      (fiveam:is (= v0 (aref *a1-v-u64* 0)))
      (fiveam:is (= v1 (aref *a1-v-u64* 1)))))

  (sb-sys:with-pinned-objects (*a2*)
    (let* ((p (sb-sys:vector-sap *a2*))
           (v0 (bops::%sap-ref-u64 p 0))
           (v1 (bops::%sap-ref-u64 p 8)))
      (fiveam:is (= v0 (aref *a2-v-u64* 0)))
      (fiveam:is (= v1 (aref *a2-v-u64* 1))))))

(defun run-tests ()
  (fiveam:run! 'all-tests))
