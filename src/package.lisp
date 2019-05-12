(uiop:define-package #:bops
  (:use #:cl :sb-ext :sb-c)
  (:import-from :sb-sys
                :system-area-pointer)
  (:import-from :sb-assem
                :inst)
  (:import-from :sb-vm
                :unsigned-reg
                :sap-reg
                :unsigned-num
                :single-sse-reg)
  (:import-from :sb-x86-64-asm
                :movaps
                :make-ea
                :divps
                :addps)
  (:import-from :sb-c
                :move)
  (:export #:bit-and-vec
           #:test
           #:test2))

(uiop:define-package #:bops2
  (:use #:cl :sb-ext :sb-c)
  (:import-from :sb-sys
                :system-area-pointer)
  (:import-from :sb-assem
                :inst)
  (:import-from :sb-vm
                :unsigned-reg
                :sap-reg
                :unsigned-num
                :single-sse-reg)
  (:import-from :sb-x86-64-asm
                :movaps
                :make-ea
                :divps
                :addps)
  (:import-from :sb-c
                :move)
  (:export #:dense
           #:test
           #:test2))

(uiop:define-package #:bops3
  (:use #:cl :sb-ext :sb-c)
  (:import-from :sb-sys
                :system-area-pointer)
  (:import-from :sb-assem
                :inst)
  (:import-from :sb-vm
                :unsigned-reg
                :sap-reg
                :unsigned-num
                :single-sse-reg)
  (:import-from :sb-x86-64-asm
                :movaps
                :make-ea
                :divps
                :addps)
  (:import-from :sb-c
                :move)
  (:export #:dense
           #:test
           #:test2))
